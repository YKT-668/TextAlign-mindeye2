#!/usr/bin/env python3
# coding: utf-8
"""Prepare + encode TextAlign teacher features for MindEye2 subj01 training using HuggingFace datasets.

This is a self-contained replacement for the old 2-step pipeline:
  1) tools/prepare_nsd_coco_captions.py (needs local COCO annotation json)
  2) tools/encode_train_coco_text_clip.py (encodes captions to CLIP text feats)

It:
  - scans WebDataset shards under src/wds/subj01/train/*.tar to collect unique image IDs (behav[:,0,0])
  - downloads COCO captions (2017) via HF datasets and builds image_id -> caption mapping
  - encodes captions with openai/clip-vit-large-patch14
  - writes data/nsd_text/train_coco_text_clip.pt compatible with Train_textalign.py

Output format:
  {
    "image_ids": LongTensor[N],
    "text_feats": FloatTensor[N, 768]
  }

Optional:
  - also saves data/nsd_text/train_coco_captions.json for debugging

Usage:
  python tools/prepare_and_encode_train_coco_text_clip_hf.py --subj 1

Env:
  HF_HOME / TRANSFORMERS_CACHE are respected by HF libraries.
"""

import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import webdataset as wds


def _proj_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


def collect_train_image_ids(data_path: str, subj: int) -> list[int]:
    wds_root = os.path.join(data_path, f"wds/subj0{subj}/train")
    shards = sorted(glob.glob(os.path.join(wds_root, "*.tar")))
    if not shards:
        raise FileNotFoundError(f"No shards found under {wds_root}")

    ids_set: set[int] = set()
    first = True
    for shard in shards:
        ds = (
            wds.WebDataset(shard)
            .decode("torch")
            .rename(behav="behav.npy")
            .to_tuple("behav")
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=512)
        for (behav,) in dl:
            arr = behav.numpy()
            if first:
                print(f"[WDS] behav.shape={arr.shape} dtype={arr.dtype}")
                first = False
            if arr.ndim == 3:
                col0 = arr[:, 0, 0]
            elif arr.ndim == 2:
                col0 = arr[:, 0]
            else:
                raise RuntimeError(f"Unexpected behav.ndim={arr.ndim}")

            for v in col0:
                vv = np.array(v)
                while vv.ndim > 0:
                    vv = vv.flat[0]
                ids_set.add(int(vv))

    ids = sorted(ids_set)
    print(f"[WDS] unique train image ids: {len(ids)} (min={ids[0]} max={ids[-1]})")
    return ids


def load_stiminfo_cocoids(pkl_path: str) -> np.ndarray:
    obj = pickle.load(open(pkl_path, "rb"), encoding="latin1")
    if hasattr(obj, "columns") and "cocoId" in obj.columns:
        coco = obj["cocoId"].to_numpy()
    elif isinstance(obj, dict) and "cocoId" in obj:
        coco = np.asarray(obj["cocoId"])
    else:
        coco = np.asarray(getattr(obj, "cocoId"))
    return coco.astype(np.int64)


def load_coco_captions_hf(local_ann_dir: str = None) -> dict[int, str]:
    """Return mapping: coco image_id -> one caption string (first seen).

    NOTE: This function is written to be robust on servers where network
    access is slow/spotty. It streams rows and prints progress periodically
    (instead of materializing the whole dataset into RAM).
    """
    # Prefer local COCO annotations if available (faster, deterministic, no HF config drift).
    ann_dir = local_ann_dir
    if ann_dir is None:
        root = _proj_root()
        ann_dir = os.path.join(root, "data", "coco_annotations", "annotations")
    train_json = os.path.join(ann_dir, "captions_train2017.json")
    val_json = os.path.join(ann_dir, "captions_val2017.json")
    if os.path.isfile(train_json) and os.path.isfile(val_json):
        img2cap: dict[int, str] = {}
        for p in (train_json, val_json):
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            for ann in obj.get("annotations", []):
                try:
                    image_id = int(ann.get("image_id"))
                    cap = str(ann.get("caption", "")).strip()
                except Exception:
                    continue
                if cap and image_id not in img2cap:
                    img2cap[image_id] = cap
        if img2cap:
            print(f"[COCO] loaded local annotations from {ann_dir}, mappings={len(img2cap)}")
            return img2cap

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from e

    # Try a few known dataset/config combos.
    # NOTE: 'coco_captions' is not always available on the hub.
    candidates = [
        ("HuggingFaceM4/COCO", "2017_captions"),
        ("HuggingFaceM4/COCO", "captions"),
    ]

    last_err = None
    for name, config in candidates:
        try:
            print(f"[HF] loading dataset={name} config={config}")
            ds_train = load_dataset(name, config, split="train")
            ds_val = None
            for vs in ("validation", "val"):
                try:
                    ds_val = load_dataset(name, config, split=vs)
                    break
                except Exception:
                    pass

            # Stream rows instead of converting to list (prevents large RAM usage)
            def _iter_all():
                for r in ds_train:
                    yield r
                if ds_val is not None:
                    for r in ds_val:
                        yield r

            # Heuristics to extract keys.
            # Common formats:
            #  - {"image_id": int, "caption": str}
            #  - {"image_id": int, "captions": [..]} etc.
            img2cap: dict[int, str] = {}
            seen = 0
            for r in _iter_all():
                if not isinstance(r, dict):
                    continue
                seen += 1
                if seen % 200000 == 0:
                    print(f"[HF] scanned {seen} rows, mappings={len(img2cap)}")
                image_id = None
                if "image_id" in r:
                    try:
                        image_id = int(r["image_id"])
                    except Exception:
                        image_id = None
                elif "img_id" in r:
                    try:
                        image_id = int(r["img_id"])
                    except Exception:
                        image_id = None

                if image_id is None:
                    continue

                cap = None
                if "caption" in r and isinstance(r["caption"], str):
                    cap = r["caption"].strip()
                elif "captions" in r and isinstance(r["captions"], (list, tuple)) and r["captions"]:
                    # take first non-empty
                    for c in r["captions"]:
                        if isinstance(c, str) and c.strip():
                            cap = c.strip()
                            break

                if cap and (image_id not in img2cap):
                    img2cap[image_id] = cap

            if not img2cap:
                raise RuntimeError("Loaded dataset but could not parse any (image_id, caption) rows")

            print(f"[HF] parsed {len(img2cap)} unique image_id -> caption mappings")
            return img2cap
        except Exception as e:
            last_err = e
            print(f"[HF][fail] {name}/{config}: {e}")

    raise RuntimeError(f"All HF dataset attempts failed: {last_err}")


def encode_clip_text(captions: list[str], device: str, batch_size: int) -> torch.Tensor:
    from transformers import CLIPTokenizer, CLIPModel

    model_name = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval().requires_grad_(False)

    feats = []
    for start in range(0, len(captions), batch_size):
        end = min(start + batch_size, len(captions))
        batch_caps = captions[start:end]
        inputs = tokenizer(batch_caps, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            f = model.get_text_features(**inputs)
        feats.append(f.detach().cpu())
        if (start // batch_size) % 20 == 0:
            print(f"[ENC] {end}/{len(captions)}")

    return torch.cat(feats, dim=0).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument("--data_path", type=str, default=None, help="MindEye2 data_path (the folder that contains wds/...). Default: <proj>/src")
    ap.add_argument("--out_json", action="store_true", help="Also write data/nsd_text/train_coco_captions.json")
    ap.add_argument("--stiminfo", type=str, default=None,
                    help="Path to nsd_stim_info_merged.pkl. Default: <proj>/nsd_stim_info_merged.pkl")
    ap.add_argument("--out_json_path", type=str, default=None,
                    help="Explicit output path for caption json (e.g. data/nsd_text/s2_train_coco_captions.json)")
    ap.add_argument("--out_pt_path", type=str, default=None,
                    help="Explicit output path for teacher pt (e.g. data/nsd_text/s2_train_coco_text_clip.pt)")
    ap.add_argument("--batch_size", type=int, default=256, help="CLIP text encoding batch size")
    args = ap.parse_args()

    root = _proj_root()
    data_path = args.data_path or os.path.join(root, "src")
    stiminfo_path = args.stiminfo or os.path.join(root, "nsd_stim_info_merged.pkl")
    out_dir = os.path.join(root, "data", "nsd_text")
    os.makedirs(out_dir, exist_ok=True)

    train_ids = collect_train_image_ids(data_path=data_path, subj=int(args.subj))
    cocoids = load_stiminfo_cocoids(stiminfo_path)
    print(f"[STIMINFO] cocoId shape={tuple(cocoids.shape)} min={int(cocoids.min())} max={int(cocoids.max())}")
    img2cap = load_coco_captions_hf()

    image_ids_out = []
    captions_out = []
    missed = []
    for iid in train_ids:
        if iid < 0 or iid >= len(cocoids):
            missed.append(int(iid))
            continue
        coco_id = int(cocoids[int(iid)])
        cap = img2cap.get(coco_id)
        if cap is None:
            missed.append(int(iid))
            continue
        image_ids_out.append(int(iid))
        captions_out.append(cap)

    print(f"[MAP] matched {len(image_ids_out)}/{len(train_ids)} train ids; missed={len(missed)}")
    if missed:
        print("[MAP] first 20 missed:", missed[:20])

    json_path = args.out_json_path or os.path.join(out_dir, "train_coco_captions.json")
    if args.out_json or args.out_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"image_ids": image_ids_out, "captions": captions_out}, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {json_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CLIP] device={device}")

    text_feats = encode_clip_text(captions_out, device=device, batch_size=int(args.batch_size))
    print(f"[ENC] text_feats shape={tuple(text_feats.shape)}")

    out_pt = args.out_pt_path or os.path.join(out_dir, "train_coco_text_clip.pt")
    os.makedirs(os.path.dirname(os.path.abspath(out_pt)), exist_ok=True)
    torch.save({"image_ids": torch.tensor(image_ids_out, dtype=torch.long), "text_feats": text_feats}, out_pt)
    print(f"[SAVE] {out_pt}")


if __name__ == "__main__":
    main()
