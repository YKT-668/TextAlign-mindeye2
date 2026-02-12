
#!/usr/bin/env python3
# coding: utf-8
import os, glob, json, pickle
from collections import defaultdict

import numpy as np
import torch
import webdataset as wds

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def load_stiminfo_cocoids(pkl_path):
    obj = pickle.load(open(pkl_path, "rb"), encoding='latin1')
    # common cases: pandas DataFrame, dict-like
    if hasattr(obj, "columns") and "cocoId" in obj.columns:
        coco = obj["cocoId"].to_numpy()
    elif isinstance(obj, dict) and "cocoId" in obj:
        coco = np.asarray(obj["cocoId"])
    else:
        # try attribute
        coco = np.asarray(getattr(obj, "cocoId"))
    coco = coco.astype(np.int64)
    return coco

def load_coco_caption_map(train_json, val_json):
    id2cap = {}
    for p in [train_json, val_json]:
        j = json.load(open(p, "r"))
        for a in j.get("annotations", []):
            iid = int(a["image_id"])
            if iid not in id2cap:
                cap = str(a.get("caption", "")).strip()
                if cap:
                    id2cap[iid] = cap
    return id2cap

def collect_train_local_ids(data_path, subj):
    wds_root = os.path.join(data_path, f"wds/subj0{subj}/train")
    shards = sorted(glob.glob(os.path.join(wds_root, "*.tar")))
    if not shards:
        raise FileNotFoundError(f"No shards under: {wds_root}")
    ids = set()
    first = True
    for shard in shards:
        ds = wds.WebDataset(shard).decode("torch").rename(behav="behav.npy").to_tuple("behav")
        dl = torch.utils.data.DataLoader(ds, batch_size=512)
        for (behav,) in dl:
            arr = behav.numpy()
            if first:
                print("[WDS] behav.shape=", arr.shape, "dtype=", arr.dtype)
                first = False
            col0 = arr[:, 0, 0] if arr.ndim == 3 else arr[:, 0]
            for v in col0:
                vv = np.array(v)
                while vv.ndim > 0:
                    vv = vv.flat[0]
                ids.add(int(vv))
    out = sorted(ids)
    print(f"[WDS] unique train local_ids: {len(out)} (min={out[0]} max={out[-1]})")
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument("--data_path", type=str, required=True, help="folder containing wds/...")
    ap.add_argument("--stiminfo", type=str, default=os.path.join(ROOT, "nsd_stim_info_merged.pkl"))
    ap.add_argument("--coco_train", type=str, default=os.path.join(ROOT, "data/coco_annotations/annotations/captions_train2017.json"))
    ap.add_argument("--coco_val", type=str, default=os.path.join(ROOT, "data/coco_annotations/annotations/captions_val2017.json"))
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "data/nsd_text/train_coco_captions.json"))
    args = ap.parse_args()

    cocoids = load_stiminfo_cocoids(args.stiminfo)
    print("[STIMINFO] cocoId array:", cocoids.shape, "min/max:", int(cocoids.min()), int(cocoids.max()))

    id2cap = load_coco_caption_map(args.coco_train, args.coco_val)
    print("[COCO] id2cap size:", len(id2cap))

    local_ids = collect_train_local_ids(args.data_path, args.subj)

    out_ids, out_caps, missed = [], [], []
    for lid in local_ids:
        if lid < 0 or lid >= len(cocoids):
            missed.append(lid); continue
        coco_id = int(cocoids[lid])
        cap = id2cap.get(coco_id, None)
        if cap is None:
            missed.append(lid); continue
        out_ids.append(int(lid))
        out_caps.append(cap)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"image_ids": out_ids, "captions": out_caps}, open(args.out, "w"), ensure_ascii=False)
    print(f"[SAVE] {args.out}")
    print(f"[MAP] matched {len(out_ids)}/{len(local_ids)}; missed={len(missed)}")
    if missed:
        print("[MAP] missed local_id head20:", missed[:20])

if __name__ == "__main__":
    main()

