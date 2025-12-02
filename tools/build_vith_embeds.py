#!/usr/bin/env python3
# tools/build_vith_embeds.py
import argparse, os, json, sys, math, gc
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import h5py

try:
    import open_clip
except Exception as e:
    print("[FATAL] open_clip not found. Please `pip install open-clip-torch`.", file=sys.stderr)
    raise

def load_images_from_dir(img_dir, limit=None):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.lower().endswith(exts)]
    if limit is not None:
        paths = paths[:limit]
    return paths

def load_images_from_hdf5(coco_hdf5, ids_json):
    with open(ids_json, "r", encoding="utf-8") as f:
        ids = json.load(f)
    # 读取 HDF5（float16，形状应为 [N, 224, 224, 3] 或 [224,224,3]）
    h5 = h5py.File(coco_hdf5, "r")
    images = h5.get("images", None)
    if images is None:
        raise FileNotFoundError(f"'images' dataset not found in {coco_hdf5}")
    return h5, images, ids

def pil_from_float(img_arr):
    # img_arr: [H,W,3] float16/float32 in [0,1] or [0,255]
    arr = np.array(img_arr)
    if arr.max() <= 1.5:
        arr = (arr * 255.0).clip(0, 255)
    arr = arr.astype(np.uint8)
    return Image.fromarray(arr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", type=str, default=None, help="Directory of images to encode (A模式)")
    ap.add_argument("--coco_hdf5", type=str, default=None, help="Path to COCO HDF5 (B模式)")
    ap.add_argument("--ids_json", type=str, default=None, help="JSON list of indices (B模式)")
    ap.add_argument("--out_pt", type=str, required=True, help="Output tensor path, e.g., img_vith.pt")
    ap.add_argument("--paths_out", type=str, default=None, help="Optional: save list of image paths/json order")
    ap.add_argument("--model", type=str, default="ViT-H-14")
    ap.add_argument("--pretrained", type=str, default="laion2b_s32b_b79k")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="Optional: limit number of images")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()

    img_paths = []
    pil_loader = None

    modeA = args.img_dir is not None
    modeB = (args.coco_hdf5 is not None and args.ids_json is not None)

    if not (modeA or modeB):
        raise ValueError("Please provide either --img_dir (A模式) or --coco_hdf5 + --ids_json (B模式).")

    embeddings = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        if modeA:
            paths = load_images_from_dir(args.img_dir, args.limit)
            if len(paths) == 0:
                raise RuntimeError(f"No images found in {args.img_dir}")
            for p in tqdm(paths, desc="Encoding images (dir)"):
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    continue
                tensor = preprocess(im).unsqueeze(0).to(device)
                feat = model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                embeddings.append(feat.squeeze(0).float().cpu())
                img_paths.append(p)
        else:
            h5, images, ids = load_images_from_hdf5(args.coco_hdf5, args.ids_json)
            try:
                if args.limit is not None:
                    ids = ids[:args.limit]
                for idx in tqdm(ids, desc="Encoding images (HDF5)"):
                    arr = images[int(idx)]
                    im = pil_from_float(arr)
                    tensor = preprocess(im).unsqueeze(0).to(device)
                    feat = model.encode_image(tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    embeddings.append(feat.squeeze(0).float().cpu())
                    img_paths.append(f"hdf5://{args.coco_hdf5}#images[{int(idx)}]")
            finally:
                try: h5.close()
                except: pass

    E = torch.stack(embeddings, dim=0)  # [N, 1024]
    os.makedirs(os.path.dirname(args.out_pt) or ".", exist_ok=True)
    torch.save(E, args.out_pt)
    print(f"[saved] image embeds -> {args.out_pt} shape={tuple(E.shape)}")

    if args.paths_out:
        with open(args.paths_out, "w", encoding="utf-8") as f:
            json.dump(img_paths, f, ensure_ascii=False, indent=2)
        print(f"[saved] image paths -> {args.paths_out} (N={len(img_paths)})")

if __name__ == "__main__":
    main()