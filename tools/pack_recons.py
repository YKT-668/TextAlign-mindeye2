#!/usr/bin/env python
import argparse, os, json, re, glob
from PIL import Image
import torch

def guess_id(fname: str):
    # 从文件名里抓样本 id：优先下划线或连字符后的纯数字；再退而求其次抓最后一段数字
    base = os.path.basename(fname)
    for pat in [r'_(\d+)\.', r'-(\d+)\.', r'(\d+)\.']:
        m = re.search(pat, base)
        if m: return int(m.group(1))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_dir", required=True, help=".../train_logs/<model>/inference")
    ap.add_argument("--clip", default="ViT-L-14@openai", help='e.g., "ViT-H-14@laion2b_s32k" or "ViT-L-14@openai"')
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    img_dir = os.path.join(args.infer_dir, "images")
    out_pt = os.path.join(args.infer_dir, "recons.pt")
    out_ids = os.path.join(args.infer_dir, "ids.json")

    pngs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not pngs:
        raise SystemExit(f"[pack] no PNGs in {img_dir}")

    # lazy import open_clip
    import open_clip
    model_name, *rest = args.clip.split("@")
    pretrained = rest[0] if rest else "openai"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    feats, ids = [], []
    batch = []
    with torch.no_grad():
        for i, p in enumerate(pngs, 1):
            ids.append(guess_id(p))
            img = Image.open(p).convert("RGB")
            batch.append(preprocess(img).unsqueeze(0))
            if len(batch) == args.batch or i == len(pngs):
                x = torch.cat(batch, 0).to(device)
                f = model.encode_image(x)         # [B, D]
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
                batch.clear()

    feats = torch.cat(feats, 0)                  # [N, D]
    torch.save(feats, out_pt)
    with open(out_ids, "w") as f:
        json.dump(ids, f)
    print(f"[pack] saved: {out_pt} shape={tuple(feats.shape)}")
    print(f"[pack] saved: {out_ids} (parsed ids; may contain null if未能解析)")
    print(f"[pack] done.")

if __name__ == "__main__":
    main()
