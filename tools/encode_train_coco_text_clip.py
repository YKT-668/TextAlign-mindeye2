# tools/encode_train_coco_text_clip.py

import os
import json
import time
import torch
from transformers import CLIPTokenizer, CLIPModel


def main():
    t0 = time.time()
    proj_root = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.dirname(proj_root)  # 上一级，项目根目录

    json_path = os.path.join(proj_root, "data/nsd_text/train_coco_captions.json")
    print(f"[LOAD] {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # ---- 这里做兼容：既支持 nsd_ids 也支持 image_ids ----
    if "nsd_ids" in obj:
        nsd_ids = obj["nsd_ids"]
        print("[INFO] using key 'nsd_ids' from json")
    elif "image_ids" in obj:
        nsd_ids = obj["image_ids"]
        print("[INFO] using key 'image_ids' from json")
    else:
        print(f"[ERROR] json keys = {list(obj.keys())}")
        raise KeyError("train_coco_captions.json 里既没有 'nsd_ids' 也没有 'image_ids'，请检查保存脚本。")

    captions = obj.get("captions", None)
    if captions is None:
        print(f"[ERROR] json keys = {list(obj.keys())}")
        raise KeyError("train_coco_captions.json 里没有 'captions' 字段。")

    if len(nsd_ids) != len(captions):
        raise RuntimeError(
            f"nsd_ids 长度 {len(nsd_ids)} 与 captions 长度 {len(captions)} 不一致，请检查。"
        )

    print(f"[INFO] num pairs = {len(nsd_ids)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CLIP] using device = {device}")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    model.eval().requires_grad_(False)

    all_feats = []
    bs = int(os.environ.get("CLIP_TEXT_BS", "256"))
    n = len(captions)

    print(f"[ENCODE] batch_size = {bs}")

    for i in range(0, n, bs):
        batch_caps = captions[i:i + bs]
        inputs = tokenizer(
            batch_caps,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            feats = model.get_text_features(**inputs)  # [B, d_text]
        all_feats.append(feats.cpu())

        if (i // bs) % 10 == 0 or (i + bs) >= n:
            done = min(i + bs, n)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            print(f"[ENCODE] {done}/{n} | elapsed={elapsed:.1f}s | eta={eta/60:.1f}m")

    all_feats = torch.cat(all_feats, dim=0)  # [N, d_text]
    print(f"[ENCODE] done, feats shape = {all_feats.shape}")

    out_dir = os.path.join(proj_root, "data/nsd_text")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "train_coco_text_clip.pt")

    torch.save(
        {
            "image_ids": torch.tensor(nsd_ids, dtype=torch.long),
            "text_feats": all_feats,
        },
        out_path,
    )
    print(f"[SAVE] wrote {out_path}")
    print(f"[DONE] total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
