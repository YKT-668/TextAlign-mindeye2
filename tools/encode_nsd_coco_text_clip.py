#!/usr/bin/env python
# coding: utf-8
"""
encode_nsd_coco_text_clip.py

功能：
- 读取 data/nsd_text/train_coco_captions.json
- 使用 openai/clip-vit-large-patch14 的文本编码器，将 caption 编码为向量
- 输出 data/nsd_text/coco_text_clip.pt:
  {
    "image_ids": LongTensor[N],
    "text_feats": FloatTensor[N, d_text]
  }
"""

import os
import json

import torch
from transformers import CLIPTokenizer, CLIPModel


def find_project_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


def main():
    proj_root = find_project_root()
    data_dir = os.path.join(proj_root, "data")
    in_path = os.path.join(data_dir, "nsd_text", "train_coco_captions.json")

    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"{in_path} 不存在，请先运行 tools/prepare_nsd_coco_captions.py"
        )

    with open(in_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    image_ids = [int(x) for x in obj["image_ids"]]
    captions = obj["captions"]
    assert len(image_ids) == len(captions), "image_ids 与 captions 长度不一致"

    print(f"[LOAD] 从 {in_path} 读到 {len(image_ids)} 对 (image_id, caption)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CLIP] device = {device}")
    model_name = "openai/clip-vit-large-patch14"

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval().requires_grad_(False)

    all_feats = []
    bs = 64
    total = len(captions)
    for start in range(0, total, bs):
        end = min(start + bs, total)
        batch_caps = captions[start:end]
        inputs = tokenizer(
            batch_caps,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)  # [B, d_text]
        all_feats.append(feats.cpu())
        if (start // bs) % 50 == 0:
            print(f"[ENC] {start}/{total}")

    all_feats = torch.cat(all_feats, dim=0)
    print(f"[ENC] all_feats shape = {all_feats.shape}")  # 期望 [N, 768]

    out_dir = os.path.join(data_dir, "nsd_text")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "coco_text_clip.pt")

    torch.save(
        {
            "image_ids": torch.tensor(image_ids, dtype=torch.long),
            "text_feats": all_feats,
        },
        out_path,
    )
    print(f"[SAVE] wrote {out_path}")


if __name__ == "__main__":
    main()
