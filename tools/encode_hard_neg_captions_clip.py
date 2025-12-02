#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
把 LLM 生成的 hard negative captions 编成 CLIP 文本向量。

输入：
  - data/nsd_text/train_coco_captions.json
      {"image_ids": [...], "captions": [...]}
  - data/nsd_text/train_coco_captions_hard_negs.jsonl
      每行一个 JSON:
        {"image_id": 71, "pos_caption": "...", "neg_caption": "..."}

输出：
  - data/nsd_text/train_coco_captions_hard_negs_clip.pt
      {
        "image_ids": LongTensor [1910],
        "neg_captions": List[str] 长度 1910,
        "neg_text_feats": FloatTensor [1910, 768]
      }
"""

import json
from pathlib import Path

import torch
from transformers import CLIPTokenizer, CLIPTextModel


ROOT = Path(__file__).resolve().parents[1]  # MindEyeV2_Project 根目录
pos_json = ROOT / "data/nsd_text/train_coco_captions.json"
hard_jsonl = ROOT / "data/nsd_text/train_coco_captions_hard_negs.jsonl"
out_pt = ROOT / "data/nsd_text/train_coco_captions_hard_negs_clip.pt"


def load_pos():
    obj = json.loads(pos_json.read_text())
    image_ids = obj["image_ids"]
    captions = obj["captions"]
    assert len(image_ids) == len(captions), "image_ids 和 captions 长度不一致"
    id2idx = {int(i): idx for idx, i in enumerate(image_ids)}
    print(f"[INFO] 正样本: {len(image_ids)} 条")
    return image_ids, captions, id2idx


def load_hard_negs(id2idx, n_total):
    neg_captions = [None] * n_total

    n_lines = 0
    with hard_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img_id = int(obj["image_id"])
            neg = obj["neg_caption"]
            if img_id not in id2idx:
                # 理论上不应该出现
                print(f"[WARN] image_id {img_id} 不在正样本里，忽略")
                continue
            idx = id2idx[img_id]
            neg_captions[idx] = neg
            n_lines += 1

    none_cnt = sum(1 for x in neg_captions if x is None)
    print(f"[INFO] 从 hard_neg jsonl 读到 {n_lines} 行，其中缺失 {none_cnt} 条")

    # 安全检查：不允许 None
    if none_cnt > 0:
        missing_idx = [i for i, x in enumerate(neg_captions) if x is None][:10]
        raise RuntimeError(
            f"有 {none_cnt} 条负样本缺失，比如 index={missing_idx}，请检查 jsonl。"
        )

    return neg_captions


def build_clip_text_encoder(device="cuda"):
    # ⚠️ 这里要用“和你之前生成 train_coco_text_clip.pt 同一个 CLIP 文本模型”
    # 如果之前用的是 openai/clip-vit-large-patch14，那么 text_dim=768，和 TextAlign 一致
    model_name = "openai/clip-vit-large-patch14"

    print(f"[INFO] 加载 CLIP 文本模型: {model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name)
    text_model.to(device)
    text_model.eval()
    for p in text_model.parameters():
        p.requires_grad_(False)
    return tokenizer, text_model


@torch.no_grad()
def encode_texts(texts, tokenizer, text_model, device="cuda", batch_size=64):
    all_feats = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        out = text_model(**enc)
        # 通常用 pooled output 作为句向量（与你原先 768 维 teacher 一致）
        feats = out.pooler_output  # [B, hidden_dim]
        all_feats.append(feats.cpu())
        print(f"[ENC] {i}/{n} ...")
    return torch.cat(all_feats, dim=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] 读取正样本 JSON:", pos_json)
    image_ids, captions_pos, id2idx = load_pos()

    print("[INFO] 读取 hard negative JSONL:", hard_jsonl)
    neg_captions = load_hard_negs(id2idx, len(image_ids))

    print("[INFO] 示例：")
    for k in range(3):
        print("  image_id =", image_ids[k])
        print("    POS:", captions_pos[k])
        print("    NEG:", neg_captions[k])

    tokenizer, text_model = build_clip_text_encoder(device=device)

    print("[INFO] 开始编码 NEG captions 为 CLIP 文本向量 ...")
    neg_feats = encode_texts(neg_captions, tokenizer, text_model, device=device, batch_size=64)
    print("[INFO] 编码完成，neg_feats.shape =", neg_feats.shape)

    out_obj = {
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "neg_captions": neg_captions,
        "neg_text_feats": neg_feats,  # [1910, 768]
    }
    torch.save(out_obj, out_pt)
    print("[DONE] 保存到:", out_pt)


if __name__ == "__main__":
    main()
