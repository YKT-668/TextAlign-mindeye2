#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据 LLM / RAG 生成的 structured_prompts.json，
把每个样本的 positive 文本编码为 CLIP 文本特征，作为“语义教师”向量。
"""

import os
import json
import argparse

import torch
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_json", required=True,
                        help="结构化提示 JSON 文件，例如 runs/.../llm_prompts/structured_prompts.json")
    parser.add_argument("--out_pt", required=True,
                        help="输出的教师特征 .pt 文件路径")
    parser.add_argument("--out_ids", default="",
                        help="可选，输出对应 id 列表的 .json 文件路径")
    parser.add_argument("--device", default="cuda",
                        help="cuda 或 cpu")
    parser.add_argument("--model_name", default="openai/clip-vit-large-patch14",
                        help="CLIP 文本编码模型名")
    args = parser.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[teacher] device = {device}")

    # 1) 读取 structured_prompts.json
    print(f"[teacher] loading prompts from: {args.prompts_json}")
    with open(args.prompts_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data 一般是 list[ { "id": int, "positive": str, "negative": str }, ... ]
    texts = []
    ids = []
    for rec in data:
        pos = rec.get("positive", "")
        if not isinstance(pos, str) or len(pos.strip()) == 0:
            # 空的就跳过
            continue
        texts.append(pos.strip())
        ids.append(rec.get("id"))

    print(f"[teacher] 有效样本数: {len(texts)}")

    if len(texts) == 0:
        raise RuntimeError("structured_prompts.json 中没有有效的 positive 文本，无法构建教师特征。")

    # 2) 构建 CLIP 文本编码器
    print(f"[teacher] loading CLIP text encoder: {args.model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    model.eval()

    # 3) 分批编码文本
    all_feats = []
    bs = 32
    for i in tqdm(range(0, len(texts), bs), desc="[teacher] encoding texts"):
        batch_texts = texts[i:i + bs]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**enc)  # [B, D]
            feats = feats / feats.norm(dim=-1, keepdim=True)  # 归一化一下，方便后面用余弦
        all_feats.append(feats.cpu())

    all_feats = torch.cat(all_feats, dim=0)  # [N, D]
    print(f"[teacher] 最终教师特征形状: {tuple(all_feats.shape)}")

    # 4) 保存
    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    torch.save(all_feats, args.out_pt)
    print(f"[teacher] saved teacher features to: {args.out_pt}")

    if args.out_ids:
        os.makedirs(os.path.dirname(args.out_ids), exist_ok=True)
        with open(args.out_ids, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)
        print(f"[teacher] saved ids to: {args.out_ids}")


if __name__ == "__main__":
    main()
