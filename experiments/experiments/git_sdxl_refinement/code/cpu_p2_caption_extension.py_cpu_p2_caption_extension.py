"""P2 Caption 指标扩展：用官方 all_git_generated_captions.pt（pscotti/mindeyev2，55,125 B，
SHA256 adfd766b...，git blob 8dca9586...）补齐 MindEye2 三项式中此前 BLOCKED 的两族：

  *_brain_img : all_predcaptions（脑→GIT）  vs all_git_generated_captions（GIT 看图）
  *_img_ref   : all_git_generated_captions  vs all_captions（GT）
  *_relative  : brain_img / img_ref（官方 cell 42/44/46/50/52 定义）

实现与 cpu_p2_caption.py 完全一致（nltk meteor / google rouge_score / MiniLM / CLIP-B / CLIP-L 文本塔）。
不重跑 GIT，不重新生成 Caption。输出写入新目录，不覆盖任何既有结果。
"""
import gc
import json
import os

import numpy as np
import torch

os.environ["HF_HOME"] = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/hf_home"
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
S1 = f"{ROOT}/run/full_stage1"
GTD = f"{ROOT}/assets/ground_truth"
OUT = f"{ROOT}/evaluation/cpu_mindeye2_p2_caption_extension"
M = "s1_textalign_stage1_FINAL_BEST_32"
os.makedirs(OUT, exist_ok=True)

pred = [str(x) for x in torch.load(f"{S1}/{M}_all_predcaptions.pt", map_location="cpu", weights_only=False)]
gt = [str(x) for x in torch.load(f"{GTD}/all_captions.pt", map_location="cpu", weights_only=False)]
imgc = [str(x) for x in torch.load(f"{GTD}/all_git_generated_captions.pt", map_location="cpu", weights_only=False)]
ids = [int(x) for x in torch.load(f"{S1}/{M}_all_ids.pt", map_location="cpu", weights_only=False)]
assert len(pred) == len(gt) == len(imgc) == len(ids) == 1000, (len(pred), len(gt), len(imgc))

res = {"n": 1000, "provenance": {
    "all_git_generated_captions.pt": {
        "source": "https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/evals/all_git_generated_captions.pt",
        "bytes": 55125,
        "git_blob_oid_sha1": "8dca9586fcbfef4c6b8cbc119c857e99555c6e5d",
        "local_sha256": "adfd766b6b17bfdab3b5170e86f2b5a2b0e039521c32cedb42489583d5d2e64a",
    },
    "note": "GIT 看图 caption 来自 MindEye2 官方；未重跑 GIT，未重新生成 Caption"},
    "metrics": {}, "relative": {}}

# ---------- ROUGE / METEOR（逐对三方向） ----------
from rouge_score import rouge_scorer  # noqa: E402

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge_mean(ps, rs, key):
    vals = [scorer.score(r, p)[key].fmeasure for p, r in zip(ps, rs)]
    return round(float(np.mean(vals)), 6)


import nltk  # noqa: E402
from nltk.translate.meteor_score import meteor_score  # noqa: E402


def meteor_mean(ps, rs):
    vals = [meteor_score([nltk.word_tokenize(r.lower())], nltk.word_tokenize(p.lower()))
            for p, r in zip(ps, rs)]
    return round(float(np.mean(vals)), 6)


for name, (ps, rs) in {"brain_ref": (pred, gt), "brain_img": (pred, imgc), "img_ref": (imgc, gt)}.items():
    res["metrics"][f"rouge1_{name}"] = rouge_mean(ps, rs, "rouge1")
    res["metrics"][f"rougeL_{name}"] = rouge_mean(ps, rs, "rougeL")
    res["metrics"][f"meteor_{name}"] = meteor_mean(ps, rs)
    print(name, "rouge1", res["metrics"][f"rouge1_{name}"], "rougeL", res["metrics"][f"rougeL_{name}"],
          "meteor", res["metrics"][f"meteor_{name}"], flush=True)

# ---------- MiniLM / CLIP-B / CLIP-L 文本相似度 ----------
from transformers import AutoModel, AutoTokenizer, CLIPTextModelWithProjection  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def sent_embeds(texts, batch=32):
    src = "/mnt/data/cpu_models/minilm"
    tok = AutoTokenizer.from_pretrained(src)
    mdl = AutoModel.from_pretrained(src).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            tk = tok(texts[i:i + batch], padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = mdl(**tk).last_hidden_state
            mask = tk["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(F.normalize(emb, dim=-1))
    del mdl
    gc.collect()
    return torch.cat(out, 0)


def clip_text_embeds(texts, name, local_dir=None, batch=64):
    tok = AutoTokenizer.from_pretrained(local_dir or name, use_fast=False)
    mdl = CLIPTextModelWithProjection.from_pretrained(local_dir or name).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            tk = tok(texts[i:i + batch], padding=True, truncation=True, max_length=77, return_tensors="pt")
            outs.append(F.normalize(mdl(**tk).text_embeds, dim=-1))
    del mdl
    gc.collect()
    return torch.cat(outs, 0)


def diag(a, b):
    return round(float((a * b).sum(1).mean()), 6)


EP_S, ES_S, EG_S = sent_embeds(pred), sent_embeds(imgc), sent_embeds(gt)
res["metrics"]["sentence_brain_img"] = diag(EP_S, ES_S)
res["metrics"]["sentence_img_ref"] = diag(ES_S, EG_S)

EP_B = clip_text_embeds(pred, "openai/clip-vit-base-patch32", local_dir="/mnt/data/cpu_models/clipb")
ES_B = clip_text_embeds(imgc, "openai/clip-vit-base-patch32", local_dir="/mnt/data/cpu_models/clipb")
EG_B = clip_text_embeds(gt, "openai/clip-vit-base-patch32", local_dir="/mnt/data/cpu_models/clipb")
res["metrics"]["clip_b_brain_img"] = diag(EP_B, ES_B)
res["metrics"]["clip_b_img_ref"] = diag(ES_B, EG_B)

EP_L = clip_text_embeds(pred, "openai/clip-vit-large-patch14")
ES_L = clip_text_embeds(imgc, "openai/clip-vit-large-patch14")
EG_L = clip_text_embeds(gt, "openai/clip-vit-large-patch14")
res["metrics"]["clip_l_brain_img"] = diag(EP_L, ES_L)
res["metrics"]["clip_l_img_ref"] = diag(ES_L, EG_L)

for fam in ("rouge1", "rougeL", "meteor", "sentence", "clip_b", "clip_l"):
    res["relative"][f"{fam}_relative"] = round(
        res["metrics"][f"{fam}_brain_img"] / res["metrics"][f"{fam}_img_ref"], 6)

with open(f"{OUT}/caption_extension_summary.json", "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=False)
print(json.dumps(res["metrics"], ensure_ascii=False, indent=1))
print(json.dumps(res["relative"], ensure_ascii=False, indent=1))
print("CAPTION_EXTENSION_DONE")
