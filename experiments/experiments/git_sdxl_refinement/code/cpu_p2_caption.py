"""P2 Caption 评价（MindEye2 官方口径：METEOR / ROUGE / Sentence / CLIP-B / CLIP-L）。

- A/B 共享同一批 GIT 预测 caption → 只算一套，不作为精炼效果证据
- 官方 evaluate.load('meteor'/'rouge') 底层即 nltk.meteor_score / google rouge_score（等价实现）
- CLIP-B/L 用 CLIPTextModelWithProjection（等价于 CLIPModel.get_text_features）
- 句子相似度：all-MiniLM-L6-v2 + mean pooling + L2（等价 sentence-transformers 默认 pipeline）
"""
import csv
import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
S1 = f"{ROOT}/run/full_stage1"
GTD = f"{ROOT}/assets/ground_truth"
OUT = f"{ROOT}/evaluation/cpu_mindeye2_p2_caption"
M = "s1_textalign_stage1_FINAL_BEST_32"
os.makedirs(OUT, exist_ok=True)

pred = [str(x) for x in torch.load(f"{S1}/{M}_all_predcaptions.pt", map_location="cpu", weights_only=False)]
gt = [str(x) for x in torch.load(f"{GTD}/all_captions.pt", map_location="cpu", weights_only=False)]
ids = [int(x) for x in torch.load(f"{S1}/{M}_all_ids.pt", map_location="cpu", weights_only=False)]
assert len(pred) == len(gt) == len(ids) == 1000, (len(pred), len(gt), len(ids))
print("captions ok", len(pred), "unique ids", len(set(ids)), flush=True)

res = {"n": 1000, "pairing": "row i <-> row i (same ascending ids)", "per_sample": {}}

from rouge_score import rouge_scorer  # noqa: E402

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
r1, r2, rl, rls = [], [], [], []
for p, g in zip(pred, gt):
    s = scorer.score(g, p)
    r1.append(s["rouge1"].fmeasure)
    r2.append(s["rouge2"].fmeasure)
    rl.append(s["rougeL"].fmeasure)
    rls.append(s["rougeLsum"].fmeasure)
res["rouge"] = {"rouge1": round(float(np.mean(r1)), 6), "rouge2": round(float(np.mean(r2)), 6),
                "rougeL": round(float(np.mean(rl)), 6), "rougeLsum": round(float(np.mean(rls)), 6),
                "use_stemmer": True, "aggregation": "mean F-measure"}
res["per_sample"]["rouge1"] = r1
res["per_sample"]["rougeL"] = rl
print("ROUGE", res["rouge"], flush=True)

import nltk  # noqa: E402
from nltk.translate.meteor_score import meteor_score  # noqa: E402

met = []
for p, g in zip(pred, gt):
    met.append(meteor_score([nltk.word_tokenize(g.lower())], nltk.word_tokenize(p.lower())))
res["meteor"] = {"meteor": round(float(np.mean(met)), 6)}
res["per_sample"]["meteor"] = met
print("METEOR", res["meteor"], flush=True)

from transformers import AutoModel, AutoTokenizer  # noqa: E402


def sent_embeds(texts, name="sentence-transformers/all-MiniLM-L6-v2", batch=32):
    local = "/mnt/data/cpu_models/minilm"
    src = local if os.path.exists(f"{local}/pytorch_model.bin") else name
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


SS = sent_embeds(pred)
SG = sent_embeds(gt)
ss = (SS * SG).sum(1).numpy()
res["sentence_similarity"] = {"model": "all-MiniLM-L6-v2", "pooling": "mean",
                              "diag_cos_mean": round(float(ss.mean()), 6),
                              "diag_cos_median": round(float(np.median(ss)), 6)}
res["per_sample"]["sentence"] = ss.tolist()
print("Sentence", res["sentence_similarity"], flush=True)

# ---------- CLIP-B / CLIP-L 文本相似度 ----------
from transformers import AutoTokenizer, CLIPTextModelWithProjection  # noqa: E402


def clip_text_sim(name, local_dir=None, batch=64):
    tok = AutoTokenizer.from_pretrained(local_dir or name, use_fast=False)
    mdl = CLIPTextModelWithProjection.from_pretrained(local_dir or name).eval()

    def emb(texts):
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                tk = tok(texts[i:i + batch], padding=True, truncation=True, max_length=77, return_tensors="pt")
                outs.append(F.normalize(mdl(**tk).text_embeds, dim=-1))
        return torch.cat(outs, 0)

    EP, EG = emb(pred), emb(gt)
    d = (EP * EG).sum(1).numpy()
    del mdl
    gc.collect()
    return {"model": name, "local_dir": local_dir, "diag_cos_mean": round(float(d.mean()), 6),
            "diag_cos_median": round(float(np.median(d)), 6)}, d


res["clip_b"], cb = None, None
_clipb_local = "/mnt/data/cpu_models/clipb"
_clipb_w = f"{_clipb_local}/pytorch_model.bin"
_clipb_ready = (os.path.exists(_clipb_w) and os.path.getsize(_clipb_w) == 605247071
                and not os.path.exists(_clipb_w + ".aria2"))
if _clipb_ready:
    res["clip_b"], cb = clip_text_sim("openai/clip-vit-base-patch32", local_dir=_clipb_local)
    res["per_sample"]["clip_b"] = cb.tolist()
    print("CLIP-B", res["clip_b"], flush=True)
else:
    res["clip_b"] = {"status": "PENDING_DOWNLOAD",
                     "have_bytes": os.path.getsize(_clipb_w) if os.path.exists(_clipb_w) else 0,
                     "need_bytes": 605247071,
                     "reason": "clip-vit-base-patch32 权重尚未完整落盘（单连接/分段下载受限）"}
    print("CLIP-B PENDING_DOWNLOAD", res["clip_b"], flush=True)

res["clip_l"], cl = clip_text_sim("openai/clip-vit-large-patch14")
res["per_sample"]["clip_l"] = cl.tolist()
print("CLIP-L", res["clip_l"], flush=True)

with open(f"{OUT}/caption_per_sample.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "meteor", "rouge1", "rougeL", "sentence", "clip_b", "clip_l", "pred_caption", "gt_caption"])
    for i in range(1000):
        w.writerow([ids[i], round(met[i], 6), round(r1[i], 6), round(rl[i], 6),
                    round(float(ss[i]), 6),
                    ("NA" if cb is None else round(float(cb[i]), 6)),
                    round(float(cl[i]), 6), pred[i], gt[i]])

summary = {k: v for k, v in res.items() if k != "per_sample"}
summary["status"] = {
    "meteor": "OFFICIAL_EQUIVALENT(nltk.meteor_score = evaluate.load('meteor') 底层)",
    "rouge": "OFFICIAL_EQUIVALENT(google rouge_score = evaluate.load('rouge') 底层)",
    "sentence": "OFFICIAL_EQUIVALENT(all-MiniLM-L6-v2 + mean pooling)",
    "clip_b": "OFFICIAL_REPRODUCED(text tower; 等价 get_text_features)",
    "clip_l": "OFFICIAL_REPRODUCED(text tower; OpenAI 权重 HF 版)",
    "relative_terms": "BLOCKED_MISSING_REFERENCE(all_git_generated_captions.pt 缺失)",
    "ab_note": "A/B 共享同一批 predcaptions → 本结果不反映 SDXL 精炼效果",
}
json.dump(summary, open(f"{OUT}/caption_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("CAPTION_DONE", json.dumps({k: summary[k] for k in ("meteor", "rouge", "sentence_similarity", "clip_b", "clip_l")}, ensure_ascii=False), flush=True)

