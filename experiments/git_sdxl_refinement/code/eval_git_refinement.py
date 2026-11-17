"""1000 张 A/B 配对评测：对齐校验 → bigG 图像相似度/检索 → CLIP-L 图文 → 统计。

只读输入；所有产物写入 evaluation/ 目录。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
S1 = f"{ROOT}/run/full_stage1"
S2 = f"{ROOT}/run/full_stage2"
GTD = f"{ROOT}/assets/ground_truth"
EVAL = f"{ROOT}/evaluation"
MODEL = "s1_textalign_stage1_FINAL_BEST_32"

ap = argparse.ArgumentParser()
ap.add_argument("--skip_text", action="store_true")
args = ap.parse_args()

os.makedirs(EVAL, exist_ok=True)
dev = torch.device("cuda")
torch.manual_seed(2027)

# ---------------- 1. 数据加载 & 结构校验 ----------------
raw = torch.load(f"{S1}/{MODEL}_all_enhancedrecons.pt", map_location="cpu", weights_only=False).float()  # (1000,3,224,224)
ref = torch.load(f"{S2}/{MODEL}_stage2_img2img.pt", map_location="cpu", weights_only=False).float()      # (1000,3,256,256)
gt = torch.load(f"{GTD}/all_images.pt", map_location="cpu", weights_only=False).float()                  # (1000,3,224,224)
caps_gt = [str(x) for x in torch.load(f"{GTD}/all_captions.pt", map_location="cpu", weights_only=False)]
caps_pred = [str(x) for x in torch.load(f"{S1}/{MODEL}_all_predcaptions.pt", map_location="cpu", weights_only=False)]
ids1 = torch.load(f"{S1}/{MODEL}_all_ids.pt", map_location="cpu", weights_only=False).long()
ids2 = torch.load(f"{S2}/{MODEL}_stage2_ids.pt", map_location="cpu", weights_only=False).long()
blurry = torch.load(f"{S1}/{MODEL}_all_blurryrecons.pt", map_location="cpu", weights_only=False).float()

chk = {}
chk["raw_shape"] = list(raw.shape)
chk["ref_shape"] = list(ref.shape)
chk["gt_shape"] = list(gt.shape)
chk["n_ids"] = int(ids1.numel())
chk["ids_unique"] = int(len(torch.unique(ids1)))
chk["ids_ab_equal"] = bool(torch.equal(ids1, ids2))
chk["ids_ascending"] = bool(torch.all(ids1[1:] > ids1[:-1]))
chk["n_caps_gt"] = len(caps_gt)
chk["n_caps_pred"] = len(caps_pred)
chk["raw_range"] = [round(float(raw.min()), 4), round(float(raw.max()), 4)]
chk["ref_range"] = [round(float(ref.min()), 4), round(float(ref.max()), 4)]
chk["gt_range"] = [round(float(gt.min()), 4), round(float(gt.max()), 4)]
chk["finite_raw"] = bool(torch.isfinite(raw).all())
chk["finite_ref"] = bool(torch.isfinite(ref).all())
chk["finite_gt"] = bool(torch.isfinite(gt).all())
struct_ok = (
    chk["raw_shape"] == [1000, 3, 224, 224]
    and chk["ref_shape"] == [1000, 3, 256, 256]
    and chk["gt_shape"] == [1000, 3, 224, 224]
    and chk["n_ids"] == 1000 and chk["ids_unique"] == 1000
    and chk["ids_ab_equal"] and chk["ids_ascending"]
    and chk["n_caps_gt"] == 1000 and chk["n_caps_pred"] == 1000
    and chk["finite_raw"] and chk["finite_ref"] and chk["finite_gt"]
)
chk["STRUCTURE_OK"] = struct_ok
print("STRUCTURE", json.dumps(chk, ensure_ascii=False), flush=True)

# ---------------- 2. GT 图像顺序校验（像素级）----------------
# blurryrecons 由模型从脑信号回归，与该图像的 GT 应显著相关；若顺序错位则相关性接近随机。
def pixcorr_rows(x, y, idx=None):
    a = x.reshape(x.shape[0], -1)
    b = y.reshape(y.shape[0], -1)
    if idx is not None:
        b = b[idx]
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    num = (a * b).sum(1)
    den = a.norm(dim=1) * b.norm(dim=1) + 1e-8
    return (num / den)


rng = np.random.default_rng(2027)
aligned = pixcorr_rows(blurry, gt)
perm = torch.from_numpy(rng.permutation(1000))
shuffled = pixcorr_rows(blurry, gt, perm)
gt_order_ok = float(aligned.mean()) > float(shuffled.mean()) + 0.10 and float(aligned.mean()) > 0.15
gt_chk = {"pixcorr_aligned_mean": round(float(aligned.mean()), 4),
          "pixcorr_aligned_median": round(float(aligned.median()), 4),
          "pixcorr_shuffled_mean": round(float(shuffled.mean()), 4),
          "frac_positive_aligned": round(float((aligned > 0).float().mean()), 4),
          "GT_IMAGE_ORDER_VERIFIED": bool(gt_order_ok)}
print("GT_IMAGE_ORDER", json.dumps(gt_chk, ensure_ascii=False), flush=True)

torch.save({"blurry_pixcorr_aligned": aligned, "gt_order_ok": gt_order_ok}, f"{EVAL}/_gt_order_check.pt")

# ---------------- 3. bigG 图像嵌入（项目原评测编码器）----------------
SRC = f"{ROOT}/source/TextAlign-mindeye2-231850a/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.append(f"{SRC}/generative_models/")

from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder  # noqa: E402

img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
img_embedder.to(dev).eval()


@torch.no_grad()
def embed_tokens(x, batch=20):
    outs = []
    for i in range(0, x.shape[0], batch):
        chunk = x[i:i + batch].to(dev)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            t = img_embedder(chunk)
        outs.append(t.reshape(t.shape[0], -1).float().half().cpu())
    return torch.cat(outs, 0)


print("embedding A/B/GT with bigG ...", flush=True)
EA = embed_tokens(raw).to(dev)
EB = embed_tokens(ref).to(dev)
EG = embed_tokens(gt).to(dev)
print("emb shapes", tuple(EA.shape), tuple(EB.shape), tuple(EG.shape), flush=True)

# ---------------- 4. 图像-图像余弦 + 1000-way 检索 ----------------
def l2n(E):
    return E / (E.norm(dim=1, keepdim=True) + 1e-8)


EAn, EBn, EGn = l2n(EA.float()), l2n(EB.float()), l2n(EG.float())

sim_A = (EAn * EGn).sum(1)          # 对角：raw vs 对应 GT
sim_B = (EBn * EGn).sum(1)          # 对角：refined vs 对应 GT
delta = sim_B - sim_A

S_A = EAn @ EGn.T                   # 1000x1000 全相似度（生成→GT）
S_B = EBn @ EGn.T

def retrieval_stats(S, label):
    n = S.shape[0]
    ar = torch.arange(n, device=S.device)
    ranks = (S > S[ar, ar].unsqueeze(1)).sum(1) + 1     # 1-based rank
    top1 = float((ranks <= 1).float().mean())
    top5 = float((ranks <= 5).float().mean())
    mrr = float((1.0 / ranks.float()).mean())
    med = float(ranks.float().median())
    print(f"RETRIEVAL[{label}] top1={top1:.4f} top5={top5:.4f} mrr={mrr:.4f} median={med:.1f}", flush=True)
    return {"direction": label, "top1": top1, "top5": top5, "mrr": mrr, "median_rank": med,
            "ranks": ranks.cpu()}


ret_A_fwd = retrieval_stats(S_A, "generated->GT (A raw)")
ret_B_fwd = retrieval_stats(S_B, "generated->GT (B refined)")
ret_A_bwd = retrieval_stats(S_A.T, "GT->generated (A raw)")
ret_B_bwd = retrieval_stats(S_B.T, "GT->generated (B refined)")

chance_top1 = 1.0 / 1000
print(f"chance top1={chance_top1:.4f}", flush=True)

torch.save({
    "sim_A": sim_A.cpu(), "sim_B": sim_B.cpu(), "delta": delta.cpu(),
    "ranks_A_fwd": ret_A_fwd.pop("ranks"), "ranks_B_fwd": ret_B_fwd.pop("ranks"),
    "ranks_A_bwd": ret_A_bwd.pop("ranks"), "ranks_B_bwd": ret_B_bwd.pop("ranks"),
}, f"{EVAL}/_image_metrics.pt")

image_metrics = {"retrieval": [ret_A_fwd, ret_B_fwd, ret_A_bwd, ret_B_bwd], "chance_top1": chance_top1}
with open(f"{EVAL}/_image_metrics.json", "w") as f:
    json.dump(image_metrics, f, indent=2)
print("IMAGE_STAGE_DONE", flush=True)

# ---------------- 5. CLIP-L（图文，项目原评测协议）----------------
from transformers import AutoTokenizer, CLIPModel  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CLIP_NAME = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(dev).eval()
clip_tok = AutoTokenizer.from_pretrained(CLIP_NAME)
clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)


@torch.no_grad()
def clip_img_feats(x, batch=32):
    outs = []
    for i in range(0, x.shape[0], batch):
        c = x[i:i + batch].to(dev).float()
        if c.shape[-1] != 224:
            c = F.interpolate(c, size=(224, 224), mode="bicubic", align_corners=False)
        c = (c - clip_mean) / clip_std
        f = clip_model.get_image_features(pixel_values=c)
        outs.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(outs, 0)


@torch.no_grad()
def clip_text_feats(texts, batch=64):
    outs = []
    for i in range(0, len(texts), batch):
        tk = clip_tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True,
                      max_length=77).to(dev)
        f = clip_model.get_text_features(**tk)
        outs.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(outs, 0)


print("CLIP-L: image features ...", flush=True)
FA = clip_img_feats(raw)
FB = clip_img_feats(ref)
FG = clip_img_feats(gt)
print("CLIP-L: text features ...", flush=True)
TG_gt = clip_text_feats(caps_gt)
TG_pred = clip_text_feats(caps_pred)

# 5b. GT caption 顺序校验：配对相似度 vs 错配均值
diag_gt = (FG * TG_gt).sum(1)
S_text = FG @ TG_gt.T
off = (S_text.sum() - diag_gt.sum()) / (1000 * 1000 - 1000)
cap_order_ok = float(diag_gt.mean()) > float(off) + 0.05 and float(diag_gt.mean()) > 0.20
cap_chk = {"diag_mean": round(float(diag_gt.mean()), 4), "offdiag_mean": round(float(off), 4),
           "diag_median": round(float(diag_gt.median()), 4),
           "CAPTION_ORDER_VERIFIED": bool(cap_order_ok)}
print("GT_CAPTION_ORDER", json.dumps(cap_chk, ensure_ascii=False), flush=True)

# 5c. 图文指标（仅当 caption 顺序 VERIFIED 才作为正式指标）
it_A = (FA * TG_gt).sum(1)
it_B = (FB * TG_gt).sum(1)
it_delta = it_B - it_A
pa_A = (FA * TG_pred).sum(1)   # prompt adherence（辅助）
pa_B = (FB * TG_pred).sum(1)
pa_delta = pa_B - pa_A

# ---------------- 6. 统计（配对 bootstrap）----------------
def bootstrap_ci(x, n_boot=10000, seed=2027, alpha=0.05):
    rng = np.random.default_rng(seed)
    a = np.asarray(x, dtype=np.float64)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = a[idx].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean()), float(np.median(a)), float(lo), float(hi), float(means.std())


def paired_stats(x, label):
    mean, med, lo, hi, se = bootstrap_ci(x)
    d = np.asarray(x)
    up = int((d > 1e-6).sum())
    tie = int((np.abs(d) <= 1e-6).sum())
    down = int((d < -1e-6).sum())
    out = {"label": label, "mean": round(mean, 5), "median": round(med, 5),
           "ci95": [round(lo, 5), round(hi, 5)], "boot_se": round(se, 5),
           "improved": up, "tied": tie, "worsened": down, "n": int(len(d)),
           "significant": bool(lo > 0 or hi < 0)}
    print("STATS", json.dumps(out, ensure_ascii=False), flush=True)
    return out


stats_img = paired_stats(delta.cpu().numpy(), "image_image_cosine(B-A)")
stats_it = paired_stats(it_delta.numpy(), "image_text_cosine(B-A)") if cap_order_ok else None
stats_pa = paired_stats(pa_delta.numpy(), "prompt_adherence(B-A)")

# Wilcoxon 符号秩（补充）
try:
    from scipy.stats import wilcoxon
    w_img = wilcoxon(delta.cpu().numpy())
    wil = {"stat": float(w_img.statistic), "pvalue": float(w_img.pvalue)}
    if cap_order_ok:
        w_it = wilcoxon(it_delta.numpy())
        wil["text_pvalue"] = float(w_it.pvalue)
except Exception as e:  # pragma: no cover
    wil = {"error": str(e)[:100]}
print("WILCOXON", json.dumps(wil), flush=True)

# ---------------- 7. 逐样本 CSV / 图像导出 / manifest ----------------
import csv  # noqa: E402

from PIL import Image  # noqa: E402

ids_np = ids1.cpu().numpy()
raw_np, ref_np, gt_np = raw.numpy(), ref.numpy(), gt.numpy()
rA = torch.load(f"{EVAL}/_image_metrics.pt", map_location="cpu", weights_only=False)
ranks_A = rA["ranks_A_fwd"].numpy()
ranks_B = rA["ranks_B_fwd"].numpy()

imgdir = f"{EVAL}/images"
for sub in ("raw", "refined", "gt"):
    os.makedirs(f"{imgdir}/{sub}", exist_ok=True)


def save_png(arr, path):
    a = np.clip(np.transpose(arr, (1, 2, 0)), 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


manifest_rows = []
for i in range(1000):
    iid = int(ids_np[i])
    p_raw = f"{imgdir}/raw/raw_id{iid}.png"
    p_ref = f"{imgdir}/refined/refined_id{iid}.png"
    p_gt = f"{imgdir}/gt/gt_id{iid}.png"
    save_png(raw_np[i], p_raw)
    save_png(ref_np[i], p_ref)
    save_png(gt_np[i], p_gt)
    manifest_rows.append({"id": iid, "raw_reconstruction": p_raw, "refined_reconstruction": p_ref,
                          "gt_image": p_gt, "predicted_caption": caps_pred[i]})

os.makedirs(f"{ROOT}/reports", exist_ok=True)
with open(f"{ROOT}/reports/REFINEMENT_EVALUATION_MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "raw_reconstruction", "refined_reconstruction", "gt_image", "predicted_caption"])
    w.writeheader()
    w.writerows(manifest_rows)
print("manifest written", flush=True)

with open(f"{EVAL}/per_sample_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "raw_clip_img_cos", "refined_clip_img_cos", "delta_img", "raw_rank", "refined_rank",
                "raw_clip_text_cos", "refined_clip_text_cos", "delta_text",
                "raw_prompt_adherence", "refined_prompt_adherence", "delta_prompt_adherence",
                "gt_caption", "predicted_caption"])
    for i in range(1000):
        w.writerow([int(ids_np[i]), round(float(sim_A[i]), 6), round(float(sim_B[i]), 6), round(float(delta[i]), 6),
                    int(ranks_A[i]), int(ranks_B[i]),
                    round(float(it_A[i]), 6), round(float(it_B[i]), 6), round(float(it_delta[i]), 6),
                    round(float(pa_A[i]), 6), round(float(pa_B[i]), 6), round(float(pa_delta[i]), 6),
                    caps_gt[i], caps_pred[i]])
print("per-sample csv written", flush=True)

# ---------------- 8. 图表 ----------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.figure(figsize=(6, 4))
plt.hist(delta.cpu().numpy(), bins=60)
plt.axvline(0, color="k", lw=1)
plt.xlabel("delta = refined - raw (bigG image cosine vs GT)")
plt.ylabel("#images")
plt.title(f"mean delta = {float(delta.mean()):.4f}")
plt.tight_layout()
plt.savefig(f"{EVAL}/fig_delta_hist_image.png", dpi=140)
plt.close()

plt.figure(figsize=(5, 5))
plt.scatter(sim_A.cpu().numpy(), sim_B.cpu().numpy(), s=6, alpha=0.5)
lo = min(float(sim_A.min()), float(sim_B.min()))
hi = max(float(sim_A.max()), float(sim_B.max()))
plt.plot([lo, hi], [lo, hi], "k--", lw=1)
plt.xlabel("raw vs GT cosine")
plt.ylabel("refined vs GT cosine")
plt.title("per-image paired (above diagonal = improved)")
plt.tight_layout()
plt.savefig(f"{EVAL}/fig_scatter_raw_vs_refined.png", dpi=140)
plt.close()

plt.figure(figsize=(6, 4))
for rk, lab in ((ranks_A, "raw"), (ranks_B, "refined")):
    xs = np.sort(np.asarray(rk))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    plt.plot(xs, ys, label=lab)
plt.xscale("log")
plt.xlabel("retrieval rank (log scale)")
plt.ylabel("cumulative fraction")
plt.legend()
plt.title("1000-way retrieval (generated -> GT)")
plt.tight_layout()
plt.savefig(f"{EVAL}/fig_retrieval_cdf.png", dpi=140)
plt.close()
print("figures written", flush=True)

# ---------------- 9. 人工盲测材料（150 张）----------------
BR = f"{EVAL}/human_blind_review"
os.makedirs(BR, exist_ok=True)
rng2 = np.random.default_rng(2027)
pick = np.sort(rng2.choice(1000, size=150, replace=False))
sheet, key = [], []
for k, i in enumerate(pick):
    iid = int(ids_np[i])
    left, right = ("raw", "refined") if rng2.random() < 0.5 else ("refined", "raw")
    src = {"raw": raw_np[i], "refined": ref_np[i]}
    for pos, which in (("left", left), ("right", right)):
        a = np.clip(np.transpose(src[which], (1, 2, 0)), 0, 1)
        Image.fromarray((a * 255).astype(np.uint8)).resize((256, 256)).save(f"{BR}/sample_{k:04d}_{pos}.png")
    a = np.clip(np.transpose(gt_np[i], (1, 2, 0)), 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(f"{BR}/sample_{k:04d}_gt.png")
    sheet.append({"sample": f"{k:04d}", "id": iid, "left_png": f"sample_{k:04d}_left.png",
                  "right_png": f"sample_{k:04d}_right.png", "gt_png": f"sample_{k:04d}_gt.png",
                  "subject": "", "attributes": "", "relations": "", "errors": "",
                  "overall_quality": "", "preference": ""})
    key.append({"sample": f"{k:04d}", "id": iid, "left_is": left, "right_is": right})

with open(f"{BR}/review_sheet.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(sheet[0].keys()))
    w.writeheader()
    w.writerows(sheet)
with open(f"{BR}/ANSWER_KEY_DO_NOT_OPEN_BEFORE_RATING.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["sample", "id", "left_is", "right_is"])
    w.writeheader()
    w.writerows(key)
print("blind review written", flush=True)

# ---------------- 10. 汇总 JSON ----------------
summary = {
    "structure": chk,
    "gt_image_order": gt_chk,
    "gt_caption_order": cap_chk,
    "image_metrics": image_metrics,
    "stats_image": stats_img,
    "stats_image_text": stats_it,
    "stats_prompt_adherence": stats_pa,
    "wilcoxon": wil,
    "means": {
        "raw_img_cos": round(float(sim_A.mean()), 5),
        "refined_img_cos": round(float(sim_B.mean()), 5),
        "raw_img_cos_median": round(float(sim_A.median()), 5),
        "refined_img_cos_median": round(float(sim_B.median()), 5),
        "raw_img_cos_std": round(float(sim_A.std()), 5),
        "refined_img_cos_std": round(float(sim_B.std()), 5),
        "raw_text_cos": round(float(it_A.mean()), 5),
        "refined_text_cos": round(float(it_B.mean()), 5),
    },
}
with open(f"{EVAL}/eval_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("SUMMARY_WRITTEN", json.dumps(summary["means"], ensure_ascii=False), flush=True)




