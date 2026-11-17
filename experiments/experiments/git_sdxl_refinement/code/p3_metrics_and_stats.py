"""P3 指标计算与统计分析（CLIP-L + SwAV，A/B 配对，1000 主 + 982 敏感性）。

命名纪律（遵守 FINAL_P1_PROTOCOL_AND_STATISTICAL_AUDIT.md）：
  - Pairwise 2-way Identification = 官方 two_way_identification（cell 17）
  - Strict N-way Top-1 = 独立补充指标，不冒充官方
  - Mean Rank = 补充指标（官方 notebook 未使用）
  - SwAV 官方指标 = scipy correlation distance（cell 31，越低越好）；
    SwAV 的 2-way/Top-1 为新增补充（官方未对 SwAV 做 identification），显式标注。
统计：Wilcoxon（log10 下界防 p=0）、bootstrap 10k（重采样单位=逐样本 delta，固定候选池）、
Holm（确证族 = 两个官方指标：CLIP-L 2-way 与 SwAV distance）、McNemar（Top-1 补充）。
"""
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import correlation as corr_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import P3, load_pt, verify_alignment  # noqa: E402

S = f"{P3}/statistics"
os.makedirs(S, exist_ok=True)
NB, SEED = 10000, 2027
SCOPES = [1000, 982]


def wilcoxon_logp(d):
    """Wilcoxon 符号秩双侧 p 的 log10（避免 double 下溢成 0）。"""
    d = np.asarray(d, float)
    nz = d[d != 0]
    n = len(nz)
    if n < 1:
        return 0.0, float("-inf")
    ranks = stats.rankdata(np.abs(nz))
    w = ranks[nz > 0].sum()
    mu = n * (n + 1) / 4.0
    sg = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mu) / sg
    p = 2 * stats.norm.sf(abs(z))
    return float(z), float(np.log10(max(p, 1e-300)))


def boot_ci(d, n_boot=NB, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(d, float)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    m = a[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_logp(b, c):
    if b + c == 0:
        return float("-inf")
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return float(np.log10(max(stats.chi2.sf(chi2, 1), 1e-300)))


def ident_metrics(R, scope):
    r = R[:scope, :scope].astype(np.float64)
    N = r.shape[0]
    c = (r < np.diag(r)[None, :]).sum(axis=0)
    return {"pairwise_2way_official": float(c.mean() / (N - 1)),
            "strict_top1": float((c == N - 1).mean()),
            "mean_rank": float((N - c).mean()), "c": c}


align = verify_alignment()
assert align["manifest_equals_tensor_order"], align
ids = [int(x) for x in load_pt("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/run/full_stage1/"
                               "s1_textalign_stage1_FINAL_BEST_32_all_ids.pt")]

DIMS = {"clip_l": 768, "swav": 2048}
feats = {}
for m, dim in DIMS.items():
    d = f"{P3}/{m}"
    feats[m] = {g: np.load(f"{d}/{g}_features.npy").astype(np.float32) for g in ("A_raw", "B_refined", "GT")}
    for g, a in feats[m].items():
        assert a.shape == (1000, dim), (m, g, a.shape)
        assert np.isfinite(a).all(), f"NaN/Inf in {m}/{g}"

results = {"alignment": align, "models": {}, "per_sample_csv_rows": []}
for model in ("clip_l", "swav"):
    A, B, GT = (feats[model][g] for g in ("A_raw", "B_refined", "GT"))
    mres = {"feature_check": {"shape": list(GT.shape), "finite": True}, "scopes": {}}
    R_A = np.corrcoef(GT, A)[:1000, 1000:]
    R_B = np.corrcoef(GT, B)[:1000, 1000:]
    for scope in SCOPES:
        sc = {}
        imA, imB = ident_metrics(R_A, scope), ident_metrics(R_B, scope)
        d2way = (imB["c"] - imA["c"]) / (scope - 1)
        hitA = imA["c"] == scope - 1
        hitB = imB["c"] == scope - 1
        dtop1 = hitB.astype(float) - hitA.astype(float)
        b, c_ = int((hitA & ~hitB).sum()), int((~hitA & hitB).sum())
        lo, hi = boot_ci(d2way)
        z, lp = wilcoxon_logp(imB["c"] - imA["c"])
        sc["pairwise_2way_official"] = {
            "label": "Pairwise 2-way Identification（MindEye2 官方 two_way_identification）",
            "official": True, "direction": "higher_better",
            "A_mean": imA["pairwise_2way_official"], "B_mean": imB["pairwise_2way_official"],
            "delta": imB["pairwise_2way_official"] - imA["pairwise_2way_official"],
            "delta_ci95_boot": [lo, hi], "wilcoxon_z": z, "wilcoxon_log10p": lp,
            "improved": int((d2way > 0).sum()), "worsened": int((d2way < 0).sum()),
            "tied": int((d2way == 0).sum()),
            "A_std_unit": float(np.std(d2way, ddof=1)),
        }
        lo2, hi2 = boot_ci(dtop1)
        z2, lp2 = wilcoxon_logp(dtop1)
        sc["strict_top1_supplementary"] = {
            "label": "Strict N-way Top-1（补充指标，非官方）", "official": False,
            "direction": "higher_better",
            "A_mean": imA["strict_top1"], "B_mean": imB["strict_top1"],
            "delta": imB["strict_top1"] - imA["strict_top1"],
            "delta_ci95_boot": [lo2, hi2], "wilcoxon_z": z2, "wilcoxon_log10p": lp2,
            "mcnemar_b": b, "mcnemar_c": c_, "mcnemar_log10p": mcnemar_logp(b, c_),
            "improved": int((dtop1 > 0).sum()), "worsened": int((dtop1 < 0).sum()),
            "tied": int((dtop1 == 0).sum()),
        }
        sc["mean_rank_supplementary"] = {
            "label": "Mean Rank（1-based，正确目标排名第 1；补充指标，官方未使用）",
            "official": False, "direction": "lower_better",
            "A_mean": imA["mean_rank"], "B_mean": imB["mean_rank"],
            "delta": imB["mean_rank"] - imA["mean_rank"],
        }
        if model == "swav":
            dA = np.array([corr_dist(GT[i], A[i]) for i in range(scope)])
            dB = np.array([corr_dist(GT[i], B[i]) for i in range(scope)])
            dd = dB - dA
            lo3, hi3 = boot_ci(dd)
            z3, lp3 = wilcoxon_logp(dd)
            sc["swav_distance_official"] = {
                "label": "SwAV correlation distance（MindEye2 官方 cell 31，越低越好）",
                "official": True, "direction": "lower_better",
                "A_mean": float(dA.mean()), "A_std": float(dA.std(ddof=1)),
                "A_median": float(np.median(dA)),
                "B_mean": float(dB.mean()), "B_std": float(dB.std(ddof=1)),
                "B_median": float(np.median(dB)),
                "delta": float(dd.mean()), "delta_median": float(np.median(dd)),
                "delta_ci95_boot": [lo3, hi3], "wilcoxon_z": z3, "wilcoxon_log10p": lp3,
                "improved": int((dd < 0).sum()), "worsened": int((dd > 0).sum()),
                "tied": int((dd == 0).sum()),
            }
        else:
            simA = np.array([float(np.corrcoef(GT[i], A[i])[0, 1]) for i in range(scope)])
            simB = np.array([float(np.corrcoef(GT[i], B[i])[0, 1]) for i in range(scope)])
            dsim = simB - simA
            lo4, hi4 = boot_ci(dsim)
            z4, lp4 = wilcoxon_logp(dsim)
            sc["img_img_pearson_supplementary"] = {
                "label": "GT↔重建 Pearson 相似度（编码器同官方；相似度形式为补充描述）",
                "official": False, "direction": "higher_better",
                "A_mean": float(simA.mean()), "A_std": float(simA.std(ddof=1)),
                "A_median": float(np.median(simA)),
                "B_mean": float(simB.mean()), "B_std": float(simB.std(ddof=1)),
                "B_median": float(np.median(simB)),
                "delta": float(dsim.mean()), "delta_median": float(np.median(dsim)),
                "delta_ci95_boot": [lo4, hi4], "wilcoxon_z": z4, "wilcoxon_log10p": lp4,
                "improved": int((dsim > 0).sum()), "worsened": int((dsim < 0).sum()),
                "tied": int((dsim == 0).sum()),
            }
        for i in range(scope):
            results["per_sample_csv_rows"].append(
                {"model": model, "scope": scope, "idx": i, "id": ids[i],
                 "pairwise2way_unit_delta": d2way[i],
                 "top1_A": int(hitA[i]), "top1_B": int(hitB[i])})
        mres["scopes"][str(scope)] = sc
    results["models"][model] = mres
# ---- Holm（确证族 = 两个官方指标 @1000）----
fam = []
cl = results["models"]["clip_l"]["scopes"]["1000"]["pairwise_2way_official"]
fam.append(("CLIP-L Pairwise 2-way", cl["wilcoxon_log10p"], cl))
sw = results["models"]["swav"]["scopes"]["1000"]["swav_distance_official"]
fam.append(("SwAV correlation distance", sw["wilcoxon_log10p"], sw))
fam.sort(key=lambda x: x[1])
holm = {}
for rank, (name, lp, ref) in enumerate(fam, start=1):
    adj = min(0.0, np.log10(len(fam) - rank + 1) + lp)
    ref["holm_log10p"] = round(adj, 2)
    ref["holm_significant_0.05"] = bool(adj < np.log10(0.05))
    holm[name] = {"log10p_raw": round(lp, 2), "holm_log10p": round(adj, 2),
                  "significant_0.05": ref["holm_significant_0.05"]}
results["holm_confirmatory_family"] = {
    "family": [n for n, _, _ in fam], "method": "Holm-Bonferroni in log10 space",
    "note": "确证族仅含两个官方指标；Top-1/MeanRank/Pearson 相似度为补充指标，不进族，p 值仅描述性",
    "holm": holm}

with open(f"{S}/p3_metrics_summary.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, ensure_ascii=False, default=float)
with open(f"{S}/p3_per_sample.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(results["per_sample_csv_rows"][0].keys()))
    w.writeheader()
    w.writerows(results["per_sample_csv_rows"])

for m in results["models"]:
    for sc, d in results["models"][m]["scopes"].items():
        for k, v in d.items():
            if isinstance(v, dict) and "A_mean" in v:
                print(m, sc, k, "A=", round(v["A_mean"], 6), "B=", round(v["B_mean"], 6),
                      "D=", round(v["delta"], 6), "log10p=", v.get("wilcoxon_log10p"))
print(json.dumps(holm, ensure_ascii=False))
print("P3_METRICS_DONE")
