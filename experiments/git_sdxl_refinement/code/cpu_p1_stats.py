"""P1 统一统计：合并 lowlevel + 各模型 per-sample CSV，输出 A/B 配对统计。

- 每个指标分别给出 1000 与 shared982 两种口径
- delta = B_refined − A_raw；距离类指标方向相反（lower is better）时按方向报告
- Bootstrap 10,000（seed 2027）95% percentile CI + Wilcoxon signed-rank
- 对全部 delta 的 p 值做 Holm 校正
"""
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
OUT = f"{ROOT}/evaluation/cpu_mindeye2_p1"
N982 = 982
DIRECTION = {"PixCorr": "higher", "SSIM": "higher",
             # 识别类逐样本 value = 该样本的 1-based 排名（越低越好）
             "AlexNet(2)": "lower", "AlexNet(5)": "lower",
             "InceptionV3": "lower", "CLIP-ViT-L14_HF_PROXY": "lower",
             "EfficientNet-B1": "lower", "SwAV-ResNet50": "lower"}


def boot_ci(x, n_boot=10000, seed=2027):
    rng = np.random.default_rng(seed)
    a = np.asarray(x, dtype=np.float64)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    m = a[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(a.mean()), float(lo), float(hi)


rows = []
p = f"{OUT}/lowlevel_per_sample.csv"
if os.path.exists(p):
    for r in csv.DictReader(open(p, encoding="utf-8")):
        for metric in ("PixCorr", "SSIM"):
            col = "pixcorr" if metric == "PixCorr" else "ssim"
            rows.append({"metric": metric, "group": r["group"], "idx": int(r["idx"]),
                         "id": int(r["id"]), "value": float(r[col])})
for f in sorted(glob.glob(f"{OUT}/p1_*_per_sample.csv")):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        rows.append({"metric": r["metric"], "group": r["group"], "idx": int(r["idx"]),
                     "id": int(r["id"]), "value": float(r["value"])})
print("per-sample rows", len(rows), flush=True)

metrics = sorted({r["metric"] for r in rows})
groups = sorted({r["group"] for r in rows})
print("metrics", metrics, "groups", groups, flush=True)

summary_rows = []
stats = {}
for metric in metrics:
    for group in groups:
        for scope, lim in (("1000", 10**9), ("982", N982)):
            v = np.array([r["value"] for r in rows
                          if r["metric"] == metric and r["group"] == group and r["idx"] < lim])
            if len(v) == 0:
                continue
            summary_rows.append({"metric": metric, "group": group, "scope": scope, "n": len(v),
                                 "mean": round(float(v.mean()), 6), "std": round(float(v.std(ddof=1)), 6),
                                 "median": round(float(np.median(v)), 6),
                                 "direction": DIRECTION.get(metric, "higher")})
    # A/B 配对
    for scope, lim in (("1000", 10**9), ("982", N982)):
        a = {r["idx"]: r["value"] for r in rows if r["metric"] == metric and r["group"] == "A_raw" and r["idx"] < lim}
        b = {r["idx"]: r["value"] for r in rows if r["metric"] == metric and r["group"] == "B_refined" and r["idx"] < lim}
        common = sorted(set(a) & set(b))
        if not common:
            continue
        d = np.array([b[i] - a[i] for i in common])
        mean, lo, hi = boot_ci(d)
        try:
            pval = float(wilcoxon(d).pvalue)
        except Exception:
            pval = float("nan")
        better = int((d > 0).sum()) if DIRECTION.get(metric, "higher") == "higher" else int((d < 0).sum())
        worse = int((d < 0).sum()) if DIRECTION.get(metric, "higher") == "higher" else int((d > 0).sum())
        key = f"{metric}|{scope}"
        stats[key] = {"n": len(d), "mean_delta": round(mean, 6), "median_delta": round(float(np.median(d)), 6),
                      "ci95": [round(lo, 6), round(hi, 6)], "p_wilcoxon": pval,
                      "better_by_direction": better, "worse_by_direction": worse,
                      "direction": DIRECTION.get(metric, "higher")}
        summary_rows.append({"metric": metric, "group": "DELTA(B-A)", "scope": scope, "n": len(d),
                             "mean": round(mean, 6), "std": round(float(d.std(ddof=1)), 6),
                             "median": round(float(np.median(d)), 6),
                             "direction": DIRECTION.get(metric, "higher")})

# Holm 校正（按 1000 口径的 delta）
keys = [k for k in stats if k.endswith("|1000")]
ps = sorted(((stats[k]["p_wilcoxon"], k) for k in keys), key=lambda t: (np.isnan(t[0]), t[0]))
m = len(ps)
adj = {}
prev = 0.0
for rank, (pv, k) in enumerate(ps, start=1):
    if np.isnan(pv):
        adj[k] = float("nan")
        continue
    val = min(1.0, (m - rank + 1) * pv)
    prev = max(prev, val)
    adj[k] = round(prev, 6)
for k, v in adj.items():
    stats[k]["p_holm"] = v
    stats[k]["significant_holm_0.05"] = bool(v < 0.05) if not np.isnan(v) else False

with open(f"{OUT}/metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["metric", "group", "scope", "n", "mean", "std", "median", "direction"])
    w.writeheader()
    w.writerows(summary_rows)
with open(f"{OUT}/per_sample_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["metric", "group", "idx", "id", "value", "direction"])
    w.writeheader()
    for r in rows:
        r["direction"] = DIRECTION.get(r["metric"], "higher")
        w.writerow(r)
with open(f"{OUT}/stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in stats.items() if k.endswith("|1000")}, indent=2, ensure_ascii=False), flush=True)

# ---- 汇总各模型的官方口径准确率（来自 p1_<model>_summary.json）----
acc_rows = []
for f in sorted(glob.glob(f"{OUT}/p1_*_summary.json")):
    d = json.load(open(f, encoding="utf-8"))
    for k, v in d.items():
        parts = k.split("|")
        if len(parts) == 3:
            acc_rows.append({"metric": parts[0], "group": parts[1], "stat": parts[2], "value": v})
if acc_rows:
    with open(f"{OUT}/p1_accuracy_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "group", "stat", "value"])
        w.writeheader()
        w.writerows(acc_rows)
    for r in acc_rows:
        if r["stat"].startswith("acc_N") or r["stat"] == "hit1":
            print(f"{r['metric']:24s} {r['group']:12s} {r['stat']:10s} {r['value']}", flush=True)
print("STATS_DONE", flush=True)
