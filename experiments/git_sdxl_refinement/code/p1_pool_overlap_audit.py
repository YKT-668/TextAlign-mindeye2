"""P1 池重叠审计：复现 20×300 随机候选池（seed=2027，与 cpu_p1_pool_audit.py 完全一致），
输出两两 ID 重叠矩阵与独立性判定。

要点：两个独立随机 300/1000 无放回子池的期望重叠 = 300*300/1000 = 90（超几何期望），
因此 20 个池必然高度重叠 → 池层面 20/20 符号检验不能作为独立重复实验的显著性证据，
只能作为候选池敏感性/稳健性结果。
"""
import json
import os

import numpy as np
from scipy.stats import hypergeom

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
F = f"{ROOT}/evaluation/cpu_mindeye2_p1/features"
STAT = f"{ROOT}/evaluation/cpu_mindeye2_p1_stat_audit"
OUT = f"{STAT}/pool_overlap_audit.json"          # 新文件，不覆盖 pool_level_audit.json

POOL, NPOOL, SEED = 300, 20, 2027
MODELS = [("AlexNet(2)", "alexnet", 0), ("AlexNet(5)", "alexnet", 1), ("InceptionV3", "inception", 0)]

# --- 1) 逐字复现原脚本的池生成 ---
rng = np.random.default_rng(SEED)
pools = [rng.choice(1000, size=POOL, replace=False) for _ in range(NPOOL)]

# --- 2) 两两重叠矩阵 ---
ov = np.zeros((NPOOL, NPOOL), dtype=int)
for i in range(NPOOL):
    for j in range(i + 1, NPOOL):
        k = len(np.intersect1d(pools[i], pools[j]))
        ov[i, j] = ov[j, i] = k
iu = np.triu_indices(NPOOL, k=1)
offdiag = ov[iu]

all_ids = np.unique(np.concatenate(pools))
covered = np.concatenate(pools)
counts = np.bincount(covered, minlength=1000)

# --- 3) 复现池层面准确率，核对与已记录结果一致（验证池复现正确）---
def acc_sub(M, idx):
    r = M[np.ix_(idx, idx)].astype(np.float64)
    diag = np.diag(r)
    c = (r < diag[None, :]).sum(axis=0)
    return float(c.mean() / (len(idx) - 1))

repro = {}
recorded = json.load(open(f"{STAT}/pool_level_audit.json", encoding="utf-8"))
for name, mdl, li in MODELS:
    RA = np.load(f"{F}/{mdl}/R_A_raw__{li}.npy")
    RB = np.load(f"{F}/{mdl}/R_B_refined__{li}.npy")
    a = np.array([acc_sub(RA, p) for p in pools])
    b = np.array([acc_sub(RB, p) for p in pools])
    rec = recorded[name]
    repro[name] = {
        "acc_A_mean_recomputed": round(float(a.mean()), 5),
        "acc_A_mean_recorded": rec["acc_A_mean"],
        "acc_B_mean_recomputed": round(float(b.mean()), 5),
        "acc_B_mean_recorded": rec["acc_B_mean"],
        "delta_mean_recomputed": round(float((b - a).mean()), 5),
        "delta_mean_recorded": rec["delta_mean"],
        "pools_B_worse_recomputed": int(((b - a) < 0).sum()),
        "pools_B_worse_recorded": rec["pools_B_worse"],
    }
    repro[name]["matches_recorded"] = (
        abs(repro[name]["acc_A_mean_recomputed"] - rec["acc_A_mean"]) < 1e-5
        and abs(repro[name]["acc_B_mean_recomputed"] - rec["acc_B_mean"]) < 1e-5
        and repro[name]["pools_B_worse_recomputed"] == rec["pools_B_worse"]
    )

# 理论参照：两个独立池重叠数 ~ Hypergeom(N=1000, K=300, n=300)
theo_mean = 300 * 300 / 1000
theo_sd = float(np.sqrt(hypergeom.var(1000, 300, 300)))
theo_p_zero = float(hypergeom.pmf(0, 1000, 300, 300))

out = {
    "pool_generation": {"script": "scripts/cpu_p1_pool_audit.py", "seed": SEED,
                        "rng": "numpy.random.default_rng", "n_pools": NPOOL, "pool_size": POOL,
                        "population": 1000, "replace": False,
                        "reproducible": True},
    "pairwise_overlap": {
        "matrix_20x20": ov.tolist(),
        "n_pairs": int(len(offdiag)),
        "min": int(offdiag.min()), "mean": round(float(offdiag.mean()), 2), "max": int(offdiag.max()),
        "sd": round(float(offdiag.std(ddof=1)), 2),
        "theoretical_mean_independent_pools": theo_mean,
        "theoretical_sd_independent_pools": round(theo_sd, 2),
        "theoretical_p_zero_overlap_independent": theo_p_zero,
        "pairs_with_overlap_ge_100": int((offdiag >= 100).sum()),
    },
    "coverage": {
        "unique_ids_covered": int(len(all_ids)),
        "ids_never_selected": int(1000 - len(all_ids)),
        "mean_times_selected_per_id": round(float(counts.mean()), 2),
        "max_times_selected_per_id": int(counts.max()),
        "min_times_selected_per_id": int(counts.min()),
    },
    "reproducibility_check_vs_pool_level_audit": repro,
    "independence_verdict": {
        "independent": False,
        "reason": f"任意两池期望重叠 ≈ {theo_mean:.0f}（实测均值 {offdiag.mean():.1f}），"
                  "20 个池共享同一 1000 候选总体且高度交叠，不是 20 次独立重复实验。",
        "consequence_for_sign_test": "『20/20 方向一致』的符号检验 p≈1.9e-6 只能作为"
                                     "候选池敏感性/稳健性证据，不能作为独立重复实验的显著性证据。",
        "formal_inference_status": "正式统计推断仍以 1000 主口径的配对检验 + bootstrap CI 为描述性证据；"
                                   "池层面结果降级为敏感性分析。",
    },
    "bootstrap_review": {
        "resampling_unit": "逐样本 delta（d_j = (cB_j - cA_j)/(N-1)），n=1000，10k 次有放回重采样，percentile 95% CI",
        "candidate_pool_fixed": True,
        "pool_note": "1000 主口径的候选池 = 全部 1000 张 GT（固定不变）；982 口径 = 同一固定池的前 982 列（等价于独立 982 相关矩阵，逐对 Pearson 不受行删除影响）",
        "applicability": "CI 仅描述『这 1000 张固定刺激上的平均 delta』的抽样不确定性；"
                         "共享候选池使样本间非独立，CI 偏窄，不能外推到新刺激总体；"
                         "与池层面敏感性分析互为补充。",
    },
    "mcnemar_review": {
        "input": "hit1_j = (c_j == N-1)，即 Strict N-way Top-1 逐样本 0/1 正确性",
        "b": "A 正确且 B 错误的对数", "c": "A 错误且 B 正确的对数",
        "implementation": "chi2 = (|b-c|-1)^2/(b+c)，连续校正，df=1",
        "scope_note": "McNemar 作用于 Strict Top-1（0/1），不作用于 pairwise 2-way 连续量；"
                      "A/B 共享候选池同样使 0/1 项非独立，p 值为描述性。",
        "recorded_values": "见 nway_audit.json scopes.*.mcnemar_b/c/log10p（本轮仅复核定义，未改动）",
    },
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, ensure_ascii=False)

print(json.dumps({
    "overlap_min": out["pairwise_overlap"]["min"],
    "overlap_mean": out["pairwise_overlap"]["mean"],
    "overlap_max": out["pairwise_overlap"]["max"],
    "theo_mean": theo_mean, "p_zero": theo_p_zero,
    "unique_covered": out["coverage"]["unique_ids_covered"],
    "repro_ok": all(v["matches_recorded"] for v in repro.values()),
}, ensure_ascii=False))
print("POOL_OVERLAP_AUDIT_DONE")