"""P1 补充：模拟官方 30×300 的随机子池评测，检验 A/B 差异在"池层面"的稳健性。

共享候选池会让 1000 个样本的逐项检验高估自由度；这里用 20 次随机 300 子池，
对每个池分别算 A/B 准确率，再做池层面的配对比较（每个池用同一批索引）。
"""
import json
import os

import numpy as np

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
P1 = f"{ROOT}/evaluation/cpu_mindeye2_p1"
OUT = f"{ROOT}/evaluation/cpu_mindeye2_p1_stat_audit"
F = f"{P1}/features"
MODELS = [("AlexNet(2)", "alexnet", 0), ("AlexNet(5)", "alexnet", 1), ("InceptionV3", "inception", 0)]
POOL, NPOOL, SEED = 300, 20, 2027


def acc_sub(M, idx):
    r = M[np.ix_(idx, idx)].astype(np.float64)
    diag = np.diag(r)
    c = (r < diag[None, :]).sum(axis=0)
    return float(c.mean() / (len(idx) - 1))


res = {}
rng = np.random.default_rng(SEED)
pools = [rng.choice(1000, size=POOL, replace=False) for _ in range(NPOOL)]
for name, mdl, li in MODELS:
    RA = np.load(f"{F}/{mdl}/R_A_raw__{li}.npy")
    RB = np.load(f"{F}/{mdl}/R_B_refined__{li}.npy")
    a = np.array([acc_sub(RA, p) for p in pools])
    b = np.array([acc_sub(RB, p) for p in pools])
    d = b - a
    from scipy.stats import wilcoxon
    try:
        p = float(wilcoxon(d).pvalue)
    except Exception:
        p = float("nan")
    res[name] = {"n_pools": NPOOL, "pool_size": POOL,
                 "acc_A_mean": round(float(a.mean()), 5), "acc_A_sd": round(float(a.std(ddof=1)), 5),
                 "acc_B_mean": round(float(b.mean()), 5), "acc_B_sd": round(float(b.std(ddof=1)), 5),
                 "delta_mean": round(float(d.mean()), 5), "delta_sd": round(float(d.std(ddof=1)), 5),
                 "p_wilcoxon_pool_level": p,
                 "pools_B_worse": int((d < 0).sum()), "pools_B_better": int((d > 0).sum())}
    print(name, res[name], flush=True)

json.dump(res, open(f"{OUT}/pool_level_audit.json", "w"), indent=2, ensure_ascii=False)
print("POOL_AUDIT_DONE", flush=True)
