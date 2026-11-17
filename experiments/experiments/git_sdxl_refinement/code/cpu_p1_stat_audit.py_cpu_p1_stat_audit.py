"""P1 统计复核：基于已保存的 R 矩阵重算 N-way，检查协议一致性并给出可靠统计。

不重跑模型；所有输出写入 evaluation/cpu_mindeye2_p1_stat_audit/。
"""
import json
import os

import numpy as np
from scipy import stats
from scipy.special import log_ndtr

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
P1 = f"{ROOT}/evaluation/cpu_mindeye2_p1"
OUT = f"{ROOT}/evaluation/cpu_mindeye2_p1_stat_audit"
F = f"{P1}/features"
os.makedirs(OUT, exist_ok=True)
N982 = 982
SEED = 2027
NB = 10000

MODELS = [("AlexNet(2)", "alexnet", 0), ("AlexNet(5)", "alexnet", 1), ("InceptionV3", "inception", 0)]


def load_json(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def success_counts(R, scope):
    """官方 two_way_identification 的逐列成功计数：c_j = #{i: R[i,j] < R[j,j]}（i≠j 自动成立）。"""
    r = R[:scope, :scope].astype(np.float64)
    diag = np.diag(r)
    return (r < diag[None, :]).sum(axis=0)          # (scope,)


def acc_from_counts(c, scope):
    return float(c.mean() / (scope - 1))


def boot_ci(x, n_boot=NB, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(x, dtype=np.float64)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    m = a[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(a.mean()), float(lo), float(hi)


def wilcoxon_logp(d):
    """Wilcoxon 符号秩（含零差处理），返回 z 与 log10(p)，避免 double 下溢成 0。"""
    d = np.asarray(d, dtype=np.float64)
    nz = d[d != 0]
    n = len(nz)
    if n < 1:
        return 0.0, float("-inf")
    ranks = stats.rankdata(np.abs(nz))
    w = ranks[nz > 0].sum()
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mu) / sigma
    logp = np.log(2.0) + log_ndtr(-abs(z))          # log(2*Phi(-|z|))
    return float(z), float(logp / np.log(10))


def signflip_logp(d, n_perm=20000, seed=SEED, batch=2000):
    """配对符号翻转置换检验（分批向量化），返回 log10(p) 及可分辨下限。"""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=np.float64)
    obs = abs(d.mean())
    cnt = 0
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        s = rng.integers(0, 2, size=(b, len(d))) * 2 - 1
        stats_ = np.abs((s * d).mean(axis=1))
        cnt += int((stats_ >= obs - 1e-12).sum())
        done += b
    return float(np.log10((cnt + 1) / (n_perm + 1))), float(np.log10(1.0 / (n_perm + 1)))


def mcnemar_logp(b, c):
    """hit1 的 McNemar 检验（连续校正），b/c 为不一致对数。"""
    if b + c == 0:
        return float("-inf")
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = stats.chi2.sf(chi2, 1)
    return float(np.log10(max(p, 1e-300)))


report = {"protocol_check": {}, "per_model": {}, "holm": {}}
stored = {m: load_json(f"{P1}/p1_{mdl}_summary.json") for m, mdl, _ in MODELS}

for name, mdl, li in MODELS:
    RA = np.load(f"{F}/{mdl}/R_A_raw__{li}.npy")
    RB = np.load(f"{F}/{mdl}/R_B_refined__{li}.npy")
    entry = {"R_shape": list(RA.shape), "scopes": {}}
    for scope in (1000, N982):
        cA = success_counts(RA, scope)
        cB = success_counts(RB, scope)
        accA, accB = acc_from_counts(cA, scope), acc_from_counts(cB, scope)
        d = (cB - cA) / (scope - 1)                # 逐样本贡献差 -> 均值即准确率差
        delta = accB - accA
        mean, lo, hi = boot_ci(d)
        z, log10p_w = wilcoxon_logp(cB - cA)
        log10p_perm, perm_floor = signflip_logp(d)
        hitA = (cA == scope - 1)
        hitB = (cB == scope - 1)
        b = int((hitA & ~hitB).sum())
        c_ = int((~hitA & hitB).sum())
        log10p_mc = mcnemar_logp(b, c_)
        entry["scopes"][str(scope)] = {
            "acc_A": round(accA, 6), "acc_B": round(accB, 6),
            "delta": round(float(delta), 6),
            "delta_ci95_boot": [round(lo, 6), round(hi, 6)],
            "delta_mean_from_counts": round(float(d.mean()), 6),
            "wilcoxon_z": round(z, 3), "wilcoxon_log10p": round(log10p_w, 2),
            "signflip_log10p": round(log10p_perm, 2), "signflip_floor_log10p": round(perm_floor, 2),
            "hit1_A": round(float(hitA.mean()), 6), "hit1_B": round(float(hitB.mean()), 6),
            "mcnemar_b": b, "mcnemar_c": c_, "mcnemar_log10p": round(log10p_mc, 2),
            "improved_items": int((d > 0).sum()), "worsened_items": int((d < 0).sum()),
            "tied_items": int((d == 0).sum()),
            "cA_mean_rank_equiv": round(float((scope - cA.mean())), 3),
        }
        # 与既有产物一致性
        st = stored[name]
        key = f"{name}|A_raw|acc_N{scope}"
        refA = st.get(key)
        key2 = f"{name}|B_refined|acc_N{scope}"
        refB = st.get(key2)
        entry["scopes"][str(scope)]["matches_stored_A"] = (refA is not None and abs(refA - accA) < 1e-6)
        entry["scopes"][str(scope)]["matches_stored_B"] = (refB is not None and abs(refB - accB) < 1e-6)
    report["per_model"][name] = entry

# 逐样本 CSV 一致性：rank == scope - c
name, mdl, li = MODELS[0]
RA = np.load(f"{F}/{mdl}/R_A_raw__{li}.npy")
cA = success_counts(RA, 1000)
import csv  # noqa: E402
rows = [r for r in csv.DictReader(open(f"{P1}/p1_alexnet_per_sample.csv", encoding="utf-8"))
        if r["metric"] == "AlexNet(2)" and r["group"] == "A_raw"]
rank_csv = np.array([float(r["value"]) for r in rows], dtype=np.float64)
pred_rank = (1000 - cA + 1).astype(np.float64)   # rank_j = 1 + #{i: r[i,j] >= r[j,j]} = N - c_j + 1
mism = int((rank_csv != pred_rank).sum())
report["protocol_check"] = {
    "rank_identity_rank_eq_N_minus_c_plus_1": mism == 0,
    "rank_identity_mismatch_count": mism,
    "rank_identity_max_abs_diff": float(np.abs(rank_csv - pred_rank).max()) if mism else 0.0,
    "n_rows_csv": len(rows),
    "note": "rank_j = 1 + #{i: R[i,j] >= R[j,j]}（含自身等式）= N - c_j + 1；c_j = #{i: R[i,j] < R[j,j]}",
}

# ---- Holm 校正（6 个指标的 delta，1000 口径，Wilcoxon log10 p）----
import csv as _csv  # noqa: E402

per = {}
for r in _csv.DictReader(open(f"{P1}/per_sample_metrics.csv", encoding="utf-8")):
    if int(r["idx"]) < 1000:
        per.setdefault((r["metric"], r["group"]), {})[int(r["idx"])] = float(r["value"])

pairs = {}
for metric in {m for m, _, _ in MODELS}:
    a = per.get((metric, "A_raw"), {})
    b = per.get((metric, "B_refined"), {})
    common = sorted(set(a) & set(b))
    if common:
        pairs[metric] = np.array([b[i] - a[i] for i in common])
for metric in ("PixCorr", "SSIM", "EfficientNet-B1"):
    a = per.get((metric, "A_raw"), {})
    b = per.get((metric, "B_refined"), {})
    common = sorted(set(a) & set(b))
    if common:
        pairs[metric] = np.array([b[i] - a[i] for i in common])

hol = {}
for metric, d in pairs.items():
    z, lp = wilcoxon_logp(d)
    hol[metric] = {"n": int(len(d)), "wilcoxon_z": round(z, 3), "log10p": round(lp, 2)}
m = len(hol)
for rank, (metric, v) in enumerate(sorted(hol.items(), key=lambda kv: kv[1]["log10p"]), start=1):
    v["holm_log10p"] = round(min(0.0, np.log10(m - rank + 1) + v["log10p"]), 2)
    v["holm_p_expr"] = f"< 1e{v['holm_log10p']:.0f}" if v["holm_log10p"] < -4 else f"{10**v['holm_log10p']:.2e}"
    v["significant_holm_0.05"] = bool(v["holm_log10p"] < float(np.log10(0.05)))
report["holm"] = {"family_size": m, "scope": 1000, "method": "Holm-Bonferroni in log10 space",
                  "metrics": hol}


with open(f"{OUT}/nway_audit.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
print("AUDIT_DONE", flush=True)
