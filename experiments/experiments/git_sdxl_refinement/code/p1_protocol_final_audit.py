"""P1 指标协议终审：判定官方 two_way_identification(cell 17) 究竟属于哪一类排名指标。

官方源代码（逐字核对，两仓库一致）：
  - source/TextAlign-mindeye2-231850a/src/final_evaluations.ipynb CELL 17
  - source/MindEyeV2/src/final_evaluations.ipynb CELL 14
  公式: r = np.corrcoef(reals, preds)[:N, N:]; success = r < diag(r)[:,None];
        c_j = sum_i success[i,j]; perf = mean_j(c_j) / (N-1)

三口径（对每个模型、A/B、1000/982 同时计算）：
  1) Pairwise 2-way Identification（官方口径）: mean_j(c_j/(N-1))
  2) Strict N-way Top-1: mean_j( c_j == N-1 )；交叉验证 argmax_i R[i,j] == j
  3) Mean Rank（官方 notebook 未使用，仅作补充）: mean_j(N - c_j)（1-based，正确目标排名第 1）

输出写入 evaluation/p1_protocol_final_audit/（全新目录，不覆盖任何既有结果）。
"""
import csv
import json
import os

import numpy as np

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
F = f"{ROOT}/evaluation/cpu_mindeye2_p1/features"
OUT = f"{ROOT}/evaluation/p1_protocol_final_audit"
os.makedirs(OUT, exist_ok=True)

MODELS = [("AlexNet(2)", "alexnet", 0), ("AlexNet(5)", "alexnet", 1), ("InceptionV3", "inception", 0)]
GROUPS = [("A_raw", "A"), ("B_refined", "B")]
SCOPES = [1000, 982]

# 既有报告中的数值，用于一致性核对
STORED = {
    ("AlexNet(2)", "A_raw", 1000): 0.977492, ("AlexNet(2)", "B_refined", 1000): 0.973730,
    ("AlexNet(2)", "A_raw", 982): 0.977733, ("AlexNet(2)", "B_refined", 982): 0.973966,
    ("AlexNet(5)", "A_raw", 1000): 0.993159, ("AlexNet(5)", "B_refined", 1000): 0.989296,
    ("AlexNet(5)", "A_raw", 982): 0.993131, ("AlexNet(5)", "B_refined", 982): 0.989287,
    ("InceptionV3", "A_raw", 1000): 0.962539, ("InceptionV3", "B_refined", 1000): 0.948914,
    ("InceptionV3", "A_raw", 982): 0.962359, ("InceptionV3", "B_refined", 982): 0.948658,
}

rows, report = [], {"official_evidence": {
    "notebook": "source/TextAlign-mindeye2-231850a/src/final_evaluations.ipynb",
    "cell": 17, "cross_check_cell_MindEyeV2": 14,
    "function": "two_way_identification",
    "official_print_label": "2-way Percent Correct",
}, "verdict": {}, "entries": []}

for name, mdl, li in MODELS:
    RA = np.load(f"{F}/{mdl}/R_A_raw__{li}.npy")
    RB = np.load(f"{F}/{mdl}/R_B_refined__{li}.npy")
    for gname, gshort in GROUPS:
        R = RA if gname == "A_raw" else RB
        for scope in SCOPES:
            r = R[:scope, :scope].astype(np.float64)
            N = r.shape[0]
            diag = np.diag(r)
            # 官方逐列成功计数（i=j 时 r<diag 恒 False，自动排除自身）
            c = (r < diag[None, :]).sum(axis=0)
            # 1) 官方口径 = 平均 pairwise 2-way
            acc2 = float(c.mean() / (N - 1))
            # 2) Strict N-way Top-1（两种实现交叉验证）
            hit_counts = (c == N - 1)
            argmax_hit = (np.argmax(r, axis=0) == np.arange(N))
            top1 = float(hit_counts.mean())
            top1_argmax = float(argmax_hit.mean())
            top1_methods_agree = bool((hit_counts == argmax_hit).all())
            # 3) Mean rank（1-based）
            rank = N - c                     # rank_j = N - c_j（恒等式已验证）
            mean_rank = float(rank.mean())
            median_rank = float(np.median(rank))
            # 恒等式: acc2 == 1 - (mean_rank - 1)/(N-1)
            identity_err = abs(acc2 - (1.0 - (mean_rank - 1) / (N - 1)))
            stored = STORED.get((name, gname, scope))
            rows.append({
                "model": name, "group": gname, "scope": scope, "N": N,
                "pairwise_2way_identification_official": round(acc2, 6),
                "strict_nway_top1": round(top1, 6),
                "strict_nway_top1_argmax_xcheck": round(top1_argmax, 6),
                "top1_methods_agree": top1_methods_agree,
                "mean_rank_1based": round(mean_rank, 3),
                "median_rank": round(median_rank, 1),
                "identity_err_2way_vs_rank": round(identity_err, 12),
                "matches_stored_report_acc": (stored is not None and abs(stored - acc2) < 1e-6),
                "stored_value": stored,
            })
            report["entries"].append(rows[-1])

# ---- 结论与修正版命名表 ----
d1000 = {(e["model"], e["group"]): e for e in report["entries"] if e["scope"] == 1000}
report["verdict"] = {
    "classification": "Mean Pairwise 2-way Identification（官方口径）",
    "not_strict_nway_top1": "Strict N-way Top-1 需要正确目标严格大于全部 N-1 个候选；官方公式只统计"
                            "『正确目标胜过的候选数比例』。同数据下 Top-1 仅 0.17~0.52，而官方口径 0.95~0.99。",
    "mean_rank_used_by_official": False,
    "naming_correction": "此前报告中的『N-way acc / acc_N1000』应改名为『2-way Identification"
                         "（mean pairwise，MindEye2 官方 two_way_identification）』",
    "evidence": "cell 17 打印标签即 '2-way Percent Correct'；cell 23/25/27 变量与注释一致。",
    "values_96_99_are": "Pairwise 2-way Identification（平均两两识别率），非 N-way Top-1。",
}

with open(f"{OUT}/protocol_audit.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)

with open(f"{OUT}/protocol_comparison.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

with open(f"{OUT}/CORRECTED_METRIC_NAMING_TABLE.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["metric_former_name", "correct_name", "group", "scope_1000", "scope_982",
                "strict_top1_1000", "strict_top1_982", "mean_rank_1000", "mean_rank_982"])
    for name, _, _ in MODELS:
        eA1 = d1000[(name, "A_raw")]; eB1 = d1000[(name, "B_refined")]
        eA9 = next(e for e in report["entries"] if e["model"] == name and e["group"] == "A_raw" and e["scope"] == 982)
        eB9 = next(e for e in report["entries"] if e["model"] == name and e["group"] == "B_refined" and e["scope"] == 982)
        w.writerow([f"{name} N-way acc", "2-way Identification (mean pairwise)", "A_raw",
                    eA1["pairwise_2way_identification_official"], eA9["pairwise_2way_identification_official"],
                    eA1["strict_nway_top1"], eA9["strict_nway_top1"],
                    eA1["mean_rank_1based"], eA9["mean_rank_1based"]])
        w.writerow([f"{name} N-way acc", "2-way Identification (mean pairwise)", "B_refined",
                    eB1["pairwise_2way_identification_official"], eB9["pairwise_2way_identification_official"],
                    eB1["strict_nway_top1"], eB9["strict_nway_top1"],
                    eB1["mean_rank_1based"], eB9["mean_rank_1based"]])

bad = [r for r in rows if not r["matches_stored_report_acc"] or not r["top1_methods_agree"]
       or r["identity_err_2way_vs_rank"] > 1e-9]
print(json.dumps({"n_rows": len(rows), "inconsistencies": len(bad)}, ensure_ascii=False))
for r in rows:
    print(r["model"], r["group"], r["scope"], "2way=", r["pairwise_2way_identification_official"],
          "top1=", r["strict_nway_top1"], "meanrank=", r["mean_rank_1based"],
          "stored_ok=", r["matches_stored_report_acc"])
print("PROTOCOL_AUDIT_DONE")
