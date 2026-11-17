"""从 nway_audit.json 生成 P1 修正后的 p 值/CI 汇总 CSV（便于阅读）。"""
import csv
import json

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
AUD = f"{ROOT}/evaluation/cpu_mindeye2_p1_stat_audit"
d = json.load(open(f"{AUD}/nway_audit.json", encoding="utf-8"))

rows = []
for name, e in d["per_model"].items():
    for scope, s in e["scopes"].items():
        rows.append({
            "metric": name, "scope": scope, "acc_A": s["acc_A"], "acc_B": s["acc_B"],
            "delta": s["delta"], "ci95_low": s["delta_ci95_boot"][0], "ci95_high": s["delta_ci95_boot"][1],
            "wilcoxon_z": s["wilcoxon_z"],
            "wilcoxon_log10p": s["wilcoxon_log10p"],
            "p_expr": (f"< 1e{int(s['wilcoxon_log10p'])}" if s["wilcoxon_log10p"] < -4 else f"{10**s['wilcoxon_log10p']:.2e}"),
            "hit1_A": s["hit1_A"], "hit1_B": s["hit1_B"],
            "mcnemar_log10p": s["mcnemar_log10p"],
            "improved": s["improved_items"], "worsened": s["worsened_items"], "tied": s["tied_items"],
        })
for metric, v in d["holm"]["metrics"].items():
    rows.append({
        "metric": metric, "scope": d["holm"]["scope"], "acc_A": "", "acc_B": "",
        "delta": "", "ci95_low": "", "ci95_high": "",
        "wilcoxon_z": v["wilcoxon_z"], "wilcoxon_log10p": v["log10p"],
        "p_expr": (f"< 1e{int(v['log10p'])}" if v["log10p"] < -4 else f"{10**v['log10p']:.2e}"),
        "hit1_A": "", "hit1_B": "", "mcnemar_log10p": "",
        "improved": "", "worsened": "", "tied": "",
        "holm_log10p": v["holm_log10p"], "holm_p_expr": v["holm_p_expr"],
        "holm_significant_0.05": v["significant_holm_0.05"],
    })

fields = ["metric", "scope", "acc_A", "acc_B", "delta", "ci95_low", "ci95_high", "wilcoxon_z",
          "wilcoxon_log10p", "p_expr", "hit1_A", "hit1_B", "mcnemar_log10p",
          "improved", "worsened", "tied", "holm_log10p", "holm_p_expr", "holm_significant_0.05"]
with open(f"{AUD}/p1_corrected_pvalues.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
print("wrote", len(rows), "rows")
