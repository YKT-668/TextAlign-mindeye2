# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0) paths
# =========================
OUT_DIR = "/mnt/work/repos/TextAlign-mindeye2/figures"
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "Fig_efficiency_ccd_acc1_v1.png")  # 不覆盖原图：换新文件名
# Also save as PDF
out_path_pdf = os.path.join(OUT_DIR, "Fig_efficiency_ccd_acc1_v1.pdf")

# 你的配色（允许搭配深灰做文字/边框）
COL_S1 = "#C4D5EB"  # 蓝
COL_S5 = "#FDE5D4"  # 杏
GRID_C = "#E8E8E8"
TITLE_C = "#222222"
EDGE_C = "#222222"

# =========================
# 1) load data
# =========================
# 自动定位 CSV (使用 efficiency_summary.csv)
csv_path = "/mnt/work/repos/TextAlign-mindeye2/results/tables/efficiency_summary.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Cannot find CSV at {csv_path}")

df = pd.read_csv(csv_path)

# =========================
# 处理列名差异
# efficiency_summary.csv 的列名通常是:
# subj, model, setting, seed, N, ccd_acc1, ccd_ci_lo, ccd_ci_hi ...
# =========================

rename_map = {
    "setting": "budget",   # setting -> budget
    # model -> method (无需改，后面有处理)
}
# 如果存在 ccd_ci_lo，将其映射为 ccd_acc1_ci95_lo 以兼容后续代码
if "ccd_ci_lo" in df.columns:
    rename_map["ccd_ci_lo"] = "ccd_acc1_ci95_lo"
if "ccd_ci_hi" in df.columns:
    rename_map["ccd_ci_hi"] = "ccd_acc1_ci95_hi"

df = df.rename(columns=rename_map)

# 确保 method, subj, budget 列存在
if "method" not in df.columns and "model" in df.columns:
    df["method"] = df["model"]
    
if "subj" not in df.columns:
    # 可能 subj 已经是数字
    pass # subj 在 csv 里就是 subj

# --- 核心逻辑：数据清理与标准化 ---

def normalize_row(row):
    # Method
    m = str(row.get("method", "")).lower()
    # efficiency_summary.csv 里已经是 "baseline", "textalign_llm"
    
    # Subj
    # csv 里是 1, 5
    s_raw = row.get("subj", "")
    try:
        s_int = int(s_raw)
        s_str = f"s{s_int}"
    except:
        s_str = str(s_raw).lower()
        if "subj" in s_str:
            s_str = s_str.replace("subj", "s")
            
    # Budget
    b_raw = str(row.get("budget", "")).lower()
    if b_raw in ["1", "1sess", "1h"]:
        b_str = "1sess"
    elif b_raw in ["2", "2sess", "2h"]:
        b_str = "2sess"
    elif b_raw in ["40", "40sess", "full"]:
        b_str = "40sess"
    else:
        b_str = b_raw # keep as is
        
    return pd.Series([m, s_str, b_str])

df[["method", "subj", "budget"]] = df.apply(normalize_row, axis=1)

# Filter
df = df[df["method"].isin(["baseline", "textalign_llm"])]
df = df[df["subj"].isin(["s1", "s5"])]
df = df[df["budget"].isin(["1sess", "2sess", "40sess"])]

print("Debug: rows after parsing/filtering:", len(df))
print(df[["method", "subj", "budget", "ccd_acc1"]].head(10))

# =========================
# 2) canonical order / labels
# =========================
# 把 “1h/2h/full” 改成论文一致的 “1 sess / 2 sess / 40 sess”
budget_order = ["1sess", "2sess", "40sess"]
budget_label = {"1sess":"1 sess", "2sess":"2 sess", "40sess":"40 sess"}
# 兼容 budget 是数字的情况 (already normalized in parse_info above but keep user logic)
df["budget"] = df["budget"].astype(str).str.replace("full", "40sess")
df["budget"] = df["budget"].str.replace("1h", "1sess").str.replace("2h", "2sess")
df.loc[df["budget"].isin(["40", "40.0"]), "budget"] = "40sess"
df.loc[df["budget"].isin(["1", "1.0"]), "budget"] = "1sess"
df.loc[df["budget"].isin(["2", "2.0"]), "budget"] = "2sess"

df["method"] = df["method"].astype(str).str.lower()

# =========================
# 3) plot config (IJCAI-like)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
fig.suptitle("Data efficiency on shared982 (CCD@1, hardneg, K=1)", fontsize=16, y=1.02, color=TITLE_C)

panel_specs = [
    ("baseline", "Baseline"),
    ("textalign_llm", "TextAlign + LLM"),
]

subj_specs = [
    ("s1", "S01", COL_S1, "o"),
    ("s5", "S05", COL_S5, "s"),
]

x = np.arange(len(budget_order))

for ax, (mkey, mtitle) in zip(axes, panel_specs):
    ax.set_title(mtitle, pad=8)
    ax.grid(True, axis="y", color=GRID_C, linewidth=1.0)
    ax.grid(False, axis="x")

    for skey, slabel, col, marker in subj_specs:
        d = df[(df["method"] == mkey) & (df["subj"] == skey)].copy()
        if d.empty:
            print(f"Warning: No data for {mkey} {skey}")
            continue
        # Drop duplicates if any (e.g. multiple seeds), take mean or best? 
        # Usually seed0 is standard, or we should aggregate. 
        # Let's aggregate by mean if multiple seeds exist, but keep it simple first
        # Sort by budget
        d["budget"] = pd.Categorical(d["budget"], categories=budget_order, ordered=True)
        # Group by budget to handle multiple entries (e.g. seeds)
        d_agg = d.groupby("budget", observed=True).agg({
            "ccd_acc1": "mean",
            "ccd_acc1_ci95_lo": "mean", # Approx
            "ccd_acc1_ci95_hi": "mean"
        }).reset_index()
        
        d = d_agg.sort_values("budget")
        print(f"Plotting {mkey} - {skey}: {len(d)} points")

        y = d["ccd_acc1"].to_numpy()
        lo = d["ccd_acc1_ci95_lo"].to_numpy()
        hi = d["ccd_acc1_ci95_hi"].to_numpy()
        
        if len(y) == 0:
             print(f"Skipping {mkey}-{skey} : No data")
             continue

        # Plotting logic for both full and partial data
        # Check if we need to align x-axis due to missing budgets
        # x starts at 0 for 1sess, 1 for 2sess, 2 for 40sess
        budget_indices = [budget_order.index(b) for b in d["budget"]]
        x_plot = np.array(budget_indices)
        y_plot = y
        lo_plot = lo
        hi_plot = hi
        yerr_plot = np.vstack([y_plot - lo_plot, hi_plot - y_plot])
        
        ax.errorbar(
            x_plot, y_plot, yerr=yerr_plot,
            fmt=marker+"-",
            linewidth=2.6,
            markersize=7.0,
            capsize=3.5,
            color=col,
            ecolor=EDGE_C,
            elinewidth=1.4,
            markeredgecolor=EDGE_C,
            markeredgewidth=1.2,
            zorder=3,
            label=slabel
        )

    ax.set_xticks(x, [budget_label[b] for b in budget_order])
    ax.set_xlabel("Training budget (sessions)")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(EDGE_C)
    ax.spines["bottom"].set_color(EDGE_C)

axes[0].set_ylabel("CCD@1")

# 全局 legend 放底部（不挤图）
# Dedup labels
handles, labels = axes[0].get_legend_handles_labels()
# If axes[0] missing some curves (e.g. s5 missing), try axes[1]
h2, l2 = axes[1].get_legend_handles_labels()
all_handles = handles + h2
all_labels = labels + l2
by_label = dict(zip(all_labels, all_handles)) 
# Sort by strict order S01, S05
final_handles = []
final_labels = []
for sl in ["S01", "S05"]:
    if sl in by_label:
        final_handles.append(by_label[sl])
        final_labels.append(sl)

fig.legend(final_handles, final_labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.02),
           frameon=False, fontsize=12)

fig.tight_layout(rect=[0, 0.06, 1, 0.94])
fig.savefig(out_path, dpi=300, bbox_inches="tight")
fig.savefig(out_path_pdf, dpi=300, bbox_inches="tight")
print("saved to:", out_path)
print("saved to:", out_path_pdf)
