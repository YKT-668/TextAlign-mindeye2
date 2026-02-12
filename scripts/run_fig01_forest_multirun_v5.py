# Fig01 CCD@1 per subject (multi-run list) — IJCAI-style polish
# Save to /mnt/work/repos/TextAlign-mindeye2/figures without overwriting

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0) Paths & I/O
# -----------------------------
CSV_PATH = "/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_ccd/ccd_summary.csv"
OUT_DIR  = "/mnt/work/repos/TextAlign-mindeye2/figures"
os.makedirs(OUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PNG = os.path.join(OUT_DIR, f"fig01_ccd_acc1_per_subj_multirun_point_{timestamp}.png")
OUT_PDF = os.path.join(OUT_DIR, f"fig01_ccd_acc1_per_subj_multirun_point_{timestamp}.pdf")

# -----------------------------
# 1) Color palette (strict)
# -----------------------------
COL_OFFICIAL = "#C4D5EB"  # blue-ish
COL_OURS     = "#E0CAEF"  # purple-ish
COL_GRID     = "#E8E8E8"
COL_CHANCE   = "#E8E8E8"  # keep light; line style will differentiate

# -----------------------------
# 2) Load & normalize schema
# -----------------------------
df = pd.read_csv(CSV_PATH)

# --- Column compat ---
# Subject column
if "subj" not in df.columns:
    raise ValueError("CSV 缺少 subj 列（你说已存在），请确认 ccd_summary.csv 表头。")

# Main metric
if "ccd_acc1" not in df.columns:
    raise ValueError("CSV 缺少 ccd_acc1 列。")

# CI columns (95% CI)
# allow a couple of naming variants just in case
ci_lo_candidates = ["ccd_acc1_ci95_lo", "ccd_acc1_ci_lo", "ccd_acc1_lo"]
ci_hi_candidates = ["ccd_acc1_ci95_hi", "ccd_acc1_ci_hi", "ccd_acc1_hi"]

ci_lo = next((c for c in ci_lo_candidates if c in df.columns), None)
ci_hi = next((c for c in ci_hi_candidates if c in df.columns), None)
if ci_lo is None or ci_hi is None:
    raise ValueError("CSV 缺少 ccd_acc1 的 95% CI 列（lo/hi）。请检查表头。")

# Filtering columns
# neg mode
neg_col = "neg_mode" if "neg_mode" in df.columns else ("neg" if "neg" in df.columns else None)
# K column (you said实际是 k_neg)
k_col = "k_neg" if "k_neg" in df.columns else ("K" if "K" in df.columns else None)
# repr column (you said实际是 eval_repr)
repr_col = "eval_repr" if "eval_repr" in df.columns else ("feature_name" if "feature_name" in df.columns else None)

if neg_col is None or k_col is None or repr_col is None:
    raise ValueError(f"过滤所需列缺失：neg_col={neg_col}, k_col={k_col}, repr_col={repr_col}。请检查 CSV 表头。")

# Tag column (you said实际列名叫 tag)
if "tag" not in df.columns:
    raise ValueError("CSV 缺少 tag 列（用于 run 标识）。")

# -----------------------------
# 3) Filter to match paper caption
#    pooled + hardneg + K=1 on shared982 (as in your Fig01 description)
# -----------------------------
# repr might be pooled / pooled_mean etc. accept pooled substring
df_f = df.copy()
df_f[repr_col] = df_f[repr_col].astype(str)

df_f = df_f[
    df_f[repr_col].str.contains("pooled", case=False, na=False) &
    (df_f[neg_col].astype(str).str.lower() == "hardneg") &
    (df_f[k_col].astype(int) == 1)
].copy()

if df_f.empty:
    raise ValueError("过滤后数据为空：请确认 eval_repr含pooled、neg_mode=hardneg、k_neg=1 是否存在于 CSV。")

# -----------------------------
# 4) Short label builder (避免重叠的关键)
#    Keep multi-run info but remove long paths / redundant prefixes
# -----------------------------
def short_run_label(tag: str) -> str:
    t = str(tag)

    # ours patterns
    # examples: ours_s1_v2, ours_s7_v10, ours_s1_from_official_full_v1 ...
    m = re.match(r"^ours[_\-]?s(\d+)[_\-]v(\d+)$", t)
    if m:
        return f"Ours-v{m.group(2)}"
    m = re.match(r"^ours[_\-]?s(\d+).*from[_\-]?official.*v(\d+)$", t)
    if m:
        return f"Ours-fromOff-v{m.group(2)}"
    if t.startswith("ours_"):
        # fallback: keep key tail, strip subject token if any
        t2 = t.replace("ours_", "Ours-")
        t2 = re.sub(r"s\d+_", "", t2)
        t2 = t2.replace("_", "-")
        return t2

    # official HF patterns:
    # examples: official_hf/final_subj01_pretrained_40sess_24bs
    # sometimes tag may contain slashes; keep only last chunk
    last = t.split("/")[-1]

    # squeeze: final_subj01_pretrained_40sess_24bs -> OfficialHF-40sess
    m = re.match(r"final_subj(\d+)_pretrained_(\d+)sess", last)
    if m:
        sess = m.group(2)
        return f"OfficialHF-{sess}sess"

    # official_s1/official_s1 or official_s1
    if "official" in t.lower():
        # Official-S1 etc.
        mm = re.search(r"s(\d+)", t.lower())
        if mm:
            return f"Official-S{mm.group(1)}"
        return "Official"

    # fallback: compact
    return last.replace("_", "-")

df_f["run_short"] = df_f["tag"].apply(short_run_label)

# group label for legend coloring
def method_group(tag: str) -> str:
    t = str(tag).lower()
    return "Ours" if t.startswith("ours") else "Official HF"

df_f["method_group"] = df_f["tag"].apply(method_group)

# -----------------------------
# 5) Plot: 1x4 facets, dot + 95%CI horizontal errorbar
# -----------------------------
subjects = ["subj01", "subj02", "subj05", "subj07"]
# Safe subj parsing if needed (depends on raw csv format "01" vs "subj01")
df_f["subj"] = df_f["subj"].astype(str).apply(lambda x: f"subj{int(x):02d}" if x.isdigit() else x)

panel_titles = {"subj01":"S01", "subj02":"S02", "subj05":"S05", "subj07":"S07"}

# layout tuning
fig_w, fig_h = 14.5, 4.2
fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h), sharex=True)
if len(subjects) == 1:
    axes = [axes]

# Leave room for suptitle + legend outside
fig.subplots_adjust(left=0.13, right=0.995, bottom=0.16, top=0.80, wspace=0.22)

# overall title
fig.suptitle("CCD@1 on shared982 (pooled, hardneg, K=1)", fontsize=14, y=0.94)

# legend handles (patched with Best Ours)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0],[0], marker='o', linestyle='None', markersize=7,
           markerfacecolor=COL_OFFICIAL, markeredgecolor='black', label="Official HF"),
    Line2D([0],[0], marker='o', linestyle='None', markersize=7,
           markerfacecolor=COL_OURS, markeredgecolor='black', label="Ours"),
    Line2D([0],[0], marker='D', linestyle='None', markersize=8,
           markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.6,
           label="Best Ours"),
    Line2D([0],[0], linestyle='--', color="gray", linewidth=1.5, label="Chance (0.5)")
]
# legend outside, above panels, ncol=4
fig.legend(handles=legend_handles, loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=12)

x_min, x_max = 0.42, 0.72

for ax, subj in zip(axes, subjects):
    d = df_f[df_f["subj"] == subj].copy()

    # sort within subject: highest CCD@1 on top
    d = d.sort_values("ccd_acc1", ascending=False).reset_index(drop=True)

    # y positions
    y = np.arange(len(d))

    # error bars: asymmetric
    x = d["ccd_acc1"].to_numpy() # float
    lo = d[ci_lo].to_numpy() # float
    hi = d[ci_hi].to_numpy() # float
    xerr = np.vstack([np.clip(x - lo, 0, None), np.clip(hi - x, 0, None)])

    colors = np.where(d["method_group"].to_numpy() == "Ours", COL_OURS, COL_OFFICIAL)

    ax.errorbar(
        x, y, xerr=xerr,
        fmt='o', markersize=7,
        ecolor="black", elinewidth=1.1, capsize=3,
        markeredgecolor="black",
        markerfacecolor="none",
        zorder=3
    )
    # draw filled markers on top
    ax.scatter(x, y, s=55, c=colors, edgecolors="black", linewidths=1.0, zorder=4)

    # ---- highlight: Best Ours (diamond outline) ----
    # patched here
    ours_mask = (d["method_group"].to_numpy() == "Ours")
    if np.any(ours_mask):
        # since d is already sorted by ccd_acc1 descending, the first Ours is the best Ours
        idx_best_ours = np.where(ours_mask)[0][0]
        x_best = x[idx_best_ours]
        y_best = y[idx_best_ours]
        ax.scatter(
            [x_best], [y_best],
            marker='D', s=95,
            facecolors='none', edgecolors='black',
            linewidths=1.8, zorder=5
        )
    # ------------------------------------------------

    # chance line
    ax.axvline(0.5, linestyle="--", linewidth=1.5, color="gray", zorder=1)

    # y tick labels (short labels)
    ax.set_yticks(y)
    ax.set_yticklabels(d["run_short"].tolist(), fontsize=11)

    # invert so best on top
    ax.invert_yaxis()

    # axes cosmetics
    ax.set_title(panel_titles.get(subj, subj), fontsize=16, pad=6)
    ax.set_xlim(x_min, x_max)
    ax.grid(axis='x', color=COL_GRID, linewidth=1.0, alpha=0.7)
    ax.grid(axis='y', visible=False)

    # spines: remove top/right for IJCAI clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Only left-most panel shows y label to reduce clutter
    if ax is axes[0]:
        ax.set_ylabel("Run (short label)", fontsize=13)
    else:
        ax.set_ylabel("")
        # keep ticks but reduce padding to avoid crowding
        ax.tick_params(axis='y', pad=2)

    ax.set_xlabel("CCD@1", fontsize=13)

# final save (high dpi)
fig.savefig(OUT_PNG, dpi=300)
fig.savefig(OUT_PDF)
plt.close(fig)

print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)
