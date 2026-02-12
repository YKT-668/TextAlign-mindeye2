import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
csv_path = "/mnt/work/repos/TextAlign-mindeye2/results/tables/ccd_ablation_difficulty.csv"
out_dir = "/mnt/work/repos/TextAlign-mindeye2/figures"
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "Fig09_ccd_ablation_difficulty_v2.png")
out_pdf = os.path.join(out_dir, "Fig09_ccd_ablation_difficulty_v2.pdf")

# =========================
# Palette (your academic colors)
# =========================
COL_HARDEST = "#C4D5EB"  # blue
COL_RANDOM  = "#FDE5D4"  # peach
COL_CONN    = "#E8E8E8"  # light gray
COL_TEXT    = "#2f2f2f"
COL_AXIS    = "#333333"
COL_CHANCE  = "#8f8f8f"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# =========================
# Load + column compat
# =========================
df = pd.read_csv(csv_path)

col_subj = _pick_col(df, ["subj", "subject", "subj_id"])
col_tag  = _pick_col(df, ["tag", "model", "run", "run_name"])
col_repr = _pick_col(df, ["eval_repr", "feature_name", "repr", "representation"])
col_k    = _pick_col(df, ["k_neg", "K", "k"])
col_negm = _pick_col(df, ["difficulty", "neg_mode", "neg_mode_name", "negmode"])

col_acc  = _pick_col(df, ["ccd_acc1", "acc1", "ccd@1"])
col_lo   = _pick_col(df, ["ccd_acc1_ci95_lo", "acc1_ci95_lo", "ci95_lo"])
col_hi   = _pick_col(df, ["ccd_acc1_ci95_hi", "acc1_ci95_hi", "ci95_hi"])

need = [col_subj, col_tag, col_repr, col_k, col_negm, col_acc, col_lo, col_hi]
if any(v is None for v in need):
    missing = [n for n,v in zip(
        ["subj","tag","repr","k","neg_mode","acc","ci_lo","ci_hi"], need
    ) if v is None]
    raise ValueError(f"Missing required columns in CSV: {missing}. "
                     f"Please check ccd_summary.csv headers.")

# =========================
# Filter: pooled + K=1 + (hardest/random)
# =========================
dff = df.copy()

# pooled filter (match your paper: pooled / pooled_mean / pooled_mean etc.)
dff = dff[dff[col_repr].astype(str).str.contains("pooled", case=False, na=False)]

# K=1 filter (or K=2 if K=1 not found)
dff[col_k] = pd.to_numeric(dff[col_k], errors="coerce")
if 1 in dff[col_k].unique():
    dff = dff[dff[col_k] == 1]
elif 2 in dff[col_k].unique():
    print("Warning: K=1 not found, using K=2")
    dff = dff[dff[col_k] == 2]

# keep only hardest/random (case-insensitive)
dff[col_negm] = dff[col_negm].astype(str).str.lower()
dff = dff[dff[col_negm].isin(["hardest", "random"])]

# =========================
# Select "best pooled model per subject"
# Rule: pick the tag that has highest ACC under HARDEST for each subject,
# then use the same tag's RANDOM record for paired comparison.
# =========================
# Normalize subject labels
def _norm_subj(x):
    s = str(x)
    # accept "1" or "subj01" or "s1"
    digits = "".join([ch for ch in s if ch.isdigit()])
    if digits != "":
        n = int(digits)
        return f"S{n:02d}"
    return s.upper()

dff["SUBJ"] = dff[col_subj].apply(_norm_subj)

# find best tag per subject based on hardest
hardest = dff[dff[col_negm] == "hardest"].copy()
best_rows = hardest.sort_values(col_acc, ascending=False).groupby("SUBJ", as_index=False).head(1)
best_map = dict(zip(best_rows["SUBJ"], best_rows[col_tag]))

# keep paired rows (hardest + random) for those best tags
keep = []
for subj, tag in best_map.items():
    tmp = dff[(dff["SUBJ"] == subj) & (dff[col_tag] == tag)]
    # require both modes
    if set(tmp[col_negm].unique()) >= {"hardest", "random"}:
        keep.append(tmp)
paired = pd.concat(keep, axis=0).copy()
paired = paired.sort_values("SUBJ")

# =========================
# Prepare plot data
# =========================
subjects = list(paired["SUBJ"].unique())
y = np.arange(len(subjects))

def _get(subj, mode, col):
    return float(paired[(paired["SUBJ"] == subj) & (paired[col_negm] == mode)][col].iloc[0])

hard_mean = np.array([_get(s, "hardest", col_acc) for s in subjects])
hard_lo   = np.array([_get(s, "hardest", col_lo) for s in subjects])
hard_hi   = np.array([_get(s, "hardest", col_hi) for s in subjects])

rand_mean = np.array([_get(s, "random", col_acc) for s in subjects])
rand_lo   = np.array([_get(s, "random", col_lo) for s in subjects])
rand_hi   = np.array([_get(s, "random", col_hi) for s in subjects])

# errorbar wants symmetric; we use [mean-lo, hi-mean]
hard_err = np.vstack([hard_mean - hard_lo, hard_hi - hard_mean])
rand_err = np.vstack([rand_mean - rand_lo, rand_hi - rand_mean])

# =========================
# Plot: Dumbbell with CI
# =========================
fig, ax = plt.subplots(figsize=(8.2, 3.8))

# grid + clean spines
ax.grid(True, axis="x", color=COL_CONN, linewidth=1.0)
ax.grid(False, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(COL_AXIS)
ax.spines["bottom"].set_color(COL_AXIS)
ax.tick_params(colors=COL_TEXT)

# connecting lines
for i in range(len(subjects)):
    ax.plot([hard_mean[i], rand_mean[i]], [y[i], y[i]], color=COL_CONN, lw=3.0, zorder=1)

# points + CI
cap = 3.5
ms = 8.0
lw = 2.0

h1 = ax.errorbar(
    hard_mean, y, xerr=hard_err,
    fmt="o", ms=ms,
    color=COL_AXIS, markerfacecolor=COL_HARDEST, markeredgecolor=COL_AXIS, markeredgewidth=1.0,
    ecolor=COL_AXIS, elinewidth=1.4, capsize=cap, capthick=1.4,
    label="Hardest (rigorous)", zorder=3
)

h2 = ax.errorbar(
    rand_mean, y, xerr=rand_err,
    fmt="D", ms=ms-0.5,
    color=COL_AXIS, markerfacecolor=COL_RANDOM, markeredgecolor=COL_AXIS, markeredgewidth=1.0,
    ecolor=COL_AXIS, elinewidth=1.4, capsize=cap, capthick=1.4,
    label="Random (easier)", zorder=3
)

# chance line at 0.5 (K=1 => chance=0.5)
ax.axvline(0.5, color=COL_CHANCE, linestyle="--", linewidth=1.6, zorder=0)
ax.text(0.5, 1.02, "Chance=0.5", transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", color=COL_CHANCE, fontsize=11)

# labels
ax.set_yticks(y, subjects)
ax.set_xlabel("CCD@1", color=COL_TEXT)
ax.set_title("CCD-hard difficulty ablation (Hardest vs Random)\n(best pooled model per subject)", pad=10, color=COL_TEXT)

# xlim: auto but keep readable margin
xmin = min(hard_lo.min(), rand_lo.min()) - 0.01
xmax = max(hard_hi.max(), rand_hi.max()) + 0.01
ax.set_xlim(max(0.0, xmin), min(1.0, xmax))

# legend at bottom (match your Fig01 style)
fig.legend(
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02)
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()

print("Saved:", out_png)
print("Saved:", out_pdf)
print("Best tag per subject (selected by Hardest):")
for k,v in best_map.items():
    print(k, "->", v)
