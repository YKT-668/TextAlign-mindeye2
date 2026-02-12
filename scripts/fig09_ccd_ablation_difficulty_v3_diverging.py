import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
# csv_path = "/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_ccd/ccd_summary.csv"
# Auto-detect path from previous context since cache file doesn't exist at that location in this env
csv_path = "/mnt/work/repos/TextAlign-mindeye2/results/tables/ccd_ablation_difficulty.csv"

out_dir = "/mnt/work/repos/TextAlign-mindeye2/figures"
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "Fig09_ccd_ablation_difficulty_v3_diverging.png")
out_pdf = os.path.join(out_dir, "Fig09_ccd_ablation_difficulty_v3_diverging.pdf")

# =========================
# Palette (your academic colors)
# =========================
COL_HARDEST = "#C4D5EB"  # blue
COL_RANDOM  = "#FDE5D4"  # peach
COL_GRID    = "#E8E8E8"  # light gray
COL_TEXT    = "#2f2f2f"
COL_AXIS    = "#333333"
COL_ZERO    = "#8f8f8f"  # center line

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# =========================
# Load + column compat
# =========================
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

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

# pooled filter (pooled / pooled_mean / pooled_* etc.)
dff = dff[dff[col_repr].astype(str).str.contains("pooled", case=False, na=False)]

# K=1 filter
dff[col_k] = pd.to_numeric(dff[col_k], errors="coerce")
# Auto-fallback from previous attempt
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
def _norm_subj(x):
    s = str(x)
    digits = "".join([ch for ch in s if ch.isdigit()])
    if digits != "":
        n = int(digits)
        return f"S{n:02d}"
    return s.upper()

dff["SUBJ"] = dff[col_subj].apply(_norm_subj)

hardest = dff[dff[col_negm] == "hardest"].copy()
best_rows = hardest.sort_values(col_acc, ascending=False).groupby("SUBJ", as_index=False).head(1)
best_map = dict(zip(best_rows["SUBJ"], best_rows[col_tag]))

pairs = []
skipped = []
for subj, tag in best_map.items():
    tmp = dff[(dff["SUBJ"] == subj) & (dff[col_tag] == tag)]
    if set(tmp[col_negm].unique()) >= {"hardest", "random"}:
        pairs.append(tmp)
    else:
        skipped.append((subj, tag))

if len(pairs) == 0:
    raise RuntimeError("No subject has both hardest+random for the selected best tag. "
                       "Check if random mode exists in the CSV for those runs.")

paired = pd.concat(pairs, axis=0).copy().sort_values("SUBJ")
subjects = list(paired["SUBJ"].unique())

def _get(subj, mode, col):
    return float(paired[(paired["SUBJ"] == subj) & (paired[col_negm] == mode)][col].iloc[0])

# raw
hard_mean = np.array([_get(s, "hardest", col_acc) for s in subjects])
hard_lo   = np.array([_get(s, "hardest", col_lo) for s in subjects])
hard_hi   = np.array([_get(s, "hardest", col_hi) for s in subjects])

rand_mean = np.array([_get(s, "random", col_acc) for s in subjects])
rand_lo   = np.array([_get(s, "random", col_lo) for s in subjects])
rand_hi   = np.array([_get(s, "random", col_hi) for s in subjects])

# chance-centered (acc - 0.5)
chance = 0.5
h_mu = hard_mean - chance
h_lo = hard_lo   - chance
h_hi = hard_hi   - chance

r_mu = rand_mean - chance
r_lo = rand_lo   - chance
r_hi = rand_hi   - chance

# =========================
# Plot: Diverging bars + CI
# =========================
# layout
fig_h = 0.72 * len(subjects) + 2.0
fig, ax = plt.subplots(figsize=(7.8, fig_h))

# y positions (two offsets per subject)
y = np.arange(len(subjects))
off = 0.18
y_h = y - off
y_r = y + off

# grid & spines
ax.grid(True, axis="x", color=COL_GRID, linewidth=1.0)
ax.grid(False, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(COL_AXIS)
ax.spines["bottom"].set_color(COL_AXIS)
ax.tick_params(colors=COL_TEXT)

# center line at 0 (chance)
ax.axvline(0.0, color=COL_ZERO, linestyle="--", linewidth=1.6, zorder=0)
ax.text(0.0, 1.02, "Chance (0.5)", transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", color=COL_ZERO, fontsize=11)

# bars
bar_h = ax.barh(y_h, h_mu, height=0.32, color=COL_HARDEST, edgecolor=COL_AXIS, linewidth=1.0,
                label="Hardest (rigorous)", zorder=2)
bar_r = ax.barh(y_r, r_mu, height=0.32, color=COL_RANDOM,  edgecolor=COL_AXIS, linewidth=1.0,
                label="Random (easier)", zorder=2)

# error bars (CI)
cap = 3.5
ax.errorbar(h_mu, y_h, xerr=[h_mu - h_lo, h_hi - h_mu], fmt="none",
            ecolor=COL_AXIS, elinewidth=1.4, capsize=cap, capthick=1.4, zorder=3)
ax.errorbar(r_mu, y_r, xerr=[r_mu - r_lo, r_hi - r_mu], fmt="none",
            ecolor=COL_AXIS, elinewidth=1.4, capsize=cap, capthick=1.4, zorder=3)

# y labels
ax.set_yticks(y, subjects)

# x label (explicitly say we center at chance)
ax.set_xlabel("Above-chance CCD@1  (CCD@1 − 0.5)", color=COL_TEXT)
# ax.set_title("CCD-hard difficulty ablation: Hardest vs Random\n(best pooled model per subject)", pad=10, color=COL_TEXT)

fig.suptitle("CCD-hard difficulty ablation: Hardest vs Random\n(best pooled model per subject)", 
             y=0.87, fontsize=14, color=COL_TEXT)

# xlim: symmetric-ish for aesthetics
xmin = min(h_lo.min(), r_lo.min()) - 0.01
xmax = max(h_hi.max(), r_hi.max()) + 0.01
m = max(abs(xmin), abs(xmax), 0.05)
ax.set_xlim(-m, m)

# legend bottom
fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.06, 1, 0.93])

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()

print("Saved:", out_png)
print("Saved:", out_pdf)
print("Best tag per subject (selected by Hardest):")
for k, v in best_map.items():
    print(k, "->", v)
if skipped:
    print("Skipped subjects (missing paired hardest+random for best tag):", skipped)
