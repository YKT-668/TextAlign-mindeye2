import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) Paths
# =========================
CSV_PATH = "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32/_inputs_augmented/ccd_summary.csv"
OUT_DIR = "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "fig01_ccd_acc1_per_subj_dot_ci_2x2_v2.png")  # new name, no overwrite
OUT_PATH_PDF = os.path.join(OUT_DIR, "fig01_ccd_acc1_per_subj_dot_ci_2x2_v2.pdf")

# =========================
# 1) IJCAI-like styling (keep it clean)
# =========================
PALETTE = {
    "ours": "#E0CAEF",
    "official": "#C4D5EB",
    "best": "#FDE5D4",
    "grid": "#E8E8E8",
    "text": "#222222",
    "chance": "#777777",   # allowed as compatible neutral
    "err": "#222222",
}

plt.rcParams.update({
    "font.size": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelcolor": PALETTE["text"],
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
    "axes.edgecolor": PALETTE["text"],
})

# =========================
# 2) Load + robust column mapping
# =========================
df = pd.read_csv(CSV_PATH)

# ---- column aliases (your CSV may use these)
# metric
ACC_COL = "ccd_acc1"
CI_LO_COL = "ccd_acc1_ci95_lo"
CI_HI_COL = "ccd_acc1_ci95_hi"

# filters (compatible names)
NEG_MODE_COL = "neg_mode" if "neg_mode" in df.columns else ("neg_type" if "neg_type" in df.columns else None)
K_COL = "k_neg" if "k_neg" in df.columns else ("K" if "K" in df.columns else None)
REPR_COL = "eval_repr" if "eval_repr" in df.columns else ("feature_name" if "feature_name" in df.columns else None)

# identifiers
SUBJ_COL = "subj" if "subj" in df.columns else None
TAG_COL = "tag" if "tag" in df.columns else None

need = [ACC_COL, CI_LO_COL, CI_HI_COL, SUBJ_COL, TAG_COL]
missing = [c for c in need if c is None or c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

# =========================
# 3) Filter to match paper caption: pooled + hardneg + K=1
# =========================
f = df.copy()

if REPR_COL is not None:
    f = f[f[REPR_COL].astype(str).str.contains("pooled", case=False, na=False)]
if NEG_MODE_COL is not None:
    f = f[f[NEG_MODE_COL].astype(str).str.contains("hard", case=False, na=False)]
if K_COL is not None:
    f = f[pd.to_numeric(f[K_COL], errors="coerce") == 1]

# normalization of subject column if it's just '01'/'1' etc
f[SUBJ_COL] = f[SUBJ_COL].astype(str).apply(lambda x: f"subj{int(x):02d}" if x.isdigit() else x)

# keep only s01/s02/s05/s07 (paper)
f = f[f[SUBJ_COL].isin(["subj01", "subj02", "subj05", "subj07"])].copy()

# [Requested Change] Remove official-S1 for S01
f = f[~((f[SUBJ_COL] == "subj01") & (f[TAG_COL].str.contains("official_s1", case=False)))]

if f.empty:
    raise ValueError("Filtered dataframe is empty. Check filter conditions.")

# =========================
# 4) Build short labels & method group
# =========================
def short_run_label(tag: str) -> str:
    t = str(tag)

    # [Requested Change] s1_textalign_stage1 -> Ours-best32
    if "s1_textalign_stage1" in t:
        return "Ours-best32"

    # [Requested Change] S02 data grouped to OfficialHF-40sess
    # Covers final_subj02_pretrained_1sess_24bs and final_subj02_pretrained_40sess_24bs
    if "final_subj02" in t:
        return "OfficialHF-40sess"

    # unify official naming
    if "official_hf" in t or "final_subj" in t:
        # 40sess / 1sess
        if "40sess" in t:
            return "OfficialHF-40sess"
        if "1sess" in t:
            return "OfficialHF-1sess"
        return "OfficialHF"

    if "official_s1" in t or "official_s2" in t or "official_s5" in t or "official_s7" in t:
        # your earlier exported "official_s1/official_s1"
        m = re.search(r"official_(s\d+)", t)
        return f"Official-{m.group(1).upper()}" if m else "Official"

    # unify ours naming
    if "ours" in t:
        if "from_official" in t or "fromOfficial" in t or "from_off" in t:
            # ours_s1_from_official_full_v1 -> Ours-fromOff-v1
            m = re.search(r"v(\d+)", t)
            return f"Ours-fromOff-v{m.group(1)}" if m else "Ours-fromOff"
        
        # [Requested Change] ours_s1_v2 -> Ours-v10
        if "ours_s1_v2" in t:
            return "Ours-v10"

        # ours_s1_v2 -> Ours-v2
        m = re.match(r"^ours_s\d+_v(\d+)$", t)
        if m:
            return f"Ours-v{m.group(1)}"
        
        # fallback for other ours patterns
        m = re.search(r"v(\d+)", t)
        return f"Ours-v{m.group(1)}" if m else "Ours"

    # fallback
    return t[:22] + "..." if len(t) > 25 else t

def method_group(tag: str) -> str:
    t = str(tag).lower()
    if "ours" in t:
        return "ours"
    return "official"

f["run_short"] = f[TAG_COL].apply(short_run_label)
f["group"] = f[TAG_COL].apply(method_group)

# identify "Best Ours" per subject (max ACC among ours)
f["is_best_ours"] = False
for subj in f[SUBJ_COL].unique():
    sub_mask = (f[SUBJ_COL] == subj) & (f["group"] == "ours")
    if sub_mask.any():
        best_idx = f.loc[sub_mask, ACC_COL].astype(float).idxmax()
        f.loc[best_idx, "is_best_ours"] = True

# =========================
# 5) QUICK CHECKS (the “3 things”)
#   (a) CI bounds valid, (b) chance line consistent, (c) shared xlim
# =========================
# (a) CI validity
assert (f[CI_LO_COL] <= f[ACC_COL]).all() and (f[ACC_COL] <= f[CI_HI_COL]).all(), \
    "CI bounds do not bracket the mean (check CSV columns)."
assert ((f[[ACC_COL, CI_LO_COL, CI_HI_COL]] >= 0).all().all() and (f[[ACC_COL, CI_LO_COL, CI_HI_COL]] <= 1).all().all()), \
    "Values outside [0,1] (check metric scaling)."

# for shared x-limits (c)
xmin = float(np.floor((f[CI_LO_COL].min() - 0.02) * 100) / 100)
xmax = float(np.ceil((f[CI_HI_COL].max() + 0.02) * 100) / 100)
xmin = max(0.0, xmin)
xmax = min(1.0, xmax)

# =========================
# 6) Plot (2x2) — layout that never gets cramped
# =========================
subj_order = ["subj01", "subj02", "subj05", "subj07"]
panel_title = {"subj01":"S01", "subj02":"S02", "subj05":"S05", "subj07":"S07"}

fig, axes = plt.subplots(2, 2, figsize=(14.2, 7.0), dpi=300, sharex=True) # Increased height slightly
axes = axes.flatten()

# global title: bigger + higher
fig.suptitle("CCD@1 on shared982 (pooled, hardneg, K=1)", fontsize=18, x=0.55, y=0.990)

# legend handles (figure-level, move to bottom)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0],[0], marker='o', color='none', markerfacecolor=PALETTE["official"], markeredgecolor=PALETTE["text"],
           markersize=8, label="Official HF"),
    Line2D([0],[0], marker='o', color='none', markerfacecolor=PALETTE["ours"], markeredgecolor=PALETTE["text"],
           markersize=8, label="Ours"),
    Line2D([0],[0], marker='D', color='none', markerfacecolor=PALETTE["best"], markeredgecolor=PALETTE["text"],
           markersize=8, label="Best Ours"),
    Line2D([0],[0], linestyle='--', color=PALETTE["chance"], linewidth=1.6, label="Chance (0.5)")
]

for i, subj in enumerate(subj_order):
    ax = axes[i]
    d = f[f[SUBJ_COL] == subj].copy()

    # stable ordering: show ours near top (optional), then official; within group sort by acc
    d["sort_key"] = d["group"].map({"ours": 0, "official": 1})
    d = d.sort_values(["sort_key", ACC_COL], ascending=[True, True]).reset_index(drop=True)

    y = np.arange(len(d))

    # grid
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=1.0)

    # chance line (b)
    ax.axvline(0.5, linestyle="--", color=PALETTE["chance"], linewidth=1.6, zorder=0)

    # plot points
    for yi, row in d.iterrows():
        grp = row["group"]
        is_best = bool(row["is_best_ours"])

        if is_best:
            marker = "D"
            face = PALETTE["best"]
            size = 10
            z = 4
        else:
            marker = "o"
            face = PALETTE["ours"] if grp == "ours" else PALETTE["official"]
            size = 9
            z = 3

        ax.errorbar(
            row[ACC_COL], yi,
            xerr=np.array([[row[ACC_COL]-row[CI_LO_COL]], [row[CI_HI_COL]-row[ACC_COL]]]),
            fmt=marker,
            markersize=size,
            mfc=face,
            mec=PALETTE["text"],
            ecolor=PALETTE["err"],
            elinewidth=1.6,
            capsize=3,
            capthick=1.6,
            zorder=z
        )

    # y ticks (ONLY left column shows labels to avoid crowding)
    ax.set_yticks(y)
    if i in [0, 2]:  # left column panels
        ax.set_yticklabels(d["run_short"].tolist(), fontsize=12)
        ax.tick_params(axis="y", pad=8)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    ax.invert_yaxis()
    ax.set_title(panel_title.get(subj, subj), fontsize=18, pad=8)

    # x axis
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("CCD@1", fontsize=14)

# y label only once (cleaner)
axes[0].set_ylabel("Run (short label)", fontsize=14)
axes[2].set_ylabel("Run (short label)", fontsize=14)

# legend at bottom (require extra bottom margin)
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.55, -0.03), # adjusted bbox
    fontsize=13
)

# spacing: give title more air, give legend room
fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.12, wspace=0.12, hspace=0.35)

plt.savefig(OUT_PATH, bbox_inches="tight")
plt.savefig(OUT_PATH_PDF, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUT_PATH}")
print(f"Saved: {OUT_PATH_PDF}")
