import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
in_csv_candidates = ["/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32/_inputs_augmented/rsa_summary.csv",
    # Prioritize main_results.csv as it contains all subjects
    "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32/_inputs_augmented/rsa_summary.csv",
    "/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_rsa/rsa_summary.csv",
    "/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_rsa/rsa_results.csv",
]
out_dir = "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32"
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.png")
out_pdf = os.path.join(out_dir, "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.pdf")

# Your palette
COL_OURS = "#E0CAEF"
COL_BASE = "#E8E8E8"
COL_BASE_EDGE = "#6b6b6b"
COL_AXIS = "#2f2f2f"
COL_GRID = "#E8E8E8"
COL_LINE = "#9a9a9a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def pick_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Cannot find RSA summary CSV. Please put the correct path into in_csv_candidates.\n"
        "Tried:\n" + "\n".join(paths)
    )

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def norm_subj(x):
    s = str(x)
    digits = "".join([ch for ch in s if ch.isdigit()])
    if digits:
        return f"S{int(digits):02d}"
    # fallback: already like S01
    return s.upper()

def is_official_40sess(tag: str):
    t = str(tag).lower()
    # 1. Official
    if ("official" in t) and ("40" in t or "40sess" in t):
        return True
    # 2. Baseline keyword
    if ("baseline" in t) and ("40" in t or "40sess" in t):
        return True
    # 3. Final pretrained 40sess
    if ("final" in t) and ("pretrained" in t) and ("40" in t):
        return True
    return False

def is_ours(tag: str):
    t = str(tag).lower()
    return ("ours" in t) or ("textalign" in t)

# =========================
# Load
# =========================
csv_path = pick_existing(in_csv_candidates)
df = pd.read_csv(csv_path)

# Column compat
col_subj = _pick_col(df, ["subj", "subject", "subj_id"])
col_tag  = _pick_col(df, ["tag", "model", "run", "run_name"])
col_repr = _pick_col(df, ["eval_repr", "feature_name", "repr", "representation"])
col_metric = _pick_col(df, ["metric", "metric_name", "rsa_metric"])

# prefer pearson pooled
col_val  = _pick_col(df, ["rsa", "rsa_val", "value", "pearson", "rsa_pearson", "rsa_mean", "mean"])
col_lo   = _pick_col(df, ["ci95_lo", "rsa_ci95_lo", "rsa_pearson_ci95_lo", "pearson_ci95_lo"])
col_hi   = _pick_col(df, ["ci95_hi", "rsa_ci95_hi", "rsa_pearson_ci95_hi", "pearson_ci95_hi"])

# If using main_results.csv, fake eval_repr to pass filtering if missing
if col_repr is None and "main_results.csv" in csv_path:
    col_repr = "eval_repr"
    df[col_repr] = "pooled"

need = [col_subj, col_tag, col_val]
if any(v is None for v in need):
    raise ValueError(f"Missing required columns. Found columns: {list(df.columns)}")

dff = df.copy()
dff["SUBJ"] = dff[col_subj].apply(norm_subj)

# pooled filter（若有 repr 列）
if col_repr is not None:
    dff = dff[dff[col_repr].astype(str).str.contains("pooled", case=False, na=False)]

# pearson filter（若有 metric 列）
if col_metric is not None:
    dff = dff[dff[col_metric].astype(str).str.contains("pearson", case=False, na=False)]

# coerce numeric
dff[col_val] = pd.to_numeric(dff[col_val], errors="coerce")
dff = dff.dropna(subset=[col_val])

# =========================
# Select per subject:
# - baseline: official 40sess
# - ours: best ours (max RSA)
# =========================
rows = []
subjects = sorted(dff["SUBJ"].unique().tolist())

for s in subjects:
    dsub = dff[dff["SUBJ"] == s].copy()

    base = dsub[dsub[col_tag].apply(is_official_40sess)]
    if len(base) == 0:
        # fallback: any "official"
        base = dsub[dsub[col_tag].astype(str).str.lower().str.contains("official", na=False)]
    if len(base) == 0:
        continue
    base = base.sort_values(col_val, ascending=False).head(1).iloc[0]

    ours = dsub[dsub[col_tag].apply(is_ours)]
    if len(ours) == 0:
        continue
    ours = ours.sort_values(col_val, ascending=False).head(1).iloc[0]

    rows.append({
        "SUBJ": s,
        "base_tag": str(base[col_tag]),
        "ours_tag": str(ours[col_tag]),
        "base": float(base[col_val]),
        "ours": float(ours[col_val]),
        "base_lo": float(base[col_lo]) if col_lo is not None and pd.notna(base.get(col_lo, np.nan)) else np.nan,
        "base_hi": float(base[col_hi]) if col_hi is not None and pd.notna(base.get(col_hi, np.nan)) else np.nan,
        "ours_lo": float(ours[col_lo]) if col_lo is not None and pd.notna(ours.get(col_lo, np.nan)) else np.nan,
        "ours_hi": float(ours[col_hi]) if col_hi is not None and pd.notna(ours.get(col_hi, np.nan)) else np.nan,
    })

plot_df = pd.DataFrame(rows)
if len(plot_df) == 0:
    raise RuntimeError("No valid subjects found after filtering. Check your CSV content/filters.")

# Keep only common NSD subjects order if present
preferred = ["S01", "S02", "S05", "S07"]
plot_df["order"] = plot_df["SUBJ"].apply(lambda x: preferred.index(x) if x in preferred else 999)
plot_df = plot_df.sort_values("order").drop(columns=["order"])

# =========================
# Plot: Dumbbell + CI
# =========================
fig_h = 1.2 + 0.75 * len(plot_df)
fig, ax = plt.subplots(figsize=(6.6, fig_h))

y = np.arange(len(plot_df))
x_base = plot_df["base"].to_numpy()
x_ours = plot_df["ours"].to_numpy()

# connecting lines
for i in range(len(y)):
    ax.plot([x_base[i], x_ours[i]], [y[i], y[i]], color=COL_LINE, linewidth=2.0, zorder=1)

# points
ax.scatter(x_base, y, s=120, facecolor=COL_BASE, edgecolor=COL_BASE_EDGE, linewidth=1.6, zorder=3, label="Official HF (40sess)")
ax.scatter(x_ours, y, s=140, facecolor=COL_OURS, edgecolor=COL_AXIS, linewidth=1.6, zorder=4, label="Ours (best)")

# error bars if CI exists
def has_ci(lo, hi):
    return np.isfinite(lo).all() and np.isfinite(hi).all()

if col_lo is not None and col_hi is not None:
    base_lo = plot_df["base_lo"].to_numpy()
    base_hi = plot_df["base_hi"].to_numpy()
    ours_lo = plot_df["ours_lo"].to_numpy()
    ours_hi = plot_df["ours_hi"].to_numpy()

    # draw only if not all nan
    if np.isfinite(base_lo).any() and np.isfinite(base_hi).any():
        ax.errorbar(x_base, y, xerr=[x_base - base_lo, base_hi - x_base],
                    fmt="none", ecolor=COL_BASE_EDGE, elinewidth=1.6, capsize=4, capthick=1.6, zorder=2)
    if np.isfinite(ours_lo).any() and np.isfinite(ours_hi).any():
        ax.errorbar(x_ours, y, xerr=[x_ours - ours_lo, ours_hi - x_ours],
                    fmt="none", ecolor=COL_AXIS, elinewidth=1.6, capsize=4, capthick=1.6, zorder=2)

# delta annotation
delta = x_ours - x_base
for i, d in enumerate(delta):
    ax.text(max(x_base[i], x_ours[i]) + 0.003, y[i],
            f"{d:+.3f}", va="center", ha="left", color=COL_AXIS, fontsize=11)

# axes, grid, labels
ax.set_yticks(y, plot_df["SUBJ"].tolist())
ax.set_xlabel("RSA (Pearson)  ↑", color=COL_AXIS)
# ax.set_title("RSA (Pearson) on shared982 (pooled)\nBest Ours vs Official 40-session baseline", pad=10)
fig.suptitle("RSA (Pearson) on shared982 (pooled)\nBest Ours vs Official 40-session baseline", y=0.96, fontsize=14, color=COL_AXIS)

ax.grid(True, axis="x", color=COL_GRID, linewidth=1.0)
ax.grid(False, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(COL_AXIS)
ax.spines["bottom"].set_color(COL_AXIS)

# xlim with padding
xmin = min(np.min(x_base), np.min(x_ours)) - 0.02
xmax = max(np.max(x_base), np.max(x_ours)) + 0.04
ax.set_xlim(xmin, xmax)

# legend bottom
fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.05, 1, 0.9])

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()

print("Saved:", out_png)
print("Saved:", out_pdf)
print("\nSelected tags per subject:")
for _, r in plot_df.iterrows():
    print(f"{r['SUBJ']} | base={r['base_tag']} | ours={r['ours_tag']}")
