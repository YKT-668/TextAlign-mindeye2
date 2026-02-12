import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) Path
# =========================
CSV_PATH = "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32/_inputs_augmented/rsa_summary.csv"
OUT_DIR = Path("/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32")

# =========================
# 1) Style (IJCAI-like + your palette)
# =========================
COL_OFFICIAL = "#C4D5EB"
COL_OURS     = "#E0CAEF"
COL_GRID     = "#E8E8E8"
COL_TEXT     = "#2F2F2F"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.edgecolor": COL_TEXT,
    "text.color": COL_TEXT,
    "axes.labelcolor": COL_TEXT,
    "xtick.color": COL_TEXT,
    "ytick.color": COL_TEXT,
    "font.family": "serif",
})

def safe_save(fig, out_path: Path, dpi=300):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        stem, suf = out_path.stem, out_path.suffix
        for i in range(2, 100):
            cand = out_path.with_name(f"{stem}_v{i}{suf}")
            if not cand.exists():
                out_path = cand
                break
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    # Also save PDF
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    print(f"[Saved] {out_path}")
    print(f"[Saved] {pdf_path}")

# =========================
# 2) Load + robust column detection
# =========================
df = pd.read_csv(CSV_PATH)
cols = {c.lower(): c for c in df.columns}

def pick_col(*cands):
    for c in cands:
        if c in cols:
            return cols[c]
    return None

subj_col   = pick_col("subj", "subject")
tag_col    = pick_col("tag", "run", "run_tag", "model", "ckpt")
repr_col   = pick_col("eval_repr", "feature_name", "repr")
metric_col = pick_col("metric", "metric_name")

# mean / ci columns (try common patterns)
# Prefer explicit rsa_pearson
mean_col = None
if "rsa_pearson" in cols:
    mean_col = cols["rsa_pearson"]
else:
    for key in ["rsa", "pearson", "rsa_mean", "mean"]:
        if key in cols:
            mean_col = cols[key]
            break

# CI columns logic: prefer _p suffix if pearson is used
ci_lo_col = None
ci_hi_col = None

if "ci95_low_p" in cols: ci_lo_col = cols["ci95_low_p"]
elif "ci95_lower_p" in cols: ci_lo_col = cols["ci95_lower_p"]

if "ci95_high_p" in cols: ci_hi_col = cols["ci95_high_p"]
elif "ci95_upper_p" in cols: ci_hi_col = cols["ci95_upper_p"]

if ci_lo_col is None:
    for key in cols:
        lk = key.lower()
        if ("ci95" in lk or "95" in lk) and ("lo" in lk or "lower" in lk):
            ci_lo_col = cols[key]
if ci_hi_col is None:
    for key in cols:
        lk = key.lower()
        if ("ci95" in lk or "95" in lk) and ("hi" in lk or "upper" in lk):
            ci_hi_col = cols[key]

if tag_col is None:
    raise ValueError("Cannot find a tag/run column. Expected one of: tag/run/run_tag/model/ckpt")

# Normalize Subject Column to subj01, subj02 etc.
if subj_col is not None:
    def normalize_subj(x):
        x_str = str(x).lower().strip()
        # if already "subj01", keep it
        if x_str.startswith("subj"):
            return x_str
        # if "01", "1", "s1", map it
        # Extract digits
        digits = re.findall(r"\d+", x_str)
        if digits:
            return f"subj{int(digits[0]):02d}"
        return x_str # fallback
    
    df[subj_col] = df[subj_col].apply(normalize_subj)
else:
    # infer from tag
    def infer_subj(x):
        x = str(x)
        m = re.search(r"subj0?(\d+)", x)
        if m: return f"subj{int(m.group(1)):02d}"
        m = re.search(r"\bs(\d+)\b", x)
        if m: return f"subj{int(m.group(1)):02d}"
        return None
    df["__subj"] = df[tag_col].apply(infer_subj)
    subj_col = "__subj"

# =========================
# 3) Filter: pooled + pearson (if possible)
# =========================
dff = df.copy()

if repr_col is not None:
    # Filter for pooled
    dff = dff[dff[repr_col].astype(str).str.contains("pooled", case=False, na=False)]

if metric_col is not None:
    dff = dff[dff[metric_col].astype(str).str.contains("pearson", case=False, na=False)]

# drop rows without mean
if mean_col is None:
    print("Available columns:", df.columns.tolist())
    raise ValueError("Cannot find RSA mean column (e.g., rsa_pearson / rsa / pearson).")
    
dff = dff.dropna(subset=[subj_col, mean_col])

# subjects in your paper
SUBJ_ORDER = ["subj01", "subj02", "subj05", "subj07"]
dff = dff[dff[subj_col].isin(SUBJ_ORDER)].copy()

print(f"Filtered DataFrame size: {len(dff)}")

# =========================
# 4) Select baseline + best ours per subject
# =========================
def is_official_hf_40(x: str):
    x = str(x).lower()
    return ("official_hf" in x or "officialhf" in x) and ("40sess" in x or "40" in x and "sess" in x)

def is_ours(x: str):
    x = str(x).lower()
    return "ours" in x

rows = []
for s in SUBJ_ORDER:
    ds = dff[dff[subj_col] == s].copy()

    # Find baseline
    base = ds[ds[tag_col].apply(is_official_hf_40)]
    if len(base) == 0:
        # fallback: any official with 40sess
        base = ds[ds[tag_col].astype(str).str.contains("40sess", case=False, na=False)]
    
    if len(base) == 0:
        print(f"Warning: No baseline row found for {s}.")
    else:
        # Take the best one if multiple
        base_row = base.sort_values(mean_col, ascending=False).iloc[0]
    
    # Find Ours
    ours = ds[ds[tag_col].apply(is_ours)]
    if len(ours) == 0:
        print(f"Warning: No ours row found for {s}.")
    else:
        ours_row = ours.sort_values(mean_col, ascending=False).iloc[0]

    # Only append if both found? Or fill NaN? 
    # The snippet implies we want to compare method vs ours.
    # If one is missing, plotting will be weird. Let's assume we need both.
    if len(base) > 0 and len(ours) > 0:
        rows.append((s, "Official HF", base_row[mean_col], base_row.get(ci_lo_col, np.nan), base_row.get(ci_hi_col, np.nan)))
        rows.append((s, "Ours",        ours_row[mean_col], ours_row.get(ci_lo_col, np.nan), ours_row.get(ci_hi_col, np.nan)))

if not rows:
    print("Debug: All rows in dff:")
    print(dff[[subj_col, tag_col, mean_col]].head(20))
    raise ValueError("No data rows generated for plotting.")

plot_df = pd.DataFrame(rows, columns=["subj", "method", "mean", "ci_lo", "ci_hi"])

# =========================
# 5) Plot (grouped bars + 95% CI)
# =========================
fig, ax = plt.subplots(figsize=(8.6, 3.6))

x_pos = np.arange(len(SUBJ_ORDER))
w = 0.34

def ci_to_yerr(m, lo, hi):
    if np.any(pd.isna([lo, hi])):
        return None
    return np.vstack([m - lo, hi - m])

official = plot_df[plot_df["method"] == "Official HF"].set_index("subj").reindex(SUBJ_ORDER)
ours     = plot_df[plot_df["method"] == "Ours"].set_index("subj").reindex(SUBJ_ORDER)

y1 = official["mean"].values
y2 = ours["mean"].values

if ci_lo_col and ci_hi_col:
    yerr1 = ci_to_yerr(y1, official["ci_lo"].values, official["ci_hi"].values)
    yerr2 = ci_to_yerr(y2, ours["ci_lo"].values, ours["ci_hi"].values)
else:
    yerr1 = None
    yerr2 = None

ax.bar(x_pos - w/2, y1, w, label="Official HF", color=COL_OFFICIAL, edgecolor=COL_TEXT, linewidth=1.0,
       yerr=yerr1, ecolor=COL_TEXT, capsize=3)
ax.bar(x_pos + w/2, y2, w, label="Ours", color=COL_OURS, edgecolor=COL_TEXT, linewidth=1.0,
       yerr=yerr2, ecolor=COL_TEXT, capsize=3)

# axes
ax.set_xticks(x_pos)
ax.set_xticklabels(["S01", "S02", "S05", "S07"])
ax.set_ylabel("RSA (Pearson)")
ax.grid(axis="y", color=COL_GRID, linewidth=1.0)
ax.set_axisbelow(True)

# y-lim
vals = plot_df["mean"].values
if ci_lo_col:
    los = plot_df["ci_lo"].dropna().values
    his = plot_df["ci_hi"].dropna().values
    
    if len(los) > 0:
        all_lo = min(np.min(vals), np.min(los))
        all_hi = max(np.max(vals), np.max(his))
    else:
        all_lo = np.min(vals)
        all_hi = np.max(vals)
else:
    all_lo = np.min(vals)
    all_hi = np.max(vals)

pad = 0.05 * (all_hi - all_lo) if all_hi != all_lo else 0.1
ax.set_ylim(max(0.0, all_lo - pad), min(1.0, all_hi + pad))

# title + legend
fig.suptitle("Representational Similarity Analysis (RSA) on shared982 (pooled)", x=0.55, y=0.90, fontsize=15, fontweight="normal")
ax.set_title("Best model vs Official 40-session baseline", y=0.97,fontsize=12, pad=20)

leg = fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
for t in leg.get_texts():
    t.set_fontsize(12)

plt.tight_layout(rect=[0, 0.07, 1, 0.93])

out = OUT_DIR / "fig03_rsa_bar_pooled_main.png"
safe_save(fig, out)
plt.close(fig)
