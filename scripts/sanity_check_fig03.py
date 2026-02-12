import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0) USER CONFIG (edit this only)
# =========================
CSV_PATH = "/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_rsa/rsa_summary.csv"
OUT_DIR  = "/mnt/work/repos/TextAlign-mindeye2/figures"
OUT_NAME = "fig03_rsa_best_vs_official_pooled_v2.png"   # do NOT overwrite old figs
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, OUT_NAME)

# =========================
# 1) Load + column auto-detect
# =========================
df = pd.read_csv(CSV_PATH)

# --- subject column ---
subj_col = None
for c in ["subj", "subject", "subj_id"]:
    if c in df.columns:
        subj_col = c
        break
if subj_col is None:
    raise ValueError(f"Cannot find subject column in CSV. Columns={list(df.columns)[:50]}")

# --- tag column (you said actual is `tag`) ---
tag_col = "tag" if "tag" in df.columns else None
if tag_col is None:
    # fallback
    for c in ["run", "run_name", "model", "model_name"]:
        if c in df.columns:
            tag_col = c
            break
if tag_col is None:
    raise ValueError(f"Cannot find tag/run column in CSV. Columns={list(df.columns)[:50]}")

# --- metric value column (RSA pearson) ---
val_col = None
for c in ["rsa", "rsa_pearson", "pearson_rsa", "rsa_value", "score", "metric_value"]:
    if c in df.columns:
        val_col = c
        break
if val_col is None:
    # Fallback to 'rsa_mean' or 'mean' if specific 'rsa' not found, often rsa is stored there
    for c in ["rsa_mean", "mean"]:
         if c in df.columns:
            val_col = c
            break

if val_col is None:
    raise ValueError(f"Cannot find RSA value column in CSV. Columns={list(df.columns)[:50]}")

# --- CI columns (95% lo/hi) ---
lo_col, hi_col = None, None
# Try various permutations including _p suffix for pearson
for lo_candidate, hi_candidate in [
    ("rsa_ci95_lo", "rsa_ci95_hi"),
    ("ci95_lo", "ci95_hi"),
    ("rsa_lo", "rsa_hi"),
    ("bootstrap_ci95_lo", "bootstrap_ci95_hi"),
    ("ci95_low_p", "ci95_high_p"),
    ("ci95_lo_p", "ci95_hi_p"),
    ("ci95_low", "ci95_high"),
]:
    
    if lo_candidate in df.columns and hi_candidate in df.columns:
        lo_col, hi_col = lo_candidate, hi_candidate
        break

# If still not found, try to be more fuzzy or pick first pair of "ci95_low" like columns
if lo_col is None or hi_col is None:
    # Attempt to find *any* pair of low/high that looks like CI
    cols = sorted(df.columns)
    lows = [c for c in cols if "ci95" in c and ("low" in c or "lo" in c)]
    highs = [c for c in cols if "ci95" in c and ("high" in c or "hi" in c)]
    if lows and highs:
        lo_col = lows[0]
        hi_col = highs[0] # Naively take first match if precise matching fails

if lo_col is None or hi_col is None:
    print(f"Warning: Cannot find complete 95% CI columns. Columns={list(df.columns)[:80]}. Plot will lack error bars.")
else:
    print(f"Using CI columns: {lo_col}, {hi_col}")


# --- optional filters: split/representation/corr type ---
def maybe_filter(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()

    # shared982
    for c in ["split", "eval_split", "subset", "testset"]:
        if c in d.columns:
            d = d[d[c].astype(str).str.contains("shared982", case=False, na=False)]
            break

    # pooled representation
    for c in ["eval_repr", "repr", "feature_name", "representation"]:
        if c in d.columns:
            d = d[d[c].astype(str).str.contains("pooled", case=False, na=False)]
            break

    # pearson (if a column exists)
    for c in ["corr", "corr_type", "rsa_type", "stat", "metric", "sim_metric"]:
         if c in d.columns:
            # Some csvs use 'cosine' in sim_metric but 'rsa_pearson' as column. 
            # If val_col is 'rsa_pearson', we are already good.
            # But let's check if there's an explicit metric column to filter.
            # Only filter if it distinguishes between pearson/spearman in ROWS, not COLUMNS.
            # If the CSV has 'rsa_pearson' and 'rsa_spearman' as separate columns, we rely on val_col selection.
            # If CSV has 'metric' column with 'pearson' or 'spearman', we filter.
            vals = d[c].astype(str).unique()
            if any("pearson" in v.lower() for v in vals):
                d = d[d[c].astype(str).str.contains("pearson", case=False, na=False)]
            break

    return d

df = maybe_filter(df)

# normalize subject names to S01/S02...
def norm_subj(x):
    s = str(x).lower()
    # Handle single digit '1' or '01'
    m_digit = re.search(r"(\d+)", s)
    if m_digit:
        num = int(m_digit.group(1))
        return f"S{num:02d}"
    return s.upper()

df["S"] = df[subj_col].map(norm_subj)
df["TAG"] = df[tag_col].astype(str)
df["VAL"] = pd.to_numeric(df[val_col], errors="coerce")

if lo_col and hi_col:
    df["LO"]  = pd.to_numeric(df[lo_col], errors="coerce")
    df["HI"]  = pd.to_numeric(df[hi_col], errors="coerce")
else:
    df["LO"] = np.nan
    df["HI"] = np.nan

df = df.dropna(subset=["S","TAG","VAL"])

# =========================
# 2) Define Official + Ours selectors
# =========================
def is_official_40sess(tag: str) -> bool:
    t = tag.lower()
    # Official HF runs in this CSV might be named "final_subjXX_pretrained_40sess..." without "official" in the tag 
    # (but "official" is in the group column). 
    # So we relax the check to include "pretrained" + "40sess" which are characteristic of the official baseline here.
    cond1 = ("official" in t or "official_hf" in t) and ("40sess" in t or "40" in t and "sess" in t)
    cond2 = ("pretrained" in t and "40sess" in t)
    return cond1 or cond2

def is_ours(tag: str) -> bool:
    t = tag.lower()
    return "ours" in t # Matches anything with 'ours'

# =========================
# 3) Build best-per-subject table (Official vs Best Ours)
# =========================
rows = []
subjects = sorted(df["S"].unique())

for s in subjects:
    dsub = df[df["S"] == s].copy()

    off = dsub[dsub["TAG"].map(is_official_40sess)]
    ours = dsub[dsub["TAG"].map(is_ours)]

    if off.empty:
        # Fallback debug
        # print(f"[{s}] Debug: Tags present: {dsub['TAG'].unique()}")
        print(f"[{s}] Warning: Missing Official 40sess row. Skipping.")
        continue
    if ours.empty:
        print(f"[{s}] Warning: Missing Ours rows. Skipping.")
        continue

    # if multiple official rows exist, take the best (should be identical but be safe)
    off_best = off.sort_values("VAL", ascending=False).iloc[0]
    ours_best = ours.sort_values("VAL", ascending=False).iloc[0]

    rows.append({
        "S": s,
        "Official_TAG": off_best["TAG"],
        "Official_VAL": off_best["VAL"],
        "Official_LO": off_best["LO"],
        "Official_HI": off_best["HI"],
        "Ours_TAG": ours_best["TAG"],
        "Ours_VAL": ours_best["VAL"],
        "Ours_LO": ours_best["LO"],
        "Ours_HI": ours_best["HI"],
    })

if not rows:
    raise ValueError("No rows found for plotting. Check filters.")

tab = pd.DataFrame(rows).sort_values("S")

# Print a compact sanity table for manual verification
print("\n=== Sanity Table (check meaning consistency) ===")
print(tab[["S","Official_TAG","Official_VAL","Ours_TAG","Ours_VAL"]].to_string(index=False))
print("\n[CI check] Example rows:")
print(tab[["S","Official_LO","Official_HI","Ours_LO","Ours_HI"]].head(4).to_string(index=False))

# =========================
# 4) Plot (IJCAI-like style; your palette)
# =========================
COL_OFF = "#C4D5EB"  # provided
COL_OURS = "#E0CAEF" # provided
COL_GRID = "#E8E8E8" # provided
COL_TEXT = "#222222"

x_pos = np.arange(len(tab["S"]))
width = 0.34

fig, ax = plt.subplots(figsize=(10.8, 4.3), dpi=220)

# bars
# Handle potential NaN in errors if CI columns were missing
if tab["Official_LO"].isna().all():
    yerr_off = None
    yerr_ours = None
else:
    yerr_off = [tab["Official_VAL"]-tab["Official_LO"], tab["Official_HI"]-tab["Official_VAL"]]
    yerr_ours = [tab["Ours_VAL"]-tab["Ours_LO"], tab["Ours_HI"]-tab["Ours_VAL"]]

ax.bar(
    x_pos - width/2, tab["Official_VAL"], width,
    yerr=yerr_off,
    capsize=3, linewidth=1.2, edgecolor=COL_TEXT,
    label="Official HF", alpha=0.95
)

ax.bar(
    x_pos + width/2, tab["Ours_VAL"], width,
    yerr=yerr_ours,
    capsize=3, linewidth=1.2, edgecolor=COL_TEXT,
    label="Ours", alpha=0.95, color=COL_OURS
)

# Official color set after first bar call so both have explicit colors
# The first bar call uses default blueish if color not set, we need to enforce COL_OFF
# bar returns container of patches.
ax.patches[0].set_facecolor(COL_OFF) # This only sets first bar if we index 0. 
# Better: Set color in bar call above? No, bar call has `label` but no color arg for first one? 
# Ah, I see `color=COL_OURS` in second call. 
# Let's fix the first call to include color=COL_OFF
# Re-plotting conceptually for safety:
ax.clear()
ax.bar(
    x_pos - width/2, tab["Official_VAL"], width,
    yerr=yerr_off,
    capsize=3, linewidth=1.2, edgecolor=COL_TEXT,
    label="Official HF", alpha=0.95, color=COL_OFF
)
ax.bar(
    x_pos + width/2, tab["Ours_VAL"], width,
    yerr=yerr_ours,
    capsize=3, linewidth=1.2, edgecolor=COL_TEXT,
    label="Ours", alpha=0.95, color=COL_OURS
)


# axes & grids
ax.set_xticks(x_pos)
ax.set_xticklabels(tab["S"], fontsize=12)
ax.set_ylabel("RSA (Pearson)", fontsize=13)
ax.grid(axis="y", color=COL_GRID, linewidth=1.0, alpha=0.9)
ax.set_axisbelow(True)

# y-lim: keep readable but not misleading (you can tune)
# Check for NaNs before min/max
min_lo = min(tab["Official_LO"].min(), tab["Ours_LO"].min())
max_hi = max(tab["Official_HI"].max(), tab["Ours_HI"].max())

if pd.isna(min_lo): min_lo = min(tab["Official_VAL"].min(), tab["Ours_VAL"].min())
if pd.isna(max_hi): max_hi = max(tab["Official_VAL"].max(), tab["Ours_VAL"].max())

y_min = max(0.0, float(min_lo - 0.015))
y_max = min(1.0, float(max_hi + 0.015))
ax.set_ylim(y_min, y_max)

# titles
fig.suptitle("Representational Similarity Analysis (RSA) on shared982 (pooled)", fontsize=18, fontweight="bold", y=0.98)
ax.set_title("Best model vs Official 40-session baseline", fontsize=14, pad=8)

# legend bottom (match Fig01 style)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=False,
    fontsize=12
)

plt.tight_layout(rect=[0, 0.05, 1, 0.92])
plt.savefig(OUT_PATH, bbox_inches="tight")
print(f"\nSaved: {OUT_PATH}")
plt.close(fig)
