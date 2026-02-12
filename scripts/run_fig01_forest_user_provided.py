import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# Config
# =========================
CSV_PATH = "/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_ccd/ccd_summary.csv"
OUT_PNG  = "/mnt/work/repos/TextAlign-mindeye2/figures/fig01_ccd_fullruns_forest_v4.png"
OUT_PDF  = "/mnt/work/repos/TextAlign-mindeye2/figures/fig01_ccd_fullruns_forest_v4.pdf"
MAP_CSV  = "/mnt/work/repos/TextAlign-mindeye2/figures/fig01_run_label_mapping_v4.csv"

MODEL_COL = "tag"  # <-- you confirmed

MEAN_COL = "ccd_acc1"
LO_COL   = "ccd_acc1_ci95_lo"
HI_COL   = "ccd_acc1_ci95_hi"

SUBJECTS = ["subj01", "subj02", "subj05", "subj07"]

# If filtering columns exist, apply them; if filtering makes df empty, auto-fallback.
STRICT_FILTER = True

# =========================
# Load
# =========================
df = pd.read_csv(CSV_PATH)

required = [MODEL_COL, MEAN_COL, LO_COL, HI_COL]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in ccd_summary.csv: {missing}\n"
                     f"Available columns: {list(df.columns)}")

df[MODEL_COL] = df[MODEL_COL].astype(str)

# =========================
# Helpers
# =========================
def parse_subj(s: str):
    m = re.search(r"subj0?(\d+)", s)
    return f"subj{int(m.group(1)):02d}" if m else None

def category(tag: str):
    t = str(tag).lower()
    if "official" in t:
        return "Official"
    if "ours" in t:
        return "Ours"
    return "Other"

def short_label(tag: str):
    """
    Keep full run list, but shorten display names aggressively.
    Also export mapping CSV for traceability.
    """
    s = str(tag)
    sl = s.lower()

    # Official HF variants
    if "official_hf" in sl:
        if "40sess" in sl: return "Official-40sess"
        if "1sess"  in sl: return "Official-1sess"
        return "Official-HF"

    # Local official (non HF)
    if sl.startswith("official_") or "/official_" in sl:
        # e.g., official_s1/official_s1
        m = re.search(r"official[_-]s(\d+)", sl)
        if m:
            return f"Official-S{m.group(1)}"
        return "Official-local"

    # Ours variants
    if "ours" in sl:
        mv = re.search(r"_v(\d+)", sl)
        v = f"v{mv.group(1)}" if mv else "v?"
        if "from_official" in sl:
            return f"Ours-fromOff-{v}"
        # if you later want to distinguish no-hard/+hard, encode it here
        if "nohard" in sl or "no-hard" in sl:
            return f"Ours-nohard-{v}"
        return f"Ours-{v}"

    # Fallback: use last 1-2 path segments trimmed
    parts = [p for p in s.split("/") if p]
    tail = parts[-1] if parts else s
    if len(parts) >= 2 and len(tail) < 10:
        tail = parts[-2] + "/" + parts[-1]
    return tail[:32]

def apply_optional_filters(dfin: pd.DataFrame):
    d = dfin.copy()

    # match your figure title intent: pooled + hardneg + K=1 when columns exist
    if "neg_mode" in d.columns:
        d = d[d["neg_mode"].astype(str).str.contains("hard", case=False, na=False)]
    if "K" in d.columns:
        # some csv store as str
        d = d[pd.to_numeric(d["K"], errors="coerce") == 1]
    
    # Check for known k_neg column as well if K is missing (User patch)
    if "K" not in d.columns and "k_neg" in d.columns:
         d = d[pd.to_numeric(d["k_neg"], errors="coerce") == 1]

    # pooled-related column name varies
    # User provided list: ["repr", "representation", "rep_name", "feature_name"]
    # Added "eval_repr" to match actual CSV
    for col in ["eval_repr", "repr", "representation", "rep_name", "feature_name"]:
        if col in d.columns:
            d = d[d[col].astype(str).str.contains("pooled", case=False, na=False)]
            break

    return d

# =========================
# Build plotting df
# =========================
if "subj" in df.columns:
    # Safe handling of numeric or string subj column "1", "01", etc.
    def norm_subj_col(x):
        try:
            return f"subj{int(x):02d}"
        except:
            return None
    df["subject"] = df["subj"].apply(norm_subj_col)
else:
    df["subject"] = df[MODEL_COL].apply(parse_subj)

df = df[df["subject"].isin(SUBJECTS)].copy()

before_n = len(df)

if STRICT_FILTER:
    df_f = apply_optional_filters(df)
    # fallback if too strict
    if len(df_f) == 0:
        print("[WARN] STRICT_FILTER removed all rows; fallback to unfiltered-by-setting within subjects.")
        df_f = df.copy()
else:
    df_f = df.copy()

after_n = len(df_f)
print(f"[INFO] Rows within subjects: {before_n}; after optional filters: {after_n}")

df_f["cat"] = df_f[MODEL_COL].apply(category)
df_f["label"] = df_f[MODEL_COL].apply(short_label)

# Dedup identical tags if any duplicates exist (keep first)
df_f = df_f.drop_duplicates(subset=["subject", MODEL_COL])

# =========================
# Style (your palette + coordinating dark gray)
# =========================
palette = {
    "Official": "#E8E8E8",
    "Ours":     "#E0CAEF",
    "Other":    "#C4D5EB",   # fallback
}
text_color   = "#333333"   # coordinating neutral (allowed)
chance_color = "#8C8C8C"   # coordinating neutral (allowed)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelcolor": text_color,
    "xtick.color": text_color,
    "ytick.color": text_color,
})

# Dynamic but bounded height (avoid absurdly tall figs)
# Determine max_n safely
counts = [df_f[df_f["subject"] == s].shape[0] for s in SUBJECTS]
max_n = max(counts) if counts else 0

fig_h = 1.2 + 0.23 * max_n
fig_h = min(max(fig_h, 2.8), 6.0)   # clamp to [2.8, 6.0]
fig_w = 12.5

fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h), sharex=True)
if len(SUBJECTS) == 1:
    axes = [axes]

# global x-limits with small padding; keep chance visible
# handle empty df case for min/max
if len(df_f) > 0:
    xmin = min(float(df_f[LO_COL].min()), 0.48)
    xmax = max(float(df_f[HI_COL].max()), 0.66)
else:
    xmin, xmax = 0.48, 0.66

pad = 0.01
xmin, xmax = xmin - pad, xmax + pad

for ax, subj in zip(axes, SUBJECTS):
    sub = df_f[df_f["subject"] == subj].copy().sort_values(MEAN_COL, ascending=True)

    y = np.arange(len(sub))
    x  = sub[MEAN_COL].to_numpy(dtype=float)
    lo = sub[LO_COL].to_numpy(dtype=float)
    hi = sub[HI_COL].to_numpy(dtype=float)

    # CI horizontal lines
    ax.hlines(y, lo, hi, color=text_color, lw=1.0, alpha=0.9, zorder=2)

    # points
    colors = sub["cat"].map(palette).fillna(palette["Other"]).to_list()
    ax.scatter(x, y, s=26, c=colors, edgecolors=text_color, linewidths=0.8, zorder=3)

    # y tick labels (short)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"].to_list(), fontsize=8)

    # chance line (K=1 -> 2AFC style)
    ax.axvline(0.5, ls="--", lw=1.0, color=chance_color, zorder=1)

    # cosmetics
    ax.set_title(subj.replace("subj", "S"), pad=6)
    ax.set_xlim(xmin, xmax)
    ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Run (short label)", fontsize=9)
for ax in axes:
    ax.set_xlabel("CCD@1", fontsize=9)

fig.suptitle("CCD@1 on shared982 (pooled, hardneg, K=1)", y=1.01)

# Legend (compact, top-centered)
legend_elems = [
    Line2D([0],[0], marker="o", color="none", label="Official HF",
           markerfacecolor=palette["Official"], markeredgecolor=text_color, markersize=6),
    Line2D([0],[0], marker="o", color="none", label="Ours",
           markerfacecolor=palette["Ours"], markeredgecolor=text_color, markersize=6),
    Line2D([0],[0], linestyle="--", color=chance_color, label="Chance (0.5)"),
]
fig.legend(handles=legend_elems, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.08), columnspacing=1.2, handlelength=1.6)

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")  # vector for paper
plt.close(fig)

print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)

# Mapping table for supplement / reproducibility
save_cols = ["subject", MODEL_COL, "label", "cat", MEAN_COL, LO_COL, HI_COL]
sort_by = ["subject", "cat", MEAN_COL]
df_save = df_f[save_cols].sort_values(sort_by, ascending=[True, True, True])
df_save.to_csv(MAP_CSV, index=False)
print("Saved mapping:", MAP_CSV)
