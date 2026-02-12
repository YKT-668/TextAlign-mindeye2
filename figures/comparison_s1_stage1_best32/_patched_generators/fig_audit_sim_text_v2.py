import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.labelsize': 13})

# =========================
# Config
# =========================
OUT_DIR = "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "Fig_audit_sim_text_v2.png")
OUT_PDF = os.path.join(OUT_DIR, "Fig_audit_sim_text_v2.pdf")

# Try to auto-find a csv/json that contains sim_text
SEARCH_ROOTS = ["/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32/_inputs_augmented", # Patched to prefer inputs

    "/mnt/work/repos/TextAlign-mindeye2/cache",
    "/mnt/work/repos/TextAlign-mindeye2",
]

# Given palette
C_PURPLE = "#E0CAEF"
C_BLUE   = "#C4D5EB"
C_PEACH  = "#FDE5D4"
C_GREEN  = "#DDEDD3"
C_GRAY   = "#E8E8E8"
C_AXIS   = "#2f2f2f"
C_GRID   = "#d9d9d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# =========================
# Utils
# =========================
def find_candidate_data():
    cands = []
    for root in SEARCH_ROOTS:
        cands += glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
        cands += glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
    
    # prefer files with "audit" or "sim" in name, and specifically hardneg
    score = []
    for p in cands:
        name = os.path.basename(p).lower()
        s = 0
        if "hardneg" in name and ".jsonl" in name: s += 5 # Top priority
        if "shared982_hardneg.jsonl" in name: s += 10 # Prefer main file
        if "ablation" in name: s -= 2 # Penalize ablation
        if "audit" in name: s += 3
        if "sim" in name: s += 2
        if "summary" in name: s -= 1 # Avoid summary tables
        score.append((s, os.path.getmtime(p), p))
    score.sort(reverse=True)
    return [p for _, _, p in score]

def pick_sim_column(df):
    # common candidates
    for col in ["sim_text", "text_sim", "sim", "similarity", "clip_text_sim", "sim_txt"]:
        if col in df.columns:
            return col
    # fallback: any column containing sim_text
    for c in df.columns:
        if isinstance(c, str) and "sim" in c.lower() and "text" in c.lower():
            return c
    raise ValueError(f"Cannot find sim_text-like column. Columns={list(df.columns)}")

def kde_gaussian(x, grid, bw=None):
    # simple gaussian KDE (no seaborn/scipy dependency)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.zeros_like(grid)

    if bw is None:
        # Scott's rule
        std = np.std(x, ddof=1)
        bw = 1.06 * std * (n ** (-1/5))
        if not np.isfinite(bw) or bw <= 1e-6:
            bw = max(std, 1e-3) * 0.2

    diffs = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diffs**2).sum(axis=1) / (n * bw * np.sqrt(2*np.pi))
    return dens

# =========================
# Load data
# =========================
csv_path = None
# Try find_candidate first
candidates = ["/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32/_inputs_augmented/s1_textalign_stage1_FINAL_BEST_32_hardneg.jsonl"] # PATCHED

for p in candidates:
    try:
        if p.endswith(".jsonl"):
            df = pd.read_json(p, lines=True)
        else:
            df = pd.read_csv(p)
        
        try:
            col = pick_sim_column(df)
            # require enough rows
            if df[col].dropna().shape[0] >= 50:
                csv_path = p
                print(f"Selected Data: {csv_path} with col {col}")
                break
        except ValueError:
            continue
    except Exception:
        continue

if csv_path is None:
    raise FileNotFoundError(
        "No suitable CSV/JSONL found containing sim_text. "
        "Please point csv_path to the audit sim_text file."
    )

if csv_path.endswith(".jsonl"):
    df = pd.read_json(csv_path, lines=True)
else:
    df = pd.read_csv(csv_path)

sim_col = pick_sim_column(df)

x = df[sim_col].to_numpy(dtype=float)
x = x[np.isfinite(x)]

# Thresholds: if your csv has explicit columns, use them; else use defaults from your caption
# Defaults: init window [0.15, 0.85], plus p05/p95 from data
init_low_default, init_high_default = 0.15, 0.85
# Check for columns, but handle missing gracefully
init_low = init_low_default
init_high = init_high_default

if "init_low" in df.columns:
    try:
        init_low = float(df["init_low"].iloc[0])
    except: pass
if "init_high" in df.columns:
    try:
        init_high = float(df["init_high"].iloc[0])
    except: pass

p05 = float(np.percentile(x, 5))
p95 = float(np.percentile(x, 95))
mean = float(np.mean(x))
med  = float(np.median(x))
n = len(x)

# =========================
# Plot
# =========================
fig = plt.figure(figsize=(12.8, 6.8))
ax = fig.add_subplot(111)

# Density histogram
bins = 48  # a bit finer looks more "audit-grade"
ax.hist(
    x, bins=bins, density=True,
    color=C_GREEN, edgecolor=C_AXIS, linewidth=0.8, alpha=0.85,
    label="Similarity histogram (density)"
)

# KDE line overlay
xmin = max(0.0, np.min(x) - 0.02)
xmax = min(1.0, np.max(x) + 0.02)
grid = np.linspace(xmin, xmax, 500)
kde = kde_gaussian(x, grid)
ax.plot(grid, kde, color=C_PURPLE, linewidth=2.2, label="KDE (Gaussian)")

# Band for [p05, p95]
ax.axvspan(p05, p95, color=C_GRAY, alpha=0.55, label="Central 90% band (p05–p95)")

# Threshold lines
ax.axvline(init_low,  color="black", linestyle="--", linewidth=1.8, label="Init low/high")
ax.axvline(init_high, color="black", linestyle="--", linewidth=1.8)
ax.axvline(p05, color="#6f6f6f", linestyle=":", linewidth=2.0)
ax.axvline(p95, color="#6f6f6f", linestyle=":", linewidth=2.0, label="p05 / p95")

# Style
ax.set_title("Hard Negative Similarity Audit (CLIP-text)", pad=30)
ax.set_xlabel(sim_col)
ax.set_ylabel("Density")

ax.grid(True, axis="y", color=C_GRID, linewidth=1.0, alpha=0.7)
ax.grid(False, axis="x")

# Clean spines
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(C_AXIS)
ax.spines["bottom"].set_color(C_AXIS)

# Limits: ensure all lines visible and leave breathing room
ax.set_xlim(min(xmin, init_low) - 0.02, max(xmax, init_high) + 0.02)

# Summary box (Moved to upper left to avoid covering high-similarity density)
summary = (
    f"N={n}\n"
    f"mean={mean:.3f}\n"
    f"median={med:.3f}\n"
    f"p05={p05:.3f}, p95={p95:.3f}\n"
    f"init=[{init_low:.2f}, {init_high:.2f}]"
)
ax.text(
    0.03, 0.98, summary, 
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=C_GRAY, alpha=0.95)
)

# Legend at bottom (outside)
handles, labels = ax.get_legend_handles_labels()
# de-duplicate legend labels
seen = set()
h2, l2 = [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l)
        h2.append(h)
        l2.append(l)

fig.legend(
    h2, l2,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02)
)

# Footer removed (Watermark)

plt.tight_layout(rect=[0, 0.06, 1, 0.95]) # Adjusted top margin for taller title
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)
print("Loaded:", csv_path, " | sim_col:", sim_col, " | N:", n)
