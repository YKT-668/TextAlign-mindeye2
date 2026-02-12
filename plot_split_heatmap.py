
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =========================
# Config
# =========================
# Paths found by investigation
PATH_BASE = "/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_isrsa/baseline/isrsa_matrix.csv"
PATH_OURS = "/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_isrsa/textalign_llm/isrsa_matrix.csv"

# Output
OUT_DIR = "/mnt/work/repos/TextAlign-mindeye2/figures/resized_figures"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "Fig_isrsa_heatmap_textalign_llm_v2.png")
OUT_PDF = os.path.join(OUT_DIR, "Fig_isrsa_heatmap_textalign_llm_v2.pdf")

# Palette
C_PURPLE = "#E0CAEF"
C_BLUE   = "#C4D5EB"
C_PEACH  = "#FDE5D4"
C_GREEN  = "#DDEDD3"
C_GRAY   = "#E8E8E8"
C_AXIS   = "#2f2f2f"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# =========================
# Load & Process
# =========================
def load_mat(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path, index_col=0)
    
    # Robust index handling: convert all to string
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    
    # Ensure standard order [1, 2, 5, 7]
    idx = ["1", "2", "5", "7"]
    if all(i in df.index for i in idx) and all(i in df.columns for i in idx):
        df = df.loc[idx, idx]
    else:
        print(f"Warning: Could not reorder {path}, using original order: {df.index.tolist()}")

    return df.values.astype(float)

M_base = load_mat(PATH_BASE)
M_ours = load_mat(PATH_OURS)

# Construct Split Matrix
# Upper Triangle (Triu): Baseline (keep j > i)
# Lower Triangle (Tril): Ours (keep i > j)
# Diagonal: 1.0

# Initialize with diagonal 1.0
N = 4
M_plot = np.eye(N)

# Fill (iterate for clarity)
for i in range(N):
    for j in range(N):
        if i < j:  # Upper triangle (Baseline)
            M_plot[i, j] = M_base[i, j]
        elif i > j: # Lower triangle (Ours)
            M_plot[i, j] = M_ours[i, j]
        # else i==j is 1.0 already

print("Matrix Constructed:")
print(M_plot)

# =========================
# Plotting
# =========================
# Custom colormap: Gray -> Blue -> Purple -> Peach
cmap = LinearSegmentedColormap.from_list("zexu_isrsa", [C_GRAY, C_BLUE, C_PURPLE, C_PEACH], N=256)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111)

# Heatmap
# vmin/vmax logic: fit to range of interest (e.g. 0.3 to 1.0 or data min/max)
# Using raw data range but capped at 1.0
vmin = min(M_base.min(), M_ours.min())
vmax = 1.0
im = ax.imshow(M_plot, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")

# Diagonal divider line
# Draw line from (-0.5, -0.5) to (3.5, 3.5)
ax.plot([-0.5, N-0.5], [-0.5, N-0.5], color="white", linewidth=4, zorder=10)

# Labels
labels = ["S01", "S02", "S05", "S07"]
ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Grid for cells
ax.set_xticks(np.arange(-.5, N, 1), minor=True)
ax.set_yticks(np.arange(-.5, N, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

# Annotations
thr = (vmin + vmax) * 0.5
for i in range(N):
    for j in range(N):
        val = M_plot[i, j]
        if i == j: continue # Skip diagonal text if desired, or show "1.0"
        
        txt = f"{val:.3f}"
        # Text color adaptive
        color = "white" if val < thr else C_AXIS
        ax.text(j, i, txt, ha="center", va="center", fontsize=14, color=color, fontweight="bold")

# Corner Labels (Upper = Baseline, Lower = Ours)
# Placing labels outside or in corners
# Upper Triangle is top-right. Lower Triangle is bottom-left.
# Note: "Upper Triangle" typically means j > i. 
# We add text patches to indicate regions clearly.

# Text: Baseline (Top Right quadrant area)
ax.text(N-0.6, -0.4, "Baseline", ha="right", va="top", fontsize=16, fontweight="bold", color=C_AXIS, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

# Text: Ours (Bottom Left quadrant area)
ax.text(-0.4, N-0.6, "Ours (LLM)", ha="left", va="bottom", fontsize=16, fontweight="bold", color=C_AXIS,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

ax.set_title("IS-RSA: Baseline (Upper) vs Ours (Lower)", pad=15)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Spearman Correlation", rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

print(f"[Success] Generated split heatmap at {OUT_PNG}")
