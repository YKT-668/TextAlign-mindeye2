import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =========================
# Config
# =========================
OUT_DIR = "/mnt/work/repos/TextAlign-mindeye2/figures"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "Fig_isrsa_heatmap_textalign_llm_v2.png")
OUT_PDF = os.path.join(OUT_DIR, "Fig_isrsa_heatmap_textalign_llm_v2.pdf")

SEARCH_ROOTS = [
    "/mnt/work/repos/TextAlign-mindeye2/cache",
    "/mnt/work/repos/TextAlign-mindeye2",
]

# Your palette
C_PURPLE = "#E0CAEF"
C_BLUE   = "#C4D5EB"
C_PEACH  = "#FDE5D4"
C_GREEN  = "#DDEDD3"
C_GRAY   = "#E8E8E8"
C_AXIS   = "#2f2f2f"
C_GRID   = "#d9d9d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# =========================
# Find data file
# =========================
def find_isrsa_file():
    pats = [
        "**/*isrsa*matrix.csv", # Added this line to match actual file
        "**/*isrsa*textalign*llm*.csv",
        "**/*isrsa*textalign*llm*.npy",
        "**/*isrsa*textalign*llm*.pt",
        "**/*isrsa*llm*.csv",
        "**/*isrsa*llm*.npy",
        "**/*isrsa*llm*.pt",
    ]
    hits = []
    for root in SEARCH_ROOTS:
        for pat in pats:
            hits += glob.glob(os.path.join(root, pat), recursive=True)
    # rank: csv/npy/pt, plus newer first
    def score(p):
        ext = os.path.splitext(p)[1].lower()
        s = 0
        if ext == ".npy": s += 3
        if ext == ".csv": s += 2
        if ext == ".pt":  s += 1
        # name hints
        name = os.path.basename(p).lower()
        if "matrix" in name or "mat" in name: s += 2
        if "heatmap" in name: s += 1
        return (s, os.path.getmtime(p))
    hits = sorted(hits, key=score, reverse=True)
    return hits[0] if hits else None

path = find_isrsa_file()
if path is None:
    raise FileNotFoundError(
        "Cannot auto-find IS-RSA matrix file for textalign_llm. "
        "Please set `path` manually to a .csv/.npy/.pt containing a 4x4 matrix."
    )

# =========================
# Load matrix
# =========================
ext = os.path.splitext(path)[1].lower()

M = None
labels = [1, 2, 5, 7]  # default ordering used in your paper

if ext == ".npy":
    M = np.load(path)
elif ext == ".csv":
    df = pd.read_csv(path, index_col=0)
    # try to coerce to numeric matrix
    df = df.apply(pd.to_numeric, errors="coerce")
    # If row/col names include subject ids, reorder to [1,2,5,7] when possible
    try:
        idx = [str(x) for x in labels]
        if all(i in df.index.astype(str) for i in idx) and all(i in df.columns.astype(str) for i in idx):
            df = df.loc[idx, idx]
        M = df.values
        # if df had good labels, use them
        labels = [int(x) for x in df.index.astype(str)]
    except Exception:
        M = df.values
elif ext == ".pt":
    import torch
    obj = torch.load(path, map_location="cpu")
    # common patterns: dict with 'matrix' / 'isrsa' / direct tensor
    if isinstance(obj, dict):
        for k in ["matrix", "isrsa", "corr", "corr_mat", "mat"]:
            if k in obj:
                M = obj[k]
                break
        if M is None:
            # fallback: first tensor-like value
            for v in obj.values():
                if hasattr(v, "shape"):
                    M = v
                    break
    else:
        M = obj
    if hasattr(M, "detach"):
        M = M.detach().cpu().numpy()
    else:
        M = np.asarray(M)
else:
    raise ValueError(f"Unsupported ext: {ext}")

M = np.asarray(M, dtype=float)

# Basic sanity: expect 4x4
if M.ndim != 2 or M.shape[0] != M.shape[1]:
    raise ValueError(f"Matrix must be square. Got {M.shape} from {path}")
if M.shape[0] != 4:
    # still plot, but label accordingly
    labels = list(range(1, M.shape[0] + 1))

# Force diagonal to 1.0 if it is numerically near 1 (avoid tiny float noise)
for i in range(min(M.shape[0], M.shape[1])):
    if np.isfinite(M[i, i]) and abs(M[i, i] - 1.0) < 1e-3:
        M[i, i] = 1.0

# =========================
# Colormap (IJCAI-ish, using your palette)
# low -> high : gray -> blue -> purple -> peach
# =========================
cmap = LinearSegmentedColormap.from_list(
    "zexu_isrsa",
    [C_GRAY, C_BLUE, C_PURPLE, C_PEACH],
    N=256
)

# Range: keep consistent across figs
vmin = max(0.0, np.nanmin(M))
vmax = 1.0

# =========================
# Plot
# =========================
fig = plt.figure(figsize=(7.4, 6.6))
ax = fig.add_subplot(111)

im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")

# Ticks/labels
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

ax.set_xlabel("Subject")
ax.set_ylabel("Subject")
ax.set_title("IS-RSA Heatmap (TextAlign + LLM)", pad=14)

# Light grid lines between cells (cleaner than heavy borders)
ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
ax.tick_params(which="minor", bottom=False, left=False)

# Annotations with adaptive text color
# choose threshold at mid of [vmin, vmax]
thr = (vmin + vmax) * 0.5
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        val = M[i, j]
        if not np.isfinite(val):
            txt = "nan"
        else:
            txt = f"{val:.3f}"
        color = "white" if (np.isfinite(val) and val < thr) else C_AXIS
        ax.text(j, i, txt, ha="center", va="center", fontsize=12, color=color, fontweight="semibold")

# Colorbar (slim)
cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
cbar.set_label("Spearman ρ", rotation=90)

# Footnote removed

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)
print("Loaded:", path, "| shape:", M.shape, "| vmin:", vmin, "| vmax:", vmax)
