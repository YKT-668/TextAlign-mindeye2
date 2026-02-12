import os, glob, re
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# =========================
# Config
# =========================
OUT_DIR = "/mnt/work/repos/TextAlign-mindeye2/figures"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "fig06_rsm_heatmaps_subj01_best_overall_pooled_v2.png")
OUT_PDF = os.path.join(OUT_DIR, "fig06_rsm_heatmaps_subj01_best_overall_pooled_v2.pdf")

CACHE_ROOT = "/mnt/work/repos/TextAlign-mindeye2/cache"

# Define paths to feature files
GT_FEAT_PATH = "/mnt/work/repos/TextAlign-mindeye2/evals/all_images_bigG_1664_mean.pt"
BRAIN_FEAT_PATH = "/mnt/work/repos/TextAlign-mindeye2/evals/brain_tokens/ours_s1_v2/subj01_brain_clip_mean.pt"

# Palette (given)
C_PURPLE = "#E0CAEF"
C_BLUE   = "#C4D5EB"
C_PEACH  = "#FDE5D4"
C_GREEN  = "#DDEDD3"
C_GRAY   = "#E8E8E8"
C_AXIS   = "#2f2f2f"
C_GRID   = "#E8E8E8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# =========================
# Colormaps (derived from your palette)
# =========================
# Sequential for similarity (low->high)
cmap_sim = LinearSegmentedColormap.from_list(
    "sim_pastel",
    [C_GRAY, C_GREEN, C_BLUE, C_PURPLE],
    N=256
)

# Diverging for diff (neg->0->pos), centered at 0
cmap_diff = LinearSegmentedColormap.from_list(
    "diff_pastel",
    [C_BLUE, C_GRAY, C_PEACH],
    N=256
)

# =========================
# Load helpers
# =========================
def compute_rsm(feats):
    # feats: [N, D]
    # normalize
    feats = F.normalize(feats, p=2, dim=1)
    # cosine similarity
    rsm = torch.mm(feats, feats.t())
    return rsm.numpy()

def load_data_computed(n_samples=200):
    print(f"Loading features from:\n  GT: {GT_FEAT_PATH}\n  Brain: {BRAIN_FEAT_PATH}")
    
    if not os.path.exists(GT_FEAT_PATH):
        raise FileNotFoundError(f"GT features not found at {GT_FEAT_PATH}")
    if not os.path.exists(BRAIN_FEAT_PATH):
        raise FileNotFoundError(f"Brain features not found at {BRAIN_FEAT_PATH}")

    gt_feats = torch.load(GT_FEAT_PATH, map_location="cpu")
    brain_feats = torch.load(BRAIN_FEAT_PATH, map_location="cpu")

    # Slice strictly the first n_samples
    if gt_feats.shape[0] > n_samples:
        gt_feats = gt_feats[:n_samples]
    if brain_feats.shape[0] > n_samples:
        brain_feats = brain_feats[:n_samples]
    
    # Ensure they match
    min_len = min(gt_feats.shape[0], brain_feats.shape[0])
    gt_feats = gt_feats[:min_len]
    brain_feats = brain_feats[:min_len]

    print(f"Computing RSMs for {min_len} samples...")
    gt_rsm = compute_rsm(gt_feats)
    brain_rsm = compute_rsm(brain_feats)
    diff_rsm = brain_rsm - gt_rsm

    meta = {
        "src": "Computed from .pt features",
        "keys": ["gt_rsm", "brain_rsm", "diff_rsm"]
    }

    return gt_rsm, brain_rsm, diff_rsm, meta

# =========================
# Main Plot
# =========================
def plot_heatmaps(gt, pred, diff):
    # Setup figure
    # We want 3 panels: GT, Pred, Diff
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)

    # Common params
    kw_sim = dict(cmap=cmap_sim, vmin=0, vmax=1) 
    
    # Check range
    print(f"GT range: {gt.min():.3f} to {gt.max():.3f}")
    print(f"Pred range: {pred.min():.3f} to {pred.max():.3f}")
    
    # Let's auto-scale vmin/vmax for clarity if strictly positive
    vmin_sim = min(gt.min(), pred.min())
    vmax_sim = max(gt.max(), pred.max())
    
    print(f"Using vmin={vmin_sim:.3f}, vmax={vmax_sim:.3f} for Similarity maps")

    im_gt = axes[0].imshow(gt, cmap=cmap_sim, vmin=vmin_sim, vmax=vmax_sim, origin='upper')
    axes[0].set_title("Ground Truth RSM", fontweight="bold", pad=12)
    
    im_pred = axes[1].imshow(pred, cmap=cmap_sim, vmin=vmin_sim, vmax=vmax_sim, origin='upper')
    axes[1].set_title("Ours (Subj01 Pooled) RSM", fontweight="bold", pad=12)
    
    # Diff map
    # Centered at 0
    max_diff = max(abs(diff.min()), abs(diff.max()))
    print(f"Diff range: {diff.min():.3f} to {diff.max():.3f} -> Using limits +/- {max_diff:.3f}")
    
    im_diff = axes[2].imshow(diff, cmap=cmap_diff, vmin=-max_diff, vmax=max_diff, origin='upper')
    axes[2].set_title("Difference (Ours - GT)", fontweight="bold", pad=12)

    # Colorbars
    cb_gt = fig.colorbar(im_gt, ax=axes[0], fraction=0.046, pad=0.04)
    cb_pred = fig.colorbar(im_pred, ax=axes[1], fraction=0.046, pad=0.04)
    cb_diff = fig.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)

    # Styling
    for ax in axes:
        ax.set_xlabel("Image Index")
        ax.set_ylabel("Image Index")
        # Remove ticks for clarity if dense
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return fig

if __name__ == "__main__":
    try:
        gt, pred, diff, meta = load_data_computed(n_samples=200)
        
        print(f"Data loaded. Shapes: GT={gt.shape}, Pred={pred.shape}")
        
        fig = plot_heatmaps(gt, pred, diff)
        
        print(f"Saving to {OUT_PNG}")
        fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
        # fig.savefig(OUT_PDF, bbox_inches='tight') # PDF optional
        print("Done.")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
