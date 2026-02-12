import os, glob, json
from pathlib import Path
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

OUT_PNG = os.path.join(OUT_DIR, "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.png")
OUT_PDF = os.path.join(OUT_DIR, "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.pdf")

CACHE_ROOT = "/mnt/work/repos/TextAlign-mindeye2/cache"
REPO_ROOT = "/mnt/work/repos/TextAlign-mindeye2"

DOWNSAMPLE_TO = 200          # match caption "downsampled 200/982"
DOWNSAMPLE_MODE = "uniform"  # "uniform" or "none"
SCALE_MODE = "panel"         # "panel" (recommended) or "shared"

# Palette (given)
C_PURPLE = "#E0CAEF"
C_BLUE   = "#C4D5EB"
C_PEACH  = "#FDE5D4"
C_GREEN  = "#DDEDD3"
C_GRAY   = "#E8E8E8"
C_AXIS   = "#2f2f2f"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# =========================
# Colormaps
# =========================
cmap_sim = LinearSegmentedColormap.from_list(
    "sim_pastel",
    [C_GRAY, C_GREEN, C_BLUE, C_PURPLE],
    N=256
)

cmap_diff = LinearSegmentedColormap.from_list(
    "diff_pastel",
    [C_BLUE, C_GRAY, C_PEACH],
    N=256
)

# =========================
# Load helpers
# =========================
def find_first(patterns):
    cand = []
    for pat in patterns:
        cand += glob.glob(pat, recursive=True)
    cand = sorted(cand, key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0] if cand else None

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    def pick(*names):
        for n in names:
            if n in z:
                return z[n]
        return None

    gt   = pick("gt", "GT", "rsm_gt", "gt_rsm", "gt_sim", "gt_similarity")
    pred = pick("pred", "PRED", "brain", "rsm_pred", "pred_rsm", "brain_rsm", "brain_sim", "brain_similarity")
    diff = pick("diff", "DIFF", "delta", "brain_minus_gt", "pred_minus_gt")

    if gt is None or pred is None:
        raise ValueError(f"NPZ missing required arrays. keys={keys}")
    if diff is None:
        diff = pred - gt

    return gt.astype(np.float32), pred.astype(np.float32), diff.astype(np.float32), keys

def compute_rsm(feats):
    # feats: [N, D]
    feats = torch.from_numpy(feats) if isinstance(feats, np.ndarray) else feats
    feats = F.normalize(feats, p=2, dim=1)
    rsm = torch.mm(feats, feats.t())
    return rsm.numpy()

def load_computed_from_features():
    print("Searching for raw features to compute RSM...")
    # Paths derived from previous context
    gt_path = os.path.join(REPO_ROOT, "evals/all_images_bigG_1664_mean.pt")
    
    # Ours S1 V2 paths
    brain_dir = os.path.join(REPO_ROOT, "evals/brain_tokens/ours_s1_v2")
    brain_path = os.path.join(brain_dir, "subj01_brain_clip_mean.pt")
    ids_path = os.path.join(brain_dir, "subj01_ids.json")
    
    # Shared masks
    shared982_path = os.path.join(REPO_ROOT, "src/shared982.npy")
    shared1000_path = os.path.join(REPO_ROOT, "src/shared1000.npy")

    if not all(os.path.exists(p) for p in [gt_path, brain_path, ids_path, shared982_path]):
        missing = [p for p in [gt_path, brain_path, ids_path, shared982_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Missing feature/aux files for recomputation: {missing}")

    print("Loading GT...")
    gt_all = torch.load(gt_path, map_location="cpu") # [1000, 1664]

    print("Loading Brain...")
    brain_all = torch.load(brain_path, map_location="cpu") # [N, 1664]
    with open(ids_path, 'r') as f:
        brain_ids = np.array(json.load(f))
    
    print("Loading Shared Masks...")
    # shared982.npy is usually a boolean mask of length 73k? Or indices?
    # Based on previous output: "Shared982 count: 982, indices: [2950 ...]"
    # It seems to be a mask on the full COCO/NSD 73k ID space.
    s982_mask = np.load(shared982_path)
    s982_indices = np.where(s982_mask > 0)[0] # These are the scalar IDs (e.g. 2950) we want.

    # Filter Brain to Shared982
    # Find which rows in brain_all correspond to s982_indices
    # We assume brain_ids contains NSD IDs.
    keep_mask = np.isin(brain_ids, s982_indices)
    brain_shared = brain_all[keep_mask]
    brain_shared_ids = brain_ids[keep_mask]
    
    # We need to sort everything to ensure alignment.
    # Let's sort by NSD ID.
    sort_idx = np.argsort(brain_shared_ids)
    brain_final = brain_shared[sort_idx]
    final_ids = brain_shared_ids[sort_idx]
    
    print(f"Brain aligned to {len(final_ids)} shared IDs.")

    # Get corresponding GT features.
    # GT file provided is 1000 images. We need to know which NSD ID corresponds to which row 0..999.
    # shared1000.npy gives us the indices in the 73k space presumably?
    # Let's verify common usage or just match IDs if GT came with IDs.
    # GT usually doesn't come with IDs in the .pt file.
    # But checking make_rsa_figures.py:
    # "shared1000_path = ...; m1000 = np.load(...); shared_ids = np.where(m1000 > 0)[0]"
    # "gt_ids = shared_ids" (implies row i of GT corresponds to shared_ids[i]?)
    # "if int(gt.shape[0]) == int(shared_ids.shape[0]): gt_ids = shared_ids"
    
    m1000 = np.load(shared1000_path)
    gt_ids_all = np.where(m1000 > 0)[0] # Usually sorted?
    
    if len(gt_ids_all) != len(gt_all):
        print(f"Warning: GT shape {gt_all.shape} != Shared1000 count {len(gt_ids_all)}")
    
    # Now find the indices in GT that correspond to final_ids (the 982 subset)
    # We need to find index `i` in gt_ids_all such that gt_ids_all[i] == final_ids[j]
    
    # Build map from NSD_ID -> GT_Row_Index
    nsd_to_gt_row = {nsd_id: i for i, nsd_id in enumerate(gt_ids_all)}
    
    gt_row_indices = []
    valid_mask = []
    
    for i, nsd_id in enumerate(final_ids):
        if nsd_id in nsd_to_gt_row:
            gt_row_indices.append(nsd_to_gt_row[nsd_id])
            valid_mask.append(i)
    
    gt_row_indices = np.array(gt_row_indices)
    valid_mask = np.array(valid_mask)
    
    if len(gt_row_indices) < len(final_ids):
        print(f"Warning: Only matched {len(gt_row_indices)}/{len(final_ids)} images between Brain and GT.")
        brain_final = brain_final[valid_mask]
        final_ids = final_ids[valid_mask]

    gt_final = gt_all[gt_row_indices]

    print(f"Final shapes: GT={gt_final.shape}, Brain={brain_final.shape}")

    # Compute RSM
    print("Computing RSMs...")
    gt_rsm = compute_rsm(gt_final)
    brain_rsm = compute_rsm(brain_final)
    diff_rsm = brain_rsm - gt_rsm
    
    meta = {"src": "Computed from .pt (ours_s1_v2)", "keys": ["computed"]}
    return gt_rsm, brain_rsm, diff_rsm, meta

def load_any():
    # 1. Try NPZ
    npz_path = find_first([
        os.path.join(CACHE_ROOT, "**", "*fig06*subj01*pooled*.npz"),
        os.path.join(CACHE_ROOT, "**", "*rsm_heatmaps*subj01*pooled*.npz"),
        os.path.join(CACHE_ROOT, "**", "*subj01*best*pooled*.npz"),
    ])
    if npz_path:
        print(f"Found NPZ: {npz_path}")
        gt, pred, diff, keys = load_npz(npz_path)
        meta = {"src": npz_path, "keys": keys}
        return gt, pred, diff, meta
    
    # 2. Fallback to compute
    print("NPZ not found. attempting recomputation from features...")
    return load_computed_from_features()

# =========================
# Robust scaling (exclude diagonal)
# =========================
def offdiag_values(M):
    assert M.ndim == 2 and M.shape[0] == M.shape[1]
    mask = ~np.eye(M.shape[0], dtype=bool)
    return M[mask]

def robust_vmin_vmax(M, lo=2, hi=98):
    v = offdiag_values(M).reshape(-1)
    vmin = float(np.percentile(v, lo))
    vmax = float(np.percentile(v, hi))
    if vmin == vmax:
        vmin, vmax = float(v.min()), float(v.max())
    return vmin, vmax

# =========================
# Downsample (uniform)
# =========================
def maybe_downsample(M, target=200, mode="uniform"):
    n = M.shape[0]
    if mode == "none" or n <= target:
        return M, np.arange(n)
    if mode == "uniform":
        idx = np.linspace(0, n - 1, target)
        idx = np.round(idx).astype(int)
        # ensure unique & sorted
        idx = np.unique(idx)
        # if uniqueness shrinks a bit, pad by adding missing indices
        if len(idx) < target:
            extras = np.setdiff1d(np.arange(n), idx)
            need = target - len(idx)
            idx = np.sort(np.concatenate([idx, extras[:need]]))
        return M[np.ix_(idx, idx)], idx
    raise ValueError(f"Unknown downsample mode: {mode}")

# =========================
# Plot
# =========================
if __name__ == "__main__":
    gt, pred, diff, meta = load_any()
    assert gt.shape == pred.shape == diff.shape, f"shape mismatch: gt={gt.shape}, pred={pred.shape}, diff={diff.shape}"

    # Downsample if needed (to match caption)
    gt_ds, idx = maybe_downsample(gt, target=DOWNSAMPLE_TO, mode=DOWNSAMPLE_MODE)
    pred_ds, _ = maybe_downsample(pred, target=DOWNSAMPLE_TO, mode=DOWNSAMPLE_MODE)
    diff_ds = pred_ds - gt_ds

    N = gt_ds.shape[0]

    # Quick sanity prints (helps you confirm GT is not constant)
    print("[GT]   min/max/std:", float(gt_ds.min()), float(gt_ds.max()), float(gt_ds.std()))
    print("[Pred] min/max/std:", float(pred_ds.min()), float(pred_ds.max()), float(pred_ds.std()))
    print("[Diff] min/max/std:", float(diff_ds.min()), float(diff_ds.max()), float(diff_ds.std()))
    print("Loaded from:", meta.get("src"))
    print("Downsample:", DOWNSAMPLE_MODE, f"{N}/{gt.shape[0]}")

    # vmin/vmax
    if SCALE_MODE == "shared":
        # shared range from BOTH matrices' off-diagonals (still excluding diagonals)
        v_all = np.concatenate([offdiag_values(gt_ds).ravel(), offdiag_values(pred_ds).ravel()])
        vmin_sim = float(np.percentile(v_all, 2))
        vmax_sim = float(np.percentile(v_all, 98))
        gt_vmin, gt_vmax = vmin_sim, vmax_sim
        pr_vmin, pr_vmax = vmin_sim, vmax_sim
    else:
        # panel-wise scaling (recommended so GT doesn't look flat)
        gt_vmin, gt_vmax = robust_vmin_vmax(gt_ds, 2, 98)
        pr_vmin, pr_vmax = robust_vmin_vmax(pred_ds, 2, 98)

    # diff scale symmetric around 0
    abs95 = float(np.percentile(np.abs(diff_ds.ravel()), 95))
    abs95 = max(abs95, 1e-6)
    norm_diff = TwoSlopeNorm(vmin=-abs95, vcenter=0.0, vmax=abs95)

    # =========================
    # V2-style Plotting: 1 row, 3 cols, vertical colorbars
    # =========================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)

    # Panel 1: GT
    im1 = axes[0].imshow(gt_ds, cmap=cmap_sim, vmin=gt_vmin, vmax=gt_vmax, origin='upper', interpolation="nearest")
    axes[0].set_title("Ground Truth RSM", fontweight="bold", pad=12, color=C_AXIS)
    cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cb1.outline.set_visible(False)
    
    # Panel 2: Brain
    im2 = axes[1].imshow(pred_ds, cmap=cmap_sim, vmin=pr_vmin, vmax=pr_vmax, origin='upper', interpolation="nearest")
    axes[1].set_title("Ours (Subj01 Pooled) RSM", fontweight="bold", pad=12, color=C_AXIS)
    cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cb2.outline.set_visible(False)

    # Panel 3: Diff
    im3 = axes[2].imshow(diff_ds, cmap=cmap_diff, norm=norm_diff, origin='upper', interpolation="nearest")
    axes[2].set_title("Difference (Ours - GT)", fontweight="bold", pad=12, color=C_AXIS)
    cb3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cb3.outline.set_visible(False)

    # Styling
    for ax in axes:
        ax.set_xlabel("Image Index")
        ax.set_ylabel("Image Index")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(C_GRAY)
        ax.spines['left'].set_color(C_GRAY)

    # Suptitle aligned with your supplement caption
    fig.suptitle(
        f"RSM heatmaps | subj01 | best overall (pooled)  (downsampled {N}/982)",
        fontsize=16, color=C_AXIS
    )

    # Watermark removed as requested

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()

    print("Saved:", OUT_PNG)
    print("Saved:", OUT_PDF)
