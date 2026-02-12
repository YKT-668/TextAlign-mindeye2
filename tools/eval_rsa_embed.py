#!/usr/bin/env python
# coding: utf-8
"""eval_rsa_embed.py

RSA (Representational Similarity Analysis) for embedding-based evaluation.

We compare representational geometry by correlating the upper-triangular entries of
within-modality similarity matrices:
  - S_img: similarity between GT image embeddings
  - S_brain: similarity between brain-predicted embeddings

Score:
  - Spearman rho between vec(triu(S_img, k=1)) and vec(triu(S_brain, k=1))

Why this implementation:
  - Uses shared982 protocol (WDS test split) for N=982.
  - Computes similarity matrices on GPU (fast), then does rank-correlation on CPU.
  - Uses analytic Fisher-z CI (very fast, typically tight due to large number of pairs).

Inputs (env vars, consistent with other tools):
- BRAIN_PATH: pooled tensor .pt [N,D] OR tokens tensor .pt [N,T,D]
- IDS_PATH: ids.json (len N, global image id)
- GT_PATH: evals/all_images.pt (images or precomputed features)
- EVAL_REPR: pooled | tokens_flatten
    NOTE: RSA operates on stimulus-level vectors. If EVAL_REPR=tokens_flatten and the
    input is 3D tokens, we use mean-pooling over tokens to obtain [N,D] before RSA.
- EVAL_SUBSET: (empty) | shared982
- SIM_METRIC: cosine | pearson  (default cosine)
- RESULT_DIR: output directory (metrics.json/metrics.txt)
- EXP_NAME: label

Outputs:
- metrics.json
- metrics.txt
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

device = "cuda" if torch.cuda.is_available() else "cpu"

BRAIN_PATH = os.environ.get("BRAIN_PATH")
IDS_PATH = os.environ.get("IDS_PATH")
GT_PATH = os.environ.get("GT_PATH")
EVAL_REPR = os.environ.get("EVAL_REPR", "pooled").strip().lower()
EVAL_SUBSET = os.environ.get("EVAL_SUBSET", "").strip().lower()
SIM_METRIC = os.environ.get("SIM_METRIC", "cosine").strip().lower()
RESULT_DIR = os.environ.get("RESULT_DIR")
EXP_NAME = os.environ.get("EXP_NAME", "rsa")
GT_POOLING = os.environ.get("GT_POOLING", "mean").strip().lower()
BOOTSTRAP = int(os.environ.get("BOOTSTRAP", "0"))
SEED = int(os.environ.get("SEED", "42"))


def _build_or_load_shared982_ids() -> np.ndarray:
    mask_path = os.path.join(_PROJ_ROOT, "src", "shared982.npy")
    if os.path.isfile(mask_path):
        m = np.load(mask_path)
        if m.dtype == np.bool_ and m.ndim == 1:
            return np.where(m > 0)[0].astype(np.int64)
        if m.ndim == 1:
            return m.astype(np.int64)
        raise RuntimeError(f"shared982.npy 形状异常: {m.shape} dtype={m.dtype}")

    try:
        import webdataset as wds
    except Exception as e:
        raise RuntimeError(
            f"EVAL_SUBSET=shared982 需要 webdataset 来自动生成 shared982.npy，但导入失败: {type(e).__name__}: {e}"
        )

    wds_root = os.path.join(_PROJ_ROOT, "src", "wds", "subj01", "test")
    urls = sorted(glob.glob(os.path.join(wds_root, "*.tar")))
    if not urls:
        raise RuntimeError(f"找不到 WDS test split shards: {wds_root}")

    ds = wds.WebDataset(urls).decode("torch").to_tuple("behav.npy")
    uniq = set()
    for (b,) in ds:
        uniq.add(int(b[0, 0]))
    ids = np.array(sorted(uniq), dtype=np.int64)
    if ids.size != 982:
        raise RuntimeError(f"从 WDS test split 得到 unique ids={ids.size}，期望 982")

    mask_len = 73000
    try:
        shared1000_path = os.path.join(_PROJ_ROOT, "src", "shared1000.npy")
        if os.path.isfile(shared1000_path):
            m1000 = np.load(shared1000_path)
            if m1000.ndim == 1 and m1000.dtype == np.bool_:
                mask_len = int(m1000.shape[0])
    except Exception:
        pass

    m = np.zeros((mask_len,), dtype=np.bool_)
    m[ids] = True
    np.save(mask_path, m)
    print(f"[SUBSET] wrote {mask_path} (len={mask_len}, nnz={int(m.sum())})")
    return ids


def _load_tensor(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        # pick first tensor that looks like features
        for _, v in obj.items():
            if torch.is_tensor(v) and v.dim() in (2, 3):
                return v
    raise RuntimeError(f"Unsupported tensor container at {path}: {type(obj)}")


def _load_ids(ids_path: str, n: int) -> np.ndarray:
    if ids_path and os.path.isfile(ids_path):
        with open(ids_path, "r") as f:
            ids = np.asarray(json.load(f), dtype=np.int64)
        if ids.shape[0] != n:
            raise RuntimeError(f"ids.json len={ids.shape[0]} != N={n}")
        return ids
    return np.arange(n, dtype=np.int64)


def _load_gt(gt_path: str) -> Tuple[torch.Tensor, np.ndarray]:
    gt_obj = torch.load(gt_path, map_location="cpu")
    gt_feats = None
    gt_ids = None

    if isinstance(gt_obj, torch.Tensor):
        if gt_obj.dim() == 2:
            gt_feats = gt_obj
        elif gt_obj.dim() == 4 and gt_obj.shape[1] == 3:
            # Use cached pooled BigG features.
            pool_tag = GT_POOLING.replace("-", "_")
            offline_feat_path = os.path.join(_PROJ_ROOT, "evals", f"all_images_bigG_1664_{pool_tag}.pt")
            legacy_feat_path = os.path.join(_PROJ_ROOT, "evals", "all_images_bigG_1664.pt")
            if os.path.isfile(offline_feat_path):
                gt_feats = torch.load(offline_feat_path, map_location="cpu")
            elif os.path.isfile(legacy_feat_path):
                gt_feats = torch.load(legacy_feat_path, map_location="cpu")
            else:
                raise RuntimeError(
                    "GT pooled 缓存不存在: evals/all_images_bigG_1664_*.pt. 请先跑 retrieval 生成缓存。"
                )
        else:
            raise RuntimeError(f"Unsupported GT tensor shape: {tuple(gt_obj.shape)}")

    elif isinstance(gt_obj, dict):
        for k, v in gt_obj.items():
            if torch.is_tensor(v) and v.dim() == 2:
                gt_feats = v
                break
        if gt_feats is None:
            for k, v in gt_obj.items():
                if torch.is_tensor(v) and v.dim() == 3:
                    gt_feats = v.mean(dim=1)
                    break
        if gt_feats is None:
            raise RuntimeError(f"GT dict 里找不到特征 tensor, keys={list(gt_obj.keys())}")
        for key in ["ids", "image_ids", "img_ids", "nsd_ids"]:
            if key in gt_obj:
                gt_ids = np.asarray(gt_obj[key], dtype=np.int64)
                break
    else:
        raise RuntimeError(f"Unsupported GT object type: {type(gt_obj)}")

    if gt_ids is None:
        shared_path = os.path.join(_PROJ_ROOT, "src", "shared1000.npy")
        if os.path.isfile(shared_path):
            m = np.load(shared_path)
            shared_ids = np.where(m > 0)[0].astype(np.int64)
            if len(shared_ids) == int(gt_feats.shape[0]):
                gt_ids = shared_ids
            else:
                gt_ids = np.arange(int(gt_feats.shape[0]), dtype=np.int64)
        else:
            gt_ids = np.arange(int(gt_feats.shape[0]), dtype=np.int64)

    return gt_feats, gt_ids


def _to_stim_vectors(x: torch.Tensor) -> Tuple[torch.Tensor, str]:
    if x.dim() == 2:
        return x, "pooled"
    if x.dim() == 3:
        # mean over tokens for RSA
        return x.mean(dim=1), "tokens_mean"
    raise RuntimeError(f"Unsupported feature rank: {x.dim()} shape={tuple(x.shape)}")


def _standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def _cosine_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    return x @ x.t()


def _pearson_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    x = _standardize(x)
    x = F.normalize(x, dim=-1)
    return x @ x.t()


def _vectorize_upper(sim: np.ndarray) -> np.ndarray:
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    return sim[iu]


def _rankdata_no_ties(x: np.ndarray) -> np.ndarray:
    # Fast rank assignment assuming very few exact ties.
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    return ranks


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).mean()) * np.sqrt((b * b).mean()))
    if denom <= 0:
        return float("nan")
    return float((a * b).mean() / denom)


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata_no_ties(a)
    rb = _rankdata_no_ties(b)
    return _pearson_corr(ra, rb)


def _fisher_ci(r: float, m: int) -> Tuple[float, float]:
    # Approximate 95% CI using Fisher z transform.
    # For Spearman, this is an approximation but commonly used.
    if not np.isfinite(r):
        return float("nan"), float("nan")
    r = float(np.clip(r, -0.999999, 0.999999))
    if m <= 3:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(m - 3)
    lo = np.tanh(z - 1.96 * se)
    hi = np.tanh(z + 1.96 * se)
    return float(lo), float(hi)


def _bootstrap_pairwise_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    corr_fn,
) -> Tuple[Tuple[float, float], float]:
    """Pairwise bootstrap CI over similarity-vector entries.

    This is much faster than bootstrapping over stimuli (which would require
    rebuilding submatrices and re-ranking O(N^2) per resample).
    """
    if n_boot <= 0:
        return (float("nan"), float("nan")), float("nan")
    if a.shape != b.shape:
        raise RuntimeError(f"bootstrap vector shape mismatch: {a.shape} vs {b.shape}")
    m = int(a.size)
    if m <= 1:
        return (float("nan"), float("nan")), float("nan")

    rng = np.random.default_rng(int(seed))
    boots = np.empty((int(n_boot),), dtype=np.float64)
    # sample indices with replacement
    for t in range(int(n_boot)):
        idx = rng.integers(0, m, size=m, dtype=np.int64)
        boots[t] = float(corr_fn(a[idx], b[idx]))
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return (lo, hi), float(np.nanstd(boots, ddof=1))


def main() -> None:
    if not BRAIN_PATH or not os.path.isfile(BRAIN_PATH):
        raise SystemExit(f"BRAIN_PATH missing or not found: {BRAIN_PATH}")
    if not GT_PATH or not os.path.isfile(GT_PATH):
        raise SystemExit(f"GT_PATH missing or not found: {GT_PATH}")
    if not RESULT_DIR:
        raise SystemExit("RESULT_DIR is required")

    os.makedirs(RESULT_DIR, exist_ok=True)

    print("[PATH] brain:", BRAIN_PATH)
    print("[PATH] ids  :", IDS_PATH)
    print("[PATH] gt   :", GT_PATH)
    print(
        f"[CFG] repr={EVAL_REPR} subset={EVAL_SUBSET or 'all'} sim_metric={SIM_METRIC} device={device} "
        f"bootstrap={BOOTSTRAP} seed={SEED}"
    )

    brain_raw = _load_tensor(BRAIN_PATH)
    brain_ids = _load_ids(IDS_PATH, int(brain_raw.shape[0]))
    gt_raw, gt_ids = _load_gt(GT_PATH)

    # Align GT to brain order
    id2row = {int(g): i for i, g in enumerate(gt_ids)}
    rows = []
    for gid in brain_ids:
        if int(gid) not in id2row:
            raise KeyError(f"GT 中找不到 image_id={int(gid)}")
        rows.append(id2row[int(gid)])
    rows = np.asarray(rows, dtype=np.int64)
    gt_sel_raw = gt_raw[rows]

    # Convert to stimulus-level vectors
    brain_vec, brain_repr = _to_stim_vectors(brain_raw)
    gt_vec, gt_repr = _to_stim_vectors(gt_sel_raw)

    # Subset
    if EVAL_SUBSET in ("shared982", "test982", "wds_test_982", "wds-test-982"):
        subset_ids = _build_or_load_shared982_ids()
        keep = np.isin(brain_ids, subset_ids)
        if int(keep.sum()) <= 0:
            raise RuntimeError("shared982 过滤后样本数为 0")
        brain_ids = brain_ids[keep]
        brain_vec = brain_vec[keep]
        gt_vec = gt_vec[keep]

    n = int(brain_vec.shape[0])
    if int(gt_vec.shape[0]) != n:
        raise RuntimeError(f"N mismatch after subset: brain={n} gt={int(gt_vec.shape[0])}")

    # Similarity matrices (GPU)
    brain_vec = brain_vec.to(device=device, dtype=torch.float32)
    gt_vec = gt_vec.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        if SIM_METRIC == "cosine":
            s_brain = _cosine_sim_matrix(brain_vec)
            s_gt = _cosine_sim_matrix(gt_vec)
        elif SIM_METRIC == "pearson":
            s_brain = _pearson_sim_matrix(brain_vec)
            s_gt = _pearson_sim_matrix(gt_vec)
        else:
            raise ValueError(f"Unknown SIM_METRIC: {SIM_METRIC}")

    s_brain_np = s_brain.detach().cpu().numpy()
    s_gt_np = s_gt.detach().cpu().numpy()

    v_brain = _vectorize_upper(s_brain_np)
    v_gt = _vectorize_upper(s_gt_np)

    m = int(v_brain.size)

    # RSA correlations
    rho_spearman = _spearman_corr(v_brain, v_gt)
    rho_pearson = _pearson_corr(v_brain, v_gt)

    ci_s = _fisher_ci(rho_spearman, m)
    ci_p = _fisher_ci(rho_pearson, m)

    ci_s_boot = None
    ci_p_boot = None
    if BOOTSTRAP and int(BOOTSTRAP) > 0:
        # Pairwise bootstrap over similarity-vector entries.
        # This is an approximation; see ci_note.
        ci_s_boot, _ = _bootstrap_pairwise_ci(
            v_brain,
            v_gt,
            n_boot=int(BOOTSTRAP),
            seed=int(SEED),
            corr_fn=_spearman_corr,
        )
        ci_p_boot, _ = _bootstrap_pairwise_ci(
            v_brain,
            v_gt,
            n_boot=int(BOOTSTRAP),
            seed=int(SEED),
            corr_fn=_pearson_corr,
        )

    metrics: Dict = {
        "exp_name": EXP_NAME,
        "N": n,
        "pairs": m,
        "eval_repr": EVAL_REPR,
        "eval_subset": EVAL_SUBSET or "all",
        "brain_repr_used": brain_repr,
        "gt_repr_used": gt_repr,
        "sim_metric": SIM_METRIC,
        "gt_pooling": GT_POOLING,
        "bootstrap": int(BOOTSTRAP),
        "seed": int(SEED),
        "rsa": {
            "spearman": {
                "rho": float(rho_spearman),
                "ci95_fisher": [ci_s[0], ci_s[1]],
                "ci95_bootstrap": [ci_s_boot[0], ci_s_boot[1]] if ci_s_boot else None,
            },
            "pearson": {
                "rho": float(rho_pearson),
                "ci95_fisher": [ci_p[0], ci_p[1]],
                "ci95_bootstrap": [ci_p_boot[0], ci_p_boot[1]] if ci_p_boot else None,
            },
        },
        "ci_note": (
            "Fisher CI is computed via Fisher-z on pair count (approx; also approx for Spearman). "
            "If BOOTSTRAP>0, ci95_bootstrap uses pairwise bootstrap over similarity-vector entries "
            "(fast approximation; not stimulus-level bootstrap)."
        ),
        "notes": [],
    }

    if EVAL_REPR == "tokens_flatten" and brain_raw.dim() == 3:
        metrics["notes"].append("RSA uses mean-pooling over tokens (tokens_mean) for efficiency.")

    out_dir = Path(RESULT_DIR)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = []
    lines.append("=== RSA ===")
    lines.append(f"N={n} pairs={m} subset={metrics['eval_subset']} sim_metric={SIM_METRIC}")
    lines.append(f"repr(requested)={EVAL_REPR} brain_repr_used={brain_repr} gt_repr_used={gt_repr}")
    lines.append(f"Spearman rho={rho_spearman:.6f}  CI95~[{ci_s[0]:.6f},{ci_s[1]:.6f}]")
    if ci_s_boot is not None:
        lines.append(f"Spearman bootstrap CI95~[{ci_s_boot[0]:.6f},{ci_s_boot[1]:.6f}]  (B={int(BOOTSTRAP)})")
    lines.append(f"Pearson  rho={rho_pearson:.6f}  CI95~[{ci_p[0]:.6f},{ci_p[1]:.6f}]")
    if ci_p_boot is not None:
        lines.append(f"Pearson  bootstrap CI95~[{ci_p_boot[0]:.6f},{ci_p_boot[1]:.6f}]  (B={int(BOOTSTRAP)})")
    if metrics["notes"]:
        lines.append("Notes: " + " ".join(metrics["notes"]))
    (out_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[DONE] wrote", str(out_dir / "metrics.json"))


if __name__ == "__main__":
    main()
