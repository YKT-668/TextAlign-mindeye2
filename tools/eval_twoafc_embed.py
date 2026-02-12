#!/usr/bin/env python
# coding: utf-8
"""eval_twoafc_embed.py

2AFC / two-way identification for embedding-based evaluation.

Inputs (env vars, similar to eval_textalign_latent_plus.py):
- BRAIN_PATH: brain embedding tensor (.pt)
  - pooled: [N, D]
  - tokens: [N, T, D]
- IDS_PATH: ids.json (len N, global image id)
- GT_PATH: evals/all_images.pt (images or precomputed features)
- EVAL_REPR: pooled | tokens_flatten
- EVAL_SUBSET: (empty)|shared982
- METRIC: cosine | pearson   (default cosine)
- BOOTSTRAP: int (default 1000; 0 disables)
- RESULT_DIR: output directory for metrics.json/metrics.txt

Protocol (easy, stable): for each i, compare the matched similarity vs ALL mismatched negatives.
- brain->image: acc_i = mean_{j!=i} [ s(b_i,g_i) > s(b_i,g_j) ]
- image->brain: acc_i = mean_{j!=i} [ s(g_i,b_i) > s(g_i,b_j) ]
Final score is mean over i. Chance = 0.5.

Note: Uses shared982 ids from WDS test split; auto-generates src/shared982.npy if missing.
"""

from __future__ import annotations

import json
import os
import sys
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
METRIC = os.environ.get("METRIC", "cosine").strip().lower()
BOOTSTRAP = int(os.environ.get("BOOTSTRAP", "1000"))
RESULT_DIR = os.environ.get("RESULT_DIR")
EXP_NAME = os.environ.get("EXP_NAME", "twoafc")
GT_POOLING = os.environ.get("GT_POOLING", "mean").strip().lower()


def _build_or_load_shared982_ids() -> np.ndarray:
    mask_path = os.path.join(_PROJ_ROOT, "src", "shared982.npy")
    if os.path.isfile(mask_path):
        m = np.load(mask_path)
        if m.dtype == np.bool_ and m.ndim == 1:
            return np.where(m > 0)[0].astype(np.int64)
        if m.ndim == 1:
            return m.astype(np.int64)
        raise RuntimeError(f"shared982.npy 形状异常: {m.shape} dtype={m.dtype}")

    import glob
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

    # Make a bool mask compatible with shared1000.npy length.
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


def _load_brain(brain_path: str) -> torch.Tensor:
    obj = torch.load(brain_path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        cand = None
        for k, v in obj.items():
            if not torch.is_tensor(v):
                continue
            if EVAL_REPR == "pooled" and v.dim() == 2:
                cand = v
                print(f"[LOAD] brain: dict['{k}']")
                break
            if EVAL_REPR == "tokens_flatten" and v.dim() == 3:
                cand = v
                print(f"[LOAD] brain: dict['{k}']")
                break
        if cand is None:
            raise RuntimeError(f"brain dict 里找不到匹配 EVAL_REPR={EVAL_REPR} 的 tensor, keys={list(obj.keys())}")
        return cand
    raise RuntimeError(f"Unsupported brain object type: {type(obj)}")


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
            if EVAL_REPR == "tokens_flatten":
                raise RuntimeError("tokens_flatten 需要 GT tokens [N,T,D]，但 GT 是 2D")
            gt_feats = gt_obj
        elif gt_obj.dim() == 4 and gt_obj.shape[1] == 3:
            # Use cached BigG features/tokens from eval_textalign_latent_plus pipeline.
            if EVAL_REPR == "tokens_flatten":
                offline_tok_path = os.path.join(_PROJ_ROOT, "evals", "all_images_bigG_tokens_256x1664.pt")
                if not os.path.isfile(offline_tok_path):
                    raise RuntimeError(
                        "GT tokens 缓存不存在: evals/all_images_bigG_tokens_256x1664.pt. "
                        "请先跑 retrieval 的 tokens_flatten 生成缓存。"
                    )
                gt_feats = torch.load(offline_tok_path, map_location="cpu")
            else:
                pool_tag = GT_POOLING.replace("-", "_")
                offline_feat_path = os.path.join(_PROJ_ROOT, "evals", f"all_images_bigG_1664_{pool_tag}.pt")
                legacy_feat_path = os.path.join(_PROJ_ROOT, "evals", "all_images_bigG_1664.pt")
                if os.path.isfile(offline_feat_path):
                    gt_feats = torch.load(offline_feat_path, map_location="cpu")
                elif os.path.isfile(legacy_feat_path):
                    gt_feats = torch.load(legacy_feat_path, map_location="cpu")
                else:
                    raise RuntimeError(
                        "GT pooled 缓存不存在: evals/all_images_bigG_1664_*.pt. "
                        "请先跑 retrieval 生成缓存。"
                    )
        else:
            raise RuntimeError(f"Unsupported GT tensor shape: {tuple(gt_obj.shape)}")

    elif isinstance(gt_obj, dict):
        # features
        if EVAL_REPR == "tokens_flatten":
            for k, v in gt_obj.items():
                if torch.is_tensor(v) and v.dim() == 3:
                    gt_feats = v
                    break
        else:
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
        # ids
        for key in ["ids", "image_ids", "img_ids", "nsd_ids"]:
            if key in gt_obj:
                gt_ids = np.asarray(gt_obj[key], dtype=np.int64)
                break
    else:
        raise RuntimeError(f"Unsupported GT object type: {type(gt_obj)}")

    if gt_ids is None:
        shared_path = os.path.join(_PROJ_ROOT, "src", "shared1000.npy")
        if os.path.isfile(shared_path):
            shared_mask = np.load(shared_path)
            shared_ids = np.where(shared_mask > 0)[0].astype(np.int64)
            if len(shared_ids) == int(gt_feats.shape[0]):
                gt_ids = shared_ids
            else:
                gt_ids = np.arange(int(gt_feats.shape[0]), dtype=np.int64)
        else:
            gt_ids = np.arange(int(gt_feats.shape[0]), dtype=np.int64)

    return gt_feats, gt_ids


def _align(brain_feats: torch.Tensor, brain_ids: np.ndarray, gt_feats: torch.Tensor, gt_ids: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    id2row = {int(g): i for i, g in enumerate(gt_ids)}
    rows = []
    for gid in brain_ids:
        if int(gid) not in id2row:
            raise KeyError(f"GT 中找不到 image_id={int(gid)}")
        rows.append(id2row[int(gid)])
    rows = np.asarray(rows, dtype=np.int64)
    gt_sel = gt_feats[rows]
    return brain_feats, brain_ids, gt_sel


def _apply_subset(brain_feats: torch.Tensor, brain_ids: np.ndarray, gt_sel: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    if EVAL_SUBSET in ("shared982", "test982", "wds_test_982", "wds-test-982"):
        subset_ids = _build_or_load_shared982_ids()
        keep = np.isin(brain_ids, subset_ids)
        if int(keep.sum()) <= 0:
            raise RuntimeError("shared982 过滤后样本数为 0")
        return brain_feats[keep], brain_ids[keep], gt_sel[keep]
    return brain_feats, brain_ids, gt_sel


def _standardize(x: torch.Tensor) -> torch.Tensor:
    # per-sample z-score for pearson
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def _compute_sim(brain: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # brain/gt already aligned and subset-applied.
    use_amp = (device == "cuda")

    if METRIC not in ("cosine", "pearson"):
        raise ValueError(f"Unknown METRIC: {METRIC}")

    brain = brain.to(device)
    gt = gt.to(device)

    if METRIC == "pearson":
        brain = _standardize(brain)
        gt = _standardize(gt)

    if EVAL_REPR == "pooled":
        brain = brain.to(dtype=torch.float16 if device == "cuda" else torch.float32)
        gt = gt.to(dtype=torch.float16 if device == "cuda" else torch.float32)
        brain_n = F.normalize(brain, dim=-1)
        gt_n = F.normalize(gt, dim=-1)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            sim = brain_n @ gt_n.t()
        return sim

    if EVAL_REPR == "tokens_flatten":
        if brain.dim() != 3 or gt.dim() != 3:
            raise RuntimeError(f"tokens_flatten expects 3D tokens: brain={tuple(brain.shape)} gt={tuple(gt.shape)}")
        # Normalize in flattened sense without materializing [N, T*D].
        # dot(i,j) = sum_{t,d} brain[i,t,d] * gt[j,t,d]
        # norm(i) = sqrt(sum_{t,d} brain[i,t,d]^2)
        # IMPORTANT: do accumulation in float32.
        # With fp16 accumulation, the 426k-term dot-products can become numerically unstable
        # and may invert ordering (observed as all-zero 2AFC).
        brain = brain.to(dtype=torch.float32)
        gt = gt.to(dtype=torch.float32)

        with torch.no_grad():
            dot = torch.einsum("ntd,mtd->nm", brain, gt)
            bn = torch.sqrt(torch.einsum("ntd,ntd->n", brain, brain).clamp_min(1e-12))
            gn = torch.sqrt(torch.einsum("mtd,mtd->m", gt, gt).clamp_min(1e-12))
            sim = dot / (bn[:, None] * gn[None, :]).clamp_min(1e-12)
        return sim

    raise ValueError(f"Unknown EVAL_REPR: {EVAL_REPR}")


def _twoafc_allpairs(sim: torch.Tensor) -> Dict[str, np.ndarray]:
    # sim: [N,N]
    with torch.no_grad():
        diag = torch.diag(sim).view(-1, 1)
        gt_mask = torch.ones_like(sim, dtype=torch.bool)
        gt_mask.fill_diagonal_(False)

        # brain->image: compare row-wise
        ok_fwd = (diag > sim) & gt_mask
        acc_fwd = ok_fwd.float().sum(dim=1) / (sim.size(1) - 1)

        # image->brain: compare column-wise (equivalent to using sim.T)
        ok_bwd = (diag.t() > sim) & gt_mask
        acc_bwd = ok_bwd.float().sum(dim=0) / (sim.size(0) - 1)

    return {
        "acc_fwd": acc_fwd.detach().cpu().numpy(),
        "acc_bwd": acc_bwd.detach().cpu().numpy(),
    }


def _bootstrap_ci(acc: np.ndarray, n_boot: int, seed: int = 0) -> Tuple[float, Tuple[float, float]]:
    mean = float(acc.mean())
    if n_boot <= 0:
        return mean, (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = acc.shape[0]
    samples = rng.integers(0, n, size=(n_boot, n))
    boot = acc[samples].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, (float(lo), float(hi))


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
    print(f"[CFG]  repr={EVAL_REPR} subset={EVAL_SUBSET or 'all'} metric={METRIC} bootstrap={BOOTSTRAP} device={device}")

    brain_feats = _load_brain(BRAIN_PATH)
    brain_ids = _load_ids(IDS_PATH, int(brain_feats.shape[0]))
    gt_feats, gt_ids = _load_gt(GT_PATH)

    # align then subset
    brain_feats, brain_ids, gt_sel = _align(brain_feats, brain_ids, gt_feats, gt_ids)
    brain_feats, brain_ids, gt_sel = _apply_subset(brain_feats, brain_ids, gt_sel)

    # compute sim and 2AFC
    sim = _compute_sim(brain_feats, gt_sel)
    acc = _twoafc_allpairs(sim)

    fwd_mean, fwd_ci = _bootstrap_ci(acc["acc_fwd"], BOOTSTRAP, seed=0)
    bwd_mean, bwd_ci = _bootstrap_ci(acc["acc_bwd"], BOOTSTRAP, seed=1)

    metrics = {
        "exp_name": EXP_NAME,
        "N": int(sim.size(0)),
        "eval_repr": EVAL_REPR,
        "eval_subset": EVAL_SUBSET or "all",
        "metric": METRIC,
        "gt_pooling": GT_POOLING,
        "twoafc": {
            "brain_to_image": {
                "mean": fwd_mean,
                "ci95": [fwd_ci[0], fwd_ci[1]],
            },
            "image_to_brain": {
                "mean": bwd_mean,
                "ci95": [bwd_ci[0], bwd_ci[1]],
            },
        },
    }

    out_json = Path(RESULT_DIR) / "metrics.json"
    out_txt = Path(RESULT_DIR) / "metrics.txt"
    out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = []
    lines.append("=== 2AFC (all-pairs) ===")
    lines.append(f"N={metrics['N']}  repr={EVAL_REPR}  subset={metrics['eval_subset']}  metric={METRIC}")
    lines.append(f"brain->image: {fwd_mean:.4f}  (95% CI {fwd_ci[0]:.4f}..{fwd_ci[1]:.4f})")
    lines.append(f"image->brain: {bwd_mean:.4f}  (95% CI {bwd_ci[0]:.4f}..{bwd_ci[1]:.4f})")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[DONE] wrote", str(out_json))


if __name__ == "__main__":
    main()
