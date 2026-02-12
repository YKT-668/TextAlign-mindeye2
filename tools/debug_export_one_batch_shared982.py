#!/usr/bin/env python
# coding: utf-8
"""debug_export_one_batch_shared982.py

诊断用途：用与导出脚本一致的模型构建/ckpt加载逻辑，只跑 WDS test split 的一条 batch，
打印：
1) 实际 ckpt 路径、state_dict 前 5 个 key、load_state_dict 的 missing/unexpected keys
2) forward 输入/输出统计（均值/方差/最小最大），确保不是全零/常数
3) 明确导出变量：tokens=[B,256,1664]，pooled=tokens.mean(dim=1)=[B,1664]
4) 随机抽样(默认 10)做小规模对齐检查：sim.diag.mean / sim.mean / 差值
5) 打印 batch 的前若干 image ids，并验证它们都在 shared982 mask 内

注意：
- 该脚本不会写任何评测结果目录，也不会覆盖历史产物。
- 该脚本仅用于证明 ckpt/导出分支是否生效。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_SRC_DIR = os.path.join(_PROJ_ROOT, "src")

import sys
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models import BrainNetwork  # noqa: E402

CLIP_SEQ_DIM = 256
CLIP_EMB_DIM = 1664


class RidgeRegression(nn.Module):
    def __init__(self, n_voxels: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(n_voxels, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.linear(x).unsqueeze(1)


class ClipBackbone(nn.Module):
    def __init__(self, n_voxels: int, hidden_dim: int, n_blocks: int):
        super().__init__()
        self.ridge = RidgeRegression(n_voxels, hidden_dim)
        self.backbone = BrainNetwork(
            h=hidden_dim,
            in_dim=hidden_dim,
            seq_len=1,
            n_blocks=n_blocks,
            clip_size=CLIP_EMB_DIM,
            out_dim=CLIP_EMB_DIM * CLIP_SEQ_DIM,
            blurry_recon=False,
            clip_scale=1,
        )

    def forward(self, vox: torch.Tensor) -> torch.Tensor:
        h = self.ridge(vox)
        _backbone, clip_voxels, _b = self.backbone(h)
        return clip_voxels


def _load_checkpoint(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _infer_hidden_and_blocks(state_dict: Dict[str, Any]) -> Tuple[int, int]:
    hidden_dim = None
    for key in ("ridge.linear.weight", "ridge.weight", "ridge.linears.0.weight"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            hidden_dim = int(state_dict[key].shape[0])
            break
    if hidden_dim is None:
        for k, v in state_dict.items():
            if k.startswith("ridge.linears.") and k.endswith(".weight") and hasattr(v, "shape"):
                hidden_dim = int(v.shape[0])
                break
    if hidden_dim is None:
        cand = None
        for k, v in state_dict.items():
            if k.endswith("backbone.mixer_blocks1.0.0.weight") and hasattr(v, "shape"):
                cand = int(v.shape[0])
                break
        if cand is None:
            raise RuntimeError("无法从 checkpoint 推断 hidden_dim")
        hidden_dim = cand

    block_ids = set()
    prefix = "backbone.mixer_blocks1."
    for k in state_dict.keys():
        if k.startswith(prefix):
            rest = k[len(prefix) :]
            try:
                idx = int(rest.split(".", 1)[0])
                block_ids.add(idx)
            except Exception:
                pass
    n_blocks = (max(block_ids) + 1) if block_ids else 4
    return hidden_dim, n_blocks


def _select_ridge_source(sd: Dict[str, Any], n_vox: int, subj_hint: int) -> Tuple[str, str]:
    if "ridge.linear.weight" in sd and "ridge.linear.bias" in sd:
        w = sd["ridge.linear.weight"]
        if hasattr(w, "shape") and int(w.shape[1]) != int(n_vox):
            raise RuntimeError(f"Checkpoint ridge.linear expects n_vox={int(w.shape[1])}, betas n_vox={int(n_vox)}")
        return "ridge.linear.weight", "ridge.linear.bias"

    import re

    head_indices = []
    for k in sd.keys():
        m = re.match(r"ridge\.linears\.(\d+)\.weight$", k)
        if m:
            head_indices.append(int(m.group(1)))
    head_indices = sorted(set(head_indices))
    if not head_indices:
        raise RuntimeError("Checkpoint does not contain ridge.linear.* or ridge.linears.*")

    dims = {}
    for i in head_indices:
        w_key = f"ridge.linears.{i}.weight"
        if w_key in sd and hasattr(sd[w_key], "shape"):
            dims[i] = int(sd[w_key].shape[1])

    candidates = [i for i, d in dims.items() if int(d) == int(n_vox)]
    if len(candidates) == 1:
        i = candidates[0]
        return f"ridge.linears.{i}.weight", f"ridge.linears.{i}.bias"
    if len(candidates) > 1:
        hint = int(subj_hint) - 1
        i = hint if hint in candidates else min(candidates)
        return f"ridge.linears.{i}.weight", f"ridge.linears.{i}.bias"

    raise RuntimeError(f"No ridge head matches n_vox={n_vox}. Available dims={dims}")


def _load_first_batch_trials(wds_tar: str, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import webdataset as wds  # type: ignore
    except Exception as e:
        raise RuntimeError(f"webdataset not available: {type(e).__name__}: {e}")

    ds = wds.WebDataset(wds_tar, resampled=False).decode().to_tuple("behav.npy")
    image_ids = []
    voxel_rows = []

    for (behav,) in ds:
        if isinstance(behav, np.ndarray):
            behav = torch.from_numpy(behav)
        if behav.ndim == 3:
            behav = behav[0]
        if behav.ndim != 2:
            raise RuntimeError(f"Unexpected behav shape: {tuple(behav.shape)}")
        image_ids.append(int(behav[0, 0].item()))
        voxel_rows.append(int(behav[0, 5].item()))
        if len(image_ids) >= batch_size:
            break

    if not image_ids:
        raise RuntimeError(f"No samples found in {wds_tar}")

    return np.asarray(image_ids, dtype=np.int64), np.asarray(voxel_rows, dtype=np.int64)


def _load_gt_pooled_features() -> Tuple[torch.Tensor, np.ndarray]:
    # 优先使用 mean pooling 缓存
    cand = [
        Path(_PROJ_ROOT) / "evals" / "all_images_bigG_1664_mean.pt",
        Path(_PROJ_ROOT) / "evals" / "all_images_bigG_1664.pt",
    ]
    gt_feats = None
    for p in cand:
        if p.is_file():
            gt_feats = torch.load(str(p), map_location="cpu")
            break
    if gt_feats is None:
        raise RuntimeError("Missing GT cached pooled features under evals/all_images_bigG_1664_*.pt")

    shared1000_mask = np.load(Path(_PROJ_ROOT) / "src" / "shared1000.npy")
    shared1000_ids = np.where(shared1000_mask > 0)[0].astype(np.int64)
    if int(gt_feats.shape[0]) != int(shared1000_ids.shape[0]):
        raise RuntimeError(f"GT feats N={gt_feats.shape[0]} != shared1000 ids N={shared1000_ids.shape[0]}")

    return gt_feats, shared1000_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subj", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--split", default="test", choices=["test", "new_test"], help="WDS split; shared982 uses test")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--debug_num_samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = str(Path(args.ckpt).resolve())
    print("[STEP1] ckpt path:", ckpt_path)

    betas_h5 = str(Path(_SRC_DIR) / f"betas_all_subj0{args.subj}_fp32_renorm.hdf5")
    wds_tar = str(Path(_SRC_DIR) / "wds" / f"subj{args.subj:02d}" / args.split / "0.tar")
    print("[IO] betas_h5:", betas_h5)
    print("[IO] wds_tar :", wds_tar)

    image_ids, voxel_rows = _load_first_batch_trials(wds_tar=wds_tar, batch_size=int(args.batch_size))
    print("[DATA] batch image_ids (first 20):", image_ids[:20].tolist())
    print("[DATA] batch voxel_rows shape:", voxel_rows.shape)

    with h5py.File(betas_h5, "r") as f:
        betas = f["betas"][:]
    vox = torch.tensor(betas[voxel_rows], dtype=torch.float32)
    n_vox = int(vox.shape[1])
    print("[DATA] vox batch shape:", tuple(vox.shape), "n_vox=", n_vox)
    print("[STEP1] vox stats: mean=", float(vox.mean()), "var=", float(vox.var(unbiased=False)))

    sd = _load_checkpoint(ckpt_path)
    keys = list(sd.keys())
    print("[STEP1] state_dict first 5 keys:")
    for k in keys[:5]:
        print(" -", k)

    hidden_dim, n_blocks = _infer_hidden_and_blocks(sd)
    print(f"[STEP1] inferred hidden_dim={hidden_dim} n_blocks={n_blocks}")

    model = ClipBackbone(n_voxels=n_vox, hidden_dim=hidden_dim, n_blocks=n_blocks).to(args.device)

    mapped = dict(sd)
    w_src, b_src = _select_ridge_source(sd, n_vox=n_vox, subj_hint=args.subj)
    mapped["ridge.linear.weight"] = sd[w_src]
    mapped["ridge.linear.bias"] = sd[b_src]

    msg = model.load_state_dict(mapped, strict=False)
    missing = list(getattr(msg, "missing_keys", []))
    unexpected = list(getattr(msg, "unexpected_keys", []))
    print("[STEP1] load_state_dict missing_keys:", len(missing))
    if missing:
        print("  ", missing)
    print("[STEP1] load_state_dict unexpected_keys:", len(unexpected))
    if unexpected:
        print("  ", unexpected)

    model.eval()
    with torch.no_grad():
        x = vox.to(args.device)
        tok = model(x)
        print("[STEP2] export variable name: clip_voxels (tokens)")
        print("[STEP2] clip_voxels shape (pre-mean):", tuple(tok.shape), "(expect [B,256,1664])")
        print("[STEP1] forward output tokens stats: mean=", float(tok.mean()), "var=", float(tok.var(unbiased=False)), "min=", float(tok.min()), "max=", float(tok.max()))

        pooled = tok.mean(dim=1)
        print("[STEP2] pooled = clip_voxels.mean(dim=1) shape:", tuple(pooled.shape), "(expect [B,1664])")
        print("[STEP1] forward output pooled stats: mean=", float(pooled.mean()), "var=", float(pooled.var(unbiased=False)), "min=", float(pooled.min()), "max=", float(pooled.max()))

    # Step3: small alignment sanity on random subset within this batch
    gt_feats, gt_ids = _load_gt_pooled_features()
    id2row = {int(gid): i for i, gid in enumerate(gt_ids)}
    k = min(int(args.debug_num_samples), int(pooled.shape[0]))
    idx = list(range(int(pooled.shape[0])))
    random.shuffle(idx)
    idx = idx[:k]

    chosen_ids = [int(image_ids[i]) for i in idx]
    missing_gt = [gid for gid in chosen_ids if gid not in id2row]
    print("[STEP3] chosen sample ids (k=%d):" % k, chosen_ids)
    print("[STEP3] GT missing ids count:", len(missing_gt))
    if missing_gt:
        print("  missing ids:", missing_gt)

    if not missing_gt:
        b = pooled[idx].float()
        g = torch.stack([gt_feats[id2row[gid]].float() for gid in chosen_ids], dim=0)
        b_n = F.normalize(b, dim=-1)
        g_n = F.normalize(g, dim=-1)
        sim = b_n @ g_n.t()
        sim_diag = sim.diag().mean().item()
        sim_mean = sim.mean().item()
        print("[STEP3] sim.diag().mean:", sim_diag)
        print("[STEP3] sim.mean:", sim_mean)
        print("[STEP3] sim.diag().mean - sim.mean:", sim_diag - sim_mean)

    # Step4-ish: ids in shared982 mask
    shared982_mask = np.load(Path(_PROJ_ROOT) / "src" / "shared982.npy")
    if shared982_mask.dtype == np.bool_:
        in_mask = [bool(shared982_mask[gid]) for gid in image_ids[:20]]
        ok = all(in_mask)
        print("[STEP4] first 20 ids in shared982 mask:", in_mask)
        print("[STEP4] all first 20 in mask?", ok)
    else:
        subset = set(shared982_mask.astype(np.int64).tolist())
        ok = all(int(gid) in subset for gid in image_ids[:20])
        print("[STEP4] shared982.npy is ids list; all first20 in set?", ok)


if __name__ == "__main__":
    main()
