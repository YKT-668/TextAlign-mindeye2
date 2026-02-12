#!/usr/bin/env python
# coding: utf-8
"""extract_brain_clip_tokens_shared1000.py

从本仓库自带的 betas_all_subj0X_fp32_renorm.hdf5 + shared1000.npy
离线提取 brain->CLIP 的 token 级特征。

输出：
- brain_clip_tokens.pt: [1000, 256, 1664]
- brain_clip_mean.pt  : [1000, 1664]  (tokens.mean(dim=1))

设计目标：
- 不依赖 WebDataset (wds)
- 只跑 ridge + backbone（不跑 prior / blurry / 图像生成）
- 兼容 TextAlign 单被试 checkpoint（.pth, 含 model_state_dict）
- 也兼容官方 MindEye2 subj01 ckpt（下载到 train_logs/.../last.pth）
"""

import argparse
import json
import os
from typing import Dict, Any, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_SRC_DIR = os.path.join(_PROJ_ROOT, "src")

# 让我们可以 import src 下模块
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
        # x: [B, n_vox]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.linear(x).unsqueeze(1)  # [B, 1, hidden]


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
        # 返回 clip_voxels tokens: [B, 256, 1664]
        h = self.ridge(vox)
        _backbone, clip_voxels, _b = self.backbone(h)
        return clip_voxels


def _load_checkpoint(path: str) -> Dict[str, Any]:
    # 对于超大 ckpt（官方 subj01 40sess 约 9GB），使用 mmap + weights_only 以降低内存压力
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        # 兜底：整个 dict 就当 state_dict
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _infer_hidden_and_blocks(state_dict: Dict[str, Any]) -> Tuple[int, int]:
    # hidden_dim：优先从 ridge.linears.0.weight 或 ridge.linear.weight 推
    hidden_dim = None
    for key in ("ridge.linear.weight", "ridge.weight", "ridge.linears.0.weight"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            hidden_dim = int(state_dict[key].shape[0])
            break

    if hidden_dim is None:
        # 兼容官方多被试：ridge.linears.{k}.weight（未必包含 0）
        for k, v in state_dict.items():
            if k.startswith("ridge.linears.") and k.endswith(".weight") and hasattr(v, "shape"):
                hidden_dim = int(v.shape[0])
                break

    if hidden_dim is None:
        # 再兜底从 backbone mixer 的 layernorm weight
        cand = None
        for k, v in state_dict.items():
            if k.endswith("backbone.mixer_blocks1.0.0.weight") and hasattr(v, "shape"):
                cand = int(v.shape[0])
                break
        if cand is None:
            raise RuntimeError("无法从 checkpoint 推断 hidden_dim（未找到 ridge/backbone 关键权重）")
        hidden_dim = cand

    # n_blocks：从 backbone.mixer_blocks1.* 统计
    block_ids = set()
    prefix = "backbone.mixer_blocks1."
    for k in state_dict.keys():
        if k.startswith(prefix):
            rest = k[len(prefix):]
            try:
                idx = int(rest.split(".", 1)[0])
                block_ids.add(idx)
            except Exception:
                pass
    n_blocks = (max(block_ids) + 1) if block_ids else 4

    return hidden_dim, n_blocks


def _adapt_state_dict_for_single_subject(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """把多被试 ridge.linears.X.* 映射到 ridge.linear.*（单线性层）。

    - TextAlign 单被试 ckpt：一般是 ridge.linears.0.*
    - 官方多被试 ckpt：可能是 ridge.linears.{subj_idx}.*

    这里不做 subj_idx 选择，选择逻辑放在外层（根据 args.subj）。
    """
    # 由调用者负责挑出正确 subj 的 ridge 权重
    return state_dict


def _select_ridge_source(sd: Dict[str, Any], n_vox: int, subj_hint: int) -> Tuple[str, str]:
    """选择用于构建 ridge.linear 的来源权重 key。

    优先策略：
    1) 若存在 ridge.linear.weight，要求其输入维度等于 n_vox；
    2) 若存在 ridge.linears.{i}.weight，按 weight.shape[1] == n_vox 匹配 i；
       - 0 个匹配：报错并打印该 ckpt 里可用 head 的输入维度（方便判断缺失 subject head）；
       - 多个匹配：优先使用 subj_hint-1，否则取最小 i。
    """
    if "ridge.linear.weight" in sd and "ridge.linear.bias" in sd:
        w = sd["ridge.linear.weight"]
        if hasattr(w, "shape") and int(w.shape[1]) != int(n_vox):
            raise RuntimeError(
                f"Checkpoint ridge.linear expects n_vox={int(w.shape[1])}, but betas provides n_vox={int(n_vox)}. "
                "Please use matching subject betas or a checkpoint trained for this subject."
            )
        return "ridge.linear.weight", "ridge.linear.bias"

    # Multi-head ridge
    import re

    head_indices = []
    for k in sd.keys():
        m = re.match(r"ridge\.linears\.(\d+)\.weight$", k)
        if m:
            head_indices.append(int(m.group(1)))
    head_indices = sorted(set(head_indices))
    if not head_indices:
        raise RuntimeError("Checkpoint does not contain ridge.linear.* or ridge.linears.* weights")

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
        if hint in candidates:
            i = hint
        else:
            i = min(candidates)
        return f"ridge.linears.{i}.weight", f"ridge.linears.{i}.bias"

    # No matching head: provide a helpful diagnostic
    subj_dims = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
    inv = {v: k for k, v in subj_dims.items()}
    avail = []
    for i in sorted(dims.keys()):
        d = dims[i]
        s = inv.get(d)
        avail.append(f"{i}:{d}" + (f"(subj{s:02d})" if s is not None else ""))

    raise RuntimeError(
        "No ridge head matches betas n_vox. "
        f"Requested subj{subj_hint:02d} betas has n_vox={int(n_vox)}; "
        "checkpoint ridge.head input dims: [" + ", ".join(avail) + "]. "
        "This checkpoint likely does not support this subject."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pth or last.pth)")
    ap.add_argument("--subj", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints: ckpt load info, one-batch forward stats, tensor shapes.",
    )
    ap.add_argument(
        "--debug_max_batches",
        type=int,
        default=0,
        help="If >0, limit number of forward batches processed (debug helper).",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="new_test",
        choices=["new_test", "test"],
        help="Which local WDS split to use. new_test is typically shared1000 (3000 trials = 1000*3).",
    )
    ap.add_argument(
        "--wds_tar",
        type=str,
        default=None,
        help="Optional explicit path to WDS tar (behav.npy). Default uses src/wds/subj0X/{split}/0.tar",
    )
    ap.add_argument(
        "--betas_h5",
        type=str,
        default=None,
        help="Optional explicit betas hdf5 path. Default uses src/betas_all_subj0X_fp32_renorm.hdf5",
    )
    ap.add_argument("--ids_out", type=str, default=None, help="Optional ids.json output path")
    ap.add_argument("--no_average_reps", action="store_true", help="If set, do NOT average repetitions (keep trial-level).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Robust device handling (avoid crash when user passes --device cuda on CPU-only env)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] --device=cuda requested but no CUDA is available; falling back to cpu")
        args.device = "cpu"

    betas_h5 = args.betas_h5 or os.path.join(_SRC_DIR, f"betas_all_subj0{args.subj}_fp32_renorm.hdf5")
    assert os.path.isfile(betas_h5), f"Missing betas: {betas_h5}"

    default_tar = os.path.join(_SRC_DIR, "wds", f"subj0{args.subj:02d}".replace("subj0", "subj"), args.split, "0.tar")
    # 注意：repo 里的目录是 src/wds/subj01/...（无前导 0 的 subj01），所以用 subj{args.subj:02d}。
    default_tar = os.path.join(_SRC_DIR, "wds", f"subj{args.subj:02d}", args.split, "0.tar")
    wds_tar = args.wds_tar or default_tar
    assert os.path.isfile(wds_tar), f"Missing WDS tar: {wds_tar}"

    print(f"[IO] betas_h5 : {betas_h5}")
    print(f"[IO] wds_tar  : {wds_tar}")
    print(f"[IO] ckpt     : {args.ckpt}")

    # 读取 behav.npy，拿到 voxel row index 与 image id
    try:
        import webdataset as wds  # type: ignore
    except Exception as e:
        raise RuntimeError(f"webdataset not available: {type(e).__name__}: {e}")

    ds = (
        wds.WebDataset(wds_tar, resampled=False)
        .decode()
        .to_tuple("behav.npy")
    )
    # new_test/test 的 0.tar 是单 shard，直接一次性读完
    behav_list = []
    for (behav,) in ds:
        if isinstance(behav, np.ndarray):
            behav = torch.from_numpy(behav)
        elif not torch.is_tensor(behav):
            raise TypeError(f"Unexpected behav type: {type(behav)}")
        behav_list.append(behav)
    if not behav_list:
        raise RuntimeError(f"No samples found in {wds_tar}")

    # 每个 sample 的 behav: 常见为 [1, 1, 6] 或 [1, 6]；拼成 [N,1,6]
    normed = []
    for b in behav_list:
        if b.ndim == 3:
            normed.append(b[0])
        elif b.ndim == 2:
            normed.append(b)
        else:
            raise RuntimeError(f"Unexpected behav shape: {tuple(b.shape)}")
    behav0 = torch.stack(normed, dim=0)
    # 兼容不同形状：取最后一维的 (image_id, voxel_row) 位置
    # 按项目代码约定：behav[:,0,0] = image idx; behav[:,0,5] = voxel row idx
    image_ids = behav0[:, 0, 0].cpu().long().numpy()
    voxel_rows = behav0[:, 0, 5].cpu().long().numpy()
    print(f"[WDS] loaded trials: {len(image_ids)}  unique images: {len(np.unique(image_ids))}")

    with h5py.File(betas_h5, "r") as f:
        betas = f["betas"][:]
    vox_trials = torch.tensor(betas[voxel_rows], dtype=torch.float32)  # [N_trials, n_vox]

    n_vox = int(vox_trials.shape[1])
    print(f"[DATA] vox_trials shape: {tuple(vox_trials.shape)}")

    sd = _load_checkpoint(args.ckpt)
    sd = _adapt_state_dict_for_single_subject(sd)

    if args.debug:
        try:
            keys = list(sd.keys())
            print("[DEBUG] ckpt_path:", args.ckpt)
            print("[DEBUG] state_dict first5 keys:", keys[:5])
            print("[DEBUG] state_dict num_keys:", len(keys))
        except Exception as e:
            print(f"[DEBUG][WARN] failed to print state_dict keys: {type(e).__name__}: {e}")
    hidden_dim, n_blocks = _infer_hidden_and_blocks(sd)
    print(f"[CKPT] inferred hidden_dim={hidden_dim}, n_blocks={n_blocks}")

    model = ClipBackbone(n_voxels=n_vox, hidden_dim=hidden_dim, n_blocks=n_blocks).to(args.device)

    mapped = dict(sd)

    # 选择与 betas 维度匹配的 ridge head
    w_src, b_src = _select_ridge_source(sd, n_vox=n_vox, subj_hint=args.subj)
    mapped["ridge.linear.weight"] = sd[w_src]
    mapped["ridge.linear.bias"] = sd[b_src]

    msg = model.load_state_dict(mapped, strict=False)
    print("[LOAD] load_state_dict:", msg)
    if args.debug:
        missing = list(getattr(msg, "missing_keys", []) or [])
        unexpected = list(getattr(msg, "unexpected_keys", []) or [])
        print("[DEBUG] load_state_dict missing_keys:", missing)
        print("[DEBUG] load_state_dict unexpected_keys:", unexpected)
        if len(missing) == 0 and len(unexpected) == 0:
            print("[DEBUG] load_state_dict OK (missing=0, unexpected=0)")

    model.eval()

    bs = int(args.batch_size)

    if args.no_average_reps:
        # 注意：trial-level tokens 体积较大（new_test 通常 3000x256x1664 float32 ≈ 5GB），
        # 这里保持原逻辑会占用较高内存；如确有需要建议在更大内存机器上运行。
        all_tokens = []
        with torch.no_grad():
            for i in range(0, vox_trials.shape[0], bs):
                if args.debug and args.debug_max_batches and (i // bs) >= int(args.debug_max_batches):
                    print(f"[DEBUG] Reached debug_max_batches={args.debug_max_batches}; stopping early")
                    break
                x = vox_trials[i : i + bs].to(args.device)
                if args.debug and i == 0:
                    x_cpu = x.detach().float().cpu()
                    print("[DEBUG] forward input vox batch shape:", tuple(x_cpu.shape))
                    print("[DEBUG] forward input vox mean/var:", float(x_cpu.mean()), float(x_cpu.var(unbiased=False)))
                tok = model(x)
                if tok.ndim != 3:
                    raise RuntimeError(f"Expected tokens [B,256,1664], got {tok.shape}")
                if args.debug and i == 0:
                    tok_cpu = tok.detach().float().cpu()
                    print("[DEBUG] forward output tokens var_name=tok (model(x) == clip_voxels) shape:", tuple(tok_cpu.shape))
                    print("[DEBUG] forward output tokens mean/var:", float(tok_cpu.mean()), float(tok_cpu.var(unbiased=False)))
                all_tokens.append(tok.detach().cpu())
        tokens = torch.cat(all_tokens, dim=0)  # [N_trials,256,1664]
        out_ids = image_ids.tolist()
    else:
        # 流式平均：避免把全部 trial_tokens 留在内存里。
        uniq = np.unique(image_ids)
        out_ids = [int(x) for x in uniq.tolist()]
        id2pos = {int(img_id): i for i, img_id in enumerate(out_ids)}

        sum_tokens = torch.zeros((len(out_ids), CLIP_SEQ_DIM, CLIP_EMB_DIM), dtype=torch.float32)
        counts = torch.zeros((len(out_ids),), dtype=torch.int32)

        ones_cpu = None
        with torch.no_grad():
            for i in range(0, vox_trials.shape[0], bs):
                if args.debug and args.debug_max_batches and (i // bs) >= int(args.debug_max_batches):
                    print(f"[DEBUG] Reached debug_max_batches={args.debug_max_batches}; stopping early")
                    break
                x = vox_trials[i : i + bs].to(args.device)
                if args.debug and i == 0:
                    x_cpu = x.detach().float().cpu()
                    print("[DEBUG] forward input vox batch shape:", tuple(x_cpu.shape))
                    print("[DEBUG] forward input vox mean/var:", float(x_cpu.mean()), float(x_cpu.var(unbiased=False)))
                tok = model(x)
                if tok.ndim != 3:
                    raise RuntimeError(f"Expected tokens [B,256,1664], got {tok.shape}")
                tok = tok.detach().cpu().to(dtype=torch.float32)

                if args.debug and i == 0:
                    print("[DEBUG] forward output tokens var_name=tok (model(x) == clip_voxels) shape:", tuple(tok.shape))
                    print("[DEBUG] forward output tokens mean/var:", float(tok.mean()), float(tok.var(unbiased=False)))

                batch_ids = image_ids[i : i + tok.shape[0]]
                idx = torch.tensor([id2pos[int(v)] for v in batch_ids], dtype=torch.long)

                sum_tokens.index_add_(0, idx, tok)

                if ones_cpu is None or ones_cpu.numel() != idx.numel():
                    ones_cpu = torch.ones((idx.numel(),), dtype=torch.int32)
                counts.index_add_(0, idx, ones_cpu)

        denom = counts.to(dtype=torch.float32).view(-1, 1, 1)
        if (denom == 0).any().item():
            raise RuntimeError("Some images have zero repetitions; behav/image_ids may be corrupted")
        tokens = sum_tokens / denom

    pooled = tokens.mean(dim=1)

    if args.debug:
        print("[DEBUG] tokens before mean-pool shape:", tuple(tokens.shape))
        print("[DEBUG] pooled after tokens.mean(dim=1) shape:", tuple(pooled.shape))
        print("[DEBUG] pooled mean/var:", float(pooled.mean()), float(pooled.var(unbiased=False)))

    out_tokens = os.path.join(args.out_dir, f"subj{args.subj:02d}_brain_clip_tokens.pt")
    out_mean = os.path.join(args.out_dir, f"subj{args.subj:02d}_brain_clip_mean.pt")
    torch.save(tokens, out_tokens)
    torch.save(pooled, out_mean)

    ids_out = args.ids_out or os.path.join(args.out_dir, f"subj{args.subj:02d}_ids.json")
    with open(ids_out, "w", encoding="utf-8") as f:
        json.dump([int(x) for x in out_ids], f)

    print(f"[SAVE] tokens: {out_tokens}  shape={tuple(tokens.shape)}")
    print(f"[SAVE] mean  : {out_mean}  shape={tuple(pooled.shape)}")
    print(f"[SAVE] ids   : {ids_out}  len={len(out_ids)} min={min(out_ids)} max={max(out_ids)}")


if __name__ == "__main__":
    main()
