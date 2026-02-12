#!/usr/bin/env python
# coding: utf-8
"""eval_ccd_embed.py

CCD (Counterfactual Caption Discrimination) for embedding-based evaluation.

Goal
- Given brain-predicted visual features and captions, evaluate whether the positive
  caption for an image is ranked above sampled (or provided) negative captions.

Key design choice (documented in metrics.json):
- Brain exports in this repo are typically 1664-d (ViT-bigG-14 visual token width;
  e.g. mean over 256 visual tokens).
- OpenCLIP joint image-text embedding space for ViT-bigG-14 is 1280-d.
- We map 1664-d visual features -> 1280-d joint space via the *fixed* OpenCLIP
  visual projection matrix (model.visual.proj) from the same pretrained weights.
  This avoids learning any new mapping on evaluation data.

Inputs (env vars):
- BRAIN_PATH: .pt tensor [N,D] OR tokens [N,T,D]
- IDS_PATH: ids.json (len N, global image id). If missing, assume arange.
- CAPTIONS_PATH: evals/all_captions.pt (default)
- HARD_NEG_JSONL: optional jsonl with per-image hard neg captions (see tools/gen_hard_neg_captions_from_json.py)
- HARD_NEG_PT: optional .pt dict with neg captions (see tools/encode_hard_neg_captions_clip.py)

- EVAL_SUBSET: (empty)|shared982
- K_NEG: number of sampled negatives per item when hard neg not provided (default 31)
- SEED: RNG seed for negative sampling (default 42)

- CLIP_MODEL: OpenCLIP arch (default ViT-bigG-14)
- CLIP_PRETRAINED: OpenCLIP pretrained tag (default laion2b_s39b_b160k)

- RESULT_DIR: output directory
- EXP_NAME: label
- BOOTSTRAP: number of bootstrap resamples for CI (default 500; 0 disables)

Outputs:
- metrics.json
- metrics.txt

"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

device = "cuda" if torch.cuda.is_available() else "cpu"

BRAIN_PATH = os.environ.get("BRAIN_PATH")
IDS_PATH = os.environ.get("IDS_PATH")
CAPTIONS_PATH = os.environ.get("CAPTIONS_PATH", os.path.join(_PROJ_ROOT, "evals", "all_captions.pt"))
HARD_NEG_JSONL = os.environ.get("HARD_NEG_JSONL")
HARD_NEG_PT = os.environ.get("HARD_NEG_PT")

# Hard-negative control
HARD_NEG_K = int(os.environ.get("HARD_NEG_K", "1"))
# If hardneg source is provided but missing ids for the evaluated subset, default to error
# (prevents silent fallback to sampled negatives).
HARD_NEG_REQUIRE_FULL = os.environ.get("HARD_NEG_REQUIRE_FULL", "1").strip() not in ("0", "false", "False")

EVAL_SUBSET = os.environ.get("EVAL_SUBSET", "").strip().lower()
K_NEG = int(os.environ.get("K_NEG", "31"))
SEED = int(os.environ.get("SEED", "42"))

CLIP_MODEL = os.environ.get("CLIP_MODEL", "ViT-bigG-14")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "laion2b_s39b_b160k")

RESULT_DIR = os.environ.get("RESULT_DIR")
EXP_NAME = os.environ.get("EXP_NAME", "ccd")
BOOTSTRAP = int(os.environ.get("BOOTSTRAP", "500"))
ASSETS_DIR = os.environ.get("ASSETS_DIR")
EVAL_KEEP_MASK_NPY = os.environ.get("EVAL_KEEP_MASK_NPY")
OPENCLIP_VISUAL_PROJ_PATH = os.environ.get("OPENCLIP_VISUAL_PROJ_PATH")


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
        for _, v in obj.items():
            if torch.is_tensor(v) and v.dim() in (2, 3):
                return v
    raise RuntimeError(f"Unsupported tensor container at {path}: {type(obj)}")


def _load_ids(ids_path: Optional[str], n: int) -> np.ndarray:
    if ids_path and os.path.isfile(ids_path):
        with open(ids_path, "r") as f:
            ids = np.asarray(json.load(f), dtype=np.int64)
        if ids.shape[0] != n:
            raise RuntimeError(f"ids.json len={ids.shape[0]} != N={n}")
        return ids
    return np.arange(n, dtype=np.int64)


def _to_stim_vectors(x: torch.Tensor) -> Tuple[torch.Tensor, str]:
    if x.dim() == 2:
        return x, "pooled"
    if x.dim() == 3:
        return x.mean(dim=1), "tokens_mean"
    raise RuntimeError(f"Unsupported feature rank: {x.dim()} shape={tuple(x.shape)}")


def _load_captions(path: str) -> List[str]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, list):
        if not obj:
            raise RuntimeError("captions is empty")
        return [str(x) for x in obj]
    raise RuntimeError(f"Unsupported captions container at {path}: {type(obj)}")


def _load_hard_neg_captions(
    ids: np.ndarray,
    hard_neg_jsonl: Optional[str],
    hard_neg_pt: Optional[str],
    k_hard: int,
    require_full: bool,
) -> Optional[List[List[str]]]:
    if hard_neg_pt and os.path.isfile(hard_neg_pt):
        obj = torch.load(hard_neg_pt, map_location="cpu")
        if not isinstance(obj, dict):
            raise RuntimeError(f"HARD_NEG_PT must be dict, got {type(obj)}")
        if "image_ids" in obj and "neg_captions" in obj:
            img_ids = obj["image_ids"]
            if torch.is_tensor(img_ids):
                img_ids = img_ids.detach().cpu().numpy().astype(np.int64)
            neg_caps = obj["neg_captions"]
            id2neg = {int(i): str(c) for i, c in zip(img_ids, neg_caps)}
            out: List[List[str]] = []
            missing = 0
            for i in ids:
                if int(i) not in id2neg:
                    missing += 1
                    continue
                out.append([id2neg[int(i)]])
            if missing > 0:
                if require_full:
                    raise RuntimeError(f"HARD_NEG_PT missing {missing}/{len(ids)} ids in evaluated set")
                return None
            return out
        # fallback: try best-effort keys
        for key in ["neg_captions", "hard_neg_captions"]:
            if key in obj and isinstance(obj[key], list) and len(obj[key]) == len(ids):
                return [[str(x)] for x in obj[key]]
        return None

    if hard_neg_jsonl and os.path.isfile(hard_neg_jsonl):
        # Collect possibly-multiple negatives per image_id.
        # If sim_text is present, sort candidates by sim_text desc for determinism.
        id2cands: Dict[int, List[Tuple[float, str]]] = {}
        with open(hard_neg_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "image_id" not in row or "neg_caption" not in row:
                    continue
                sim = float(row.get("sim_text", float("nan")))
                if not np.isfinite(sim):
                    sim = 0.0
                img_id = int(row["image_id"])
                id2cands.setdefault(img_id, []).append((sim, str(row["neg_caption"])))

        if not id2cands:
            return None
        out: List[List[str]] = []
        missing = 0
        k = max(1, int(k_hard))
        for i in ids:
            cands = id2cands.get(int(i))
            if not cands:
                missing += 1
                continue
            cands_sorted = sorted(cands, key=lambda x: x[0], reverse=True)
            out.append([c for _, c in cands_sorted[:k]])
        if missing > 0:
            if require_full:
                raise RuntimeError(f"HARD_NEG_JSONL missing {missing}/{len(ids)} ids in evaluated set")
            return None
        return out

    return None


@torch.no_grad()
def _encode_captions_openclip(
    captions: List[str],
    arch: str,
    pretrained: str,
    device: str,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    tok = open_clip.get_tokenizer(arch)

    feats: List[torch.Tensor] = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i : i + batch_size]
        tokens = tok(batch).to(device)
        z = model.encode_text(tokens)
        z = z.float()
        feats.append(z.detach().cpu())
    feats_t = torch.cat(feats, dim=0)
    return feats_t, {
        "clip_text_dim": int(feats_t.shape[1]),
    }


@torch.no_grad()
def _get_openclip_visual_proj(
    arch: str,
    pretrained: str,
) -> torch.Tensor:
    # Prefer an explicit cached proj path when provided (offline-friendly).
    if OPENCLIP_VISUAL_PROJ_PATH and os.path.isfile(OPENCLIP_VISUAL_PROJ_PATH):
        proj = torch.load(OPENCLIP_VISUAL_PROJ_PATH, map_location="cpu")
        if not torch.is_tensor(proj):
            raise RuntimeError(f"OPENCLIP_VISUAL_PROJ_PATH must be a Tensor .pt, got {type(proj)}")
        if proj.ndim != 2:
            raise RuntimeError(f"Invalid proj tensor rank: {proj.ndim} shape={tuple(proj.shape)}")
        return proj.detach().cpu().float()

    # Fallback to open_clip if available; also cache the proj into assets_dir when possible.
    try:
        import open_clip
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: open_clip. Either install `open-clip-torch` or provide a cached proj via "
            "OPENCLIP_VISUAL_PROJ_PATH (a torch Tensor saved with shape [1664,1280]). "
            f"Import error: {type(e).__name__}: {e}"
        )

    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    proj = getattr(model.visual, "proj", None)
    if proj is None:
        raise RuntimeError("open_clip model.visual.proj is missing; cannot map 1664->joint")
    return proj.detach().cpu().float()


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _sample_neg_indices(n: int, k: int, seed: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("K_NEG must be > 0")
    if n <= 1:
        raise ValueError("Need at least 2 samples for negatives")
    rng = np.random.default_rng(seed)
    neg = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        # sample from [0,n) excluding i
        # (fast approach: sample from n-1 then shift)
        r = rng.integers(0, n - 1, size=k, dtype=np.int64)
        r = r + (r >= i)
        neg[i] = r
    return neg


@dataclass
class CCDMetrics:
    acc1: float
    acc5: float
    mrr: float
    mean_rank: float
    median_rank: float
    margin_mean: float
    margin_median: float
    margin_std: float


def _compute_metrics_from_scores(pos: torch.Tensor, neg: torch.Tensor) -> Tuple[CCDMetrics, torch.Tensor, torch.Tensor]:
    # pos: [N]
    # neg: [N,K]
    # ranks: 1 + count(neg > pos) + 0.5*count(neg == pos)
    gt = pos[:, None]
    better = (neg > gt).sum(dim=1).float()
    ties = (neg == gt).sum(dim=1).float()
    rank = 1.0 + better + 0.5 * ties

    acc1 = (rank <= 1.0).float().mean().item()
    acc5 = (rank <= 5.0).float().mean().item()
    mrr = (1.0 / rank).mean().item()

    margin = pos - neg.max(dim=1).values

    metrics = CCDMetrics(
        acc1=acc1,
        acc5=acc5,
        mrr=mrr,
        mean_rank=rank.mean().item(),
        median_rank=rank.median().item(),
        margin_mean=margin.mean().item(),
        margin_median=margin.median().item(),
        margin_std=margin.std(unbiased=False).item(),
    )
    return metrics, rank, margin


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(values[idx].mean())
    lo, hi = np.quantile(np.asarray(boots), [0.025, 0.975]).tolist()
    return float(lo), float(hi)


def main() -> None:
    t0 = time.time()
    if not BRAIN_PATH or not os.path.isfile(BRAIN_PATH):
        raise RuntimeError("BRAIN_PATH missing or not a file")
    if not RESULT_DIR:
        raise RuntimeError("RESULT_DIR is required")

    out_dir = Path(RESULT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    brain_raw = _load_tensor(BRAIN_PATH)
    brain_vec, brain_repr = _to_stim_vectors(brain_raw)
    n_total = int(brain_vec.shape[0])
    ids = _load_ids(IDS_PATH, n_total)

    captions_all = _load_captions(CAPTIONS_PATH)
    if len(captions_all) != n_total:
        raise RuntimeError(f"captions len={len(captions_all)} != N={n_total}. CAPTIONS_PATH must align with ids order.")

    # subset filter
    keep = np.ones((n_total,), dtype=np.bool_)
    if EVAL_SUBSET in ("shared982", "test982", "wds_test_982", "wds-test-982"):
        subset_ids = set(_build_or_load_shared982_ids().tolist())
        keep = np.asarray([int(i) in subset_ids for i in ids], dtype=np.bool_)
        if int(keep.sum()) == 0:
            raise RuntimeError("shared982 过滤后样本数为 0")

    subset_idx = np.where(keep)[0]
    if EVAL_KEEP_MASK_NPY:
        if not os.path.isfile(EVAL_KEEP_MASK_NPY):
            raise RuntimeError(f"EVAL_KEEP_MASK_NPY not found: {EVAL_KEEP_MASK_NPY}")
        m = np.load(EVAL_KEEP_MASK_NPY)
        if m.dtype != np.bool_ or m.ndim != 1:
            raise RuntimeError(f"EVAL_KEEP_MASK_NPY must be 1D bool array, got shape={m.shape} dtype={m.dtype}")
        if int(m.shape[0]) != int(subset_idx.shape[0]):
            raise RuntimeError(
                f"EVAL_KEEP_MASK_NPY length={int(m.shape[0])} does not match subset size={int(subset_idx.shape[0])}"
            )
        subset_idx = subset_idx[m]

    brain_vec = brain_vec[subset_idx]
    ids = ids[subset_idx]
    captions = [captions_all[i] for i in subset_idx.tolist()]

    # Load hard negatives *after* subset/mask filtering so missing ids outside the evaluated set
    # do not force a fallback to sampled negatives.
    hard_negs = _load_hard_neg_captions(
        ids,
        HARD_NEG_JSONL,
        HARD_NEG_PT,
        k_hard=HARD_NEG_K,
        require_full=HARD_NEG_REQUIRE_FULL,
    )
    neg_mode = "hardneg" if hard_negs is not None else "sampled"

    n_eval = int(brain_vec.shape[0])

    # Encode captions (and hard neg captions if provided)
    assets_dir = Path(ASSETS_DIR).resolve() if ASSETS_DIR else (out_dir / "assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    subset_tag = EVAL_SUBSET or "full"
    cap_cache = assets_dir / f"captions_openclip_{CLIP_MODEL}_{CLIP_PRETRAINED}_{subset_tag}_n{n_eval}.pt"

    if cap_cache.is_file():
        obj = torch.load(cap_cache, map_location="cpu")
        text_feats = obj["text_feats"]
        text_meta = obj.get("meta", {})
    else:
        text_feats, text_meta = _encode_captions_openclip(captions, CLIP_MODEL, CLIP_PRETRAINED, device=device)
        torch.save({"text_feats": text_feats, "meta": text_meta}, cap_cache)

    if hard_negs is not None:
        # hard_negs: List[List[str]] of shape [N][K]
        k_hard_used = int(max(len(x) for x in hard_negs))
        if any(len(x) != k_hard_used for x in hard_negs):
            raise RuntimeError("Inconsistent hardneg K per image after filtering; please rebuild hardneg jsonl")

        neg_cache = assets_dir / f"hardnegs_openclip_{CLIP_MODEL}_{CLIP_PRETRAINED}_{subset_tag}_n{n_eval}_k{k_hard_used}.pt"
        if neg_cache.is_file():
            neg_text_feats = torch.load(neg_cache, map_location="cpu")["text_feats"]
        else:
            flat = [c for row in hard_negs for c in row]
            neg_text_feats, _ = _encode_captions_openclip(flat, CLIP_MODEL, CLIP_PRETRAINED, device=device)
            torch.save({"text_feats": neg_text_feats}, neg_cache)
    else:
        neg_text_feats = None

    # Map brain -> joint space if needed
    brain_vec = brain_vec.float()
    d_in = int(brain_vec.shape[1])

    proj = None
    proj_tag = "none"
    if d_in != int(text_feats.shape[1]):
        proj = _get_openclip_visual_proj(CLIP_MODEL, CLIP_PRETRAINED)  # [1664,1280]
        if d_in != int(proj.shape[0]) or int(text_feats.shape[1]) != int(proj.shape[1]):
            raise RuntimeError(
                f"Dimension mismatch: brain_dim={d_in}, proj={tuple(proj.shape)}, text_dim={int(text_feats.shape[1])}"
            )
        brain_vec = brain_vec @ proj
        proj_tag = "openclip_visual_proj"

    # Move to device & normalize
    brain = _l2norm(brain_vec.to(device))
    text = _l2norm(text_feats.to(device))

    if neg_text_feats is not None:
        neg_text = _l2norm(neg_text_feats.to(device))

    # Score computation
    if neg_mode == "hardneg":
        pos_score = (brain * text).sum(dim=1)
        # neg_text_feats is flattened [N*K, D]
        k_used = int(neg_text.shape[0] // n_eval)
        if k_used <= 0 or int(n_eval * k_used) != int(neg_text.shape[0]):
            raise RuntimeError(f"Invalid hardneg text feats shape: {tuple(neg_text.shape)} n_eval={n_eval}")
        neg_text_3d = neg_text.view(n_eval, k_used, -1)
        neg_score_mat = (brain[:, None, :] * neg_text_3d).sum(dim=2)  # [N,K]
        metrics, rank, margin = _compute_metrics_from_scores(pos_score, neg_score_mat)

        hardest = neg_score_mat.max(dim=1).values
        twoafc_hardest = ((pos_score > hardest).float() + 0.5 * (pos_score == hardest).float()).mean().item()
    else:
        neg_idx = _sample_neg_indices(n_eval, K_NEG, SEED)
        neg_idx_t = torch.from_numpy(neg_idx).to(device)

        batch = 256
        pos_scores: List[torch.Tensor] = []
        neg_scores: List[torch.Tensor] = []

        for i in range(0, n_eval, batch):
            j = min(n_eval, i + batch)
            b = brain[i:j]  # [B,D]
            # positive is aligned by row index
            t_pos = text[i:j]
            s_pos = (b * t_pos).sum(dim=1)

            idx = neg_idx_t[i:j]  # [B,K]
            t_neg = text[idx]  # [B,K,D]
            s_neg = (b[:, None, :] * t_neg).sum(dim=2)  # [B,K]

            pos_scores.append(s_pos.detach().cpu())
            neg_scores.append(s_neg.detach().cpu())

        pos_score = torch.cat(pos_scores, dim=0)
        neg_score_mat = torch.cat(neg_scores, dim=0)
        metrics, rank, margin = _compute_metrics_from_scores(pos_score, neg_score_mat)
        k_used = K_NEG

        hardest = neg_score_mat.max(dim=1).values
        twoafc_hardest = ((pos_score > hardest).float() + 0.5 * (pos_score == hardest).float()).mean().item()

    # Bootstrap CIs (optional)
    ci = {}
    if BOOTSTRAP > 0:
        acc1_vals = (rank.cpu().numpy() <= 1.0).astype(np.float32)
        ci["acc1_ci95"] = _bootstrap_ci(acc1_vals, BOOTSTRAP, SEED)
        ci["margin_mean_ci95"] = _bootstrap_ci(margin.cpu().numpy().astype(np.float32), BOOTSTRAP, SEED)
        # 2AFC-hardest CI: use strict win indicator (margin>0). Ties are rare under float scores.
        win_vals = (margin.cpu().numpy() > 0).astype(np.float32)
        ci["twoafc_hardest_acc_ci95"] = _bootstrap_ci(win_vals, BOOTSTRAP, SEED)

    dt = time.time() - t0

    out = {
        "exp": EXP_NAME,
        "brain_path": str(BRAIN_PATH),
        "ids_path": str(IDS_PATH) if IDS_PATH else None,
        "captions_path": str(CAPTIONS_PATH),
        "hard_neg_jsonl": str(HARD_NEG_JSONL) if HARD_NEG_JSONL else None,
        "hard_neg_pt": str(HARD_NEG_PT) if HARD_NEG_PT else None,
        "subset": EVAL_SUBSET,
        "n_total": n_total,
        "n_eval": n_eval,
        "brain_repr": brain_repr,
        "brain_in_dim": d_in,
        "text_dim": int(text_feats.shape[1]),
        "proj": proj_tag,
        "neg_mode": neg_mode,
        "k_neg": int(k_used),
        "seed": int(SEED),
        "sim": "cosine(l2norm+dot)",
        "clip_model": CLIP_MODEL,
        "clip_pretrained": CLIP_PRETRAINED,
        "device": device,
        "bootstrap": int(BOOTSTRAP),
        "runtime_sec": float(dt),
        "metrics": {
            "ccd_acc1": metrics.acc1,
            "ccd_acc5": metrics.acc5,
            "mrr": metrics.mrr,
            "mean_rank": metrics.mean_rank,
            "median_rank": metrics.median_rank,
            "margin_mean": metrics.margin_mean,
            "margin_median": metrics.margin_median,
            "margin_std": metrics.margin_std,
            "twoafc_hardest": float(twoafc_hardest),
        },
        "ci": ci,
        "text_meta": text_meta,
    }

    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    lines = []
    lines.append(f"CCD ({EXP_NAME})")
    lines.append(f"subset={EVAL_SUBSET or 'full'} n_eval={n_eval} neg_mode={neg_mode} k={k_used}")
    lines.append(f"brain_repr={brain_repr} brain_dim={d_in} -> text_dim={int(text_feats.shape[1])} proj={proj_tag}")
    lines.append(f"acc@1={metrics.acc1:.4f} acc@5={metrics.acc5:.4f} mrr={metrics.mrr:.4f}")
    lines.append(f"rank mean={metrics.mean_rank:.2f} median={metrics.median_rank:.2f}")
    lines.append(f"margin mean={metrics.margin_mean:.4f} median={metrics.margin_median:.4f} std={metrics.margin_std:.4f}")
    lines.append(f"2AFC-hardest={twoafc_hardest:.4f}")
    if "acc1_ci95" in ci:
        lo, hi = ci["acc1_ci95"]
        lines.append(f"acc@1 95% CI (bootstrap) [{lo:.4f}, {hi:.4f}]")
    if "margin_mean_ci95" in ci:
        lo, hi = ci["margin_mean_ci95"]
        lines.append(f"margin_mean 95% CI (bootstrap) [{lo:.4f}, {hi:.4f}]")
    if "twoafc_hardest_acc_ci95" in ci:
        lo, hi = ci["twoafc_hardest_acc_ci95"]
        lines.append(f"2AFC-hardest(win) 95% CI (bootstrap) [{lo:.4f}, {hi:.4f}]")
    lines.append(f"runtime_sec={dt:.2f} device={device}")

    (out_dir / "metrics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
