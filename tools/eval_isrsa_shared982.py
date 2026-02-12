#!/usr/bin/env python
# coding: utf-8
"""eval_isrsa_shared982.py

Inter-subject consistency metrics on shared982.

Outputs (per model_tag):
- cache/model_eval_results/shared982_isrsa/<model_tag>/mean_cos_matrix.csv
- cache/model_eval_results/shared982_isrsa/<model_tag>/mean_cos_ci.csv
- cache/model_eval_results/shared982_isrsa/<model_tag>/isrsa_matrix.csv
- cache/model_eval_results/shared982_isrsa/<model_tag>/isrsa_ci.csv
- cache/model_eval_results/shared982_isrsa/<model_tag>/metrics.json

Also writes:
- results/tables/isrsa_summary.csv
- updates results/tables/main_results.csv (adds isrsa columns + group-level rows)

Design notes
- Canonical shared982 order is derived from an ids.json in shared1000 order,
  filtered by src/shared982.npy mask, then used to align all subjects.
- Cosine uses L2-normalized embeddings.
- IS-RSA uses Spearman correlation between upper-triangular vectors of the
  cosine-similarity matrices (subject-wise). Bootstrap resamples images.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


_PROJ_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT_ROOT = _PROJ_ROOT / "cache" / "model_eval_results" / "shared982_isrsa"


@dataclass(frozen=True)
class EmbedAsset:
    embed_path: Path
    ids_path: Path


def _load_ids_json(p: Path) -> np.ndarray:
    arr = np.asarray(json.load(open(p, "r", encoding="utf-8")), dtype=np.int64)
    if arr.ndim != 1:
        raise RuntimeError(f"ids.json must be 1D list, got shape={arr.shape} at {p}")
    return arr


def _load_pt_2d_tensor(p: Path) -> torch.Tensor:
    obj = torch.load(p, map_location="cpu")
    if torch.is_tensor(obj):
        t = obj
    elif isinstance(obj, dict):
        t = None
        for _, v in obj.items():
            if torch.is_tensor(v) and v.ndim == 2:
                t = v
                break
        if t is None:
            raise RuntimeError(f"No 2D tensor found in dict at {p}")
    else:
        raise RuntimeError(f"Unsupported tensor container at {p}: {type(obj)}")

    if t.ndim != 2:
        raise RuntimeError(f"Expected 2D tensor at {p}, got shape={tuple(t.shape)}")
    return t.contiguous()


def _load_shared982_mask() -> np.ndarray:
    mask_path = _PROJ_ROOT / "src" / "shared982.npy"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing shared982 mask: {mask_path}")
    m = np.load(mask_path)
    if m.ndim != 1 or m.dtype != np.bool_:
        raise RuntimeError(f"shared982.npy must be 1D bool mask, got {m.shape} {m.dtype}")
    if int(m.sum()) != 982:
        raise RuntimeError(f"shared982 mask nnz={int(m.sum())} != 982")
    return m


def _pick_canonical_ids_json() -> Path:
    """Pick an ids.json that is known to be in shared1000 order."""
    candidates = []

    # Prefer subj01 official 40sess export if present.
    p = _PROJ_ROOT / "evals" / "brain_tokens" / "official_hf" / "final_subj01_pretrained_40sess_24bs" / "subj01_ids.json"
    if p.is_file():
        return p

    # Fallback: any subj01 ids.json under evals/brain_tokens
    candidates.extend(sorted(_PROJ_ROOT.glob("evals/brain_tokens/**/subj01_ids.json")))
    if candidates:
        return candidates[0]

    raise FileNotFoundError("Could not find a canonical subj01_ids.json under evals/brain_tokens")


def _build_shared982_order_ids() -> np.ndarray:
    """Return canonical shared982 ids in a deterministic evaluation order.

    We define the order by taking a shared1000-order ids.json, filtering by
    src/shared982.npy, and preserving that shared1000 order.
    """
    m982 = _load_shared982_mask()
    ids1000 = _load_ids_json(_pick_canonical_ids_json())
    if ids1000.shape[0] != 1000:
        raise RuntimeError(f"Canonical ids.json expected len=1000, got {ids1000.shape[0]}")
    keep = m982[ids1000]
    ids982 = ids1000[keep]
    if ids982.shape[0] != 982:
        raise RuntimeError(f"After filtering canonical ids1000 by shared982 mask, got {ids982.shape[0]} != 982")
    return ids982


def _align_to_ids_order(embed: torch.Tensor, embed_ids: np.ndarray, target_ids: np.ndarray) -> torch.Tensor:
    if embed.shape[0] != embed_ids.shape[0]:
        raise RuntimeError(f"embed N={embed.shape[0]} != ids N={embed_ids.shape[0]}")

    id2row = {int(gid): int(i) for i, gid in enumerate(embed_ids.tolist())}
    rows = []
    missing = []
    for gid in target_ids.tolist():
        r = id2row.get(int(gid))
        if r is None:
            missing.append(int(gid))
        else:
            rows.append(r)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} target ids in embedding ids.json; first10={missing[:10]}")

    rows_t = torch.as_tensor(rows, dtype=torch.long)
    return embed.index_select(0, rows_t)


def _discover_assets(model_tag: str, subjects: List[int]) -> Dict[int, EmbedAsset]:
    """Discover embedding (brain_clip_mean.pt) + ids.json for each subject."""
    out: Dict[int, EmbedAsset] = {}

    def add(subj: int, embed_path: Path, ids_path: Path) -> None:
        if not embed_path.is_file():
            raise FileNotFoundError(f"Missing embed for {model_tag} subj{subj}: {embed_path}")
        if not ids_path.is_file():
            raise FileNotFoundError(f"Missing ids for {model_tag} subj{subj}: {ids_path}")
        out[subj] = EmbedAsset(embed_path=embed_path, ids_path=ids_path)

    for subj in subjects:
        s = f"{subj:02d}"
        if model_tag == "baseline":
            base = _PROJ_ROOT / "evals" / "brain_tokens" / "official_hf" / f"final_subj{s}_pretrained_40sess_24bs"
            add(subj, base / f"subj{s}_brain_clip_mean.pt", base / f"subj{s}_ids.json")
            continue

        if model_tag == "textalign_llm":
            if subj in (2, 5, 7):
                base = _PROJ_ROOT / "evals" / "brain_tokens" / f"ours_s{subj}_v10"
                add(subj, base / f"subj{s}_brain_clip_mean.pt", base / f"subj{s}_ids.json")
                continue

            # subj01: best available (no v10 in this repo snapshot)
            # Try: ours_s1_v10 -> ours_s1_v2 -> ours_s1_from_official_full_v1
            candidates = [
                _PROJ_ROOT / "evals" / "brain_tokens" / "ours_s1_v10",
                _PROJ_ROOT / "evals" / "brain_tokens" / "ours_s1_v2",
                _PROJ_ROOT / "evals" / "brain_tokens" / "ours_s1_from_official_full_v1",
            ]
            picked = None
            for c in candidates:
                if (c / f"subj{s}_brain_clip_mean.pt").is_file() and (c / f"subj{s}_ids.json").is_file():
                    picked = c
                    break
            if picked is None:
                raise FileNotFoundError(
                    "Missing subj01 embedding for textalign_llm. Tried: "
                    + ", ".join(str(c) for c in candidates)
                )
            add(subj, picked / f"subj{s}_brain_clip_mean.pt", picked / f"subj{s}_ids.json")
            continue

        raise ValueError(f"Unknown model_tag: {model_tag}")

    return out


def _rankdata_dense(x: torch.Tensor) -> torch.Tensor:
    """Rank data for Spearman (ties are extremely unlikely here).

    Returns float32 ranks in [0..n-1].
    """
    order = torch.argsort(x)
    ranks = torch.empty_like(x, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.std(unbiased=False) * b.std(unbiased=False)).clamp_min(1e-12)
    return (a * b).mean() / denom


def _write_matrix_csv(p: Path, mat: np.ndarray, subjects: List[int]) -> None:
    df = pd.DataFrame(mat, index=[str(s) for s in subjects], columns=[str(s) for s in subjects])
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=True)


def _file_fingerprint(p: Path) -> dict:
    st = p.stat()
    return {
        "path": str(p),
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", default="shared982")
    ap.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 5, 7])
    ap.add_argument("--model_tags", nargs="+", default=["baseline", "textalign_llm"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if str(args.eval_set).lower() != "shared982":
        raise ValueError("Only --eval_set shared982 is supported in this script")

    subjects = [int(s) for s in args.subjects]
    if subjects != [1, 2, 5, 7]:
        print(f"[WARN] Non-standard subjects order: {subjects} (expected [1,2,5,7])")

    target_ids = _build_shared982_order_ids()
    N = int(target_ids.shape[0])
    print(f"[OK] eval_set=shared982 N={N}")

    out_root = _DEFAULT_OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    results_rows = []

    # Prepare main_results updater
    main_results_path = _PROJ_ROOT / "results" / "tables" / "main_results.csv"
    if not main_results_path.is_file():
        raise FileNotFoundError(f"Missing main_results.csv at {main_results_path}")
    main_df = pd.read_csv(main_results_path)
    for c in [
        "isrsa_N",
        "mean_offdiag_isrsa",
        "mean_offdiag_isrsa_ci95",
        "mean_offdiag_mean_cos",
        "mean_offdiag_mean_cos_ci95",
    ]:
        if c not in main_df.columns:
            main_df[c] = np.nan

    for model_tag in args.model_tags:
        print(f"\n=== [{model_tag}] discovering embeddings ===")
        assets = _discover_assets(model_tag, subjects)
        for s in subjects:
            a = assets[s]
            print(f"[PATH] subj{s}: embed={a.embed_path} ids={a.ids_path}")

        # Load + align
        Z: Dict[int, torch.Tensor] = {}
        D = None
        for s in subjects:
            a = assets[s]
            ids = _load_ids_json(a.ids_path)
            embed = _load_pt_2d_tensor(a.embed_path).to(dtype=torch.float32)
            embed = _align_to_ids_order(embed, ids, target_ids)
            if embed.shape[0] != N:
                raise RuntimeError(f"Aligned N mismatch for subj{s}: {embed.shape[0]} vs {N}")
            if D is None:
                D = int(embed.shape[1])
            if int(embed.shape[1]) != int(D):
                raise RuntimeError(f"Dim mismatch: subj{s} D={int(embed.shape[1])} vs expected {D}")
            Z[s] = embed

        print(f"[OK] loaded embeddings: N={N} D={D}")

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available; falling back to CPU")
            device = torch.device("cpu")

        # Move + normalize
        for s in subjects:
            Z[s] = F.normalize(Z[s].to(device=device), dim=1)

        # ------------------------------
        # Metric (1): image-wise cross-subject cosine
        # ------------------------------
        pairs: List[Tuple[int, int]] = []
        for i, a in enumerate(subjects):
            for b in subjects[i + 1 :]:
                pairs.append((a, b))

        mean_cos = np.eye(len(subjects), dtype=np.float64)
        mean_cos_pair = {}
        cos_vectors = {}
        for a, b in pairs:
            sim = (Z[a] * Z[b]).sum(dim=1)  # [N]
            cos_vectors[(a, b)] = sim
            m = float(sim.mean().item())
            mean_cos_pair[(a, b)] = m
            ia = subjects.index(a)
            ib = subjects.index(b)
            mean_cos[ia, ib] = m
            mean_cos[ib, ia] = m

        # bootstrap CI for mean cosine
        B = int(args.bootstrap)
        g = torch.Generator(device=device)
        g.manual_seed(int(args.seed))
        cos_boot = {p: torch.empty((B,), device=device, dtype=torch.float32) for p in pairs}
        t0 = time.time()
        for bi in range(B):
            idx = torch.randint(0, N, (N,), generator=g, device=device)
            for a, b in pairs:
                sim = cos_vectors[(a, b)]
                cos_boot[(a, b)][bi] = sim.index_select(0, idx).mean()
            if (bi + 1) % 100 == 0:
                dt = time.time() - t0
                print(f"[bootstrap mean-cos] {bi+1}/{B} elapsed={dt:.1f}s")

        mean_cos_ci_rows = []
        for a, b in pairs:
            vals = cos_boot[(a, b)].detach().cpu().numpy()
            lo, hi = np.quantile(vals, [0.025, 0.975]).tolist()
            mean_cos_ci_rows.append(
                {
                    "model_tag": model_tag,
                    "subj_a": a,
                    "subj_b": b,
                    "mean": float(mean_cos_pair[(a, b)]),
                    "ci95_lo": float(lo),
                    "ci95_hi": float(hi),
                }
            )

        # ------------------------------
        # Metric (2): IS-RSA (Spearman between similarity matrices)
        # ------------------------------
        # Precompute subject similarity matrices once
        S = {}
        for s in subjects:
            S[s] = (Z[s] @ Z[s].t()).to(torch.float32)

        tri = torch.triu_indices(N, N, offset=1, device=device)
        I = tri[0]
        J = tri[1]

        # Full IS-RSA
        ranks_full = {}
        for s in subjects:
            v = S[s][I, J]
            ranks_full[s] = _rankdata_dense(v)

        isrsa = np.eye(len(subjects), dtype=np.float64)
        isrsa_pair = {}
        for a, b in pairs:
            rho = float(_pearson_corr(ranks_full[a], ranks_full[b]).item())
            isrsa_pair[(a, b)] = rho
            ia = subjects.index(a)
            ib = subjects.index(b)
            isrsa[ia, ib] = rho
            isrsa[ib, ia] = rho

        # Bootstrap CI for IS-RSA (resample images -> induced resample of edges)
        isrsa_boot = {p: torch.empty((B,), device=device, dtype=torch.float32) for p in pairs}
        isrsa_offdiag_boot = torch.empty((B,), device=device, dtype=torch.float32)

        t1 = time.time()
        for bi in range(B):
            idx = torch.randint(0, N, (N,), generator=g, device=device)
            A = idx.index_select(0, I)
            Bidx = idx.index_select(0, J)

            ranks = {}
            for s in subjects:
                v = S[s][A, Bidx]
                ranks[s] = _rankdata_dense(v)

            off = []
            for a, b in pairs:
                rho = _pearson_corr(ranks[a], ranks[b])
                isrsa_boot[(a, b)][bi] = rho
                off.append(rho)
            isrsa_offdiag_boot[bi] = torch.stack(off).mean()

            if (bi + 1) % 50 == 0:
                dt = time.time() - t1
                print(f"[bootstrap IS-RSA] {bi+1}/{B} elapsed={dt:.1f}s")

        isrsa_ci_rows = []
        for a, b in pairs:
            vals = isrsa_boot[(a, b)].detach().cpu().numpy()
            lo, hi = np.quantile(vals, [0.025, 0.975]).tolist()
            isrsa_ci_rows.append(
                {
                    "model_tag": model_tag,
                    "subj_a": a,
                    "subj_b": b,
                    "rho": float(isrsa_pair[(a, b)]),
                    "ci95_lo": float(lo),
                    "ci95_hi": float(hi),
                }
            )

        off_vals = isrsa_offdiag_boot.detach().cpu().numpy()
        off_lo, off_hi = np.quantile(off_vals, [0.025, 0.975]).tolist()
        mean_offdiag_isrsa = float(isrsa[np.triu_indices(len(subjects), k=1)].mean())

        mean_offdiag_mean_cos = float(mean_cos[np.triu_indices(len(subjects), k=1)].mean())
        # CI for mean_offdiag_mean_cos from pairwise bootstraps (average over pairs each bootstrap)
        cos_offdiag_boot = torch.stack([cos_boot[p] for p in pairs], dim=1).mean(dim=1).detach().cpu().numpy()
        cos_off_lo, cos_off_hi = np.quantile(cos_offdiag_boot, [0.025, 0.975]).tolist()

        # ------------------------------
        # Write outputs
        # ------------------------------
        model_dir = out_root / model_tag
        model_dir.mkdir(parents=True, exist_ok=True)

        _write_matrix_csv(model_dir / "mean_cos_matrix.csv", mean_cos, subjects)
        pd.DataFrame(mean_cos_ci_rows).to_csv(model_dir / "mean_cos_ci.csv", index=False)

        _write_matrix_csv(model_dir / "isrsa_matrix.csv", isrsa, subjects)
        pd.DataFrame(isrsa_ci_rows).to_csv(model_dir / "isrsa_ci.csv", index=False)

        metrics = {
            "eval_set": "shared982",
            "N": N,
            "D": int(D),
            "subjects": subjects,
            "model_tag": model_tag,
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "device": str(device),
            "mean_offdiag_isrsa": mean_offdiag_isrsa,
            "mean_offdiag_isrsa_ci95": [float(off_lo), float(off_hi)],
            "mean_offdiag_mean_cos": mean_offdiag_mean_cos,
            "mean_offdiag_mean_cos_ci95": [float(cos_off_lo), float(cos_off_hi)],
            "assets": {
                str(s): {
                    "embed": _file_fingerprint(assets[s].embed_path),
                    "ids": _file_fingerprint(assets[s].ids_path),
                }
                for s in subjects
            },
            "canonical_ids_json": str(_pick_canonical_ids_json()),
            "notes": {
                "canonical_order": "Filter a shared1000-order ids.json by src/shared982.npy, preserve that order.",
                "cosine": "L2-normalize then dot product.",
                "isrsa": "Spearman implemented as Pearson correlation over dense ranks (ties ignored).",
                "bootstrap": "Resample images with replacement (length N); induce edge resample via (idx[i], idx[j]) pairs.",
            },
        }
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        results_rows.append(
            {
                "model_tag": model_tag,
                "N": N,
                "mean_offdiag_isrsa": mean_offdiag_isrsa,
                "mean_offdiag_isrsa_ci95_lo": float(off_lo),
                "mean_offdiag_isrsa_ci95_hi": float(off_hi),
                "mean_offdiag_mean_cos": mean_offdiag_mean_cos,
                "mean_offdiag_mean_cos_ci95_lo": float(cos_off_lo),
                "mean_offdiag_mean_cos_ci95_hi": float(cos_off_hi),
            }
        )

        # Update main_results: add a group-level row
        new_row = {c: np.nan for c in main_df.columns}
        new_row.update(
            {
                "group": "shared982_isrsa",
                "tag": model_tag,
                "subj": "all",
                "isrsa_N": N,
                "mean_offdiag_isrsa": mean_offdiag_isrsa,
                "mean_offdiag_isrsa_ci95": f"[{off_lo:.4f}, {off_hi:.4f}]",
                "mean_offdiag_mean_cos": mean_offdiag_mean_cos,
                "mean_offdiag_mean_cos_ci95": f"[{cos_off_lo:.4f}, {cos_off_hi:.4f}]",
            }
        )
        main_df = pd.concat([main_df, pd.DataFrame([new_row])], ignore_index=True)

    # Write summary tables
    results_dir = _PROJ_ROOT / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "isrsa_summary.csv"
    pd.DataFrame(results_rows).to_csv(summary_path, index=False)
    print(f"\n[OK] wrote {summary_path}")

    main_df.to_csv(main_results_path, index=False)
    print(f"[OK] updated {main_results_path}")


if __name__ == "__main__":
    main()
