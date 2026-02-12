#!/usr/bin/env python
# coding: utf-8
"""make_ccd_by_type.py

Compute CCD-Hard by hardneg type (object/attribute/relation).

This script:
1) Builds per-type keep masks (len=982, in shared982 evaluation order)
2) Runs batch CCD for each type by reusing tools/rerun_all_ccd_shared982.py
3) Consolidates outputs into:
   - cache/model_eval_results/shared982_ccd/ccd_by_type.csv
   - cache/model_eval_results/shared982_ccd/ccd_by_type.md
4) Generates a simple figure into:
   - cache/model_eval_results/shared982_ccd/figures/Fig06_ccd_by_type.png

Notes
- The evaluated set for each type is an intersection of hardneg_valid_mask and that type's coverage.
- CI columns are sourced from per-model metrics.json (bootstrap).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJ = Path(__file__).resolve().parents[1]
CCD_DIR = PROJ / "cache" / "model_eval_results" / "shared982_ccd"
HARDNEG_DIR = PROJ / "cache" / "hardneg"


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _find_default_ids_json() -> Path:
    # Prefer a stable reference ids.json used by hardneg generation.
    cands = sorted((PROJ / "evals" / "brain_tokens").glob("**/*_ids.json"))
    if not cands:
        raise RuntimeError("No *_ids.json found under evals/brain_tokens")
    # Heuristic: prefer official_hf/subj01
    for p in cands:
        if "official_hf" in str(p) and "subj01" in p.name:
            return p
    return cands[0]


def _load_shared982_order_ids(ids_json: Path) -> np.ndarray:
    ids = np.asarray(json.loads(ids_json.read_text(encoding="utf-8")), dtype=np.int64)
    if ids.ndim != 1:
        raise RuntimeError(f"ids_json must be 1D list, got shape={ids.shape}")

    shared982_mask_path = PROJ / "src" / "shared982.npy"
    if not shared982_mask_path.is_file():
        raise RuntimeError(f"Missing {shared982_mask_path}")

    m = np.load(shared982_mask_path)
    if m.dtype != np.bool_ or m.ndim != 1:
        raise RuntimeError(f"shared982.npy must be 1D bool mask, got {m.dtype} shape={m.shape}")

    keep = np.asarray([bool(m[int(i)]) for i in ids], dtype=np.bool_)
    order = ids[np.where(keep)[0]]
    if int(order.shape[0]) != 982:
        raise RuntimeError(f"shared982 order size={int(order.shape[0])} != 982 (check ids_json alignment)")
    return order


@dataclass
class TypeAssets:
    type_name: str
    keep_mask_path: Path
    neg_jsonl_path: Path
    out_dir: Path


def _build_type_assets(
    shared982_order: np.ndarray,
    valid_mask: np.ndarray,
    hardneg_for_ccd_jsonl: Path,
) -> List[TypeAssets]:
    rows = _read_jsonl(hardneg_for_ccd_jsonl)

    # one line per image_id (if duplicates exist, keep max sim_text)
    best: Dict[int, Tuple[float, dict]] = {}
    for r in rows:
        img_id = int(r.get("image_id"))
        sim = float(r.get("sim_text", 0.0))
        if img_id not in best or sim > best[img_id][0]:
            best[img_id] = (sim, r)

    type2ids: Dict[str, List[int]] = {"object": [], "attribute": [], "relation": []}
    for img_id in shared982_order.tolist():
        if img_id not in best:
            continue
        t = str(best[img_id][1].get("type", "")).strip().lower()
        if t in type2ids:
            type2ids[t].append(img_id)

    out_assets: List[TypeAssets] = []
    masks_dir = HARDNEG_DIR / "by_type_masks"
    jsonl_dir = HARDNEG_DIR / "by_type_jsonl"
    masks_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    for t in ["object", "attribute", "relation"]:
        keep = np.zeros((982,), dtype=np.bool_)
        idset = set(type2ids[t])
        for i, img_id in enumerate(shared982_order.tolist()):
            if valid_mask[i] and int(img_id) in idset:
                keep[i] = True

        # write type jsonl (one per image_id)
        out_jsonl = jsonl_dir / f"shared982_hardneg_{t}.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for img_id in shared982_order[keep].tolist():
                f.write(json.dumps(best[int(img_id)][1], ensure_ascii=False) + "\n")

        out_mask = masks_dir / f"hardneg_valid_mask_{t}.npy"
        np.save(out_mask, keep)

        out_dir = CCD_DIR / "by_type" / t
        out_assets.append(TypeAssets(t, out_mask, out_jsonl, out_dir))

    return out_assets


def _run_batch_for_type(a: TypeAssets, bootstrap: int, seed: int) -> None:
    cmd = [
        "python",
        str(PROJ / "tools" / "rerun_all_ccd_shared982.py"),
        "--neg_jsonl",
        str(a.neg_jsonl_path),
        "--use_valid_mask",
        str(a.keep_mask_path),
        "--bootstrap",
        str(int(bootstrap)),
        "--seed",
        str(int(seed)),
        "--out_dir",
        str(a.out_dir),
    ]
    subprocess.check_call(cmd, cwd=str(PROJ))


def _consolidate_by_type(type_assets: List[TypeAssets]) -> pd.DataFrame:
    parts = []
    for a in type_assets:
        csv_path = a.out_dir / "ccd_summary.csv"
        if not csv_path.is_file():
            raise RuntimeError(f"Missing by-type summary: {csv_path}")
        df = pd.read_csv(csv_path)
        df.insert(0, "type", a.type_name)
        parts.append(df)

    out = pd.concat(parts, axis=0, ignore_index=True)
    # keep only columns we care about
    return out


def _write_md(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "type",
        "group",
        "subj",
        "eval_repr",
        "tag",
        "n_eval",
        "neg_mode",
        "k_neg",
        "bootstrap",
        "ccd_acc1",
        "ccd_acc1_ci95_lo",
        "ccd_acc1_ci95_hi",
        "twoafc_hardest",
        "margin_mean",
        "margin_mean_ci95_lo",
        "margin_mean_ci95_hi",
        "metrics.json",
    ]
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = ""
    d = d[cols]

    lines = []
    lines.append("# CCD-Hard by type (shared982)\n\n")
    lines.append(f"CSV: {path.with_suffix('.csv')}\n\n")
    lines.append("| type | group | subj | eval_repr | tag | N | neg_mode | K | bootstrap | CCD@1 | CI95_lo | CI95_hi | 2AFC-hardest | margin_mean | margin_lo | margin_hi | metrics.json |\n")
    lines.append("|---|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for _, r in d.iterrows():
        lines.append(
            "| {type} | {group} | {subj} | {eval_repr} | {tag} | {n_eval} | {neg_mode} | {k_neg} | {bootstrap} | {ccd_acc1:.4f} | {lo} | {hi} | {twoafc:.4f} | {margin:.4f} | {mlo} | {mhi} | {mp} |\n".format(
                type=str(r["type"]),
                group=str(r["group"]),
                subj=str(r["subj"]),
                eval_repr=str(r["eval_repr"]),
                tag=str(r["tag"]),
                n_eval=int(r["n_eval"]),
                neg_mode=str(r["neg_mode"]),
                k_neg=int(r["k_neg"]),
                bootstrap=int(r.get("bootstrap", 0) or 0),
                ccd_acc1=float(r["ccd_acc1"]),
                lo="" if r["ccd_acc1_ci95_lo"] == "" else f"{float(r['ccd_acc1_ci95_lo']):.4f}",
                hi="" if r["ccd_acc1_ci95_hi"] == "" else f"{float(r['ccd_acc1_ci95_hi']):.4f}",
                twoafc=float(r.get("twoafc_hardest", float("nan"))),
                margin=float(r["margin_mean"]),
                mlo="" if r["margin_mean_ci95_lo"] == "" else f"{float(r['margin_mean_ci95_lo']):.4f}",
                mhi="" if r["margin_mean_ci95_hi"] == "" else f"{float(r['margin_mean_ci95_hi']):.4f}",
                mp=str(r["metrics.json"]),
            )
        )

    path.write_text("".join(lines), encoding="utf-8")


def _plot_best_by_subj(df_by_type: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    main_summary = pd.read_csv(CCD_DIR / "ccd_summary.csv")
    main_pooled = main_summary[main_summary["eval_repr"] == "pooled_mean"].copy()
    best = (
        main_pooled.sort_values("ccd_acc1", ascending=False)
        .groupby(["subj"], as_index=False)
        .head(1)
        .loc[:, ["group", "tag", "subj", "eval_repr"]]
    )

    d = df_by_type[df_by_type["eval_repr"] == "pooled_mean"].copy()
    j = d.merge(best, on=["group", "tag", "subj", "eval_repr"], how="inner")
    if j.empty:
        return

    types = ["object", "attribute", "relation"]
    subjs = sorted(j["subj"].astype(str).unique().tolist())

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.8))

    width = 0.22
    x = np.arange(len(types))
    for si, s in enumerate(subjs):
        js = j[j["subj"].astype(str) == s]
        ys = []
        yerr_lo = []
        yerr_hi = []
        for t in types:
            r = js[js["type"] == t]
            if r.empty:
                ys.append(np.nan)
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
                continue
            rr = r.iloc[0]
            y = float(rr["ccd_acc1"])
            lo = rr.get("ccd_acc1_ci95_lo", "")
            hi = rr.get("ccd_acc1_ci95_hi", "")
            lo = float(lo) if lo != "" else y
            hi = float(hi) if hi != "" else y
            ys.append(y)
            yerr_lo.append(y - lo)
            yerr_hi.append(hi - y)

        offs = (si - (len(subjs) - 1) / 2.0) * width
        ax.bar(x + offs, ys, width=width, label=f"subj{s}")
        ax.errorbar(x + offs, ys, yerr=[yerr_lo, yerr_hi], fmt="none", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylim(0, 1)
    ax.set_ylabel("CCD@1 (hardneg)")
    ax.set_title("CCD-Hard by type (best pooled model per subject)")
    ax.legend(frameon=True)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardneg_for_ccd", default=str(HARDNEG_DIR / "shared982_hardneg_for_ccd.jsonl"))
    ap.add_argument("--valid_mask", default=str(HARDNEG_DIR / "hardneg_valid_mask.npy"))
    ap.add_argument("--ids_json", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hardneg_for_ccd = Path(args.hardneg_for_ccd).resolve()
    valid_mask_path = Path(args.valid_mask).resolve()

    if not hardneg_for_ccd.is_file():
        raise RuntimeError(f"Missing {hardneg_for_ccd}")
    if not valid_mask_path.is_file():
        raise RuntimeError(f"Missing {valid_mask_path}")

    ids_json = Path(args.ids_json).resolve() if args.ids_json else _find_default_ids_json()
    shared982_order = _load_shared982_order_ids(ids_json)

    valid_mask = np.load(valid_mask_path)
    if valid_mask.dtype != np.bool_ or valid_mask.ndim != 1 or int(valid_mask.shape[0]) != 982:
        raise RuntimeError(f"valid_mask must be 1D bool len=982, got dtype={valid_mask.dtype} shape={valid_mask.shape}")

    type_assets = _build_type_assets(shared982_order, valid_mask, hardneg_for_ccd)

    for a in type_assets:
        _run_batch_for_type(a, bootstrap=int(args.bootstrap), seed=int(args.seed))

    df = _consolidate_by_type(type_assets)

    out_csv = CCD_DIR / "ccd_by_type.csv"
    out_md = CCD_DIR / "ccd_by_type.md"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    _write_md(df, out_md)

    fig_path = CCD_DIR / "figures" / "Fig06_ccd_by_type.png"
    _plot_best_by_subj(df, fig_path)

    print(f"wrote: {out_csv}")
    print(f"wrote: {out_md}")
    print(f"wrote: {fig_path}")


if __name__ == "__main__":
    main()
