#!/usr/bin/env python
# coding: utf-8
"""run_ccd_ablation.py

Minimal CCD-Hard ablations without retraining.

Ablations:
1) K ablation: evaluate with hard negatives per image K=2,4,8 (requires candidate pool; K=8 may be unavailable).
2) Similarity window ablation: tight vs loose sim_text filtering on candidate pool.
3) Difficulty ablation: hardest-vs-random selection (text-side sim_text only), with K fixed (default 4).

Outputs:
- cache/model_eval_results/shared982_ccd/ablation/** (per-setting CCD summaries)
- cache/model_eval_results/shared982_ccd/ccd_ablation_k.csv
- cache/model_eval_results/shared982_ccd/ccd_ablation_window.csv
- cache/model_eval_results/shared982_ccd/ccd_ablation_difficulty.csv
- figures:
  - cache/model_eval_results/shared982_ccd/figures/Fig07_ccd_ablation_k.png
  - cache/model_eval_results/shared982_ccd/figures/Fig08_ccd_ablation_window.png
    - cache/model_eval_results/shared982_ccd/figures/Fig09_ccd_ablation_difficulty.png

Paper/supplement tables:
- results/tables/ccd_ablation_k.csv
- results/tables/ccd_ablation_difficulty.csv
- results/tables/main_results.csv (append-only ablation rows)

Implementation detail:
- Uses tools/eval_ccd_embed.py (now supports HARD_NEG_K>1 when HARD_NEG_JSONL has multiple rows per image_id).
- Uses per-setting keep masks (len=982) to ensure every evaluated image has >=K candidates after filtering.
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
RESULTS_TABLES_DIR = PROJ / "results" / "tables"


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
    cands = sorted((PROJ / "evals" / "brain_tokens").glob("**/*_ids.json"))
    if not cands:
        raise RuntimeError("No *_ids.json found under evals/brain_tokens")
    for p in cands:
        if "official_hf" in str(p) and "subj01" in p.name:
            return p
    return cands[0]


def _load_shared982_order_ids(ids_json: Path) -> np.ndarray:
    ids = np.asarray(json.loads(ids_json.read_text(encoding="utf-8")), dtype=np.int64)
    shared982_mask_path = PROJ / "src" / "shared982.npy"
    if not shared982_mask_path.is_file():
        raise RuntimeError(f"Missing {shared982_mask_path}")
    m = np.load(shared982_mask_path)
    keep = np.asarray([bool(m[int(i)]) for i in ids], dtype=np.bool_)
    order = ids[np.where(keep)[0]]
    if int(order.shape[0]) != 982:
        raise RuntimeError(f"shared982 order size={int(order.shape[0])} != 982")
    return order


def _count_candidates_by_image(rows: List[dict]) -> Dict[int, List[Tuple[float, dict]]]:
    out: Dict[int, List[Tuple[float, dict]]] = {}
    for r in rows:
        if "image_id" not in r or "neg_caption" not in r:
            continue
        img_id = int(r["image_id"])
        sim = float(r.get("sim_text", 0.0))
        out.setdefault(img_id, []).append((sim, r))
    # sort for determinism
    for k, v in out.items():
        out[k] = sorted(v, key=lambda x: x[0], reverse=True)
    return out


def _write_selected_jsonl_random(
    shared982_order: np.ndarray,
    pool: Dict[int, List[Tuple[float, dict]]],
    k: int,
    seed: int,
    out_path: Path,
) -> Dict[int, int]:
    """Write a jsonl with exactly K negatives per image, sampled uniformly from the candidate pool.

    Important: selection uses only the candidate pool + RNG seed; it does NOT look at any brain features.
    Returns: per-image candidate counts in the *base pool* (for coverage bookkeeping).
    """

    rng = np.random.default_rng(int(seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[int, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for img_id in shared982_order.tolist():
            cands = pool.get(int(img_id), [])
            counts[int(img_id)] = len(cands)
            if len(cands) < int(k):
                continue
            idx = rng.choice(len(cands), size=int(k), replace=False)
            # Deterministic order for stable file diff: sort selected by original sim_text desc
            chosen = [cands[int(i)][1] for i in idx]
            chosen = sorted(chosen, key=lambda r: float(r.get("sim_text", 0.0)), reverse=True)
            for r in chosen:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return counts


def _write_filtered_jsonl(
    shared982_order: np.ndarray,
    pool: Dict[int, List[Tuple[float, dict]]],
    low: float | None,
    high: float | None,
    out_path: Path,
) -> Dict[int, int]:
    """Write a filtered multi-row jsonl preserving all candidates within window.

    Returns: per-image candidate counts after filtering.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[int, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for img_id in shared982_order.tolist():
            cands = pool.get(int(img_id), [])
            kept = []
            for sim, r in cands:
                if low is not None and sim < low:
                    continue
                if high is not None and sim > high:
                    continue
                kept.append(r)
            counts[int(img_id)] = len(kept)
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return counts


def _write_keep_mask(shared982_order: np.ndarray, base_valid_mask: np.ndarray, counts: Dict[int, int], k: int, out_path: Path) -> np.ndarray:
    keep = np.zeros((982,), dtype=np.bool_)
    for i, img_id in enumerate(shared982_order.tolist()):
        if not base_valid_mask[i]:
            continue
        if counts.get(int(img_id), 0) >= k:
            keep[i] = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, keep)
    return keep


@dataclass
class AblSetting:
    name: str
    hardneg_jsonl: Path
    keep_mask: Path
    hardneg_k: int
    out_dir: Path


def _run_setting(s: AblSetting, bootstrap: int, seed: int) -> None:
    env = os.environ.copy()
    env["HARD_NEG_K"] = str(int(s.hardneg_k))
    env["HARD_NEG_REQUIRE_FULL"] = "1"

    cmd = [
        "python",
        str(PROJ / "tools" / "rerun_all_ccd_shared982.py"),
        "--neg_jsonl",
        str(s.hardneg_jsonl),
        "--use_valid_mask",
        str(s.keep_mask),
        "--bootstrap",
        str(int(bootstrap)),
        "--seed",
        str(int(seed)),
        "--out_dir",
        str(s.out_dir),
    ]
    subprocess.check_call(cmd, cwd=str(PROJ), env=env)


def _best_models_pooled() -> pd.DataFrame:
    main = pd.read_csv(CCD_DIR / "ccd_summary.csv")
    d = main[main["eval_repr"] == "pooled_mean"].copy()
    return (
        d.sort_values("ccd_acc1", ascending=False)
        .groupby(["subj"], as_index=False)
        .head(1)
        .loc[:, ["group", "tag", "subj", "eval_repr"]]
    )


def _plot_ablation_k(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    best = _best_models_pooled()
    d = df[df["eval_repr"] == "pooled_mean"].merge(best, on=["group", "tag", "subj", "eval_repr"], how="inner")
    if d.empty:
        return

    ks = sorted(d["hardneg_k"].unique().tolist())
    subjs = sorted(d["subj"].astype(str).unique().tolist())

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.4))
    all_lo: List[float] = []
    all_hi: List[float] = []
    for s in subjs:
        ds = d[d["subj"].astype(str) == s].sort_values("hardneg_k")
        y = ds["ccd_acc1"].astype(float).to_numpy()
        lo = ds["ccd_acc1_ci95_lo"].replace("", np.nan).astype(float).to_numpy()
        hi = ds["ccd_acc1_ci95_hi"].replace("", np.nan).astype(float).to_numpy()
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(ks, y, yerr=yerr, marker="o", capsize=3, label=f"subj{s}")

        for v in lo:
            if np.isfinite(v):
                all_lo.append(float(v))
        for v in hi:
            if np.isfinite(v):
                all_hi.append(float(v))

    ax.set_xlabel("#hard negatives per image (K)")
    ax.set_ylabel("CCD@1")
    # Auto-tighten y-range (with padding) to avoid compressing curves into a tiny band.
    if all_lo and all_hi:
        y_min = min(all_lo)
        y_max = max(all_hi)
    else:
        y_min = float(np.nanmin(d["ccd_acc1"].astype(float).to_numpy()))
        y_max = float(np.nanmax(d["ccd_acc1"].astype(float).to_numpy()))

    span = max(1e-6, y_max - y_min)
    # Smaller padding to visually separate close curves.
    pad = max(0.005, 0.08 * span)
    y_min = max(0.0, y_min - pad)
    y_max = min(1.0, y_max + pad)
    ax.set_ylim(y_min, y_max)

    span2 = max(1e-6, y_max - y_min)
    # Pick a "nice" step so there are ~5-8 major ticks.
    candidates = [0.01, 0.02, 0.025, 0.04, 0.05, 0.1]
    step = candidates[-1]
    for c in candidates:
        n = span2 / c
        if 5 <= n <= 9:
            step = c
            break
    # If still too sparse, prefer a slightly denser grid.
    if span2 / step < 5 and step > 0.02:
        step = 0.02
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_title("CCD-Hard K ablation (best pooled model per subject)")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_ablation_window(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    best = _best_models_pooled()
    d = df[df["eval_repr"] == "pooled_mean"].merge(best, on=["group", "tag", "subj", "eval_repr"], how="inner")
    if d.empty:
        return

    labels = ["loose", "tight"]
    subjs = sorted(d["subj"].astype(str).unique().tolist())

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    x = np.arange(len(labels))
    width = 0.22
    for si, s in enumerate(subjs):
        ds = d[d["subj"].astype(str) == s].set_index("window")
        y = [float(ds.loc[l, "ccd_acc1"]) if l in ds.index else np.nan for l in labels]
        lo = [float(ds.loc[l, "ccd_acc1_ci95_lo"]) if l in ds.index and ds.loc[l, "ccd_acc1_ci95_lo"] != "" else y[i] for i, l in enumerate(labels)]
        hi = [float(ds.loc[l, "ccd_acc1_ci95_hi"]) if l in ds.index and ds.loc[l, "ccd_acc1_ci95_hi"] != "" else y[i] for i, l in enumerate(labels)]
        yerr = [np.asarray(y) - np.asarray(lo), np.asarray(hi) - np.asarray(y)]
        offs = (si - (len(subjs) - 1) / 2.0) * width
        ax.bar(x + offs, y, width=width, label=f"subj{s}")
        ax.errorbar(x + offs, y, yerr=yerr, fmt="none", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CCD@1")
    ax.set_ylim(0, 1)
    ax.set_title("CCD-Hard similarity-window ablation (best pooled model per subject)")
    ax.legend(frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_ablation_difficulty(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    best = _best_models_pooled()
    d = df[df["eval_repr"] == "pooled_mean"].merge(best, on=["group", "tag", "subj", "eval_repr"], how="inner")
    if d.empty:
        return

    labels = ["hardest", "random"]
    subjs = sorted(d["subj"].astype(str).unique().tolist())

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    x = np.arange(len(labels))
    width = 0.22
    for si, s in enumerate(subjs):
        ds = d[d["subj"].astype(str) == s].set_index("difficulty")
        y = [float(ds.loc[l, "ccd_acc1"]) if l in ds.index else np.nan for l in labels]
        lo = [
            float(ds.loc[l, "ccd_acc1_ci95_lo"]) if l in ds.index and ds.loc[l, "ccd_acc1_ci95_lo"] != "" else y[i]
            for i, l in enumerate(labels)
        ]
        hi = [
            float(ds.loc[l, "ccd_acc1_ci95_hi"]) if l in ds.index and ds.loc[l, "ccd_acc1_ci95_hi"] != "" else y[i]
            for i, l in enumerate(labels)
        ]
        yerr = [np.asarray(y) - np.asarray(lo), np.asarray(hi) - np.asarray(y)]
        offs = (si - (len(subjs) - 1) / 2.0) * width
        ax.bar(x + offs, y, width=width, label=f"subj{s}")
        ax.errorbar(x + offs, y, yerr=yerr, fmt="none", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CCD@1")
    ax.set_ylim(0, 1)
    ax.set_title("CCD-Hard difficulty ablation (hardest vs random; best pooled model per subject)")
    ax.legend(frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _best_rows_from_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce a ccd_summary-like df to one row per subject (best pooled model)."""

    best = _best_models_pooled()
    d = df[df["eval_repr"] == "pooled_mean"].merge(best, on=["group", "tag", "subj", "eval_repr"], how="inner")
    cols = [
        c
        for c in [
            "setting",
            "hardneg_k",
            "window",
            "difficulty",
            "group",
            "tag",
            "subj",
            "eval_repr",
            "n_eval",
            "neg_mode",
            "k_neg",
            "seed",
            "bootstrap",
            "ccd_acc1",
            "ccd_acc1_ci95_lo",
            "ccd_acc1_ci95_hi",
            "twoafc_hardest",
            "twoafc_hardest_ci95_lo",
            "twoafc_hardest_ci95_hi",
            "margin_mean",
            "margin_mean_ci95_lo",
            "margin_mean_ci95_hi",
        ]
        if c in d.columns
    ]
    return d.loc[:, cols].copy()


def _append_main_results(rows: pd.DataFrame, group_name: str) -> None:
    """Append ablation rows into results/tables/main_results.csv (do not overwrite existing)."""

    main_path = RESULTS_TABLES_DIR / "main_results.csv"
    if not main_path.is_file():
        raise RuntimeError(f"Missing {main_path}")

    main = pd.read_csv(main_path)
    # Create ablation-compatible rows (only ccd_* columns populated).
    out_rows = []
    for _, r in rows.iterrows():
        tag = str(r.get("tag", ""))
        subj = str(int(r.get("subj"))) if pd.notna(r.get("subj")) else "all"
        setting = str(r.get("setting", ""))
        difficulty = str(r.get("difficulty", ""))
        window = str(r.get("window", ""))
        hardneg_k = r.get("hardneg_k")

        k_part = f"k{int(hardneg_k)}" if pd.notna(hardneg_k) else ""
        # Avoid redundant tags like "k2:k2:<model>".
        if setting == k_part:
            k_part = ""
        tag2 = ":".join([x for x in [setting, window, difficulty, k_part, tag] if x and x != "nan"])

        ccd_acc1 = float(r.get("ccd_acc1")) if pd.notna(r.get("ccd_acc1")) else np.nan
        if "ccd_acc1_ci95_lo" in r and pd.notna(r.get("ccd_acc1_ci95_lo")):
            ci = f"[{float(r.get('ccd_acc1_ci95_lo')):.4f},  {float(r.get('ccd_acc1_ci95_hi')):.4f}]"
        else:
            ci = ""

        out_rows.append(
            {
                "group": group_name,
                "tag": tag2,
                "subj": subj,
                "neg_mode": "hardneg",
                "ccd_N": float(r.get("n_eval")) if pd.notna(r.get("n_eval")) else np.nan,
                "ccd_acc1": ccd_acc1,
                "ccd_acc1_ci95": ci,
                "twoafc_hardest": float(r.get("twoafc_hardest")) if pd.notna(r.get("twoafc_hardest")) else np.nan,
                "margin_mean": float(r.get("margin_mean")) if pd.notna(r.get("margin_mean")) else np.nan,
            }
        )

    add = pd.DataFrame(out_rows)
    # Align columns
    for c in main.columns:
        if c not in add.columns:
            add[c] = np.nan
    add = add.loc[:, main.columns]

    # De-dup by (group,tag,subj)
    key_cols = ["group", "tag", "subj"]
    existing = set(tuple(x) for x in main[key_cols].astype(str).to_numpy())
    keep_mask = [tuple(x) not in existing for x in add[key_cols].astype(str).to_numpy()]
    add2 = add[np.asarray(keep_mask, dtype=bool)]
    if add2.empty:
        return
    out = pd.concat([main, add2], axis=0, ignore_index=True)
    out.to_csv(main_path, index=False)


def _drop_group_from_main_results(group_name: str) -> None:
    """Remove all rows with group==group_name from results/tables/main_results.csv."""

    main_path = RESULTS_TABLES_DIR / "main_results.csv"
    if not main_path.is_file():
        raise RuntimeError(f"Missing {main_path}")
    main = pd.read_csv(main_path)
    keep = main["group"].astype(str) != str(group_name)
    if int(keep.sum()) == int(main.shape[0]):
        return
    main.loc[keep].to_csv(main_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_pool_jsonl", default=str(HARDNEG_DIR / "shared982_hardneg.jsonl"))
    ap.add_argument("--valid_mask", default=str(HARDNEG_DIR / "hardneg_valid_mask.npy"))
    ap.add_argument("--ids_json", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42, help="Seed for K/window ablations")
    ap.add_argument("--k_values", default="2,4,8")
    ap.add_argument("--difficulty_k", type=int, default=2, help="K_fixed for difficulty ablation (default 2)")
    ap.add_argument("--difficulty_seed", type=int, default=0, help="Seed for difficulty ablation (default 0)")
    ap.add_argument(
        "--only",
        default="",
        choices=["", "all", "k", "window", "difficulty"],
        help="Run only a subset of ablations (default: run all)",
    )
    args = ap.parse_args()

    base_jsonl = Path(args.base_pool_jsonl).resolve()
    if not base_jsonl.is_file():
        raise RuntimeError(f"Missing {base_jsonl}")

    valid_mask = np.load(Path(args.valid_mask).resolve())
    if valid_mask.dtype != np.bool_ or valid_mask.ndim != 1 or int(valid_mask.shape[0]) != 982:
        raise RuntimeError(f"valid_mask must be 1D bool len=982, got dtype={valid_mask.dtype} shape={valid_mask.shape}")

    ids_json = Path(args.ids_json).resolve() if args.ids_json else _find_default_ids_json()
    shared982_order = _load_shared982_order_ids(ids_json)

    pool_rows = _read_jsonl(base_jsonl)
    pool = _count_candidates_by_image(pool_rows)

    ab_dir = CCD_DIR / "ablation"
    mask_dir = HARDNEG_DIR / "ablation_masks"
    jsonl_dir = HARDNEG_DIR / "ablation_jsonl"
    ab_dir.mkdir(parents=True, exist_ok=True)

    # candidate counts without filtering
    counts_all = {int(img_id): len(pool.get(int(img_id), [])) for img_id in shared982_order.tolist()}

    run_only = (args.only or "").strip().lower()
    run_all = run_only in ("", "all")

    # K ablation (no sim filtering)
    df_k = None
    out_k = CCD_DIR / "ccd_ablation_k.csv"
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    if run_all or run_only == "k":
        k_settings: List[AblSetting] = []
        for k in k_values:
            keep_mask_path = mask_dir / f"hardneg_valid_mask_k{k}.npy"
            keep = _write_keep_mask(shared982_order, valid_mask, counts_all, k, keep_mask_path)
            if int(keep.sum()) == 0:
                print(
                    f"[SKIP] K={k} has zero coverage under current hardneg pool (max candidates per image is {max(counts_all.values())})."
                )
                continue
            out_dir = ab_dir / f"k{k}"
            k_settings.append(
                AblSetting(name=f"k{k}", hardneg_jsonl=base_jsonl, keep_mask=keep_mask_path, hardneg_k=k, out_dir=out_dir)
            )

        for s in k_settings:
            _run_setting(s, bootstrap=int(args.bootstrap), seed=int(args.seed))

        if k_settings:
            parts = []
            for s in k_settings:
                df = pd.read_csv(s.out_dir / "ccd_summary.csv")
                df.insert(0, "setting", s.name)
                df.insert(1, "hardneg_k", int(s.hardneg_k))
                parts.append(df)

            df_k = pd.concat(parts, axis=0, ignore_index=True)
            df_k.to_csv(out_k, index=False)
        else:
            print("[WARN] No valid K settings produced; skipping ccd_ablation_k.csv")
            df_k = pd.DataFrame()

    # similarity-window ablation
    df_w = None
    out_w = CCD_DIR / "ccd_ablation_window.csv"
    if run_all or run_only == "window":
        sim_vals = np.asarray([float(r.get("sim_text", 0.0)) for r in pool_rows if "sim_text" in r], dtype=np.float32)
        if sim_vals.size == 0:
            raise RuntimeError("base_pool_jsonl has no sim_text; cannot do window ablation")

        p05 = float(np.quantile(sim_vals, 0.05))
        p50 = float(np.quantile(sim_vals, 0.50))
        p95 = float(np.quantile(sim_vals, 0.95))

        # Define two windows
        loose = (p05, p95)
        tight = (p50, p95)

        window_settings: List[Tuple[str, Tuple[float, float]]] = [("loose", loose), ("tight", tight)]
        win_runs: List[AblSetting] = []

        for name, (low, high) in window_settings:
            out_jsonl = jsonl_dir / f"shared982_hardneg_window_{name}.jsonl"
            counts = _write_filtered_jsonl(shared982_order, pool, low=low, high=high, out_path=out_jsonl)
            keep_mask_path = mask_dir / f"hardneg_valid_mask_window_{name}.npy"
            # use K=4 by default for window ablation
            keep = _write_keep_mask(shared982_order, valid_mask, counts, 4, keep_mask_path)
            if int(keep.sum()) == 0:
                print(f"[SKIP] window={name} has zero coverage for K=4 after sim_text filtering.")
                continue
            out_dir = ab_dir / f"window_{name}"
            win_runs.append(
                AblSetting(name=f"window_{name}", hardneg_jsonl=out_jsonl, keep_mask=keep_mask_path, hardneg_k=4, out_dir=out_dir)
            )

        for s in win_runs:
            _run_setting(s, bootstrap=int(args.bootstrap), seed=int(args.seed))

        if win_runs:
            parts = []
            for s in win_runs:
                df = pd.read_csv(s.out_dir / "ccd_summary.csv")
                df.insert(0, "setting", s.name)
                df.insert(1, "window", s.name.replace("window_", ""))
                df.insert(2, "hardneg_k", int(s.hardneg_k))
                parts.append(df)

            df_w = pd.concat(parts, axis=0, ignore_index=True)
            df_w.to_csv(out_w, index=False)
        else:
            print("[WARN] No valid window settings produced; skipping ccd_ablation_window.csv")
            df_w = pd.DataFrame()

    # difficulty ablation: hardest vs random (K fixed)
    diff_k = int(args.difficulty_k)
    diff_seed = int(args.difficulty_seed)
    diff_runs: List[AblSetting] = []

    # hardest: just reuse base pool, let eval_ccd_embed pick top-K by sim_text
    keep_mask_hardest = mask_dir / f"hardneg_valid_mask_difficulty_hardest_k{diff_k}.npy"
    keep_h = _write_keep_mask(shared982_order, valid_mask, counts_all, diff_k, keep_mask_hardest)
    if int(keep_h.sum()) > 0:
        diff_runs.append(
            AblSetting(
                name=f"difficulty_hardest",
                hardneg_jsonl=base_jsonl,
                keep_mask=keep_mask_hardest,
                hardneg_k=diff_k,
                out_dir=ab_dir / f"difficulty_hardest_k{diff_k}",
            )
        )
    else:
        print(f"[SKIP] difficulty=hardest has zero coverage for K={diff_k}")

    # random: materialize a jsonl with exactly K sampled negatives per image
    out_jsonl_rand = jsonl_dir / f"shared982_hardneg_random_k{diff_k}_seed{diff_seed}.jsonl"
    counts_rand = _write_selected_jsonl_random(shared982_order, pool, k=diff_k, seed=diff_seed, out_path=out_jsonl_rand)
    keep_mask_random = mask_dir / f"hardneg_valid_mask_difficulty_random_k{diff_k}.npy"
    keep_r = _write_keep_mask(shared982_order, valid_mask, counts_rand, diff_k, keep_mask_random)
    if int(keep_r.sum()) > 0:
        diff_runs.append(
            AblSetting(
                name=f"difficulty_random",
                hardneg_jsonl=out_jsonl_rand,
                keep_mask=keep_mask_random,
                hardneg_k=diff_k,
                out_dir=ab_dir / f"difficulty_random_k{diff_k}",
            )
        )
    else:
        print(f"[SKIP] difficulty=random has zero coverage for K={diff_k}")

    for s in diff_runs:
        _run_setting(s, bootstrap=int(args.bootstrap), seed=int(diff_seed))

    df_d = None
    out_d = CCD_DIR / "ccd_ablation_difficulty.csv"
    if diff_runs and (run_all or run_only == "difficulty"):
        parts = []
        for s in diff_runs:
            df = pd.read_csv(s.out_dir / "ccd_summary.csv")
            df.insert(0, "setting", s.name)
            df.insert(1, "difficulty", s.name.replace("difficulty_", ""))
            df.insert(2, "hardneg_k", int(s.hardneg_k))
            parts.append(df)
        df_d = pd.concat(parts, axis=0, ignore_index=True)
        df_d.to_csv(out_d, index=False)
    else:
        df_d = pd.DataFrame()
        if run_all or run_only == "difficulty":
            print("[WARN] No valid difficulty runs produced; skipping ccd_ablation_difficulty.csv")

    # figures
    fig_k = CCD_DIR / "figures" / "Fig07_ccd_ablation_k.png"
    fig_w = CCD_DIR / "figures" / "Fig08_ccd_ablation_window.png"
    fig_d = CCD_DIR / "figures" / "Fig09_ccd_ablation_difficulty.png"

    if run_all or run_only == "k":
        _plot_ablation_k(df_k, fig_k)
    if run_all or run_only == "window":
        _plot_ablation_window(df_w, fig_w)
    if run_all or run_only == "difficulty":
        _plot_ablation_difficulty(df_d, fig_d)

    # write lightweight tables for paper/supplement + update main_results
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if (run_all or run_only == "k") and df_k is not None:
        best_k = _best_rows_from_summary(df_k)
        # Ensure requested K values show up; if a K is unavailable, add explicit placeholder rows per subject.
        subjs = sorted(best_k["subj"].unique().tolist()) if not best_k.empty else [1, 2, 5, 7]
        need_ks = [int(x) for x in k_values]
        have_ks = set(best_k["hardneg_k"].astype(int).unique().tolist()) if ("hardneg_k" in best_k.columns and not best_k.empty) else set()
        missing_ks = [k for k in need_ks if k not in have_ks]
        if missing_ks:
            placeholders = []
            for k in missing_ks:
                for subj in subjs:
                    placeholders.append(
                        {
                            "setting": f"k{k}",
                            "hardneg_k": k,
                            "group": "",
                            "tag": "",
                            "subj": int(subj),
                            "eval_repr": "pooled_mean",
                            "n_eval": 0,
                            "neg_mode": "hardneg",
                            "k_neg": k,
                            "seed": int(args.seed),
                            "bootstrap": int(args.bootstrap),
                            "ccd_acc1": np.nan,
                            "ccd_acc1_ci95_lo": np.nan,
                            "ccd_acc1_ci95_hi": np.nan,
                            "twoafc_hardest": np.nan,
                            "margin_mean": np.nan,
                            "note": "unavailable under current hardneg pool (insufficient candidates per image)",
                        }
                    )
            best_k = pd.concat([best_k, pd.DataFrame(placeholders)], axis=0, ignore_index=True)

        out_best_k = RESULTS_TABLES_DIR / "ccd_ablation_k.csv"
        best_k.to_csv(out_best_k, index=False)
        # Append to main_results (only rows with real metrics)
        _append_main_results(best_k[best_k["group"].astype(str) != ""].copy(), group_name="shared982_ccd_ablation_k")

    if (run_all or run_only == "difficulty") and df_d is not None and not df_d.empty:
        best_d = _best_rows_from_summary(df_d)
        # Add canonical columns requested by the spec (keep existing columns too).
        best_d.insert(0, "model_tag", best_d.get("tag", ""))
        if "difficulty" in best_d.columns:
            best_d.insert(1, "difficulty_mode", best_d["difficulty"])
        best_d.insert(2, "K_fixed", best_d.get("hardneg_k", np.nan))
        best_d.insert(3, "N", best_d.get("n_eval", np.nan))
        best_d.insert(4, "acc1", best_d.get("ccd_acc1", np.nan))
        best_d.insert(5, "ci_lo", best_d.get("ccd_acc1_ci95_lo", np.nan))
        best_d.insert(6, "ci_hi", best_d.get("ccd_acc1_ci95_hi", np.nan))

        out_best_d = RESULTS_TABLES_DIR / "ccd_ablation_difficulty.csv"
        best_d.to_csv(out_best_d, index=False)

        # Overwrite previous difficulty entries in main_results to keep it consistent.
        _drop_group_from_main_results("shared982_ccd_ablation_difficulty")
        _append_main_results(best_d.copy(), group_name="shared982_ccd_ablation_difficulty")

    if run_all or run_only == "k":
        print(f"wrote: {out_k}")
        print(f"wrote: {fig_k}")
    if run_all or run_only == "window":
        print(f"wrote: {out_w}")
        print(f"wrote: {fig_w}")
    if run_all or run_only == "difficulty":
        print(f"wrote: {out_d}")
        print(f"wrote: {fig_d}")


if __name__ == "__main__":
    main()
