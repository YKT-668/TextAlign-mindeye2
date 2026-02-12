#!/usr/bin/env python
# coding: utf-8

"""regen_fig07_ccd_ablation_k_nopandas.py

Regenerates:
  cache/model_eval_results/shared982_ccd/figures/Fig07_ccd_ablation_k.png

This is a lightweight, pandas-free plotter intended for environments where
`tools/run_ccd_ablation.py` (which depends on pandas) can't be executed.

Logic matches the original intent:
- Use pooled_mean rows.
- For each subject, pick the best pooled model (max CCD@1) from ccd_summary.csv.
- In ccd_ablation_k.csv, keep only rows matching that (group, tag, subj, pooled_mean).
- Plot CCD@1 vs hard-negative K with 95% CI error bars.
- Auto-tighten y-limits so lines don't stick together.

"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


PROJ = Path(__file__).resolve().parents[1]
CCD_DIR = PROJ / "cache" / "model_eval_results" / "shared982_ccd"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: str) -> float:
    x = (x or "").strip()
    if x == "":
        return float("nan")
    return float(x)


def _best_models_pooled(ccd_summary_csv: Path) -> Dict[str, Tuple[str, str]]:
    rows = _read_csv(ccd_summary_csv)
    best: Dict[str, Tuple[float, str, str]] = {}
    for r in rows:
        if str(r.get("eval_repr", "")) != "pooled_mean":
            continue
        subj_raw = str(r.get("subj", ""))
        if subj_raw == "":
            continue
        try:
            subj = str(int(subj_raw))
        except Exception:
            subj = subj_raw.strip()
        score = _f(r.get("ccd_acc1", ""))
        if not np.isfinite(score):
            continue
        group = str(r.get("group", ""))
        tag = str(r.get("tag", ""))
        prev = best.get(subj)
        if prev is None or score > prev[0]:
            best[subj] = (score, group, tag)

    return {subj: (group, tag) for subj, (_, group, tag) in best.items()}


def plot_fig07(
    *,
    ccd_summary_csv: Path,
    ablation_csv: Path,
    out_png: Path,
) -> None:
    best = _best_models_pooled(ccd_summary_csv)
    if not best:
        raise SystemExit(f"No pooled_mean rows found in {ccd_summary_csv}")

    rows = _read_csv(ablation_csv)
    # Keep only pooled_mean and match best model per subject.
    kept: List[Dict[str, str]] = []
    for r in rows:
        if str(r.get("eval_repr", "")) != "pooled_mean":
            continue
        subj_raw = str(r.get("subj", ""))
        try:
            subj = str(int(subj_raw))
        except Exception:
            subj = subj_raw.strip()
        if subj == "" or subj not in best:
            continue
        g, t = best[subj]
        if str(r.get("group", "")) == g and str(r.get("tag", "")) == t:
            kept.append(r)

    if not kept:
        raise SystemExit("No matching rows in ablation csv after filtering to best pooled model per subject.")

    # Determine K values and subjects.
    ks = sorted({int(float(r.get("hardneg_k", "nan"))) for r in kept if (r.get("hardneg_k") or "").strip() != ""})
    subjs = sorted({str(int(str(r.get("subj", "")).strip())) for r in kept if str(r.get("subj", "")).strip() != ""})

    # Index rows by (subj, K)
    idx: Dict[Tuple[str, int], Dict[str, str]] = {}
    for r in kept:
        subj = str(int(str(r.get("subj", "")).strip()))
        k = int(float(r.get("hardneg_k", "nan")))
        idx[(subj, k)] = r

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.4))

    all_lo: List[float] = []
    all_hi: List[float] = []

    for subj in subjs:
        ys: List[float] = []
        los: List[float] = []
        his: List[float] = []
        xs: List[int] = []
        for k in ks:
            r = idx.get((subj, k))
            if r is None:
                continue
            y = _f(r.get("ccd_acc1", ""))
            lo = _f(r.get("ccd_acc1_ci95_lo", ""))
            hi = _f(r.get("ccd_acc1_ci95_hi", ""))
            xs.append(k)
            ys.append(y)
            los.append(lo)
            his.append(hi)
            if np.isfinite(lo):
                all_lo.append(lo)
            if np.isfinite(hi):
                all_hi.append(hi)

        if not xs:
            continue

        y = np.asarray(ys, dtype=float)
        lo = np.asarray(los, dtype=float)
        hi = np.asarray(his, dtype=float)

        # If CI missing, fall back to zero errorbar.
        yerr_low = np.where(np.isfinite(lo), y - lo, 0.0)
        yerr_high = np.where(np.isfinite(hi), hi - y, 0.0)
        ax.errorbar(xs, y, yerr=[yerr_low, yerr_high], marker="o", capsize=3, label=f"subj{subj}")

    ax.set_xlabel("#hard negatives per image (K)")
    ax.set_ylabel("CCD@1")
    ax.set_title("CCD-Hard K ablation (best pooled model per subject)")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", alpha=0.25)

    # Auto-tight y-limits + tick step.
    if all_lo and all_hi:
        y_min = min(all_lo)
        y_max = max(all_hi)
    else:
        # fallback to plotted values
        y_vals = [line.get_ydata() for line in ax.get_lines()]
        y_flat = np.concatenate([np.asarray(v, dtype=float) for v in y_vals]) if y_vals else np.asarray([0.0, 1.0])
        y_min = float(np.nanmin(y_flat))
        y_max = float(np.nanmax(y_flat))

    span = max(1e-6, y_max - y_min)
    pad = max(0.005, 0.08 * span)
    y_min = max(0.0, y_min - pad)
    y_max = min(1.0, y_max + pad)
    ax.set_ylim(y_min, y_max)

    span2 = max(1e-6, y_max - y_min)
    candidates = [0.01, 0.02, 0.025, 0.04, 0.05, 0.1]
    step = candidates[-1]
    for c in candidates:
        n = span2 / c
        if 5 <= n <= 9:
            step = c
            break
    if span2 / step < 5 and step > 0.02:
        step = 0.02
    ax.yaxis.set_major_locator(MultipleLocator(step))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    plot_fig07(
        ccd_summary_csv=CCD_DIR / "ccd_summary.csv",
        ablation_csv=CCD_DIR / "ccd_ablation_k.csv",
        out_png=CCD_DIR / "figures" / "Fig07_ccd_ablation_k.png",
    )
    print("[DONE] regenerated Fig07_ccd_ablation_k.png")


if __name__ == "__main__":
    main()
