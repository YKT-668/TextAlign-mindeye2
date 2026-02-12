#!/usr/bin/env python
# coding: utf-8

"""make_efficiency_figures.py

Reads results/tables/efficiency_summary.csv and generates efficiency curve figures:
- results/figures_main/Fig_efficiency_ccd_acc1.png
- results/figures_main/Fig_efficiency_twoafc_hard.png

Also writes:
- cache/model_eval_results/shared982_efficiency/figures/figures_manifest.json
- cache/model_eval_results/shared982_efficiency/figures/figures_report.md

Protocol:
- x-axis: sessions {1, 2, 40} (display as 1h/2h/full)
- lines: subj=1 and subj=5
- error bars: 95% CI from summary table

"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Enforce minimum font size > 7pt (using 10pt as base)
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})


PROJ = Path(__file__).resolve().parents[1]
SUMMARY_CSV = PROJ / "results" / "tables" / "efficiency_summary.csv"
FIG_MAIN_DIR = PROJ / "results" / "figures_main"
EFF_FIG_DIR = PROJ / "cache" / "model_eval_results" / "shared982_efficiency" / "figures"


@dataclass
class Row:
    subj: int
    model: str
    setting: str
    seed: int
    sessions: int
    N: int
    ccd_acc1: float
    ccd_ci_lo: float
    ccd_ci_hi: float
    twoafc_hard: float
    twoafc_ci_lo: float
    twoafc_ci_hi: float
    rsa_rho: float
    rsa_ci_lo: float
    rsa_ci_hi: float


def _parse_setting_to_sessions(setting: str) -> int:
    s = setting.strip().lower()
    if s in ("1sess", "1", "1h"):
        return 1
    if s in ("2sess", "2", "2h"):
        return 2
    if s in ("40sess", "40", "full"):
        return 40
    raise ValueError(f"Unknown setting: {setting}")


def _read_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sessions = _parse_setting_to_sessions(r["setting"])
            rows.append(
                Row(
                    subj=int(r["subj"]),
                    model=str(r["model"]),
                    setting=str(r["setting"]),
                    seed=int(r["seed"]),
                    sessions=int(sessions),
                    N=int(float(r.get("N", "0") or 0)),
                    ccd_acc1=float(r["ccd_acc1"]),
                    ccd_ci_lo=float(r["ccd_ci_lo"]),
                    ccd_ci_hi=float(r["ccd_ci_hi"]),
                    twoafc_hard=float(r["twoafc_hard"]),
                    twoafc_ci_lo=float(r["twoafc_ci_lo"]),
                    twoafc_ci_hi=float(r["twoafc_ci_hi"]),
                    rsa_rho=float(r["rsa_rho"]),
                    rsa_ci_lo=float(r["rsa_ci_lo"]),
                    rsa_ci_hi=float(r["rsa_ci_hi"]),
                )
            )
    return rows


def _plot_metric(
    rows: List[Row],
    *,
    metric_key: str,
    y_label: str,
    out_path: Path,
) -> Dict:
    sessions_order = [1, 2, 40]
    x_labels = ["1h", "2h", "full"]
    x_positions = list(range(len(sessions_order)))
    sess_to_x = {s: i for i, s in enumerate(sessions_order)}

    # Prepare series keyed by (model, subj)
    series_map: Dict[Tuple[str, int], Dict[int, Tuple[float, float, float]]] = {}
    for r in rows:
        if r.subj not in (1, 5):
            continue
        if r.model not in ("baseline", "textalign_llm"):
            continue
        key = (r.model, r.subj)
        series_map.setdefault(key, {})
        if metric_key == "ccd_acc1":
            series_map[key][r.sessions] = (r.ccd_acc1, r.ccd_ci_lo, r.ccd_ci_hi)
        elif metric_key == "twoafc_hard":
            series_map[key][r.sessions] = (r.twoafc_hard, r.twoafc_ci_lo, r.twoafc_ci_hi)
        elif metric_key == "rsa_rho":
            series_map[key][r.sessions] = (r.rsa_rho, r.rsa_ci_lo, r.rsa_ci_hi)
        else:
            raise ValueError(f"Unknown metric_key: {metric_key}")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=150, sharey=True)

    # Spec: each figure shows subj=1 and subj=5 two curves; we use two subplots for models.
    models = ["baseline", "textalign_llm"]
    subj_styles = {
        1: {"linestyle": "-", "marker": "o"},
        5: {"linestyle": "-", "marker": "s"},
    }

    plotted = []
    for ax, model in zip(axes, models):
        for subj in (1, 5):
            series = series_map.get((model, subj), {})
            xs: List[int] = []
            ys: List[float] = []
            yerr_lo: List[float] = []
            yerr_hi: List[float] = []
            for s in sessions_order:
                if s not in series:
                    continue
                y, lo, hi = series[s]
                xs.append(sess_to_x[s])
                ys.append(y)
                yerr_lo.append(max(0.0, y - lo))
                yerr_hi.append(max(0.0, hi - y))
            if not xs:
                continue
            st = subj_styles[subj]
            ax.errorbar(
                xs,
                ys,
                yerr=[yerr_lo, yerr_hi],
                marker=st["marker"],
                linestyle=st["linestyle"],
                linewidth=2.2,
                capsize=3,
                label=f"s{subj}",
            )
            plotted.append({"subj": subj, "model": model})

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Data (sessions)")
        ax.set_title(model)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.25, x_positions[-1] + 0.25)

    axes[0].set_ylabel(y_label)
    fig.suptitle("Efficiency curve (shared982)")
    axes[0].legend(loc="best")

    # Tighten y-range to improve readability when curves are close.
    all_lo: List[float] = []
    all_hi: List[float] = []
    for series in series_map.values():
        for s in sessions_order:
            if s not in series:
                continue
            _, lo, hi = series[s]
            all_lo.append(lo)
            all_hi.append(hi)

    if all_lo and all_hi:
        y_lo = min(all_lo)
        y_hi = max(all_hi)
        span = max(1e-6, y_hi - y_lo)
        pad = max(0.02, 0.12 * span)
        y_min = y_lo - pad
        y_max = y_hi + pad
        if metric_key != "rsa_rho":
            y_min = max(0.0, y_min)
            y_max = min(1.0, y_max)

        for ax in axes:
            ax.set_ylim(y_min, y_max)

        # Use a small set of major ticks ("each grid cell" spacing) to avoid clutter.
        span2 = max(1e-6, y_max - y_min)
        if span2 <= 0.08:
            step = 0.02
        elif span2 <= 0.16:
            step = 0.04
        elif span2 <= 0.30:
            step = 0.05
        else:
            step = 0.10
        for ax in axes:
            ax.yaxis.set_major_locator(MultipleLocator(step))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "metric": metric_key,
        "y_label": y_label,
        "out_path": str(out_path.relative_to(PROJ)),
        "x": sessions_order,
        "x_labels": x_labels,
        "subplots": models,
        "lines": sorted(plotted, key=lambda d: (d["model"], d["subj"])) ,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(SUMMARY_CSV))
    ap.add_argument("--out_main", default=str(FIG_MAIN_DIR))
    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.is_file():
        raise SystemExit(f"Missing summary csv: {summary_path}")

    rows = _read_rows(summary_path)

    FIG_MAIN_DIR.mkdir(parents=True, exist_ok=True)
    EFF_FIG_DIR.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict] = {
        "source_csv": str(summary_path.relative_to(PROJ)),
        "figures": {},
    }

    # Required main figures
    fig1 = PROJ / "results" / "figures_main" / "Fig_efficiency_ccd_acc1.png"
    fig2 = PROJ / "results" / "figures_main" / "Fig_efficiency_twoafc_hard.png"

    manifest["figures"]["Fig_efficiency_ccd_acc1.png"] = _plot_metric(
        rows,
        metric_key="ccd_acc1",
        y_label="CCD@1",
        out_path=fig1,
    )
    manifest["figures"]["Fig_efficiency_twoafc_hard.png"] = _plot_metric(
        rows,
        metric_key="twoafc_hard",
        y_label="Hard-2AFC",
        out_path=fig2,
    )

    # Report: keep it minimal and structured
    report_lines = []
    report_lines.append("# Efficiency figures (shared982)")
    report_lines.append("")
    report_lines.append(f"- Source: `{summary_path.relative_to(PROJ)}`")
    report_lines.append(f"- Output dir: `{EFF_FIG_DIR.relative_to(PROJ)}`")
    report_lines.append("")
    report_lines.append("## Main")
    report_lines.append(f"- results/figures_main/Fig_efficiency_ccd_acc1.png")
    report_lines.append(f"- results/figures_main/Fig_efficiency_twoafc_hard.png")

    (EFF_FIG_DIR / "figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (EFF_FIG_DIR / "figures_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("[DONE] wrote figures + manifest/report")


if __name__ == "__main__":
    main()
