#!/usr/bin/env python
# coding: utf-8
"""make_isrsa_figures.py

Generate figures for shared982 IS-RSA analysis.

Reads:
- cache/model_eval_results/shared982_isrsa/baseline/isrsa_matrix.csv
- cache/model_eval_results/shared982_isrsa/textalign_llm/isrsa_matrix.csv
(and optionally mean_cos_matrix.csv)

Writes:
- cache/model_eval_results/shared982_isrsa/figures/Fig_isrsa_heatmap_baseline.png
- cache/model_eval_results/shared982_isrsa/figures/Fig_isrsa_heatmap_textalign_llm.png
- cache/model_eval_results/shared982_isrsa/figures/Fig_isrsa_delta.png
- cache/model_eval_results/shared982_isrsa/figures/Fig_mean_cos_delta.png (if available)
- cache/model_eval_results/shared982_isrsa/figures/figures_manifest.json
- cache/model_eval_results/shared982_isrsa/figures/figures_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Enforce minimum font size > 7pt
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})


_PROJ_ROOT = Path(__file__).resolve().parents[1]
_OUT_ROOT = _PROJ_ROOT / "cache" / "model_eval_results" / "shared982_isrsa"
_FIG_DIR = _OUT_ROOT / "figures"


def _read_matrix(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, index_col=0)
    # keep stable subject order by index order
    return df


def _heatmap(df: pd.DataFrame, out_path: Path, title: str, vmin=None, vmax=None, cmap="viridis", center=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat = df.values.astype(np.float32)

    plt.figure(figsize=(4.2, 3.6), dpi=220)
    ax = plt.gca()

    if center is not None:
        from matplotlib.colors import TwoSlopeNorm

        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im = ax.imshow(mat, cmap=cmap, norm=norm)
    else:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(df.shape[1]))
    ax.set_yticks(range(df.shape[0]))
    ax.set_xticklabels(df.columns.tolist())
    ax.set_yticklabels(df.index.tolist())
    ax.set_title(title)

    # annotate
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white" if abs(mat[i, j]) > 0.5 else "black", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    baseline_p = _OUT_ROOT / "baseline" / "isrsa_matrix.csv"
    textalign_p = _OUT_ROOT / "textalign_llm" / "isrsa_matrix.csv"

    if not baseline_p.is_file():
        raise FileNotFoundError(f"Missing {baseline_p}")
    if not textalign_p.is_file():
        raise FileNotFoundError(f"Missing {textalign_p}")

    df_base = _read_matrix(baseline_p)
    df_ta = _read_matrix(textalign_p)

    # align indices
    df_ta = df_ta.reindex(index=df_base.index, columns=df_base.columns)
    df_delta = df_ta - df_base

    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    out1 = _FIG_DIR / "Fig_isrsa_heatmap_baseline.png"
    out2 = _FIG_DIR / "Fig_isrsa_heatmap_textalign_llm.png"
    out3 = _FIG_DIR / "Fig_isrsa_delta.png"

    _heatmap(df_base, out1, "IS-RSA (baseline)", vmin=0.0, vmax=1.0, cmap="magma")
    _heatmap(df_ta, out2, "IS-RSA (textalign_llm)", vmin=0.0, vmax=1.0, cmap="magma")

    # delta centered at 0
    vmax = float(np.nanmax(np.abs(df_delta.values)))
    vmax = max(vmax, 1e-6)
    _heatmap(df_delta, out3, "IS-RSA Δ (textalign_llm - baseline)", vmin=-vmax, vmax=vmax, cmap="coolwarm", center=0.0)

    figures = [
        {
            "name": "Fig_isrsa_heatmap_baseline.png",
            "title": "IS-RSA heatmap (baseline)",
            "what": "4×4 subject-by-subject Spearman RSA matrix for baseline embeddings.",
            "usage": "Supplement: cross-subject consistency of brain embeddings.",
        },
        {
            "name": "Fig_isrsa_heatmap_textalign_llm.png",
            "title": "IS-RSA heatmap (textalign_llm)",
            "what": "4×4 subject-by-subject Spearman RSA matrix for textalign_llm embeddings.",
            "usage": "Supplement: cross-subject consistency after TextAlign.",
        },
        {
            "name": "Fig_isrsa_delta.png",
            "title": "IS-RSA delta heatmap",
            "what": "Difference heatmap: textalign_llm - baseline.",
            "usage": "Supplement: show where inter-subject RSA improves/degrades.",
        },
    ]

    # Optional mean-cos delta
    base_cos_p = _OUT_ROOT / "baseline" / "mean_cos_matrix.csv"
    ta_cos_p = _OUT_ROOT / "textalign_llm" / "mean_cos_matrix.csv"
    if base_cos_p.is_file() and ta_cos_p.is_file():
        df_cos_base = _read_matrix(base_cos_p)
        df_cos_ta = _read_matrix(ta_cos_p).reindex(index=df_cos_base.index, columns=df_cos_base.columns)
        df_cos_delta = df_cos_ta - df_cos_base
        out4 = _FIG_DIR / "Fig_mean_cos_delta.png"
        vmax2 = float(np.nanmax(np.abs(df_cos_delta.values)))
        vmax2 = max(vmax2, 1e-6)
        _heatmap(df_cos_delta, out4, "MeanCos Δ (textalign_llm - baseline)", vmin=-vmax2, vmax=vmax2, cmap="coolwarm", center=0.0)
        figures.append(
            {
                "name": "Fig_mean_cos_delta.png",
                "title": "Image-wise mean cosine delta",
                "what": "Difference heatmap of mean image-wise cross-subject cosine (textalign_llm - baseline).",
                "usage": "Supplement: same-image embedding alignment across subjects.",
            }
        )

    manifest = {
        "out_dir": str(_FIG_DIR),
        "figures": figures,
    }
    (_FIG_DIR / "figures_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # report md
    lines = [
        "# shared982 IS-RSA figures report",
        "",
        "本文件汇总 IS-RSA 补充材料图表的含义与用途。",
        "",
    ]
    for f in figures:
        lines.extend(
            [
                f"## {f['name']}",
                "",
                f"- 标题：{f['title']}",
                f"- 内容：{f['what']}",
                f"- 用途：{f['usage']}",
                "",
                f"![](figures/{f['name']})",
                "",
            ]
        )

    (_FIG_DIR / "figures_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"[OK] wrote figures to: {_FIG_DIR}")


if __name__ == "__main__":
    main()
