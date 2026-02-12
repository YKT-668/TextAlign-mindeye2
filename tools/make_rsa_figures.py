#!/usr/bin/env python
# coding: utf-8
"""Generate RSA figures (fast, many, publication-friendly) and a manifest.

Outputs under:
  cache/model_eval_results/shared982_rsa/figures/
  - fig01_rsa_pearson_pooled_by_subj.png
  - fig02_rsa_pearson_tokensmean_by_subj.png
  - fig03_rsa_best_per_group_subj_pooled.png
  - fig04_rsa_best_per_group_subj_tokensmean.png
  - fig05_rsa_vs_2afc_scatter_pooled.png
    - fig06+_rsm_heatmaps_subjXX_best_overall_pooled.png
    - fig??_rsm_heatmaps_subjXX_best_officialhf_pooled.png
  - figures_manifest.json

Notes:
- Uses RSA summary CSV.
- Optionally uses 2AFC summary CSV for scatter.
- Heatmaps use a downsampled subset (default 200) for readability and speed.

Env vars:
- RSA_DIR: root dir containing rsa_summary.csv (default: cache/model_eval_results/shared982_rsa)
- TWOAFC_CSV: path to twoafc_summary.csv (default: cache/model_eval_results/shared982_twoafc/twoafc_summary.csv)
- HEATMAP_N: number of stimuli to plot in heatmaps (default: 200)
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FigureItem:
    fig_id: int
    file: str  # relative to rsa_dir
    title: str
    what: str
    comment: str
    usage: str
    mapping: Optional[List[Dict[str, str]]] = None


def _safe_import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    # Academic-ish defaults (best effort; falls back gracefully)
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex", "grid"])  # type: ignore
    except Exception:
        pass

    if sns is not None:
        try:
            sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
        except Exception:
            pass

    plt.rcParams.update(
        {
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        }
    )

    return plt, sns


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _ensure_cols(rows: List[Dict[str, str]], cols: List[str], name: str):
    if not rows:
        raise RuntimeError(f"{name} is empty")
    keys = set(rows[0].keys())
    missing = [c for c in cols if c not in keys]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}; got={sorted(keys)}")


def _subj_order(rows: List[Dict[str, str]]) -> List[str]:
    s = sorted({str(r.get("subj", "")) for r in rows if str(r.get("subj", "")) != ""})
    # prefer 01,02,05,07 order
    pref = ["01", "02", "05", "07"]
    out = [p for p in pref if p in s]
    out += [x for x in s if x not in out]
    return out


def _model_label(row: Dict[str, str]) -> str:
    # Compact label: group:tag
    g = str(row.get("group", ""))
    t = str(row.get("tag", ""))
    if g == t or t == "":
        return g
    return f"{g}:{t}"


def _pick_best(rows: List[Dict[str, str]], eval_repr: str) -> List[Dict[str, str]]:
    best: Dict[str, Dict[str, str]] = {}
    for r in rows:
        if str(r.get("eval_repr")) != eval_repr:
            continue
        key = f"{r.get('group','')}|{r.get('subj','')}|{eval_repr}"
        v = float(r.get("rsa_pearson", "nan"))
        if key not in best or v > float(best[key].get("rsa_pearson", "nan")):
            best[key] = r
    out = list(best.values())
    out.sort(key=lambda x: (x.get("group", ""), x.get("subj", ""), x.get("eval_repr", ""), -float(x.get("rsa_pearson", "0"))))
    return out


def _barplot_by_subj(
    rows: List[Dict[str, str]],
    eval_repr: str,
    out_path: Path,
    title: str,
    *,
    layout: str = "auto",
    letterize_labels: bool = False,
    alphabet: str | None = None,
):
    plt, sns = _safe_import_plotting()
    d = [r for r in rows if str(r.get("eval_repr")) == eval_repr]
    if not d:
        return

    subjects = _subj_order(d)

    # Decide whether to use compact A/B/C... labels.
    raw_labels_all: List[str] = [_model_label(r) for r in d]
    if not letterize_labels and raw_labels_all:
        max_len = max(len(s) for s in raw_labels_all)
        if max_len >= 14 or len(set(raw_labels_all)) >= 8:
            letterize_labels = True

    label_to_letter: Dict[str, str] = {}
    mapping_lines: List[str] = []
    if letterize_labels:
        uniq = sorted(set(raw_labels_all))
        if not alphabet:
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, lab in enumerate(uniq):
            letter = alphabet[i] if i < len(alphabet) else f"{alphabet[0]}{i}"
            label_to_letter[lab] = letter
        mapping_lines = [f"{label_to_letter[lab]} = {lab}" for lab in uniq]

    # Layout: for 4 subjects, prefer a clean 2x2.
    if layout == "grid2x2":
        ncols = 2
        nrows = 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14.0, 9.2), squeeze=False, sharey=True)
    else:
        ncols = 2
        nrows = int(np.ceil(len(subjects) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14.5, 4.6 * nrows), squeeze=False, sharey=True)

    for idx, subj in enumerate(subjects):
        ax = axes[idx // ncols][idx % ncols]
        ds = [r for r in d if str(r.get("subj", "")) == subj]
        ds.sort(key=lambda r: float(r.get("rsa_pearson", "0")), reverse=True)
        x = np.arange(len(ds))
        y = np.array([float(r.get("rsa_pearson", "nan")) for r in ds], dtype=float)
        ci_low = np.array([float(r.get("ci95_low_p", "nan")) for r in ds], dtype=float)
        ci_high = np.array([float(r.get("ci95_high_p", "nan")) for r in ds], dtype=float)
        yerr_low = y - ci_low
        yerr_high = ci_high - y
        color = "#4C78A8" if sns is None else sns.color_palette("deep")[0]
        ax.bar(x, y, color=color)
        ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", elinewidth=1, capsize=2)
        ax.set_xticks(x)
        if letterize_labels:
            xt = [label_to_letter[_model_label(r)] for r in ds]
            ax.set_xticklabels(xt, rotation=0, ha="center", fontsize=10)
        else:
            ax.set_xticklabels([_model_label(r) for r in ds], rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0.0, max(0.45, float(np.nanmax(y)) + 0.05))
        ax.set_title(f"subj{subj} | {eval_repr}")
        ax.set_ylabel("RSA (pearson)")
        ax.grid(True, axis="y", alpha=0.25)

    # hide unused axes
    for j in range(len(subjects), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=14)
    if letterize_labels and mapping_lines:
        chunks: List[str] = []
        line = []
        for item in mapping_lines:
            candidate = ("  |  ".join(line + [item])) if line else item
            if len(candidate) > 145:
                chunks.append("  |  ".join(line))
                line = [item]
            else:
                line.append(item)
        if line:
            chunks.append("  |  ".join(line))
        fig.text(0.5, 0.02, "\n".join(chunks), ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _barplot_best(
    rows: List[Dict[str, str]],
    eval_repr: str,
    out_path: Path,
    title: str,
    *,
    layout: str = "row",
    letterize_labels: bool = False,
    bar_color: str | None = None,
):
    plt, sns = _safe_import_plotting()
    d = _pick_best(rows, eval_repr)
    if not d:
        return

    subjects = _subj_order(d)

    # Decide whether to use compact A/B/C... labels.
    # We letterize when requested (e.g., for paper-quality multi-panel layouts)
    # or when labels are likely to overlap.
    raw_labels_all: List[str] = []
    for r in d:
        raw_labels_all.append(_model_label(r))
    if not letterize_labels:
        if raw_labels_all:
            max_len = max(len(s) for s in raw_labels_all)
            if max_len >= 14 or len(set(raw_labels_all)) >= 8:
                letterize_labels = True

    label_to_letter: Dict[str, str] = {}
    mapping_lines: List[str] = []
    if letterize_labels:
        uniq = sorted(set(raw_labels_all))
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, lab in enumerate(uniq):
            if i < len(alphabet):
                letter = alphabet[i]
            else:
                # Fallback if there are many labels.
                letter = f"A{i}"
            label_to_letter[lab] = letter
        # Build compact mapping text for the bottom of the figure.
        mapping_lines = [f"{label_to_letter[lab]} = {lab}" for lab in uniq]

    # Layout
    if layout == "grid2x2":
        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11.6, 8.2), squeeze=False, sharey=True)
        flat_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    else:
        fig, axes = plt.subplots(nrows=1, ncols=len(subjects), figsize=(5.9 * len(subjects), 4.8), squeeze=False, sharey=True)
        flat_axes = [axes[0][i] for i in range(len(subjects))]

    # Consistent y-limits across panels.
    all_y = np.array([float(r.get("rsa_pearson", "nan")) for r in d], dtype=float)
    y_top = float(np.nanmax(all_y)) if np.isfinite(np.nanmax(all_y)) else 0.45
    y_top = max(0.45, y_top + 0.05)

    for i, subj in enumerate(subjects[: len(flat_axes)]):
        ax = flat_axes[i]
        ds = [r for r in d if str(r.get("subj", "")) == subj]
        ds.sort(key=lambda r: float(r.get("rsa_pearson", "0")), reverse=True)
        x = np.arange(len(ds))
        y = np.array([float(r.get("rsa_pearson", "nan")) for r in ds], dtype=float)
        ci_low = np.array([float(r.get("ci95_low_p", "nan")) for r in ds], dtype=float)
        ci_high = np.array([float(r.get("ci95_high_p", "nan")) for r in ds], dtype=float)
        yerr_low = y - ci_low
        yerr_high = ci_high - y
        if bar_color is not None:
            color = bar_color
        else:
            # Old default was green; orange tends to pair better with the repo's default blue.
            color = "#F28E2B" if sns is None else sns.color_palette("deep")[1]
        ax.bar(x, y, color=color)
        ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", elinewidth=1, capsize=2)
        ax.set_xticks(x)
        if letterize_labels:
            xt = [label_to_letter[_model_label(r)] for r in ds]
            ax.set_xticklabels(xt, rotation=0, ha="center", fontsize=10)
        else:
            ax.set_xticklabels([_model_label(r) for r in ds], rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0.0, y_top)
        ax.set_title(f"Best per group | subj{subj}")
        ax.set_ylabel("RSA (pearson)")
        ax.grid(True, axis="y", alpha=0.25)

    # Hide unused panels for grid layout.
    for j in range(len(subjects), len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    if letterize_labels and mapping_lines:
        # Put the legend text at the bottom spanning the whole figure.
        # Use multiple lines to keep it readable.
        chunks: List[str] = []
        line = []
        for item in mapping_lines:
            # Wrap roughly by character count; keep simple and robust.
            candidate = ("  |  ".join(line + [item])) if line else item
            if len(candidate) > 130:
                chunks.append("  |  ".join(line))
                line = [item]
            else:
                line.append(item)
        if line:
            chunks.append("  |  ".join(line))
        legend_text = "\n".join(chunks)
        fig.text(0.5, 0.02, legend_text, ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _scatter_rsa_vs_2afc(rows_rsa: List[Dict[str, str]], twoafc_csv: Path, out_path: Path, title: str):
    if not twoafc_csv.is_file():
        return
    plt, sns = _safe_import_plotting()

    rows_2 = _read_csv_rows(twoafc_csv)
    # twoafc_summary.csv in this repo uses snake_case columns.
    # Prefer forward direction mean as the B→I analogue.
    b2i_key_candidates = [
        "2AFC B→I",
        "2AFC B->I",
        "twoafc_b2i",
        "twoafc_b2i_mean",
        "twoafc_fwd_mean",
        "twoafc_fwd",
    ]
    b2i_key = None
    if rows_2:
        keys = set(rows_2[0].keys())
        for k in b2i_key_candidates:
            if k in keys:
                b2i_key = k
                break
    if b2i_key is None:
        # Can't make this plot; silently skip.
        return
    _ensure_cols(rows_2, ["group", "subj", "eval_repr", "tag", b2i_key], "twoafc")

    rsa_pooled = [r for r in rows_rsa if str(r.get("eval_repr")) == "pooled"]
    two_pooled = [r for r in rows_2 if str(r.get("eval_repr")) == "pooled"]

    # index twoafc by (group,subj,tag)
    idx: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for r in two_pooled:
        idx[(str(r.get("group", "")), str(r.get("subj", "")), str(r.get("tag", "")))] = r

    merged: List[Tuple[Dict[str, str], Dict[str, str]]] = []
    for r in rsa_pooled:
        k = (str(r.get("group", "")), str(r.get("subj", "")), str(r.get("tag", "")))
        if k in idx:
            merged.append((r, idx[k]))

    if not merged:
        return

    x = np.array([float(a.get("rsa_pearson", "nan")) for a, _ in merged], dtype=float)
    y = np.array([float(b.get(b2i_key, "nan")) for _, b in merged], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    # --- Colorize overlapping points (no leader lines) ---
    # We detect “close” points in display space (pixels) and assign distinct colors
    # within each connected component, so overlapping regions remain distinguishable.
    pts = [(float(a.get("rsa_pearson", "nan")), float(b.get(b2i_key, "nan"))) for a, b in merged]
    # build mapping list (also used for report table)
    mapping: List[Dict[str, str]] = []
    for i, (ra, rb) in enumerate(merged, start=1):
        mapping.append(
            {
                "id": str(i),
                "subj": str(ra.get("subj", "")),
                "group": str(ra.get("group", "")),
                "tag": str(ra.get("tag", "")),
                "model": _model_label(ra),
                "rsa_pearson": str(ra.get("rsa_pearson", "")),
                "twoafc": str(rb.get(b2i_key, "")),
            }
        )

    # Initialize axes limits early for stable transforms
    ax.set_xlabel("RSA (pearson, pooled)")
    ax.set_ylabel(f"2AFC B→I (pooled) [{b2i_key}]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(max(0.0, float(np.min(x) - 0.02)), min(1.0, float(np.max(x) + 0.02)))
    ax.set_ylim(max(0.45, float(np.min(y) - 0.02)), min(1.0, float(np.max(y) + 0.02)))

    # After limits are set, transform points into display coords
    disp = ax.transData.transform(np.asarray(pts, dtype=float))  # (N,2) pixels
    n = disp.shape[0]
    close_px = 18.0
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(disp[i, 0] - disp[j, 0])
            dy = float(disp[i, 1] - disp[j, 1])
            if dx * dx + dy * dy <= close_px * close_px:
                adj[i][j] = True
                adj[j][i] = True

    # connected components
    comp_id = [-1] * n
    comps: List[List[int]] = []
    for i in range(n):
        if comp_id[i] != -1:
            continue
        stack = [i]
        comp_id[i] = len(comps)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if adj[u][v] and comp_id[v] == -1:
                    comp_id[v] = comp_id[i]
                    stack.append(v)
        comps.append(sorted(comp))

    # color palette
    if sns is not None:
        palette = list(sns.color_palette("tab10", 10))
    else:
        import matplotlib

        palette = list(matplotlib.colormaps["tab10"].colors)

    point_colors: List = []
    for i in range(n):
        c = comps[comp_id[i]]
        if len(c) <= 1:
            point_colors.append((0.2, 0.2, 0.2, 0.85))
        else:
            # distinct color within component
            k = c.index(i)
            point_colors.append((*palette[k % len(palette)], 0.9))

    ax.scatter(x, y, s=60, c=point_colors, edgecolors="white", linewidths=0.6, zorder=2)

    # --- Place numeric labels around points without overlaps ---
    # Greedy placement using candidate offsets; no leader lines to keep the plot clean.
    import matplotlib.patheffects as pe

    candidates = [
        (0, 12),
        (12, 0),
        (0, -12),
        (-12, 0),
        (10, 10),
        (-10, 10),
        (10, -10),
        (-10, -10),
        (0, 16),
        (16, 0),
        (0, -16),
        (-16, 0),
    ]

    # label crowded components first
    order = list(range(n))
    order.sort(key=lambda i: (-len(comps[comp_id[i]]), i))

    placed_bboxes = []
    renderer = None
    for idx in order:
        xi, yi = pts[idx]
        label = str(idx + 1)
        col = point_colors[idx]

        # Manual tweak: put label "6" below its point for readability.
        cand = candidates
        if (idx + 1) == 6:
            cand = [
                (0, -14),
                (0, -18),
                (10, -10),
                (-10, -10),
                (12, 0),
                (-12, 0),
                (0, 12),
                (10, 10),
                (-10, 10),
            ]

        best = None
        best_overlap = None
        best_text = None

        for (dx, dy) in cand:
            t = ax.annotate(
                label,
                xy=(xi, yi),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                ha="center",
                va="center",
                color=col,
                zorder=3,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white", alpha=0.95)],
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer() if renderer is None else renderer
            bb = t.get_window_extent(renderer=renderer).expanded(1.08, 1.12)

            overlap = 0.0
            for ob in placed_bboxes:
                ix0 = max(bb.x0, ob.x0)
                iy0 = max(bb.y0, ob.y0)
                ix1 = min(bb.x1, ob.x1)
                iy1 = min(bb.y1, ob.y1)
                if ix1 > ix0 and iy1 > iy0:
                    overlap += (ix1 - ix0) * (iy1 - iy0)

            if best_overlap is None or overlap < best_overlap:
                # keep this candidate
                if best_text is not None:
                    best_text.remove()
                best_overlap = overlap
                best = bb
                best_text = t
                if overlap == 0.0:
                    break
            else:
                t.remove()

        if best is not None:
            placed_bboxes.append(best)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return mapping


def _load_feat(path: Path) -> np.ndarray:
    import torch

    x = torch.load(path, map_location="cpu")
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        return x.numpy()
    raise RuntimeError(f"Unsupported tensor in {path}")


def _cosine_sim(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    return a @ a.T


def _heatmaps_best_pair(
    rsa_dir: Path,
    rows_rsa: List[Dict[str, str]],
    subj: str,
    out_path: Path,
    heatmap_n: int,
    *,
    group_filter: Optional[str] = None,
    title: Optional[str] = None,
):
    # Pick best pooled model for subject by RSA(pearson), optionally within a group.
    d = [
        r
        for r in rows_rsa
        if str(r.get("subj", "")) == str(subj)
        and str(r.get("eval_repr", "")) == "pooled"
        and (group_filter is None or str(r.get("group", "")) == group_filter)
    ]
    if not d:
        return None
    d.sort(key=lambda r: float(r.get("rsa_pearson", "0")), reverse=True)
    d = d[0]

    metrics_path = Path(str(d["metrics.json"]))
    # reconstruct brain feature path from metrics.json location: .../<group>/<tag>/pooled_mean/metrics.json
    pooled_dir = metrics_path.parent
    tag_dir = pooled_dir.parent
    group_dir = tag_dir.parent
    group = group_dir.name
    tag = tag_dir.name

    # Find original exported brain mean feature
    brain_export_dir = Path(rsa_dir).parents[2] / "evals" / "brain_tokens" / group
    brain_path = None
    # Try group/tag or group
    cand = list((brain_export_dir / tag).glob("*_brain_clip_mean.pt"))
    if cand:
        brain_path = cand[0]
    else:
        cand = list(brain_export_dir.glob("*_brain_clip_mean.pt"))
        if cand:
            brain_path = cand[0]
    if brain_path is None:
        return None

    # Load GT pooled features in the aligned order for this brain set
    # We can reuse eval_rsa_embed alignment by reading its metrics.json? For speed, we approximate:
    # The RSA eval aligned GT to brain ids internally; here we visualize using the same aligned order
    # by recomputing on ids.json.
    ids_path = None
    cand_ids = list((brain_path.parent).glob("*_ids.json"))
    if cand_ids:
        ids_path = cand_ids[0]

    # Load brain features + ids
    brain = _load_feat(brain_path)
    if brain.ndim == 3:
        brain = brain.mean(axis=1)

    ids = None
    if ids_path is not None and ids_path.is_file():
        ids = np.asarray(json.loads(ids_path.read_text(encoding="utf-8")), dtype=np.int64)
    else:
        ids = np.arange(brain.shape[0], dtype=np.int64)

    # Subset shared982
    shared982_mask = np.load(Path(rsa_dir).parents[2] / "src" / "shared982.npy")
    subset_ids = np.where(shared982_mask > 0)[0].astype(np.int64)
    keep = np.isin(ids, subset_ids)
    ids = ids[keep]
    brain = brain[keep]

    # Downsample deterministic subset for readability
    n = ids.shape[0]
    k = min(int(heatmap_n), n)
    # choose evenly-spaced indices
    idx = np.linspace(0, n - 1, k).round().astype(int)
    ids_small = ids[idx]
    brain_small = brain[idx]

    # GT pooled cache
    gt_path = Path(rsa_dir).parents[2] / "evals" / "all_images_bigG_1664_mean.pt"
    gt = _load_feat(gt_path)
    # IMPORTANT: GT cache rows correspond to shared1000 id order (not 0..N-1).
    gt_ids = None
    shared1000_path = Path(rsa_dir).parents[2] / "src" / "shared1000.npy"
    if shared1000_path.is_file():
        m1000 = np.load(shared1000_path)
        if m1000.ndim == 1 and m1000.dtype == np.bool_:
            shared_ids = np.where(m1000 > 0)[0].astype(np.int64)
            if int(shared_ids.shape[0]) == int(gt.shape[0]):
                gt_ids = shared_ids
            elif int(shared_ids.shape[0]) > int(gt.shape[0]):
                gt_ids = shared_ids[: int(gt.shape[0])]
    if gt_ids is None:
        gt_ids = np.arange(gt.shape[0], dtype=np.int64)
    id2row = {int(i): j for j, i in enumerate(gt_ids)}
    rows = np.asarray([id2row[int(i)] for i in ids_small], dtype=np.int64)
    gt_small = gt[rows]

    s_gt = _cosine_sim(gt_small)
    s_br = _cosine_sim(brain_small)
    s_diff = s_br - s_gt

    plt, sns = _safe_import_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0))
    vmin, vmax = -0.2, 1.0
    im0 = axes[0].imshow(s_gt, vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
    axes[0].set_title("GT similarity (cosine)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(s_br, vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
    axes[1].set_title("Brain similarity (cosine)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    im2 = axes[2].imshow(s_diff, vmin=-0.5, vmax=0.5, cmap="coolwarm", interpolation="nearest")
    axes[2].set_title("Brain - GT")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    if title is None:
        title = "RSM heatmaps"
    fig.suptitle(f"{title}\n(downsampled {k}/{n}) | model={group}:{tag}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "group": group,
        "tag": tag,
        "downsample_k": int(k),
        "total_n": int(n),
    }


def main() -> None:
    proj = Path(__file__).resolve().parents[1]
    rsa_dir = Path(os.environ.get("RSA_DIR", proj / "cache" / "model_eval_results" / "shared982_rsa")).resolve()
    rsa_csv = rsa_dir / "rsa_summary.csv"
    fig_dir = rsa_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    twoafc_csv = Path(os.environ.get("TWOAFC_CSV", proj / "cache" / "model_eval_results" / "shared982_twoafc" / "twoafc_summary.csv")).resolve()
    heatmap_n = int(os.environ.get("HEATMAP_N", "200"))

    rows = _read_csv_rows(rsa_csv)
    _ensure_cols(
        rows,
        [
            "group",
            "subj",
            "eval_repr",
            "tag",
            "rsa_pearson",
            "ci95_low_p",
            "ci95_high_p",
            "rsa_spearman",
            "ci95_low",
            "ci95_high",
            "metrics.json",
        ],
        "rsa",
    )

    figures: List[FigureItem] = []

    # Fig 01 pooled by subject
    f1 = fig_dir / "fig01_rsa_pearson_pooled_by_subj.png"
    # Use compact letter labels (requested: replace subscripts/indices with letters)
    # Start with the custom sequence "acbd", then continue with the remaining lowercase letters.
    _custom_alpha = "acbd" + "efghijklmnopqrstuvwxyz"
    _barplot_by_subj(
        rows,
        "pooled",
        f1,
        "RSA (PRIMARY=Pearson) by subject | pooled",
        layout="grid2x2",
        letterize_labels=True,
        alphabet=_custom_alpha,
    )
    figures.append(
        FigureItem(
            1,
            str(f1.relative_to(rsa_dir)),
            "RSA(pearson) 分布（pooled，各被试分面）",
            "每个被试单独一张子图；横轴是模型（group:tag），纵轴是 RSA(pearson)；误差条是 95% CI。",
            "最直观的主文图候选：能一眼看出 1sess→40sess、official→ours 的提升趋势。",
            "建议放主文或主文补充：用于展示整体趋势与被试差异。",
        )
    )

    # Fig 02 tokens_mean by subject
    f2 = fig_dir / "fig02_rsa_pearson_tokensmean_by_subj.png"
    _barplot_by_subj(
        rows,
        "tokens_mean",
        f2,
        "RSA (PRIMARY=Pearson) by subject | tokens_mean",
        layout="grid2x2",
        letterize_labels=True,
    )
    figures.append(
        FigureItem(
            2,
            str(f2.relative_to(rsa_dir)),
            "RSA(pearson) 分布（tokens_mean，各被试分面）",
            "tokens_flatten 导出在 RSA 里会先对 tokens 做 mean pooling（每图一个向量），再做 RSA；其余同 Figure 1。",
            "通常与 pooled 结果高度一致（因为都是 stimulus-level 向量），用于证明结论不依赖特定表征导出。",
            "建议放补充材料：作为 pooled 版本的稳健性对照。",
        )
    )

    # Fig 03 best per group/subj pooled
    f3 = fig_dir / "fig03_rsa_best_per_group_subj_pooled.png"
    _barplot_best(
        rows,
        "pooled",
        f3,
        "Best RSA(pearson) per group within each subject | pooled",
        layout="grid2x2",
        letterize_labels=True,
        bar_color="#F28E2B",
    )
    figures.append(
        FigureItem(
            3,
            str(f3.relative_to(rsa_dir)),
            "每个被试：各 group 取 best（pooled）",
            "对每个 (group,subj) 只保留 RSA(pearson) 最高的 tag，得到更紧凑的对比图。",
            "更适合论文主文：减少标签数量，读者更容易看出我们方法在每个被试上的优势。",
            "建议主文优先：如果 Figure 1 太拥挤，用这个替代。",
        )
    )

    # Fig 04 best per group/subj tokens_mean
    f4 = fig_dir / "fig04_rsa_best_per_group_subj_tokensmean.png"
    _barplot_best(
        rows,
        "tokens_mean",
        f4,
        "Best RSA(pearson) per group within each subject | tokens_mean",
        layout="grid2x2",
        letterize_labels=True,
        bar_color="#F28E2B",
    )
    figures.append(
        FigureItem(
            4,
            str(f4.relative_to(rsa_dir)),
            "每个被试：各 group 取 best（tokens_mean）",
            "同 Figure 3，但用 tokens_mean 口径。",
            "用于展示 best 的结论在 tokens_mean 下依旧成立。",
            "建议补充材料：作为 Figure 3 的稳健性对照。",
        )
    )

    # Fig 05 scatter RSA vs 2AFC
    f5 = fig_dir / "fig05_rsa_vs_2afc_scatter_pooled.png"
    mapping_5 = _scatter_rsa_vs_2afc(rows, twoafc_csv, f5, "RSA(pearson, pooled) vs 2AFC B→I (pooled)")
    figures.append(
        FigureItem(
            5,
            str(f5.relative_to(rsa_dir)),
            "RSA vs 2AFC 的一致性（pooled）",
            "把每个模型的 RSA(pearson, pooled) 与同一模型的 2AFC B→I (pooled) 画成散点图；为避免长标签遮挡，图中每个点只标编号。",
            "如果点云呈单调关系，叙事更强：表示几何对齐（RSA）与识别性能（2AFC）一致提升。",
            "建议放补充材料或方法分析部分；编号→模型的对应表写在报告里便于查阅。",
            mapping=mapping_5,
        )
    )

    # Fig 06+ heatmaps: for each subject, generate (A) best overall pooled and (B) best official_hf pooled
    fig_id = 6
    subjects = _subj_order(rows)
    for subj in subjects:
        # A) best overall
        f = fig_dir / f"fig{fig_id:02d}_rsm_heatmaps_subj{subj}_best_overall_pooled.png"
        meta = _heatmaps_best_pair(
            rsa_dir,
            rows,
            subj,
            f,
            heatmap_n,
            group_filter=None,
            title=f"RSM heatmaps | subj{subj} | best overall (pooled)",
        )
        if meta is not None:
            tag_note = f"（best={meta['group']}:{meta['tag']}，downsample={meta['downsample_k']}/{meta['total_n']}）"
            figures.append(
                FigureItem(
                    fig_id,
                    str(f.relative_to(rsa_dir)),
                    f"RSM 热力图：GT vs Brain（subj{subj}，best overall pooled）{tag_note}",
                    f"三联图：GT 相似度矩阵、Brain 相似度矩阵、差分（Brain-GT）。为可视化与速度，使用等距抽样的 {heatmap_n} 个刺激子集。",
                    "补充主指标的定性证据：直观看模型是否复现 GT 的块结构/相对关系。",
                    "建议补充材料：与 Figure 1/3 的定量趋势配套展示。",
                )
            )
            fig_id += 1

        # B) best official_hf baseline
        f = fig_dir / f"fig{fig_id:02d}_rsm_heatmaps_subj{subj}_best_officialhf_pooled.png"
        meta = _heatmaps_best_pair(
            rsa_dir,
            rows,
            subj,
            f,
            heatmap_n,
            group_filter="official_hf",
            title=f"RSM heatmaps | subj{subj} | best official_hf (pooled)",
        )
        if meta is not None:
            tag_note = f"（best_official_hf={meta['group']}:{meta['tag']}，downsample={meta['downsample_k']}/{meta['total_n']}）"
            figures.append(
                FigureItem(
                    fig_id,
                    str(f.relative_to(rsa_dir)),
                    f"RSM 热力图：GT vs Brain（subj{subj}，best official_hf pooled）{tag_note}",
                    f"同样的三联图口径，但模型限定在 official_hf 组内取 best，便于与 best overall 做定性对照。抽样 {heatmap_n} 个刺激。",
                    "用于展示基线在几何结构上的典型误差模式，帮助解释我们方法为何提升 RSA。",
                    "建议补充材料：与对应 subj 的 best overall 热力图成对阅读。",
                )
            )
            fig_id += 1

    manifest = {
        "rsa_dir": str(rsa_dir),
        "generated": True,
        "figures": [asdict(f) for f in figures],
        "notes": [
            "PRIMARY score in report is RSA(pearson); Spearman is retained as robustness.",
            "Heatmap uses downsampled subset for readability; quantitative RSA uses full N=982.",
        ],
    }

    (fig_dir / "figures_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[DONE] wrote {fig_dir / 'figures_manifest.json'}")


if __name__ == "__main__":
    main()
