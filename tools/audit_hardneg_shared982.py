#!/usr/bin/env python
# coding: utf-8
"""audit_hardneg_shared982.py

Negative Quality Audit for shared982 hard negatives.

Outputs (default under cache/model_eval_results/shared982_ccd/audit/):
- audit_tables.csv
- Fig_audit_len_words.png
- Fig_audit_len_chars.png
- Fig_audit_negation_rate.png
- Fig_audit_sim_text.png
- Fig_audit_type_coverage.png

The goal is to address reviewer concerns about hard negatives being hackable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJ = Path(__file__).resolve().parents[1]

NEGATION_WORDS = [
    "not",
    "no",
    "never",
    "without",
    "none",
    "nobody",
    "nothing",
    "nowhere",
    "neither",
]


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _has_negation(s: str) -> bool:
    s = s.lower()
    # whole-word match
    for w in NEGATION_WORDS:
        if re.search(rf"\\b{re.escape(w)}\\b", s):
            return True
    return False


def _word_count(s: str) -> int:
    return len([t for t in re.split(r"\s+", s.strip()) if t])


def _setup_style():
    import matplotlib as mpl

    mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})
    try:
        import scienceplots  # noqa: F401

        import matplotlib.pyplot as plt

        plt.style.use(["science", "no-latex", "grid"])
    except Exception:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")

    # Enforce minimum font size > 7pt
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hardneg_jsonl",
        default=str(PROJ / "cache" / "hardneg" / "shared982_hardneg.jsonl"),
        help="Full hardneg pool jsonl (may contain multiple candidates per image)",
    )
    ap.add_argument(
        "--hardneg_for_ccd",
        default=str(PROJ / "cache" / "hardneg" / "shared982_hardneg_for_ccd.jsonl"),
        help="One-per-image jsonl used by CCD (for per-image type coverage)",
    )
    ap.add_argument(
        "--hardneg_audit_json",
        default=str(PROJ / "cache" / "hardneg" / "shared982_hardneg_audit.json"),
        help="Audit meta json produced during hardneg generation",
    )
    ap.add_argument(
        "--out_dir",
        default=str(PROJ / "cache" / "model_eval_results" / "shared982_ccd" / "audit"),
    )
    args = ap.parse_args()

    hardneg_jsonl = Path(args.hardneg_jsonl).resolve()
    hardneg_for_ccd = Path(args.hardneg_for_ccd).resolve()
    audit_json = Path(args.hardneg_audit_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not hardneg_jsonl.is_file():
        raise RuntimeError(f"Missing {hardneg_jsonl}")

    rows = _read_jsonl(hardneg_jsonl)
    df = pd.DataFrame(rows)

    # Basic fields
    df["pos_caption"] = df["pos_caption"].astype(str)
    df["neg_caption"] = df["neg_caption"].astype(str)
    df["type"] = df.get("type", "unknown").astype(str).str.lower()

    df["pos_words"] = df["pos_caption"].map(_word_count)
    df["neg_words"] = df["neg_caption"].map(_word_count)
    df["pos_chars"] = df["pos_caption"].map(len)
    df["neg_chars"] = df["neg_caption"].map(len)

    df["pos_has_negation"] = df["pos_caption"].map(_has_negation)
    df["neg_has_negation"] = df["neg_caption"].map(_has_negation)

    negation_rate = float((df["pos_has_negation"] | df["neg_has_negation"]).mean())

    # sim_text stats
    if "sim_text" in df.columns:
        sim = df["sim_text"].astype(float).to_numpy()
    else:
        sim = np.asarray([], dtype=np.float32)

    sim_stats: Dict[str, float] = {}
    if sim.size:
        sim_stats = {
            "mean": float(sim.mean()),
            "std": float(sim.std()),
            "min": float(sim.min()),
            "p05": float(np.quantile(sim, 0.05)),
            "p50": float(np.quantile(sim, 0.50)),
            "p95": float(np.quantile(sim, 0.95)),
            "max": float(sim.max()),
        }

    # per-type coverage (candidates)
    type_counts = df["type"].value_counts(dropna=False).to_dict()
    type_image_counts = df.groupby("type")["image_id"].nunique().to_dict() if "image_id" in df.columns else {}

    # per-image chosen type coverage (from for_ccd jsonl)
    chosen_type_counts = {}
    chosen_image_counts = {}
    if hardneg_for_ccd.is_file():
        d2 = pd.DataFrame(_read_jsonl(hardneg_for_ccd))
        if not d2.empty and "type" in d2.columns:
            d2["type"] = d2["type"].astype(str).str.lower()
            chosen_type_counts = d2["type"].value_counts(dropna=False).to_dict()
            chosen_image_counts = d2.groupby("type")["image_id"].nunique().to_dict() if "image_id" in d2.columns else {}

    # audit json thresholds
    sim_low = None
    sim_high = None
    k_raw = None
    k_final = None
    if audit_json.is_file():
        j = json.loads(audit_json.read_text(encoding="utf-8"))
        params = j.get("params", {})
        sim_low = params.get("sim_low_init")
        sim_high = params.get("sim_high_init")
        k_raw = params.get("k_raw")
        k_final = params.get("k_final")

    tables = []
    tables.append({"section": "overall", "key": "n_rows", "value": int(df.shape[0])})
    tables.append({"section": "overall", "key": "n_unique_images", "value": int(df["image_id"].nunique()) if "image_id" in df.columns else ""})
    tables.append({"section": "text", "key": "negation_rate_any", "value": negation_rate})
    tables.append({"section": "params", "key": "k_raw", "value": k_raw if k_raw is not None else ""})
    tables.append({"section": "params", "key": "k_final", "value": k_final if k_final is not None else ""})
    tables.append({"section": "params", "key": "sim_low_init", "value": sim_low if sim_low is not None else ""})
    tables.append({"section": "params", "key": "sim_high_init", "value": sim_high if sim_high is not None else ""})
    for k, v in sim_stats.items():
        tables.append({"section": "sim_text", "key": k, "value": v})

    for t, c in sorted(type_counts.items(), key=lambda x: (-x[1], x[0])):
        tables.append({"section": "type_pool", "key": f"count_{t}", "value": int(c)})
    for t, c in sorted(type_image_counts.items(), key=lambda x: (-x[1], x[0])):
        tables.append({"section": "type_pool", "key": f"unique_images_{t}", "value": int(c)})

    for t, c in sorted(chosen_type_counts.items(), key=lambda x: (-x[1], x[0])):
        tables.append({"section": "type_for_ccd", "key": f"count_{t}", "value": int(c)})
    for t, c in sorted(chosen_image_counts.items(), key=lambda x: (-x[1], x[0])):
        tables.append({"section": "type_for_ccd", "key": f"unique_images_{t}", "value": int(c)})

    out_tables = out_dir / "audit_tables.csv"
    pd.DataFrame(tables).to_csv(out_tables, index=False)

    _setup_style()
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1) length distributions
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    sns.histplot(df["pos_words"], bins=30, stat="density", color=sns.color_palette("deep")[0], label="pos", ax=ax, alpha=0.55)
    sns.histplot(df["neg_words"], bins=30, stat="density", color=sns.color_palette("deep")[3], label="neg", ax=ax, alpha=0.55)
    ax.set_title("Caption length (words): pos vs neg")
    ax.set_xlabel("#words")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_audit_len_words.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    sns.histplot(df["pos_chars"], bins=30, stat="density", color=sns.color_palette("deep")[0], label="pos", ax=ax, alpha=0.55)
    sns.histplot(df["neg_chars"], bins=30, stat="density", color=sns.color_palette("deep")[3], label="neg", ax=ax, alpha=0.55)
    ax.set_title("Caption length (chars): pos vs neg")
    ax.set_xlabel("#chars")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_audit_len_chars.png")
    plt.close(fig)

    # 2) negation rate
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.2))
    vals = {
        "pos_has_negation": float(df["pos_has_negation"].mean()),
        "neg_has_negation": float(df["neg_has_negation"].mean()),
        "any": negation_rate,
    }
    ax.bar(list(vals.keys()), list(vals.values()), color=sns.color_palette("deep")[:3])
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Negation word rate (should be ~0)")
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_audit_negation_rate.png")
    plt.close(fig)

    # 3) sim_text distribution
    if sim.size:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
        sns.histplot(sim, bins=40, stat="density", color=sns.color_palette("deep")[2], ax=ax, alpha=0.6)
        ax.set_title("CLIP text similarity sim_text distribution")
        ax.set_xlabel("sim_text")
        ax.set_ylabel("density")
        if sim_low is not None and sim_high is not None:
            ax.axvline(float(sim_low), color="black", linestyle="--", linewidth=1, label="init low/high")
            ax.axvline(float(sim_high), color="black", linestyle="--", linewidth=1)
        if "p05" in sim_stats:
            ax.axvline(float(sim_stats["p05"]), color="gray", linestyle=":", linewidth=1, label="p05/p95")
            ax.axvline(float(sim_stats["p95"]), color="gray", linestyle=":", linewidth=1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "Fig_audit_sim_text.png")
        plt.close(fig)

    # 4) type coverage
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    d_type = pd.DataFrame(
        {
            "type": list(type_counts.keys()),
            "n_candidates": [type_counts[k] for k in type_counts.keys()],
            "n_unique_images": [type_image_counts.get(k, 0) for k in type_counts.keys()],
        }
    )
    d_type = d_type.sort_values("n_unique_images", ascending=False)
    ax.bar(d_type["type"], d_type["n_unique_images"], color=sns.color_palette("deep")[0], alpha=0.8, label="unique images")
    ax2 = ax.twinx()
    ax2.plot(d_type["type"], d_type["n_candidates"], color=sns.color_palette("deep")[3], marker="o", label="#candidates")
    ax.set_title("Type coverage in hardneg pool")
    ax.set_xlabel("type")
    ax.set_ylabel("#unique images")
    ax2.set_ylabel("#candidate rows")
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_audit_type_coverage.png")
    plt.close(fig)

    print(f"wrote: {out_tables}")
    print(f"wrote figures under: {out_dir}")


if __name__ == "__main__":
    main()
