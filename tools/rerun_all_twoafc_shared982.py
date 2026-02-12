#!/usr/bin/env python
# coding: utf-8
"""rerun_all_twoafc_shared982.py

批量在 shared982 protocol 上跑 2AFC（embedding 版），覆盖仓库里所有已导出 brain_tokens 的模型。

扫描：evals/brain_tokens/**
  - subj??_brain_clip_mean.pt
  - subj??_brain_clip_tokens.pt
  - subj??_ids.json

评测：
  - pooled (1664d)
  - tokens_flatten (256x1664)

输出：
  cache/model_eval_results/shared982_twoafc/<group>/<tag>/{pooled_mean|tokens_flatten}/metrics.json
  cache/model_eval_results/shared982_twoafc/twoafc_summary.csv
  cache/model_eval_results/shared982_twoafc/twoafc_summary.md

实时进度：脚本会增量写 CSV/MD，配合 tools/watch_twoafc_progress.py 或 `tail -f` 使用。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Row:
    group: str
    tag: str
    subj: int
    eval_repr: str
    N: int
    metric: str
    twoafc_fwd_mean: float
    twoafc_fwd_ci_lo: float
    twoafc_fwd_ci_hi: float
    twoafc_bwd_mean: float
    twoafc_bwd_ci_lo: float
    twoafc_bwd_ci_hi: float
    metrics_json: str


def _python() -> str:
    return sys.executable


def _run(cmd: List[str], env: Dict[str, str]) -> subprocess.CompletedProcess:
    print("[CMD]", " ".join(cmd))
    return subprocess.run(cmd, env=env, text=True, capture_output=True)


def _read_metrics(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_group_tag(root: Path, mean_path: Path) -> Tuple[str, str]:
    rel = mean_path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return parts[0], parts[0]
    raise RuntimeError(f"Unexpected brain_tokens path: {mean_path}")


def _parse_subj_from_filename(name: str) -> int:
    m = re.search(r"subj(\d+)_brain_clip_mean\.pt$", name)
    if not m:
        raise RuntimeError(f"Cannot parse subject from: {name}")
    return int(m.group(1))


def _collect_row(group: str, tag: str, subj: int, eval_repr: str, mj: Path) -> Row:
    d = _read_metrics(mj)
    two = d["twoafc"]
    fwd = two["brain_to_image"]
    bwd = two["image_to_brain"]
    return Row(
        group=group,
        tag=tag,
        subj=subj,
        eval_repr=eval_repr,
        N=int(d["N"]),
        metric=str(d.get("metric", "")),
        twoafc_fwd_mean=float(fwd["mean"]),
        twoafc_fwd_ci_lo=float(fwd["ci95"][0]),
        twoafc_fwd_ci_hi=float(fwd["ci95"][1]),
        twoafc_bwd_mean=float(bwd["mean"]),
        twoafc_bwd_ci_lo=float(bwd["ci95"][0]),
        twoafc_bwd_ci_hi=float(bwd["ci95"][1]),
        metrics_json=str(mj),
    )


def _write_csv(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "group",
                "tag",
                "subj",
                "eval_repr",
                "N",
                "metric",
                "twoafc_fwd_mean",
                "twoafc_fwd_ci_lo",
                "twoafc_fwd_ci_hi",
                "twoafc_bwd_mean",
                "twoafc_bwd_ci_lo",
                "twoafc_bwd_ci_hi",
                "metrics_json",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.group,
                    r.tag,
                    f"{r.subj:02d}",
                    r.eval_repr,
                    r.N,
                    r.metric,
                    f"{r.twoafc_fwd_mean:.6f}",
                    f"{r.twoafc_fwd_ci_lo:.6f}",
                    f"{r.twoafc_fwd_ci_hi:.6f}",
                    f"{r.twoafc_bwd_mean:.6f}",
                    f"{r.twoafc_bwd_ci_lo:.6f}",
                    f"{r.twoafc_bwd_ci_hi:.6f}",
                    r.metrics_json,
                ]
            )


def _write_md(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def score(r: Row) -> float:
        return 0.5 * (r.twoafc_fwd_mean + r.twoafc_bwd_mean)

    pooled = [r for r in rows if r.eval_repr == "pooled"]
    pooled_sorted = sorted(pooled, key=lambda r: (r.group, r.subj, -score(r)))

    lines: List[str] = []
    lines.append("# 2AFC results on shared982 (our eval pipeline)")
    lines.append("")
    lines.append(f"CSV: {path.with_suffix('.csv')}")
    lines.append("")
    lines.append("## Best pooled per (group, subj)")
    lines.append("")
    lines.append("| group | subj | best tag | N | metric | 2AFC B→I | 95% CI | 2AFC I→B | 95% CI | metrics.json |")
    lines.append("|---|---:|---|---:|---|---:|---|---:|---|---|")

    best: Dict[Tuple[str, int], Row] = {}
    for r in pooled_sorted:
        k = (r.group, r.subj)
        if k not in best:
            best[k] = r

    for (group, subj), r in sorted(best.items(), key=lambda x: (x[0][0], x[0][1])):
        lines.append(
            "| "
            + f"{group} | {subj:02d} | {r.tag} | {r.N} | {r.metric} | {r.twoafc_fwd_mean:.3f} | [{r.twoafc_fwd_ci_lo:.3f}, {r.twoafc_fwd_ci_hi:.3f}] | {r.twoafc_bwd_mean:.3f} | [{r.twoafc_bwd_ci_lo:.3f}, {r.twoafc_bwd_ci_hi:.3f}] | {r.metrics_json} |"
        )

    lines.append("")
    lines.append("## Full results")
    lines.append("")
    lines.append("| group | subj | eval_repr | tag | N | metric | 2AFC B→I | 95% CI | 2AFC I→B | 95% CI | metrics.json |")
    lines.append("|---|---:|---|---|---:|---|---:|---|---:|---|---|")

    rows_sorted = sorted(rows, key=lambda r: (r.group, r.subj, r.eval_repr, -score(r) if r.eval_repr == "pooled" else 0.0))
    for r in rows_sorted:
        lines.append(
            "| "
            + f"{r.group} | {r.subj:02d} | {r.eval_repr} | {r.tag} | {r.N} | {r.metric} | {r.twoafc_fwd_mean:.3f} | [{r.twoafc_fwd_ci_lo:.3f}, {r.twoafc_fwd_ci_hi:.3f}] | {r.twoafc_bwd_mean:.3f} | [{r.twoafc_bwd_ci_lo:.3f}, {r.twoafc_bwd_ci_hi:.3f}] | {r.metrics_json} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--subset", default="shared982")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "pearson"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--groups", default="", help="Comma-separated brain_tokens groups to include (default all)")
    args = ap.parse_args()

    proj_root = Path(__file__).resolve().parents[1]
    brain_tokens_root = proj_root / "evals" / "brain_tokens"
    eval_py = proj_root / "tools" / "eval_twoafc_embed.py"

    gt_path = proj_root / "evals" / "all_images.pt"
    if not gt_path.exists():
        gt_path = proj_root / "src" / "evals" / "all_images.pt"

    out_root = proj_root / "cache" / "model_eval_results" / "shared982_twoafc"
    out_csv = out_root / "twoafc_summary.csv"
    out_md = out_root / "twoafc_summary.md"

    allow_groups = None
    if args.groups.strip():
        allow_groups = {x.strip() for x in args.groups.split(",") if x.strip()}

    mean_files = sorted(brain_tokens_root.rglob("subj*_brain_clip_mean.pt"))
    if allow_groups is not None:
        mean_files = [p for p in mean_files if p.relative_to(brain_tokens_root).parts[0] in allow_groups]

    rows: List[Row] = []
    failures: List[str] = []

    for mean_path in mean_files:
        subj = _parse_subj_from_filename(mean_path.name)
        group, tag = _infer_group_tag(brain_tokens_root, mean_path)
        base_dir = mean_path.parent

        tokens_path = base_dir / f"subj{subj:02d}_brain_clip_tokens.pt"
        ids_path = base_dir / f"subj{subj:02d}_ids.json"

        if not ids_path.exists():
            failures.append(f"[MISS] ids missing: {ids_path}")
            continue

        # pooled
        pooled_dir = out_root / group / tag / "pooled_mean"
        pooled_mj = pooled_dir / "metrics.json"
        if (not args.skip_existing) or (not pooled_mj.exists()):
            env = os.environ.copy()
            env.update(
                {
                    "EXP_NAME": f"{group}_{tag}_subj{subj:02d}",
                    "BRAIN_PATH": str(mean_path),
                    "IDS_PATH": str(ids_path),
                    "GT_PATH": str(gt_path),
                    "EVAL_REPR": "pooled",
                    "EVAL_SUBSET": args.subset,
                    "METRIC": args.metric,
                    "BOOTSTRAP": str(args.bootstrap),
                    "GT_POOLING": "mean",
                    "RESULT_DIR": str(pooled_dir),
                }
            )
            cp = _run([_python(), str(eval_py)], env=env)
            if cp.returncode != 0:
                err = (cp.stderr or cp.stdout or "").strip()
                failures.append(
                    f"[FAIL][pooled] group={group} tag={tag} subj={subj:02d}\n" + "\n".join(err.splitlines()[-40:])
                )
        if pooled_mj.exists():
            rows.append(_collect_row(group, tag, subj, "pooled", pooled_mj))

        # tokens_flatten
        if not tokens_path.exists():
            failures.append(f"[MISS] tokens missing: {tokens_path}")
        else:
            tok_dir = out_root / group / tag / "tokens_flatten"
            tok_mj = tok_dir / "metrics.json"
            if (not args.skip_existing) or (not tok_mj.exists()):
                env = os.environ.copy()
                env.update(
                    {
                        "EXP_NAME": f"{group}_{tag}_subj{subj:02d}",
                        "BRAIN_PATH": str(tokens_path),
                        "IDS_PATH": str(ids_path),
                        "GT_PATH": str(gt_path),
                        "EVAL_REPR": "tokens_flatten",
                        "EVAL_SUBSET": args.subset,
                        "METRIC": args.metric,
                        "BOOTSTRAP": str(args.bootstrap),
                        "GT_POOLING": "mean",
                        "RESULT_DIR": str(tok_dir),
                    }
                )
                cp = _run([_python(), str(eval_py)], env=env)
                if cp.returncode != 0:
                    err = (cp.stderr or cp.stdout or "").strip()
                    failures.append(
                        f"[FAIL][tokens_flatten] group={group} tag={tag} subj={subj:02d}\n" + "\n".join(err.splitlines()[-40:])
                    )
            if tok_mj.exists():
                rows.append(_collect_row(group, tag, subj, "tokens_flatten", tok_mj))

        _write_csv(out_csv, rows)
        _write_md(out_md, rows)

    _write_csv(out_csv, rows)
    _write_md(out_md, rows)

    if failures:
        fail_path = out_root / "twoafc_summary.failures.txt"
        fail_path.write_text("\n\n".join(failures) + "\n", encoding="utf-8")
        print("[DONE] wrote failures:", fail_path)

    print("[DONE] wrote", out_csv)
    print("[DONE] wrote", out_md)


if __name__ == "__main__":
    main()
