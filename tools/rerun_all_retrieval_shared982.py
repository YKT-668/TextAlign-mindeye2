#!/usr/bin/env python
# coding: utf-8
"""rerun_all_retrieval_shared982.py

目的：把仓库里之前“已经导出 brain_tokens 的所有模型”统一按 shared982 protocol
重新跑双向检索（FWD/BWD top-k），并汇总成 CSV/MD。

覆盖范围：扫描 evals/brain_tokens/** 下的
  - subj??_brain_clip_mean.pt  (pooled)
  - subj??_brain_clip_tokens.pt (tokens_flatten)
  - subj??_ids.json

输出：
  cache/model_eval_results/shared982/<group>/<tag>/{pooled_mean|tokens_flatten}/metrics.json
  cache/model_eval_results/shared982/retrieval_summary.csv
  cache/model_eval_results/shared982/retrieval_summary.md

备注：shared982 定义来自 WDS test split 的 982 unique image indices。
本脚本依赖 tools/eval_textalign_latent_plus.py 的 EVAL_SUBSET=shared982 支持。
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
    fwd_top1: float
    fwd_top5: float
    bwd_top1: float
    bwd_top5: float
    rand_fwd_top1_mean: Optional[float]
    rand_fwd_top1_std: Optional[float]
    rand_bwd_top1_mean: Optional[float]
    rand_bwd_top1_std: Optional[float]
    metrics_json: str


def _python() -> str:
    return sys.executable


def _run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    print("[CMD]", " ".join(cmd))
    return subprocess.run(cmd, env=env, text=True, capture_output=True)


def _read_metrics(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dig(d: Dict, *path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _infer_group_tag(root: Path, mean_path: Path) -> Tuple[str, str]:
    rel = mean_path.relative_to(root)
    parts = rel.parts
    # layouts:
    #  - <group>/subj01_brain_clip_mean.pt                 => tag=<group>
    #  - <group>/<tag>/subj01_brain_clip_mean.pt           => tag=<tag>
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

    def req(x, name):
        if x is None:
            raise RuntimeError(f"Missing key {name} in {mj}")
        return x

    rand_fwd = _dig(d, "random_retrieval", "fwd")
    rand_bwd = _dig(d, "random_retrieval", "bwd")

    return Row(
        group=group,
        tag=tag,
        subj=subj,
        eval_repr=eval_repr,
        N=int(req(d.get("N"), "N")),
        fwd_top1=float(req(_dig(d, "retrieval", "fwd", "top1"), "retrieval.fwd.top1")),
        fwd_top5=float(req(_dig(d, "retrieval", "fwd", "top5"), "retrieval.fwd.top5")),
        bwd_top1=float(req(_dig(d, "retrieval", "bwd", "top1"), "retrieval.bwd.top1")),
        bwd_top5=float(req(_dig(d, "retrieval", "bwd", "top5"), "retrieval.bwd.top5")),
        rand_fwd_top1_mean=float(rand_fwd["top1_mean"]) if rand_fwd else None,
        rand_fwd_top1_std=float(rand_fwd["top1_std"]) if rand_fwd else None,
        rand_bwd_top1_mean=float(rand_bwd["top1_mean"]) if rand_bwd else None,
        rand_bwd_top1_std=float(rand_bwd["top1_std"]) if rand_bwd else None,
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
                "fwd_top1",
                "fwd_top5",
                "bwd_top1",
                "bwd_top5",
                "rand_fwd_top1_mean",
                "rand_fwd_top1_std",
                "rand_bwd_top1_mean",
                "rand_bwd_top1_std",
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
                    f"{r.fwd_top1:.6f}",
                    f"{r.fwd_top5:.6f}",
                    f"{r.bwd_top1:.6f}",
                    f"{r.bwd_top5:.6f}",
                    "" if r.rand_fwd_top1_mean is None else f"{r.rand_fwd_top1_mean:.6f}",
                    "" if r.rand_fwd_top1_std is None else f"{r.rand_fwd_top1_std:.6f}",
                    "" if r.rand_bwd_top1_mean is None else f"{r.rand_bwd_top1_mean:.6f}",
                    "" if r.rand_bwd_top1_std is None else f"{r.rand_bwd_top1_std:.6f}",
                    r.metrics_json,
                ]
            )


def _write_md(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def score(r: Row) -> float:
        return 0.5 * (r.fwd_top1 + r.bwd_top1)

    pooled = [r for r in rows if r.eval_repr == "pooled"]
    pooled_sorted = sorted(pooled, key=lambda r: (r.subj, -score(r), -0.5 * (r.fwd_top5 + r.bwd_top5)))

    lines: List[str] = []
    lines.append("# Retrieval results on shared982 (our eval pipeline)")
    lines.append("")
    lines.append(f"CSV: {path.with_suffix('.csv')}")
    lines.append("")
    lines.append("## Best pooled per (group, subj)")
    lines.append("")
    lines.append("| group | subj | best tag | N | FWD@1 | FWD@5 | BWD@1 | BWD@5 | metrics.json |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---|")

    best_by_key: Dict[Tuple[str, int], Row] = {}
    for r in pooled_sorted:
        k = (r.group, r.subj)
        if k not in best_by_key:
            best_by_key[k] = r

    for (group, subj), r in sorted(best_by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        lines.append(
            "| "
            + f"{group} | {subj:02d} | {r.tag} | {r.N} | {r.fwd_top1:.2f} | {r.fwd_top5:.2f} | {r.bwd_top1:.2f} | {r.bwd_top5:.2f} | {r.metrics_json} |"
        )

    lines.append("")
    lines.append("## Full results")
    lines.append("")
    lines.append("| group | subj | eval_repr | tag | N | FWD@1 | FWD@5 | BWD@1 | BWD@5 | 300-wayx30 FWD@1 (mean±std) | 300-wayx30 BWD@1 (mean±std) | metrics.json |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

    def fmt_ms(mean: Optional[float], std: Optional[float]) -> str:
        if mean is None or std is None:
            return ""
        return f"{mean:.2f}±{std:.2f}"

    # Sort for readability
    rows_sorted = sorted(rows, key=lambda r: (r.group, r.subj, r.eval_repr, -score(r) if r.eval_repr == "pooled" else 0.0))
    for r in rows_sorted:
        lines.append(
            "| "
            + f"{r.group} | {r.subj:02d} | {r.eval_repr} | {r.tag} | {r.N} | {r.fwd_top1:.2f} | {r.fwd_top5:.2f} | {r.bwd_top1:.2f} | {r.bwd_top5:.2f} | "
            + f"{fmt_ms(r.rand_fwd_top1_mean, r.rand_fwd_top1_std)} | {fmt_ms(r.rand_bwd_top1_mean, r.rand_bwd_top1_std)} | {r.metrics_json} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rand_k", default="300")
    ap.add_argument("--rand_trials", default="30")
    ap.add_argument("--rand_seed", default="0")
    ap.add_argument(
        "--subset",
        default="shared982",
        help="EVAL_SUBSET passed to eval script (default shared982)",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip if metrics.json already exists in the target result dir",
    )
    ap.add_argument(
        "--groups",
        default="",
        help="Comma-separated brain_tokens groups to include (default: all)",
    )
    args = ap.parse_args()

    proj_root = Path(__file__).resolve().parents[1]
    brain_tokens_root = proj_root / "evals" / "brain_tokens"
    eval_py = proj_root / "tools" / "eval_textalign_latent_plus.py"
    gt_path = proj_root / "evals" / "all_images.pt"
    if not gt_path.exists():
        gt_path = proj_root / "src" / "evals" / "all_images.pt"

    out_root = proj_root / "cache" / "model_eval_results" / "shared982"
    out_csv = out_root / "retrieval_summary.csv"
    out_md = out_root / "retrieval_summary.md"

    allow_groups = None
    if args.groups.strip():
        allow_groups = {x.strip() for x in args.groups.split(",") if x.strip()}

    mean_files = sorted(brain_tokens_root.rglob("subj*_brain_clip_mean.pt"))
    if allow_groups is not None:
        mean_files = [p for p in mean_files if p.relative_to(brain_tokens_root).parts[0] in allow_groups]

    if not mean_files:
        raise SystemExit(f"No mean exports found under: {brain_tokens_root}")

    failures: List[str] = []
    rows: List[Row] = []

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
                    "GT_POOLING": "mean",
                    "RAND_K": str(args.rand_k),
                    "RAND_TRIALS": str(args.rand_trials),
                    "RAND_SEED": str(args.rand_seed),
                    "EVAL_SUBSET": str(args.subset),
                    "RESULT_DIR": str(pooled_dir),
                }
            )
            cp = _run([_python(), str(eval_py)], env=env)
            if cp.returncode != 0:
                err = (cp.stderr or cp.stdout or "").strip()
                failures.append(
                    f"[FAIL][pooled] group={group} tag={tag} subj={subj:02d}\n" + "\n".join(err.splitlines()[-30:])
                )
        if pooled_mj.exists():
            rows.append(_collect_row(group, tag, subj, "pooled", pooled_mj))

        # tokens_flatten
        if not tokens_path.exists():
            failures.append(f"[MISS] tokens missing: {tokens_path}")
            continue

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
                    "GT_POOLING": "mean",
                    "RAND_K": str(args.rand_k),
                    "RAND_TRIALS": str(args.rand_trials),
                    "RAND_SEED": str(args.rand_seed),
                    "EVAL_SUBSET": str(args.subset),
                    "RESULT_DIR": str(tok_dir),
                }
            )
            cp = _run([_python(), str(eval_py)], env=env)
            if cp.returncode != 0:
                err = (cp.stderr or cp.stdout or "").strip()
                failures.append(
                    f"[FAIL][tokens_flatten] group={group} tag={tag} subj={subj:02d}\n" + "\n".join(err.splitlines()[-30:])
                )
        if tok_mj.exists():
            rows.append(_collect_row(group, tag, subj, "tokens_flatten", tok_mj))

        # incremental write
        _write_csv(out_csv, rows)
        _write_md(out_md, rows)

    _write_csv(out_csv, rows)
    _write_md(out_md, rows)

    if failures:
        fail_path = out_root / "retrieval_summary.failures.txt"
        fail_path.write_text("\n\n".join(failures) + "\n", encoding="utf-8")
        print("[DONE] wrote failures:", fail_path)

    print("[DONE] wrote", out_csv)
    print("[DONE] wrote", out_md)


if __name__ == "__main__":
    main()
