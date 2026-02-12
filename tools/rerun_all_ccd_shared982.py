#!/usr/bin/env python
# coding: utf-8
"""Batch CCD over all exported models under evals/brain_tokens/**.

Outputs:
- cache/model_eval_results/shared982_ccd/ccd_summary.csv
- cache/model_eval_results/shared982_ccd/ccd_summary.md
- per model: cache/model_eval_results/shared982_ccd/<group>/<tag>/<eval_repr>/metrics.json

Realtime progress:
- summary CSV/MD is updated incrementally; use watcher to tail changes.

Notes
- CCD evaluates caption discrimination using OpenCLIP ViT-bigG-14 joint embedding (1280).
- Brain exports are typically 1664-d visual token width; we apply OpenCLIP visual.proj
  (1664->1280) as a fixed mapping (no training on eval data).
- If hard negatives are not provided, we sample K negatives uniformly from the same
  evaluation set (shared982) with a fixed RNG seed.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJ = Path(__file__).resolve().parents[1]
BRAIN_ROOT = PROJ / "evals" / "brain_tokens"
OUT_ROOT = PROJ / "cache" / "model_eval_results" / "shared982_ccd"
EVAL_SUBSET = "shared982"

CAPTIONS_PATH = Path(os.environ.get("CCD_CAPTIONS_PATH", str(PROJ / "evals" / "all_captions.pt"))).resolve()

CLIP_MODEL = os.environ.get("CCD_CLIP_MODEL", "ViT-bigG-14").strip()
CLIP_PRETRAINED = os.environ.get("CCD_CLIP_PRETRAINED", "laion2b_s39b_b160k").strip()
K_NEG = int(os.environ.get("CCD_K_NEG", "31"))
SEED = int(os.environ.get("CCD_SEED", "42"))
# default to 1000 to match paper-style CI requirements
BOOTSTRAP = int(os.environ.get("CCD_BOOTSTRAP", "1000"))

# Optional hard-negative sources (if present)
HARD_NEG_JSONL = os.environ.get("CCD_HARD_NEG_JSONL", "").strip() or None
HARD_NEG_PT = os.environ.get("CCD_HARD_NEG_PT", "").strip() or None

# Optional: extra keep mask applied after shared982 filtering (must be len=982 for shared982)
EVAL_KEEP_MASK_NPY = os.environ.get("CCD_EVAL_KEEP_MASK_NPY", "").strip() or None

# Optional override for shared assets dir (caption/hardneg embedding caches)
ASSETS_DIR_OVERRIDE = os.environ.get("CCD_ASSETS_DIR", "").strip() or None

ASSETS_DIR = Path(ASSETS_DIR_OVERRIDE).resolve() if ASSETS_DIR_OVERRIDE else (OUT_ROOT / "assets")  # shared caches


def _reload_from_env() -> None:
    """Refresh module-level config from environment.

    Needed because this script historically read env vars at import time.
    When using CLI flags, we update os.environ and then call this to keep
    the runtime config consistent.
    """

    global OUT_ROOT, ASSETS_DIR, CAPTIONS_PATH
    global K_NEG, SEED, BOOTSTRAP
    global HARD_NEG_JSONL, HARD_NEG_PT, EVAL_KEEP_MASK_NPY

    # output dirs
    out_dir = os.environ.get("CCD_DIR", "").strip()
    if out_dir:
        OUT_ROOT = Path(out_dir).resolve()

    assets_override = os.environ.get("CCD_ASSETS_DIR", "").strip() or None
    ASSETS_DIR = Path(assets_override).resolve() if assets_override else (OUT_ROOT / "assets")

    CAPTIONS_PATH = Path(os.environ.get("CCD_CAPTIONS_PATH", str(PROJ / "evals" / "all_captions.pt"))).resolve()

    # core knobs
    K_NEG = int(os.environ.get("CCD_K_NEG", "31"))
    SEED = int(os.environ.get("CCD_SEED", "42"))
    BOOTSTRAP = int(os.environ.get("CCD_BOOTSTRAP", "1000"))

    HARD_NEG_JSONL = os.environ.get("CCD_HARD_NEG_JSONL", "").strip() or None
    HARD_NEG_PT = os.environ.get("CCD_HARD_NEG_PT", "").strip() or None
    EVAL_KEEP_MASK_NPY = os.environ.get("CCD_EVAL_KEEP_MASK_NPY", "").strip() or None


@dataclass
class Job:
    group: str
    tag: str
    subj: str
    eval_repr: str
    brain_path: Path
    ids_path: Optional[Path]


def _find_ids_path(folder: Path) -> Optional[Path]:
    candidates = sorted(folder.glob("*_ids.json"))
    if candidates:
        return candidates[0]
    p = folder / "ids.json"
    return p if p.is_file() else None


def _infer_subj_from_path(p: Path) -> str:
    m = re.search(r"subj(\d{2})", p.name)
    if m:
        return m.group(1)
    m = re.search(r"subj(\d{2})", str(p.parent))
    if m:
        return m.group(1)
    # also handle s1 style in folder/tag names
    m = re.search(r"(?:^|_)s(\d)(?:_|$)", "_" + p.parent.name + "_")
    if m:
        return m.group(1).zfill(2)
    return ""


def _discover_jobs() -> List[Job]:
    jobs: List[Job] = []
    if not BRAIN_ROOT.is_dir():
        raise RuntimeError(f"Missing {BRAIN_ROOT}")

    for group_dir in sorted(BRAIN_ROOT.iterdir()):
        if not group_dir.is_dir():
            continue
        group = group_dir.name

        # Case A: files directly under group_dir (tag == group)
        ids_path_root = _find_ids_path(group_dir)
        pooled_root = sorted(group_dir.glob("*_brain_clip_mean.pt"))
        tokens_root = sorted(group_dir.glob("*_brain_clip_tokens.pt"))

        if pooled_root:
            subj = _infer_subj_from_path(pooled_root[0])
            jobs.append(Job(group, group, subj, "pooled", pooled_root[0], ids_path_root))
        if tokens_root:
            subj = _infer_subj_from_path(tokens_root[0])
            jobs.append(Job(group, group, subj, "tokens_flatten", tokens_root[0], ids_path_root))

        for tag_dir in sorted(group_dir.iterdir()):
            if not tag_dir.is_dir():
                continue
            tag = tag_dir.name
            ids_path = _find_ids_path(tag_dir)

            pooled_candidates = sorted(tag_dir.glob("*_brain_clip_mean.pt"))
            token_candidates = sorted(tag_dir.glob("*_brain_clip_tokens.pt"))

            if pooled_candidates:
                subj = _infer_subj_from_path(pooled_candidates[0])
                jobs.append(Job(group, tag, subj, "pooled", pooled_candidates[0], ids_path))
            if token_candidates:
                subj = _infer_subj_from_path(token_candidates[0])
                jobs.append(Job(group, tag, subj, "tokens_flatten", token_candidates[0], ids_path))

    return jobs


def _read_metrics(metrics_path: Path) -> Dict:
    import json

    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _run_job(job: Job) -> Tuple[bool, str, Path]:
    out_dir = OUT_ROOT / job.group / job.tag / ("pooled_mean" if job.eval_repr == "pooled" else "tokens_mean")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"

    desired_neg_mode = "hardneg" if (HARD_NEG_JSONL or HARD_NEG_PT) else "sampled"
    if metrics_path.is_file():
        try:
            m = _read_metrics(metrics_path)
            if (
                str(m.get("subset", "")) == EVAL_SUBSET
                and str(m.get("neg_mode", "")) == desired_neg_mode
                and int(m.get("k_neg", -1)) == int(1 if desired_neg_mode == "hardneg" else K_NEG)
                and str(m.get("clip_model", "")) == CLIP_MODEL
                and str(m.get("clip_pretrained", "")) == CLIP_PRETRAINED
                and str(m.get("captions_path", "")) == str(CAPTIONS_PATH)
                and int(m.get("bootstrap", -1)) == int(BOOTSTRAP)
                and (desired_neg_mode != "hardneg" or str(m.get("hard_neg_jsonl", "")) == str(HARD_NEG_JSONL or ""))
                and (not EVAL_KEEP_MASK_NPY or str(m.get("eval_keep_mask_npy", "")) == str(EVAL_KEEP_MASK_NPY))
                and ("twoafc_hardest" in (m.get("metrics", {}) or {}))
                and ("twoafc_hardest_acc_ci95" in (m.get("ci", {}) or {}))
            ):
                return True, "reuse", metrics_path
        except Exception:
            pass

    env = os.environ.copy()
    env.update(
        {
            "BRAIN_PATH": str(job.brain_path),
            "EVAL_SUBSET": EVAL_SUBSET,
            "CAPTIONS_PATH": str(CAPTIONS_PATH),
            "K_NEG": str(K_NEG),
            "SEED": str(SEED),
            "CLIP_MODEL": CLIP_MODEL,
            "CLIP_PRETRAINED": CLIP_PRETRAINED,
            "RESULT_DIR": str(out_dir),
            "ASSETS_DIR": str(ASSETS_DIR),
            "EXP_NAME": f"ccd_{job.group}_{job.tag}_{job.eval_repr}",
            "BOOTSTRAP": str(BOOTSTRAP),
        }
    )
    if job.ids_path is not None:
        env["IDS_PATH"] = str(job.ids_path)
    if HARD_NEG_JSONL:
        env["HARD_NEG_JSONL"] = HARD_NEG_JSONL
    if HARD_NEG_PT:
        env["HARD_NEG_PT"] = HARD_NEG_PT
    if EVAL_KEEP_MASK_NPY:
        env["EVAL_KEEP_MASK_NPY"] = EVAL_KEEP_MASK_NPY

    # In hardneg mode, forbid silent fallback to sampled negatives.
    if HARD_NEG_JSONL or HARD_NEG_PT:
        env.setdefault("HARD_NEG_REQUIRE_FULL", "1")

    # Optional: hardneg ablations
    if os.environ.get("HARD_NEG_K"):
        env["HARD_NEG_K"] = os.environ.get("HARD_NEG_K")
    if os.environ.get("HARD_NEG_REQUIRE_FULL"):
        env["HARD_NEG_REQUIRE_FULL"] = os.environ.get("HARD_NEG_REQUIRE_FULL")

    cmd = ["python", str(PROJ / "tools" / "eval_ccd_embed.py")]

    try:
        subprocess.check_output(cmd, cwd=str(PROJ), env=env, stderr=subprocess.STDOUT)
        return True, "ok", metrics_path
    except subprocess.CalledProcessError as e:
        msg = e.output.decode("utf-8", errors="replace")[-4000:]
        (out_dir / "error.log").write_text(msg + "\n", encoding="utf-8")
        return False, "failed", out_dir / "error.log"


def _write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "group",
        "subj",
        "eval_repr",
        "tag",
        "n_eval",
        "neg_mode",
        "k_neg",
        "seed",
        "bootstrap",
        "clip_pretrained",
        "ccd_acc1",
        "ccd_acc1_ci95_lo",
        "ccd_acc1_ci95_hi",
        "ccd_acc5",
        "mrr",
        "mean_rank",
        "median_rank",
        "margin_mean",
        "margin_mean_ci95_lo",
        "margin_mean_ci95_hi",
        "margin_median",
        "margin_std",
        "twoafc_hardest",
        "twoafc_hardest_ci95_lo",
        "twoafc_hardest_ci95_hi",
        "runtime_sec",
        "metrics.json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _write_md(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def norm_subj(x) -> str:
        s = str(x) if x is not None else ""
        return s.zfill(2) if s.isdigit() else s

    def md_table(rs: List[Dict]) -> str:
        header = (
            "| group | subj | eval_repr | tag | N | neg_mode | K | CCD@1 | CCD@1 CI95 | 2AFC-hardest | margin_mean | margin CI95 | metrics.json |\n"
            "|---|---:|---|---|---:|---|---:|---:|---|---:|---:|---|---|\n"
        )
        lines = [header]
        for r in rs:
            lo = r.get("ccd_acc1_ci95_lo", "")
            hi = r.get("ccd_acc1_ci95_hi", "")
            acc_ci = "" if (lo == "" or hi == "") else f"[{float(lo):.4f}, {float(hi):.4f}]"
            mlo = r.get("margin_mean_ci95_lo", "")
            mhi = r.get("margin_mean_ci95_hi", "")
            margin_ci = "" if (mlo == "" or mhi == "") else f"[{float(mlo):.4f}, {float(mhi):.4f}]"
            lines.append(
                "| {group} | {subj} | {eval_repr} | {tag} | {n_eval} | {neg_mode} | {k_neg} | {ccd_acc1:.4f} | {acc_ci} | {twoafc_hardest:.4f} | {margin_mean:.4f} | {margin_ci} | {metrics_path} |\n".format(
                    group=r["group"],
                    subj=norm_subj(r.get("subj", "")),
                    eval_repr=r["eval_repr"],
                    tag=r["tag"],
                    n_eval=int(r["n_eval"]),
                    neg_mode=r["neg_mode"],
                    k_neg=int(r["k_neg"]),
                    ccd_acc1=float(r["ccd_acc1"]),
                    acc_ci=acc_ci,
                    twoafc_hardest=float(r.get("twoafc_hardest", float("nan"))),
                    margin_mean=float(r["margin_mean"]),
                    margin_ci=margin_ci,
                    metrics_path=r["metrics.json"],
                )
            )
        return "".join(lines)

    # best per (group, subj, eval_repr) by acc@1
    best: Dict[Tuple[str, str, str], Dict] = {}
    for r in rows:
        key = (r.get("group", ""), norm_subj(r.get("subj", "")), r.get("eval_repr", ""))
        if key not in best or float(r["ccd_acc1"]) > float(best[key]["ccd_acc1"]):
            best[key] = r

    best_rows = list(best.values())
    best_rows.sort(key=lambda x: (x.get("group", ""), norm_subj(x.get("subj", "")), x.get("eval_repr", ""), -float(x.get("ccd_acc1", 0.0))))

    md: List[str] = []
    md.append("# CCD results on shared982 (our eval pipeline)\n\n")
    md.append(f"CSV: {OUT_ROOT / 'ccd_summary.csv'}\n\n")
    md.append("## Method notes\n\n")
    md.append(
        "- Task: caption discrimination (K+1-way). For each image, rank the positive caption against K negative captions.\n"
        "- Positive captions: `evals/all_captions.pt` aligned with exported `*_ids.json` order (1000 shared images).\n"
        "- Subset: shared982 (WDS test split unique images).\n"
        f"- Text encoder: OpenCLIP {CLIP_MODEL} ({CLIP_PRETRAINED}).\n"
        "- Brain feature mapping: exported brain features are typically 1664-d (visual token width). We apply the fixed OpenCLIP `visual.proj` (1664→1280) to obtain joint image-text space, then cosine similarity.\n"
        "- Negatives: if hard negatives are provided (optional), CCD uses 2AFC (pos vs hard-neg). Otherwise, negatives are uniformly sampled within the evaluation set with a fixed RNG seed (reported).\n\n"
    )

    md.append("## Best per (group, subj, eval_repr)\n\n")
    md.append(md_table(best_rows))

    md.append("\n## Full results\n\n")
    md.append(md_table(rows))

    # Figures (optional, generated by tools/make_ccd_figures.py)
    manifest_path = OUT_ROOT / "figures" / "figures_manifest.json"
    if manifest_path.is_file():
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            figs = manifest.get("figures", [])
            if figs:
                md.append("\n## Figures\n\n")
                md.append(
                    "说明：下方图片由脚本生成并保存在 `cache/model_eval_results/shared982_ccd/figures/`。\n"
                    "每张图都包含：图编号、内容说明、评价、使用建议。\n\n"
                )
                for f in figs:
                    fig_id = int(f.get("fig_id"))
                    file_rel = str(f.get("file"))
                    title = str(f.get("title", ""))
                    what = str(f.get("what", ""))
                    comment = str(f.get("comment", ""))
                    usage = str(f.get("usage", ""))
                    md.append(f"### Figure {fig_id}: {title}\n\n")
                    md.append(f"- 内容：{what}\n")
                    md.append(f"- 评价：{comment}\n")
                    md.append(f"- 建议：{usage}\n\n")
                    md.append(f"![]({file_rel})\n\n")
        except Exception as e:
            md.append("\n## Figures\n\n")
            md.append(f"[WARN] Figures manifest failed to parse: {type(e).__name__}: {e}\n")

    path.write_text("".join(md), encoding="utf-8")


def main() -> None:
    # Ensure we honor any CLI-provided env overrides (this module reads env at import time).
    _reload_from_env()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    def norm_subj(x) -> str:
        s = str(x) if x is not None else ""
        s = "" if s.lower() == "nan" else s
        return s.zfill(2) if s.isdigit() else s

    jobs = _discover_jobs()
    if not jobs:
        raise RuntimeError(f"No jobs found under {BRAIN_ROOT}")

    rows: List[Dict] = []
    csv_path = OUT_ROOT / "ccd_summary.csv"
    md_path = OUT_ROOT / "ccd_summary.md"

    # If previous CSV exists, load it and keep rows to support incremental reruns
    if csv_path.is_file():
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            rows = df.to_dict(orient="records")
        except Exception:
            rows = []

    # index existing rows to avoid duplicates
    existing = {}
    for r in rows:
        r = dict(r)
        r["subj"] = norm_subj(r.get("subj", ""))
        key = (r.get("group", ""), r.get("tag", ""), r.get("eval_repr", ""), r.get("subj", ""))
        existing[key] = r

    for idx, job in enumerate(jobs, start=1):
        ok, status, p = _run_job(job)
        print(f"[{idx}/{len(jobs)}] {job.group}/{job.tag} subj={job.subj} repr={job.eval_repr}: {status}")
        if not ok:
            continue

        m = _read_metrics(p)
        met = m.get("metrics", {})
        ci = m.get("ci", {}) or {}

        acc_ci = ci.get("acc1_ci95") or ["", ""]
        margin_ci = ci.get("margin_mean_ci95") or ["", ""]
        twoafc_ci = ci.get("twoafc_hardest_acc_ci95") or ["", ""]

        row = {
            "group": job.group,
            "tag": job.tag,
            "subj": norm_subj(job.subj),
            "eval_repr": "pooled_mean" if job.eval_repr == "pooled" else "tokens_mean",
            "n_eval": int(m.get("n_eval", 0)),
            "neg_mode": str(m.get("neg_mode", "")),
            "k_neg": int(m.get("k_neg", 0)),
            "seed": int(m.get("seed", 0)),
            "bootstrap": int(m.get("bootstrap", 0)),
            "clip_pretrained": str(m.get("clip_pretrained", "")),
            "ccd_acc1": float(met.get("ccd_acc1", float("nan"))),
            "ccd_acc1_ci95_lo": float(acc_ci[0]) if len(acc_ci) == 2 and acc_ci[0] != "" else "",
            "ccd_acc1_ci95_hi": float(acc_ci[1]) if len(acc_ci) == 2 and acc_ci[1] != "" else "",
            "ccd_acc5": float(met.get("ccd_acc5", float("nan"))),
            "mrr": float(met.get("mrr", float("nan"))),
            "mean_rank": float(met.get("mean_rank", float("nan"))),
            "median_rank": float(met.get("median_rank", float("nan"))),
            "margin_mean": float(met.get("margin_mean", float("nan"))),
            "margin_mean_ci95_lo": float(margin_ci[0]) if len(margin_ci) == 2 and margin_ci[0] != "" else "",
            "margin_mean_ci95_hi": float(margin_ci[1]) if len(margin_ci) == 2 and margin_ci[1] != "" else "",
            "margin_median": float(met.get("margin_median", float("nan"))),
            "margin_std": float(met.get("margin_std", float("nan"))),
            "twoafc_hardest": float(met.get("twoafc_hardest", float("nan"))),
            "twoafc_hardest_ci95_lo": float(twoafc_ci[0]) if len(twoafc_ci) == 2 and twoafc_ci[0] != "" else "",
            "twoafc_hardest_ci95_hi": float(twoafc_ci[1]) if len(twoafc_ci) == 2 and twoafc_ci[1] != "" else "",
            "runtime_sec": float(m.get("runtime_sec", float("nan"))),
            "metrics.json": str(p.relative_to(PROJ)),
        }

        key = (row["group"], row["tag"], row["eval_repr"], row.get("subj", ""))
        existing[key] = row

        rows = list(existing.values())
        rows.sort(key=lambda x: (x.get("group", ""), str(x.get("subj", "")), x.get("eval_repr", ""), x.get("tag", "")))

        _write_csv(rows, csv_path)
        _write_md(rows, md_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch CCD over evals/brain_tokens/**")
    ap.add_argument("--neg_jsonl", default=None, help="Hard negative jsonl (enables neg_mode=hardneg)")
    ap.add_argument("--use_valid_mask", default=None, help="Path to 1D bool npy mask applied after shared982 filtering")
    ap.add_argument("--bootstrap", type=int, default=None, help="Number of bootstrap resamples (default 1000)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--k_neg", type=int, default=None, help="Number of sampled negatives when neg_mode=sampled")
    ap.add_argument("--out_dir", default=None, help="Override output directory")
    args = ap.parse_args()

    if args.neg_jsonl:
        os.environ["CCD_HARD_NEG_JSONL"] = str(args.neg_jsonl)
    if args.use_valid_mask:
        os.environ["CCD_EVAL_KEEP_MASK_NPY"] = str(args.use_valid_mask)
    if args.bootstrap is not None:
        os.environ["CCD_BOOTSTRAP"] = str(int(args.bootstrap))
    if args.seed is not None:
        os.environ["CCD_SEED"] = str(int(args.seed))
    if args.k_neg is not None:
        os.environ["CCD_K_NEG"] = str(int(args.k_neg))
    if args.out_dir:
        os.environ["CCD_DIR"] = str(args.out_dir)

    # Refresh module-level config computed at import time
    _reload_from_env()

    main()
