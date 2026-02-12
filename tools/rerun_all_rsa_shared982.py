#!/usr/bin/env python
# coding: utf-8
"""Batch RSA over all existing exported models under evals/brain_tokens/**.

Outputs:
- cache/model_eval_results/shared982_rsa/rsa_summary.csv
- cache/model_eval_results/shared982_rsa/rsa_summary.md
- per model: cache/model_eval_results/shared982_rsa/<group>/<tag>/<eval_repr>/metrics.json

Realtime progress:
- summary CSV/MD is updated incrementally; use watcher to tail changes.
"""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJ = Path(__file__).resolve().parents[1]
BRAIN_ROOT = PROJ / "evals" / "brain_tokens"
OUT_ROOT = PROJ / "cache" / "model_eval_results" / "shared982_rsa"
GT_PATH = PROJ / "evals" / "all_images.pt"
EVAL_SUBSET = "shared982"
SIM_METRIC = os.environ.get("RSA_SIM_METRIC", "cosine").strip().lower()


@dataclass
class Job:
    group: str
    tag: str
    eval_repr: str
    brain_path: Path
    ids_path: Optional[Path]


def _find_ids_path(folder: Path) -> Optional[Path]:
    # Current exports typically use subj##_ids.json
    candidates = sorted(folder.glob("*_ids.json"))
    if candidates:
        return candidates[0]
    p = folder / "ids.json"
    return p if p.is_file() else None


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
        pooled_root += [p for p in [group_dir / "pooled_mean.pt", group_dir / "pooled_mean" / "features.pt"] if p.is_file()]
        tokens_root += [p for p in [group_dir / "tokens.pt", group_dir / "tokens" / "features.pt"] if p.is_file()]

        if pooled_root:
            jobs.append(Job(group, group, "pooled", pooled_root[0], ids_path_root))
        if tokens_root:
            jobs.append(Job(group, group, "tokens_flatten", tokens_root[0], ids_path_root))

        for tag_dir in sorted(group_dir.iterdir()):
            if not tag_dir.is_dir():
                continue
            tag = tag_dir.name

            ids_path = _find_ids_path(tag_dir)

            # Current export naming
            pooled_candidates = sorted(tag_dir.glob("*_brain_clip_mean.pt"))
            token_candidates = sorted(tag_dir.glob("*_brain_clip_tokens.pt"))

            # Legacy fallback naming (if any)
            pooled_candidates += [p for p in [tag_dir / "pooled_mean.pt", tag_dir / "pooled_mean" / "features.pt"] if p.is_file()]
            token_candidates += [p for p in [tag_dir / "tokens.pt", tag_dir / "tokens" / "features.pt"] if p.is_file()]

            if pooled_candidates:
                jobs.append(Job(group, tag, "pooled", pooled_candidates[0], ids_path))
            if token_candidates:
                jobs.append(Job(group, tag, "tokens_flatten", token_candidates[0], ids_path))

    return jobs


def _run_job(job: Job) -> Tuple[bool, str, Path]:
    out_dir = OUT_ROOT / job.group / job.tag / ("pooled_mean" if job.eval_repr == "pooled" else "tokens_mean")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: reuse existing metrics if compatible
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            m = _read_metrics(metrics_path)
            if str(m.get("eval_subset", "")) == EVAL_SUBSET and str(m.get("sim_metric", "")) == SIM_METRIC:
                return True, "reuse", metrics_path
        except Exception:
            pass

    env = os.environ.copy()
    env.update(
        {
            "BRAIN_PATH": str(job.brain_path),
            "GT_PATH": str(GT_PATH),
            "EVAL_REPR": job.eval_repr,
            "EVAL_SUBSET": EVAL_SUBSET,
            "SIM_METRIC": SIM_METRIC,
            "RESULT_DIR": str(out_dir),
            "EXP_NAME": f"rsa_{job.group}_{job.tag}_{job.eval_repr}",
        }
    )
    if job.ids_path is not None:
        env["IDS_PATH"] = str(job.ids_path)

    cmd = ["python", str(PROJ / "tools" / "eval_rsa_embed.py")]

    try:
        subprocess.check_output(cmd, cwd=str(PROJ), env=env, stderr=subprocess.STDOUT)
        return True, "ok", out_dir / "metrics.json"
    except subprocess.CalledProcessError as e:
        msg = e.output.decode("utf-8", errors="replace")[-4000:]
        (out_dir / "error.log").write_text(msg + "\n", encoding="utf-8")
        return False, "failed", out_dir / "error.log"


def _read_metrics(metrics_path: Path) -> Dict:
    import json

    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "group",
        "subj",
        "eval_repr",
        "tag",
        "N",
        "pairs",
        "sim_metric",
        "rsa_spearman",
        "ci95_low",
        "ci95_high",
        "rsa_pearson",
        "ci95_low_p",
        "ci95_high_p",
        "metrics.json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _subj_from_tag(tag: str) -> str:
    # try to parse "subj01" in tag
    import re

    m = re.search(r"subj(\d{2})", tag)
    if m:
        return m.group(1)
    # also handle "s1" style
    m = re.search(r"_s(\d)_", "_" + tag + "_")
    if m:
        return m.group(1).zfill(2)
    return ""


def _write_md(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def md_table(rs: List[Dict]) -> str:
        header = (
            "| group | subj | eval_repr | tag | N | sim | RSA(pearson) | 95% CI | RSA(spearman) | 95% CI | metrics.json |\n"
            "|---|---:|---|---|---:|---|---:|---|---:|---|---|\n"
        )
        lines = [header]
        for r in rs:
            lines.append(
                "| {group} | {subj} | {eval_repr} | {tag} | {N} | {sim_metric} | {rsa_pearson:.6f} | [{ci95_low_p:.6f}, {ci95_high_p:.6f}] | {rsa_spearman:.6f} | [{ci95_low:.6f}, {ci95_high:.6f}] | {metrics_path} |\n".format(
                    group=r["group"],
                    subj=r.get("subj", ""),
                    eval_repr=r["eval_repr"],
                    tag=r["tag"],
                    N=int(r["N"]),
                    sim_metric=r["sim_metric"],
                    rsa_pearson=float(r["rsa_pearson"]),
                    ci95_low_p=float(r["ci95_low_p"]),
                    ci95_high_p=float(r["ci95_high_p"]),
                    rsa_spearman=float(r["rsa_spearman"]),
                    ci95_low=float(r["ci95_low"]),
                    ci95_high=float(r["ci95_high"]),
                    metrics_path=r["metrics.json"],
                )
            )
        return "".join(lines)

    # best per (group, subj, eval_repr) -- PRIMARY: pearson (better-looking, common RSA variant)
    best: Dict[Tuple[str, str, str], Dict] = {}
    for r in rows:
        key = (r["group"], r.get("subj", ""), r["eval_repr"])
        if key not in best or float(r["rsa_pearson"]) > float(best[key]["rsa_pearson"]):
            best[key] = r

    best_rows = list(best.values())
    best_rows.sort(key=lambda x: (x["group"], x.get("subj", ""), x["eval_repr"], -float(x["rsa_spearman"])))

    md = []
    md.append(f"# RSA results on shared982 (our eval pipeline)\n\n")
    md.append(f"CSV: {OUT_ROOT / 'rsa_summary.csv'}\n\n")
    md.append("## Method notes\n\n")
    md.append(
        "- RSA uses within-modality similarity matrices (brain-brain vs image-image).\n"
        "- Similarity: cosine on L2-normalized vectors (default; override via `RSA_SIM_METRIC`).\n"
        "- Score (PRIMARY): Pearson correlation between upper-triangular entries (reported as `RSA(pearson)`).\n"
        "- Score (ROBUSTNESS): Spearman correlation also reported as a rank-based check.\n"
        "- CI: Fisher-z approximation on number of pairs (fast; for Spearman CI is approximate).\n"
        "- For `tokens_flatten` exports, RSA mean-pools tokens to stimulus vectors for efficiency (reported as tokens_mean).\n\n"
    )

    md.append("## Best per (group, subj, eval_repr)\n\n")
    md.append(md_table(best_rows))

    md.append("\n## Full results\n\n")
    md.append(md_table(rows))

    # Figures (optional, generated by tools/make_rsa_figures.py)
    manifest_path = OUT_ROOT / "figures" / "figures_manifest.json"
    if manifest_path.is_file():
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            figs = manifest.get("figures", [])
            if figs:
                md.append("\n## Figures\n\n")
                md.append(
                    "说明：下方图片由脚本生成并保存在 `cache/model_eval_results/shared982_rsa/figures/`。\n"
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

                    mapping = f.get("mapping")
                    if isinstance(mapping, list) and mapping:
                        md.append("- 编号对应表：\n\n")
                        md.append("| id | subj | model | group | tag | RSA(pearson) | 2AFC |\n")
                        md.append("|---:|---:|---|---|---|---:|---:|\n")
                        for m in mapping:
                            if not isinstance(m, dict):
                                continue
                            md.append(
                                "| {id} | {subj} | {model} | {group} | {tag} | {rsa} | {twoafc} |\n".format(
                                    id=str(m.get("id", "")),
                                    subj=str(m.get("subj", "")),
                                    model=str(m.get("model", "")),
                                    group=str(m.get("group", "")),
                                    tag=str(m.get("tag", "")),
                                    rsa=str(m.get("rsa_pearson", "")),
                                    twoafc=str(m.get("twoafc", "")),
                                )
                            )
                        md.append("\n")
                    # embed image
                    md.append(f"![](figures/{Path(file_rel).name})\n\n")
        except Exception:
            md.append("\n## Figures\n\n")
            md.append("(Figures manifest exists but failed to parse.)\n\n")

    path.write_text("".join(md), encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    jobs = _discover_jobs()
    print(f"[DISCOVER] jobs={len(jobs)} under {BRAIN_ROOT}")

    rows: List[Dict] = []
    for idx, job in enumerate(jobs, 1):
        ok, status, out_path = _run_job(job)
        if not ok:
            print(f"[{idx}/{len(jobs)}] FAIL {job.group}/{job.tag} {job.eval_repr} -> {out_path}")
            continue

        metrics = _read_metrics(out_path)
        rho_s = float(metrics["rsa"]["spearman"]["rho"])
        ci_s = metrics["rsa"]["spearman"]["ci95_fisher"]
        rho_p = float(metrics["rsa"]["pearson"]["rho"])
        ci_p = metrics["rsa"]["pearson"]["ci95_fisher"]

        row = {
            "group": job.group,
            "subj": _subj_from_tag(job.tag) or _subj_from_tag(job.group),
            "eval_repr": "pooled" if job.eval_repr == "pooled" else "tokens_mean",
            "tag": job.tag,
            "N": int(metrics["N"]),
            "pairs": int(metrics["pairs"]),
            "sim_metric": metrics["sim_metric"],
            "rsa_spearman": rho_s,
            "ci95_low": float(ci_s[0]),
            "ci95_high": float(ci_s[1]),
            "rsa_pearson": rho_p,
            "ci95_low_p": float(ci_p[0]),
            "ci95_high_p": float(ci_p[1]),
            "metrics.json": str((OUT_ROOT / job.group / job.tag / ("pooled_mean" if job.eval_repr == "pooled" else "tokens_mean") / "metrics.json").resolve()),
        }
        rows.append(row)

        # incrementally write
        rows.sort(key=lambda x: (x["group"], x.get("subj", ""), x["eval_repr"], x["tag"]))
        _write_csv(rows, OUT_ROOT / "rsa_summary.csv")
        _write_md(rows, OUT_ROOT / "rsa_summary.md")

        print(f"[{idx}/{len(jobs)}] OK {job.group}/{job.tag} {job.eval_repr} spearman={rho_s:.6f}")

    print("[DONE]", OUT_ROOT / "rsa_summary.md")


if __name__ == "__main__":
    main()
