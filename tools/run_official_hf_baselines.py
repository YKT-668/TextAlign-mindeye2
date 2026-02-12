#!/usr/bin/env python
# coding: utf-8
"""run_official_hf_baselines.py

目标：从 HuggingFace dataset `pscotti/mindeyev2` 下载官方 ckpt，并用本仓库统一流程跑 shared1000 retrieval。

特点：
- 自动列出 train_logs/**/last.pth
- 仅对本地具备数据 (betas + wds) 的被试运行（默认 subj01/02/05/07）
- 逐个 ckpt 执行：download -> extract shared1000 tokens -> eval pooled + tokens_flatten
- 写出汇总 CSV，并可选删除下载的 ckpt 省磁盘

依赖：huggingface_hub
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional


def _is_truncated_zip_checkpoint(path: Path) -> bool:
    """Heuristic to detect a partially-downloaded HF .pth.

    Most HF-hosted large PyTorch checkpoints are zip-based (start with 'PK').
    If the file starts with 'PK' but the end-of-central-directory record is missing,
    `torch.load` fails with: "failed finding central directory".

    This check is O(1) and avoids loading multi-GB files.
    """
    try:
        if not path.exists() or path.stat().st_size < 1024 * 1024:
            return True
        with path.open("rb") as f:
            head = f.read(4)
            if not head.startswith(b"PK"):
                return False
            # Search EOCD signature in the last 64KiB (zip spec maximum comment length).
            tail_len = min(65536, path.stat().st_size)
            f.seek(-tail_len, os.SEEK_END)
            tail = f.read(tail_len)
            return (b"PK\x05\x06" not in tail)
    except Exception:
        # If we can't inspect it, be conservative and treat as bad.
        return True


def _run(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    print("[CMD]", " ".join(cmd))
    return subprocess.run(
        cmd,
        check=check,
        env=env,
        text=True,
        capture_output=capture,
    )


def _which(cmd: str) -> Optional[str]:
    from shutil import which

    return which(cmd)


def _download_ckpt_aria2(url: str, out_path: Path) -> Path:
    aria2c = _which("aria2c")
    if not aria2c:
        raise RuntimeError("aria2c not found. Install aria2 or use --download_backend hub")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve signed CAS bridge URL to avoid xet client stalls.
    resolved = url
    try:
        import requests

        r = requests.head(url, allow_redirects=False, timeout=20)
        loc = r.headers.get("location")
        if loc:
            resolved = loc
            print(f"[DL] resolved redirect -> {resolved[:128]}...")
    except Exception as e:
        print(f"[WARN] redirect HEAD failed: {e}, using original URL")

    # aria2c resume + multi-conn, IPv4-only, user-agent to avoid filtering
    cmd = [
        aria2c,
        "--continue=true",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=4M",
        "--max-tries=0",
        "--retry-wait=5",
        "--timeout=60",
        "--disable-ipv6=true",
        "--user-agent=Mozilla/5.0",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "-d",
        str(out_path.parent),
        "-o",
        out_path.name,
        resolved,
    ]
    _run(cmd)
    if not out_path.exists() or out_path.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Download did not produce expected file: {out_path}")
    return out_path


def _dataset_resolve_url(endpoint: str, repo_id: str, filename: str) -> str:
    ep = endpoint.rstrip("/")
    # For datasets, resolve URL is /datasets/{repo_id}/resolve/main/{filename}
    return f"{ep}/datasets/{repo_id}/resolve/main/{filename}"


def _python() -> str:
    return sys.executable


@dataclass
class EvalResult:
    model_id: str
    subj: int
    tag: str
    eval_repr: str
    gt_pooling: str
    fwd_top1: float
    fwd_top5: float
    bwd_top1: float
    bwd_top5: float
    rand_fwd_top1_mean: float
    rand_fwd_top1_std: float
    rand_fwd_top5_mean: float
    rand_fwd_top5_std: float
    rand_bwd_top1_mean: float
    rand_bwd_top1_std: float
    rand_bwd_top5_mean: float
    rand_bwd_top5_std: float
    metrics_json: str


def _read_metrics(metrics_json: Path) -> Dict:
    import json

    with metrics_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dig(d: Dict, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _collect_eval_result(model_id: str, subj: int, tag: str, metrics_json: Path) -> EvalResult:
    d = _read_metrics(metrics_json)

    def req(x, name):
        if x is None:
            raise RuntimeError(f"Missing key {name} in {metrics_json}")
        return float(x)

    return EvalResult(
        model_id=model_id,
        subj=subj,
        tag=tag,
        eval_repr=str(d.get("eval_repr")),
        gt_pooling=str(d.get("gt_pooling")),
        fwd_top1=req(_dig(d, "retrieval", "fwd", "top1"), "retrieval.fwd.top1"),
        fwd_top5=req(_dig(d, "retrieval", "fwd", "top5"), "retrieval.fwd.top5"),
        bwd_top1=req(_dig(d, "retrieval", "bwd", "top1"), "retrieval.bwd.top1"),
        bwd_top5=req(_dig(d, "retrieval", "bwd", "top5"), "retrieval.bwd.top5"),
        rand_fwd_top1_mean=req(_dig(d, "random_retrieval", "fwd", "top1_mean"), "random_retrieval.fwd.top1_mean"),
        rand_fwd_top1_std=req(_dig(d, "random_retrieval", "fwd", "top1_std"), "random_retrieval.fwd.top1_std"),
        rand_fwd_top5_mean=req(_dig(d, "random_retrieval", "fwd", "top5_mean"), "random_retrieval.fwd.top5_mean"),
        rand_fwd_top5_std=req(_dig(d, "random_retrieval", "fwd", "top5_std"), "random_retrieval.fwd.top5_std"),
        rand_bwd_top1_mean=req(_dig(d, "random_retrieval", "bwd", "top1_mean"), "random_retrieval.bwd.top1_mean"),
        rand_bwd_top1_std=req(_dig(d, "random_retrieval", "bwd", "top1_std"), "random_retrieval.bwd.top1_std"),
        rand_bwd_top5_mean=req(_dig(d, "random_retrieval", "bwd", "top5_mean"), "random_retrieval.bwd.top5_mean"),
        rand_bwd_top5_std=req(_dig(d, "random_retrieval", "bwd", "top5_std"), "random_retrieval.bwd.top5_std"),
        metrics_json=str(metrics_json),
    )


def _write_csv(path: Path, rows: List[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model_id",
                "subj",
                "tag",
                "eval_repr",
                "gt_pooling",
                "fwd_top1",
                "fwd_top5",
                "bwd_top1",
                "bwd_top5",
                "rand_fwd_top1_mean",
                "rand_fwd_top1_std",
                "rand_fwd_top5_mean",
                "rand_fwd_top5_std",
                "rand_bwd_top1_mean",
                "rand_bwd_top1_std",
                "rand_bwd_top5_mean",
                "rand_bwd_top5_std",
                "metrics_json",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.model_id,
                    r.subj,
                    r.tag,
                    r.eval_repr,
                    r.gt_pooling,
                    r.fwd_top1,
                    r.fwd_top5,
                    r.bwd_top1,
                    r.bwd_top5,
                    r.rand_fwd_top1_mean,
                    r.rand_fwd_top1_std,
                    r.rand_fwd_top5_mean,
                    r.rand_fwd_top5_std,
                    r.rand_bwd_top1_mean,
                    r.rand_bwd_top1_std,
                    r.rand_bwd_top5_mean,
                    r.rand_bwd_top5_std,
                    r.metrics_json,
                ]
            )


def _read_existing_csv(path: Path) -> List[EvalResult]:
    if not path.exists():
        return []

    rows: List[EvalResult] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    EvalResult(
                        model_id=str(r["model_id"]),
                        subj=int(r["subj"]),
                        tag=str(r["tag"]),
                        eval_repr=str(r["eval_repr"]),
                        gt_pooling=str(r["gt_pooling"]),
                        fwd_top1=float(r["fwd_top1"]),
                        fwd_top5=float(r["fwd_top5"]),
                        bwd_top1=float(r["bwd_top1"]),
                        bwd_top5=float(r["bwd_top5"]),
                        rand_fwd_top1_mean=float(r["rand_fwd_top1_mean"]),
                        rand_fwd_top1_std=float(r["rand_fwd_top1_std"]),
                        rand_fwd_top5_mean=float(r["rand_fwd_top5_mean"]),
                        rand_fwd_top5_std=float(r["rand_fwd_top5_std"]),
                        rand_bwd_top1_mean=float(r["rand_bwd_top1_mean"]),
                        rand_bwd_top1_std=float(r["rand_bwd_top1_std"]),
                        rand_bwd_top5_mean=float(r["rand_bwd_top5_mean"]),
                        rand_bwd_top5_std=float(r["rand_bwd_top5_std"]),
                        metrics_json=str(r["metrics_json"]),
                    )
                )
            except Exception:
                # tolerate partially-written/older CSV formats
                continue
    return rows


def _safe_mean(*xs: float) -> float:
    vs = [float(x) for x in xs]
    return sum(vs) / max(1, len(vs))


def _render_markdown_report(csv_path: Path, md_path: Path) -> None:
    rows = _read_existing_csv(csv_path)
    if not rows:
        return

    # Keep only pooled for baseline selection (tokens_flatten is near-saturated and less meaningful).
    pooled = [r for r in rows if r.eval_repr == "pooled"]
    # Group by subject
    by_subj: Dict[int, List[EvalResult]] = {}
    for r in pooled:
        by_subj.setdefault(r.subj, []).append(r)

    def score(r: EvalResult) -> float:
        # Primary: mean of fwd/bwd top1; secondary: mean of top5
        return _safe_mean(r.fwd_top1, r.bwd_top1) * 1000.0 + _safe_mean(r.fwd_top5, r.bwd_top5)

    lines: List[str] = []
    lines.append("# Official MindEye2 HF baselines (our eval pipeline)")
    lines.append("")
    lines.append(f"CSV: {csv_path}")
    lines.append("")
    lines.append("## Recommended baselines (pooled)")
    lines.append("")
    lines.append("We pick, for each subject, the model with best pooled retrieval (primary: mean Top-1 of FWD/BWD; tie-breaker: mean Top-5).")
    lines.append("")
    lines.append("| subj | baseline tag | FWD@1 | FWD@5 | BWD@1 | BWD@5 | metrics.json |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")
    for subj in sorted(by_subj.keys()):
        cand = sorted(by_subj[subj], key=score, reverse=True)
        best = cand[0]
        lines.append(
            f"| {subj:02d} | {best.tag} | {best.fwd_top1:.2f} | {best.fwd_top5:.2f} | {best.bwd_top1:.2f} | {best.bwd_top5:.2f} | {best.metrics_json} |"
        )

    lines.append("")
    lines.append("## Full results")
    lines.append("")
    lines.append("| subj | eval_repr | tag | FWD@1 | FWD@5 | BWD@1 | BWD@5 | 300-wayx30 FWD@1 (mean±std) | 300-wayx30 BWD@1 (mean±std) | metrics.json |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
    # Sort: subject, eval_repr, best-first
    def sort_key(r: EvalResult):
        return (r.subj, r.eval_repr, -score(r) if r.eval_repr == "pooled" else 0.0)

    for r in sorted(rows, key=sort_key):
        lines.append(
            "| "
            + f"{r.subj:02d} | {r.eval_repr} | {r.tag} | {r.fwd_top1:.2f} | {r.fwd_top5:.2f} | {r.bwd_top1:.2f} | {r.bwd_top5:.2f} | "
            + f"{r.rand_fwd_top1_mean:.2f}±{r.rand_fwd_top1_std:.2f} | {r.rand_bwd_top1_mean:.2f}±{r.rand_bwd_top1_std:.2f} | {r.metrics_json} |"
        )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="pscotti/mindeyev2")
    ap.add_argument("--subjs", default="1,2,5,7", help="Comma-separated subject ids to run")
    ap.add_argument(
        "--ckpt_regex",
        default=r"train_logs/.*/last\.pth$",
        help="Regex over HF repo file paths to choose ckpts",
    )
    ap.add_argument("--keep_ckpt", action="store_true", help="Do not delete downloaded ckpt after eval")
    ap.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_csv", default="cache/model_eval_results/official_hf_baselines_summary.csv")
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skip extraction/eval when outputs already exist (resume-friendly)",
    )
    ap.add_argument(
        "--ckpt_list_cache",
        default="cache/hf_ckpt_lists/pscotti_mindeyev2_last_pth.txt",
        help="Where to cache the HF file list (one path per line).",
    )
    ap.add_argument(
        "--no_cache_write",
        action="store_true",
        help="Do not write ckpt list cache (always query HF).",
    )
    ap.add_argument(
        "--hf_endpoint",
        default=os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com",
        help="HF endpoint base URL. For CN networks, https://hf-mirror.com is usually fastest.",
    )
    ap.add_argument(
        "--download_backend",
        choices=["aria2", "hub"],
        default="aria2",
        help="Checkpoint download backend. aria2 is fastest when available.",
    )
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    proj_root = Path(__file__).resolve().parents[1]
    extract_py = proj_root / "tools" / "extract_brain_clip_tokens_shared1000.py"
    eval_py = proj_root / "tools" / "eval_textalign_latent_plus.py"
    gt_path = proj_root / "evals" / "all_images.pt"

    subjs = [int(s) for s in args.subjs.split(",") if s.strip()]

    from huggingface_hub import list_repo_files, hf_hub_download

    cache_path = proj_root / args.ckpt_list_cache

    def _read_cached_file_list() -> Optional[List[str]]:
        if cache_path.exists():
            txt = cache_path.read_text(encoding="utf-8").splitlines()
            return [t.strip() for t in txt if t.strip()]
        return None

    def _write_cached_file_list(file_list: List[str]) -> None:
        if args.no_cache_write:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(file_list) + "\n", encoding="utf-8")

    files = _read_cached_file_list()
    if files is None:
        last_err: Optional[Exception] = None
        for attempt in range(1, 6):
            try:
                files = list_repo_files(repo_id=args.repo_id, repo_type="dataset")
                _write_cached_file_list(files)
                break
            except Exception as e:
                last_err = e
                wait_s = min(30, 2**attempt)
                print(f"[WARN] list_repo_files failed (attempt {attempt}/5): {e}")
                print(f"[WARN] retrying in {wait_s}s...")
                time.sleep(wait_s)

        if files is None:
            print(f"[WARN] list_repo_files failed after retries: {last_err}")
            print("[WARN] falling back to baked ckpt list (may be incomplete if HF repo changed).")
            files = [
                "train_logs/final_subj01_pretrained_40sess_24bs/last.pth",
                "train_logs/final_subj02_pretrained_40sess_24bs/last.pth",
                "train_logs/final_subj05_pretrained_40sess_24bs/last.pth",
                "train_logs/final_subj07_pretrained_40sess_24bs/last.pth",
                "train_logs/final_subj01_pretrained_1sess_24bs/last.pth",
                "train_logs/final_subj02_pretrained_1sess_24bs/last.pth",
                "train_logs/final_subj05_pretrained_1sess_24bs/last.pth",
                "train_logs/final_subj07_pretrained_1sess_24bs/last.pth",
                "train_logs/final_multisubject_subj01/last.pth",
                "train_logs/final_multisubject_subj02/last.pth",
                "train_logs/final_multisubject_subj05/last.pth",
                "train_logs/final_multisubject_subj07/last.pth",
                "train_logs/multisubject_subj01_1024hid_nolow_300ep/last.pth",
            ]
    pat = re.compile(args.ckpt_regex)
    ckpts = [f for f in files if pat.search(f)]
    # Speed-first ordering:
    # - Prefer non-multisubject models first
    # - Prefer already-present (and non-truncated) local ckpts to avoid blocking on huge downloads
    def _ckpt_sort_key(rel_path: str):
        tag = rel_path.split("/")[1] if "/" in rel_path else rel_path
        is_multi = 1 if "multisubject" in tag else 0
        ckpt_local = proj_root / rel_path
        has_valid_local = False
        try:
            if ckpt_local.exists() and ckpt_local.stat().st_size > 1024 * 1024:
                has_valid_local = not _is_truncated_zip_checkpoint(ckpt_local)
        except Exception:
            has_valid_local = False
        # 0 means earlier
        return (is_multi, 0 if has_valid_local else 1, tag)

    ckpts = sorted(ckpts, key=_ckpt_sort_key)

    if not ckpts:
        raise SystemExit(f"No ckpts matched regex: {args.ckpt_regex}")

    out_csv = Path(__file__).resolve().parents[1] / args.out_csv
    existing = _read_existing_csv(out_csv)
    # Use metrics_json path as a stable unique key.
    seen_metrics = {r.metrics_json for r in existing}
    results: List[EvalResult] = list(existing)

    failures_path = out_csv.with_suffix(".failures.txt")
    failures: List[str] = []

    for rel in ckpts:
        m_subj = re.search(r"subj(0?\d+)", rel)
        subj = int(m_subj.group(1)) if m_subj else None
        if subj is None or subj not in subjs:
            continue

        tag = rel.split("/")[1]  # e.g. final_subj01_pretrained_40sess_24bs
        model_id = f"hf:{rel}"

        # Pre-compute paths for resume checks.
        out_dir = proj_root / "evals" / "brain_tokens" / "official_hf" / tag
        tok_path = out_dir / f"subj{subj:02d}_brain_clip_tokens.pt"
        mean_path = out_dir / f"subj{subj:02d}_brain_clip_mean.pt"
        ids_path = out_dir / f"subj{subj:02d}_ids.json"
        result_root = proj_root / "cache" / "model_eval_results" / "official_hf" / tag
        mj_pooled_path = result_root / "pooled_mean" / "metrics.json"
        mj_tok_path = result_root / "tokens_flatten" / "metrics.json"

        if args.skip_existing and tok_path.exists() and mean_path.exists() and ids_path.exists() and mj_pooled_path.exists() and mj_tok_path.exists():
            print("\n====================")
            print("[HF]", rel)
            print("[SUBJ]", subj, "[TAG]", tag)
            print(f"[SKIP] all outputs exist (extract+eval): {tag}")
            # Ensure results are present in CSV
            if str(mj_pooled_path) not in seen_metrics:
                results.append(_collect_eval_result(model_id, subj, tag, mj_pooled_path))
                seen_metrics.add(str(mj_pooled_path))
                _write_csv(out_csv, results)
            if str(mj_tok_path) not in seen_metrics:
                results.append(_collect_eval_result(model_id, subj, tag, mj_tok_path))
                seen_metrics.add(str(mj_tok_path))
                _write_csv(out_csv, results)
            continue

        print("\n====================")
        print("[HF]", rel)
        print("[SUBJ]", subj, "[TAG]", tag)

        # 1) download ckpt (can be skipped if extraction already exists)
        ckpt_path = proj_root / rel
        need_extract = not (tok_path.exists() and mean_path.exists() and ids_path.exists())

        if not need_extract:
            if args.skip_existing:
                print(f"[SKIP] ckpt download (extract exists): {tag}")
        else:
            # Validate existing ckpt quickly; if it's a truncated zip, delete and re-download.
            if ckpt_path.exists() and ckpt_path.stat().st_size > 1024 * 1024:
                if _is_truncated_zip_checkpoint(ckpt_path):
                    try:
                        ckpt_path.unlink()
                    except Exception:
                        pass
                    print(f"[WARN] ckpt looks corrupted/truncated, re-downloading: {ckpt_path}")
                else:
                    print(f"[SKIP] ckpt exists: {ckpt_path}")

            # Download if missing (or deleted due to corruption)
            if (not ckpt_path.exists()) or ckpt_path.stat().st_size < 1024 * 1024:
                if args.download_backend == "aria2":
                    url = _dataset_resolve_url(args.hf_endpoint, args.repo_id, rel)
                    print(f"[DL] aria2: {url}")
                    _download_ckpt_aria2(url, ckpt_path)
                else:
                    local_dir = proj_root
                    ckpt_path = Path(
                        hf_hub_download(
                            repo_id=args.repo_id,
                            filename=rel,
                            repo_type="dataset",
                            local_dir=str(local_dir),
                            local_dir_use_symlinks=False,
                        )
                    )

        # 2) extract tokens (resume-friendly)
        if args.skip_existing and not need_extract:
            print(f"[SKIP] extraction exists: {out_dir}")
        else:
            cp = _run(
                [
                    _python(),
                    str(extract_py),
                    "--ckpt",
                    str(ckpt_path),
                    "--subj",
                    str(subj),
                    "--out_dir",
                    str(out_dir),
                    "--batch_size",
                    str(args.batch_size),
                    "--device",
                    str(args.device),
                    "--split",
                    "new_test",
                ],
                check=False,
                capture=True,
            )
            if cp.returncode != 0:
                err = (cp.stderr or cp.stdout or "").strip()
                err_tail = "\n".join(err.splitlines()[-20:]) if err else "(no output)"
                msg = f"[FAIL][extract] rel={rel} subj={subj:02d} tag={tag} rc={cp.returncode}\n{err_tail}\n"
                print(msg)
                failures.append(msg)

                # optional cleanup ckpt even on failure
                if not args.keep_ckpt:
                    try:
                        ckpt_path.unlink()
                        print(f"[CLEAN] deleted {ckpt_path}")
                    except Exception as e:
                        print(f"[WARN] failed to delete ckpt: {e}")
                continue

        # 3) eval pooled + tokens_flatten
        def run_eval(eval_repr: str, brain_path: Path, result_dir: Path) -> Path:
            mj = result_dir / "metrics.json"
            if args.skip_existing and mj.exists():
                print(f"[SKIP] eval exists: {mj}")
                return mj
            env = os.environ.copy()
            env.update(
                {
                    "EXP_NAME": f"official_hf_{tag}",
                    "BRAIN_PATH": str(brain_path),
                    "IDS_PATH": str(ids_path),
                    "GT_PATH": str(gt_path),
                    "EVAL_REPR": eval_repr,
                    "GT_POOLING": "mean",
                    "RAND_K": "300",
                    "RAND_TRIALS": "30",
                    "RAND_SEED": "0",
                    "RESULT_DIR": str(result_dir),
                }
            )
            cp = _run([_python(), str(eval_py)], env=env, check=False, capture=True)
            if cp.returncode != 0:
                err = (cp.stderr or cp.stdout or "").strip()
                err_tail = "\n".join(err.splitlines()[-20:]) if err else "(no output)"
                msg = f"[FAIL][eval:{eval_repr}] rel={rel} subj={subj:02d} tag={tag} rc={cp.returncode}\n{err_tail}\n"
                print(msg)
                failures.append(msg)
                raise RuntimeError(msg)
            if not mj.exists():
                raise RuntimeError(f"metrics.json not found: {mj}")
            return mj

        mj_pooled = run_eval("pooled", mean_path, result_root / "pooled_mean")
        if str(mj_pooled) not in seen_metrics:
            results.append(_collect_eval_result(model_id, subj, tag, mj_pooled))
            seen_metrics.add(str(mj_pooled))
            _write_csv(out_csv, results)

        mj_tok = run_eval("tokens_flatten", tok_path, result_root / "tokens_flatten")
        if str(mj_tok) not in seen_metrics:
            results.append(_collect_eval_result(model_id, subj, tag, mj_tok))
            seen_metrics.add(str(mj_tok))
            _write_csv(out_csv, results)

        # 4) optional cleanup ckpt
        if not args.keep_ckpt:
            try:
                ckpt_path.unlink()
                print(f"[CLEAN] deleted {ckpt_path}")
            except Exception as e:
                print(f"[WARN] failed to delete ckpt: {e}")

    _write_csv(out_csv, results)
    report_md = out_csv.with_suffix(".md")
    _render_markdown_report(out_csv, report_md)
    if failures:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        failures_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print("[DONE] wrote", failures_path)
    print("\n[DONE] wrote", out_csv)
    print("[DONE] wrote", report_md)


if __name__ == "__main__":
    main()
