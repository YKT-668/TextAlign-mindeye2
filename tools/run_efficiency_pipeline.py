#!/usr/bin/env python
# coding: utf-8

"""run_efficiency_pipeline.py

One-command pipeline for efficiency curves (1h/2h/full):
- Subjects: 1, 5
- Models: baseline, textalign_llm
- Settings: 1, 2, 40 sessions
- Seed: 0
- Eval subset: shared982
- Bootstrap: 1000

Deliverables (fixed paths):
- runs/train/<subj>/<model>/<setting>/seed0/ (logs + checkpoints/last.pth)
- runs/infer/<subj>/<model>/<setting>/seed0/ (brain_clip_mean.pt + ids.json + brain.npy + ids.npy)
- cache/model_eval_results/shared982_efficiency/embeds/<subj>/<model>/<setting>/seed0/{brain.npy,ids.npy}
- cache/model_eval_results/shared982_efficiency/metrics/<subj>/<model>/<setting>/seed0/{ccd.json,twoafc_hard.json,rsa.json}
- results/tables/efficiency_summary.csv
- results/tables/main_results.csv (append group rows)
- results/RESULTS_INDEX.md (append section)
- results/figures_main/Fig_efficiency_ccd_acc1.png
- results/figures_main/Fig_efficiency_twoafc_hard.png

Notes on speed:
- Automatically skips stages if outputs already exist.
- Will symlink existing pretrained checkpoints when available (baseline 1sess/40sess; textalign full).

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def _apply_hf_accel_env_inplace() -> None:
    """Apply the same HF accel settings to *this* parent process.

    Subprocesses already receive an env dict, but `hf_hub_download()` runs
    inside the pipeline process, so we set defaults here too.
    """
    patched = _apply_hf_accel_env(os.environ.copy())
    for k, v in patched.items():
        os.environ.setdefault(k, v)


def _maybe_download_official_baseline_ckpt(job: "Job") -> bool:
    """Best-effort download of official MindEyeV2 ckpts for baseline.

    The upstream dataset provides official checkpoints for some subjects/settings
    (notably 1sess/40sess). Downloading is usually much faster than training.
    """
    # NOTE: The A3.* protocol in this repo uses the Train_textalign* scripts
    # (hidden_dim=1024 family). The official MindEyeV2 baseline ckpts are a
    # different training recipe/architecture and can cause shape mismatches.
    # Keep this helper for historical compatibility, but disable it by default.
    if os.environ.get("MINDEYE_EFFICIENCY_ALLOW_OFFICIAL_BASELINE", "0") != "1":
        return False
    if job.model != "baseline":
        return False
    if int(job.num_sessions) not in (1, 40):
        return False

    # Only attempt for our protocol subjects.
    if int(job.subj) not in (1, 5):
        return False

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        print(f"[WARN] huggingface_hub not available, cannot download official ckpt: {e}")
        return False

    sess_tag = "1sess" if int(job.num_sessions) == 1 else "40sess"
    rel = f"train_logs/final_subj{int(job.subj):02d}_pretrained_{sess_tag}_24bs/last.pth"
    try:
        print(f"[hf] downloading official ckpt: {rel}")
        hf_hub_download(
            repo_id="pscotti/mindeyev2",
            repo_type="dataset",
            revision="main",
            filename=rel,
            local_dir=str(PROJ),
            local_dir_use_symlinks=False,
        )
        return (PROJ / rel).is_file()
    except Exception as e:
        print(f"[WARN] failed to download official ckpt for {job.model_name}: {e}")
        return False


def _fix_broken_runs_symlink() -> None:
    """Some environments ship `runs` as a symlink to a non-existent mount.

    If it's broken, replace it with a local directory so pipeline outputs can be written.
    """
    runs = PROJ / "runs"
    if runs.is_symlink() and not runs.exists():
        target = os.readlink(runs)
        print(f"[WARN] Broken symlink: runs -> {target}. Replacing with local directory.")
        runs.unlink()
        (PROJ / "runs" / "train").mkdir(parents=True, exist_ok=True)
        (PROJ / "runs" / "infer").mkdir(parents=True, exist_ok=True)


def _load_state_dict_for_ckpt(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        # assume it's already a state dict
        return ckpt  # type: ignore[return-value]
    raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")


def _infer_dims_from_ckpt(ckpt_path: Path) -> Tuple[int, int]:
    """Infer (hidden_dim, n_blocks) from a MindEye checkpoint state_dict."""
    sd = _load_state_dict_for_ckpt(ckpt_path)

    # hidden_dim from ridge regression output size
    ridge_key = None
    for k in ("ridge.linears.0.weight", "model.ridge.linears.0.weight"):
        if k in sd:
            ridge_key = k
            break
    if ridge_key is None:
        # fallback: find any ridge weight
        for k in sd.keys():
            if k.endswith("ridge.linears.0.weight"):
                ridge_key = k
                break
    if ridge_key is None:
        raise RuntimeError(f"Cannot infer hidden_dim: missing ridge weights in {ckpt_path}")
    hidden_dim = int(sd[ridge_key].shape[0])

    # n_blocks from mixer block indices
    max_idx = -1
    for k in sd.keys():
        if ".mixer_blocks1." in k:
            try:
                part = k.split(".mixer_blocks1.", 1)[1]
                idx = int(part.split(".", 1)[0])
                max_idx = max(max_idx, idx)
            except Exception:
                continue
    n_blocks = int(max_idx + 1) if max_idx >= 0 else 4
    return hidden_dim, n_blocks


PROJ = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJ / "src"


def _apply_hf_accel_env(env: Dict[str, str]) -> Dict[str, str]:
    """Best-effort HuggingFace download acceleration.

    Goal: make OpenCLIP weight downloads reliable under CN networks.
    - Use HF mirror endpoint by default (can be overridden by user's env).
    - Enable hf_transfer if installed.
    - Force all caches into repo-local cache/ to avoid repeated downloads.
    """
    e = env.copy()

    # Mirror endpoint (user can override externally).
    # Common CN mirror: https://hf-mirror.com
    e.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # Some libs still read HF_HUB_ENDPOINT.
    e.setdefault("HF_HUB_ENDPOINT", e["HF_ENDPOINT"])

    cache_root = PROJ / "cache" / "hf"
    cache_root.mkdir(parents=True, exist_ok=True)

    e.setdefault("HF_HOME", str(cache_root))
    e.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    e.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    e.setdefault("XDG_CACHE_HOME", str(PROJ / "cache"))

    # Download acceleration (requires `pip install hf-transfer`).
    e.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Avoid Xet/CAS bridge which is often blocked/unstable on some networks.
    # Fall back to regular HTTP/Git-LFS style downloads.
    e.setdefault("HF_HUB_DISABLE_XET", "1")

    # Be quiet + stable.
    e.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return e


def _cpu_workers() -> int:
    c = os.cpu_count() or 16
    return max(2, min(16, c))


def _run(cmd: List[str], *, cwd: Path, env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = env.copy()
    # Avoid "silent" long runs due to Python stdio buffering.
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"[run] {' '.join(cmd)}")
    print(f"[run] log -> {log_path}")

    def _read_log_tail_last_line() -> str:
        try:
            import os

            if not log_path.exists():
                return ""
            size = os.path.getsize(log_path)
            with log_path.open("rb") as fb:
                fb.seek(max(0, size - 8192))
                data = fb.read()
            txt = data.decode("utf-8", errors="ignore").replace("\r", "\n")
            lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
            return lines[-1] if lines else ""
        except Exception:
            return ""

    import time

    start = time.time()
    last_heartbeat = start

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()

        p = subprocess.Popen(cmd, cwd=str(cwd), env=run_env, stdout=f, stderr=subprocess.STDOUT)
        while True:
            ret = p.poll()
            if ret is not None:
                if ret != 0:
                    raise RuntimeError(f"Command failed (exit={ret}). See log: {log_path}")
                return

            now = time.time()
            if now - last_heartbeat >= 30:
                elapsed = int(now - start)
                mm, ss = divmod(elapsed, 60)
                hh, mm = divmod(mm, 60)
                tail = _read_log_tail_last_line()
                if tail:
                    print(f"[run] still running ({hh:02d}:{mm:02d}:{ss:02d}) | {tail}")
                else:
                    print(f"[run] still running ({hh:02d}:{mm:02d}:{ss:02d}) | see log")
                last_heartbeat = now

            time.sleep(1)


def _ensure_symlink(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def _setting_tag(num_sessions: int) -> str:
    if int(num_sessions) == 1:
        return "1sess"
    if int(num_sessions) == 2:
        return "2sess"
    if int(num_sessions) == 40:
        return "40sess"
    raise ValueError(f"Unsupported sessions: {num_sessions}")


def _tag(subj: int, model: str, num_sessions: int, seed: int) -> str:
    st = _setting_tag(num_sessions)
    if model == "baseline":
        return f"s{subj:02d}_baseline_{st}_seed{seed}"
    if model == "textalign_llm":
        return f"s{subj:02d}_textalignllm_{st}_seed{seed}"
    raise ValueError(f"Unknown model: {model}")


@dataclass(frozen=True)
class Job:
    subj: int
    model: str
    num_sessions: int
    seed: int

    @property
    def setting(self) -> str:
        return _setting_tag(self.num_sessions)

    @property
    def model_name(self) -> str:
        return _tag(self.subj, self.model, self.num_sessions, self.seed)

    @property
    def train_run_dir(self) -> Path:
        return PROJ / "runs" / "train" / f"subj{self.subj:02d}" / self.model / self.setting / f"seed{self.seed}"

    @property
    def infer_run_dir(self) -> Path:
        return PROJ / "runs" / "infer" / f"subj{self.subj:02d}" / self.model / self.setting / f"seed{self.seed}"

    @property
    def ckpt_link(self) -> Path:
        return self.train_run_dir / "checkpoints" / "last.pth"

    @property
    def train_log(self) -> Path:
        return self.train_run_dir / "train.log"

    @property
    def infer_log(self) -> Path:
        return self.infer_run_dir / "infer.log"

    @property
    def brain_pt(self) -> Path:
        return self.infer_run_dir / "brain_clip_mean.pt"

    @property
    def ids_json(self) -> Path:
        return self.infer_run_dir / "ids.json"

    @property
    def brain_npy(self) -> Path:
        return self.infer_run_dir / "brain.npy"

    @property
    def ids_npy(self) -> Path:
        return self.infer_run_dir / "ids.npy"

    @property
    def eff_embed_dir(self) -> Path:
        return (
            PROJ
            / "cache"
            / "model_eval_results"
            / "shared982_efficiency"
            / "embeds"
            / f"subj{self.subj:02d}"
            / self.model
            / self.setting
            / f"seed{self.seed}"
        )

    @property
    def eff_metrics_dir(self) -> Path:
        return (
            PROJ
            / "cache"
            / "model_eval_results"
            / "shared982_efficiency"
            / "metrics"
            / f"subj{self.subj:02d}"
            / self.model
            / self.setting
            / f"seed{self.seed}"
        )


def _maybe_link_pretrained_ckpt(job: Job) -> bool:
    """Try to provide train_logs/<tag>/last.pth by symlinking known existing ckpts."""

    train_logs = PROJ / "train_logs"
    dst_dir = train_logs / job.model_name
    dst = dst_dir / "last.pth"
    if dst.is_file() or dst.is_symlink():
        return True

    # Baseline: optionally reuse official HF pretrained checkpoints if present.
    # Default is OFF for the A3 efficiency protocol (which uses 1024-dim backbone).
    allow_official = os.environ.get("MINDEYE_EFFICIENCY_ALLOW_OFFICIAL_BASELINE", "0") == "1"
    if allow_official and job.model == "baseline" and job.num_sessions in (1, 40):
        src_name = f"final_subj{job.subj:02d}_pretrained_{job.num_sessions}sess_24bs"
        src = train_logs / src_name / "last.pth"
        if src.is_file():
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
            return True

    # Baseline 2sess: do NOT reuse local ckpts by default.
    # Older runs may have incompatible training regime (e.g., TextAlign-only). Opt-in only.
    allow_local_reuse = os.environ.get("MINDEYE_EFFICIENCY_ALLOW_LOCAL_REUSE", "0") == "1"
    if allow_local_reuse and job.model == "baseline" and job.num_sessions == 2:
        src = train_logs / f"s{job.subj:02d}_baseline_2sess_seed{job.seed}" / "last.pth"
        if src.is_file():
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
            return True

    # TextAlign full: reuse exported model weights repo if present.
    if job.model == "textalign_llm" and job.num_sessions == 40:
        model_repo = Path("/mnt/work/repos/textalign-mindeye2-model")
        if model_repo.exists():
            if job.subj == 1:
                # Prefer v2 if available
                cand = [
                    model_repo / "models" / "subj01" / "s1_textalign_coco_train_long_v2_last.pth",
                    model_repo / "models" / "subj01" / "s1_textalign_coco_train_long_v1_last.pth",
                    model_repo / "models" / "subj01" / "s1_textalign_coco_from_final_subj01_full_v1_last.pth",
                ]
            elif job.subj == 5:
                cand = [
                    model_repo / "models" / "subj05" / "s5_textalign_coco_train_long_v10_last.pth",
                ]
            else:
                cand = []
            for src in cand:
                if src.is_file():
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst.symlink_to(src)
                    return True

    return False


def _ensure_ckpt(job: Job, *, epochs: int, batch_size: int, do_train: bool) -> None:
    train_logs = PROJ / "train_logs"
    ckpt = train_logs / job.model_name / "last.pth"

    meta_path = job.train_run_dir / "ckpt_meta.json"

    def _expected_meta(_job: Job) -> dict:
        # A3.* protocol contract
        return {
            "protocol": "A3",
            "hidden_dim": 1024,
            "n_blocks": 4,
            "model": _job.model,
            "subj": int(_job.subj),
            "num_sessions": int(_job.num_sessions),
            "seed": int(_job.seed),
            "MINDEYE_TEXTALIGN": "0" if _job.model == "baseline" else "1",
            "MINDEYE_TEXTALIGN_HARDNEG": "0" if _job.model == "baseline" else "1",
        }

    def _meta_matches(_p: Path, expected: dict) -> bool:
        if not _p.is_file():
            return False
        try:
            got = json.loads(_p.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(got, dict):
            return False
        for k, v in expected.items():
            if str(got.get(k)) != str(v):
                return False
        return True

    def _expected_hidden_dim(_job: Job) -> int:
        # A3.* protocol backbone dim
        return 1024

    def _ckpt_hidden_dim_guess(p: Path) -> int | None:
        try:
            checkpoint = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(p, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        else:
            return None
        if not isinstance(state_dict, dict):
            return None
        # Prefer keys where hidden_dim is unambiguous.
        for k in ("ridge.linears.0.weight", "backbone.mixer_blocks1.0.0.weight"):
            t = state_dict.get(k)
            if torch.is_tensor(t) and t.ndim >= 2:
                return int(t.shape[0])
        t = state_dict.get("backbone.backbone_linear.weight")
        if torch.is_tensor(t) and t.ndim >= 2:
            return int(t.shape[-1])
        return None

    def _is_ckpt_compatible(p: Path, *, expected: int) -> bool:
        got = _ckpt_hidden_dim_guess(p)
        return got is None or int(got) == int(expected)

    expected_meta = _expected_meta(job)

    if ckpt.is_file() or ckpt.is_symlink():
        expected = _expected_hidden_dim(job)
        # Reuse when weights are compatible. ckpt_meta.json is best-effort provenance and
        # must NOT block eval-only reruns.
        if _is_ckpt_compatible(ckpt, expected=expected):
            _ensure_symlink(job.ckpt_link, ckpt)
            # Refresh meta when training is requested.
            if do_train and not _meta_matches(meta_path, expected_meta):
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(expected_meta, indent=2) + "\n", encoding="utf-8")
            return

        # Existing ckpt is incompatible (e.g., official 4096-dim). If training is allowed,
        # remove the local pointer and regenerate a compatible ckpt.
        if do_train:
            try:
                ckpt.unlink(missing_ok=True)
            except TypeError:
                # py<3.8 compatibility not needed here; defensive
                if ckpt.exists() or ckpt.is_symlink():
                    ckpt.unlink()
        else:
            raise RuntimeError(
                f"Incompatible checkpoint for {job.model_name}: expected hidden_dim={expected}. "
                f"Refusing to run inference with mismatched weights at {ckpt}. "
                "Set --do_train to regenerate a compatible checkpoint, or set "
                "MINDEYE_EFFICIENCY_ALLOW_OFFICIAL_BASELINE=1 and adjust protocol (not recommended for A3)."
            )

    # Try linking known ckpts
    if _maybe_link_pretrained_ckpt(job):
        ckpt = train_logs / job.model_name / "last.pth"
        if ckpt.exists():
            expected = _expected_hidden_dim(job)
            if _is_ckpt_compatible(ckpt, expected=expected):
                _ensure_symlink(job.ckpt_link, ckpt)
                return
            if do_train:
                try:
                    ckpt.unlink(missing_ok=True)
                except TypeError:
                    if ckpt.exists() or ckpt.is_symlink():
                        ckpt.unlink()
            else:
                raise RuntimeError(
                    f"Linked checkpoint for {job.model_name} is incompatible with hidden_dim={expected}: {ckpt}"
                )

    # Try downloading official baseline ckpts (1sess/40sess) to avoid slow training.
    if _maybe_download_official_baseline_ckpt(job):
        official_candidates = [
            train_logs / f"final_subj{job.subj:02d}_pretrained_{job.setting}_24bs" / "last.pth",
        ]
        for src in official_candidates:
            if src.is_file():
                dst_dir = train_logs / job.model_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / "last.pth"
                if not dst.exists() and not dst.is_symlink():
                    dst.symlink_to(src)
                _ensure_symlink(job.ckpt_link, dst)
                return

    if not do_train:
        raise RuntimeError(f"Missing checkpoint for {job.model_name} and --do_train not set")

    env = _apply_hf_accel_env(os.environ.copy())
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["MINDEYE_DTYPE"] = env.get("MINDEYE_DTYPE", "bf16")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if job.model == "baseline":
        # A3.1 baseline: use Train_textalign.py implementation but disable TextAlign.
        script = PROJ / "src" / "Train_textalign_A3_1_baseline_COPY.py"
        env["MINDEYE_TEXTALIGN"] = "0"
        env["MINDEYE_TEXTALIGN_HARDNEG"] = "0"
        cmd = [
            sys.executable,
            str(script),
            "--subj",
            str(job.subj),
            "--num_sessions",
            str(job.num_sessions),
            "--num_epochs",
            str(int(epochs)),
            "--batch_size",
            str(int(batch_size)),
            "--hidden_dim",
            "1024",
            "--n_blocks",
            "4",
            "--seed",
            str(int(job.seed)),
            "--model_name",
            job.model_name,
            "--data_path",
            str(DATA_ROOT),
            "--cache_dir",
            str(DATA_ROOT),
            "--no-blurry_recon",
            "--no-use_prior",
            "--num_workers",
            str(_cpu_workers()),
        ]
    else:
        # A3.2 textalign_llm: use Train_textalign_v1_backup.py implementation (COPY)
        # and enable TextAlign + (optional) LLM hard-negative via env.
        script = PROJ / "src" / "Train_textalign_A3_2_textalignllm_COPY.py"
        env["MINDEYE_TEXTALIGN"] = "1"
        env["MINDEYE_TEXTALIGN_HARDNEG"] = "1"
        cmd = [
            sys.executable,
            str(script),
            "--subj",
            str(job.subj),
            "--num_sessions",
            str(job.num_sessions),
            "--num_epochs",
            str(int(epochs)),
            "--batch_size",
            str(int(batch_size)),
            "--seed",
            str(int(job.seed)),
            "--model_name",
            job.model_name,
            "--data_path",
            str(DATA_ROOT),
            "--cache_dir",
            str(DATA_ROOT),
            "--no-blurry_recon",
            "--no-use_prior",
        ]

    _run(cmd, cwd=PROJ, env=env, log_path=job.train_log)

    if not ckpt.exists():
        raise RuntimeError(f"Training finished but ckpt missing: {ckpt}")

    _ensure_symlink(job.ckpt_link, ckpt)

    # Record meta so future runs can safely skip.
    try:
        job.train_run_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(expected_meta)
        meta.update(
            {
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "script": str(script.relative_to(PROJ)),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        pass


def _ensure_infer(job: Job, *, do_infer: bool) -> None:
    if job.brain_pt.is_file() and job.ids_json.is_file():
        return
    if not do_infer:
        raise RuntimeError(f"Missing inference outputs for {job.model_name} and --do_infer not set")

    env = _apply_hf_accel_env(os.environ.copy())
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    # Use latent_only to avoid heavy SDXL generation; only dump brain->CLIP vectors.
    if job.model == "baseline":
        script = PROJ / "src" / "recon_inference_run_latent.py"
    else:
        script = PROJ / "src" / "recon_inference_run_latent_textalign.py"

    ckpt_path = PROJ / "train_logs" / job.model_name / "last.pth"
    if not ckpt_path.exists():
        raise RuntimeError(f"Missing ckpt for inference: {ckpt_path}")

    # For the A3.* protocol, both baseline and textalign_llm use the 1024-dim backbone.
    # (Official MindEyeV2 baselines are 4096-dim; those are disabled by default in this pipeline.)
    hidden_dim, n_blocks = 1024, 4

    cmd = [
        sys.executable,
        str(script),
        "--model_name",
        job.model_name,
        "--ckpt_root",
        str(PROJ / "train_logs"),
        "--data_path",
        str(DATA_ROOT),
        "--cache_dir",
        str(DATA_ROOT),
        "--subj",
        str(job.subj),
        "--hidden_dim",
        str(int(hidden_dim)),
        "--n_blocks",
        str(int(n_blocks)),
        "--new_test",
        "--seed",
        str(int(job.seed)),
        "--output_dir",
        str(job.infer_run_dir),
        "--no-blurry_recon",
        "--latent_only",
        "--dump_clip_vecs",
        "--dump_ids",
        "--clip_out",
        str(job.brain_pt),
        "--max_save",
        "1000",
    ]

    if job.model == "baseline":
        cmd.extend(["--clip_pooling", "mean"])

    _run(cmd, cwd=PROJ, env=env, log_path=job.infer_log)

    if not job.brain_pt.is_file() or not job.ids_json.is_file():
        raise RuntimeError(f"Inference finished but outputs missing: {job.brain_pt} / {job.ids_json}")


def _convert_and_copy_embeds(job: Job) -> None:
    # Convert pt -> npy (required deliverables)
    if not job.brain_npy.is_file():
        V = torch.load(job.brain_pt, map_location="cpu")
        if torch.is_tensor(V):
            Vn = V.detach().cpu().numpy().astype(np.float32)
        else:
            raise RuntimeError(f"Unexpected brain pt type: {type(V)}")
        np.save(job.brain_npy, Vn)

    if not job.ids_npy.is_file():
        ids = json.loads(job.ids_json.read_text(encoding="utf-8"))
        ids = np.asarray(ids, dtype=np.int64)
        np.save(job.ids_npy, ids)

    # Copy to shared982_efficiency embeds dir
    job.eff_embed_dir.mkdir(parents=True, exist_ok=True)
    dst_brain = job.eff_embed_dir / "brain.npy"
    dst_ids = job.eff_embed_dir / "ids.npy"
    if not dst_brain.is_file():
        dst_brain.write_bytes(job.brain_npy.read_bytes())
    if not dst_ids.is_file():
        dst_ids.write_bytes(job.ids_npy.read_bytes())


def _eval_ccd_and_twoafc(job: Job, *, bootstrap: int, do_eval: bool) -> Tuple[Path, Path]:
    ccd_json = job.eff_metrics_dir / "ccd.json"
    twoafc_json = job.eff_metrics_dir / "twoafc_hard.json"
    # Backward-compat: previous versions incorrectly wrote twoafc_hard.json from CCD outputs.
    # If we detect that stale format, recompute 2AFC via eval_twoafc_embed.py.
    stale_twoafc = False
    if twoafc_json.is_file():
        try:
            old = json.loads(twoafc_json.read_text(encoding="utf-8"))
            src = (old.get("source") or {}).get("raw_metrics_json")
            if isinstance(src, str) and "_tmp_ccd" in src:
                stale_twoafc = True
        except Exception:
            stale_twoafc = True

    if ccd_json.is_file() and twoafc_json.is_file() and not stale_twoafc:
        return ccd_json, twoafc_json
    if not do_eval:
        raise RuntimeError(f"Missing metrics for {job.model_name} and --do_eval not set")

    env = _apply_hf_accel_env(os.environ.copy())

    # 1) CCD
    if not ccd_json.is_file():
        tmp_ccd = job.eff_metrics_dir / "_tmp_ccd"
        tmp_ccd.mkdir(parents=True, exist_ok=True)

        env_ccd = env.copy()
        env_ccd.update(
            {
                "BRAIN_PATH": str(job.brain_pt),
                "IDS_PATH": str(job.ids_json),
                "EVAL_SUBSET": "shared982",
                "CAPTIONS_PATH": str(PROJ / "evals" / "all_captions.pt"),
                "HARD_NEG_JSONL": str(PROJ / "cache" / "hardneg" / "shared982_hardneg_for_ccd.jsonl"),
                "HARD_NEG_REQUIRE_FULL": "0",
                "HARD_NEG_K": "1",
                "BOOTSTRAP": str(int(bootstrap)),
                "RESULT_DIR": str(tmp_ccd),
                "EXP_NAME": f"eff_ccd_{job.model_name}",
            }
        )

        cmd = [sys.executable, str(PROJ / "tools" / "eval_ccd_embed.py")]
        _run(cmd, cwd=PROJ, env=env_ccd, log_path=job.eff_metrics_dir / "ccd_eval.log")

        raw = json.loads((tmp_ccd / "metrics.json").read_text(encoding="utf-8"))
        met = raw.get("metrics", {})
        ci = raw.get("ci", {})

        job.eff_metrics_dir.mkdir(parents=True, exist_ok=True)

        out_ccd = {
            "subj": job.subj,
            "model": job.model,
            "setting": job.setting,
            "seed": job.seed,
            "N": int(raw.get("n_eval", 0)),
            "bootstrap": int(raw.get("bootstrap", bootstrap)),
            "neg_mode": raw.get("neg_mode"),
            "hard_neg_jsonl": raw.get("hard_neg_jsonl"),
            "metrics": {
                "ccd_acc1": float(met.get("ccd_acc1", float("nan"))),
            },
            "ci": {
                "ccd_acc1_ci95": [
                    float(x) for x in (ci.get("acc1_ci95") or [float("nan"), float("nan")])
                ],
            },
            "source": {
                "brain_pt": str(job.brain_pt),
                "ids_json": str(job.ids_json),
                "raw_metrics_json": str((tmp_ccd / "metrics.json")),
            },
        }
        ccd_json.write_text(json.dumps(out_ccd, indent=2) + "\n", encoding="utf-8")

    # 2) Hard-2AFC (embedding-based 2AFC on shared982)
    if stale_twoafc or not twoafc_json.is_file():
        tmp_two = job.eff_metrics_dir / "_tmp_twoafc"
        tmp_two.mkdir(parents=True, exist_ok=True)

        env_two = env.copy()
        env_two.update(
            {
                "BRAIN_PATH": str(job.brain_pt),
                "IDS_PATH": str(job.ids_json),
                "GT_PATH": str(PROJ / "evals" / "all_images.pt"),
                "EVAL_REPR": "pooled",
                "EVAL_SUBSET": "shared982",
                "METRIC": "cosine",
                "BOOTSTRAP": str(int(bootstrap)),
                "RESULT_DIR": str(tmp_two),
                "EXP_NAME": f"eff_twoafc_{job.model_name}",
            }
        )

        cmd = [sys.executable, str(PROJ / "tools" / "eval_twoafc_embed.py")]
        _run(cmd, cwd=PROJ, env=env_two, log_path=job.eff_metrics_dir / "twoafc_eval.log")

        raw = json.loads((tmp_two / "metrics.json").read_text(encoding="utf-8"))
        two = raw.get("twoafc", {}) or {}
        fwd = two.get("brain_to_image", {}) or {}

        out_two = {
            "subj": job.subj,
            "model": job.model,
            "setting": job.setting,
            "seed": job.seed,
            "N": int(raw.get("N", 0)),
            "bootstrap": int(bootstrap),
            "metrics": {
                "twoafc_hard": float(fwd.get("mean", float("nan"))),
            },
            "ci": {
                "twoafc_hard_ci95": [
                    float(x) for x in (fwd.get("ci95") or [float("nan"), float("nan")])
                ],
            },
            "source": {
                "raw_metrics_json": str((tmp_two / "metrics.json")),
            },
        }
        twoafc_json.write_text(json.dumps(out_two, indent=2) + "\n", encoding="utf-8")

    return ccd_json, twoafc_json


def _eval_rsa(job: Job, *, bootstrap: int, do_eval: bool) -> Path:
    rsa_json = job.eff_metrics_dir / "rsa.json"
    if rsa_json.is_file():
        return rsa_json
    if not do_eval:
        raise RuntimeError(f"Missing metrics for {job.model_name} and --do_eval not set")

    tmp_dir = job.eff_metrics_dir / "_tmp_rsa"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env = _apply_hf_accel_env(os.environ.copy())
    env.update(
        {
            "BRAIN_PATH": str(job.brain_pt),
            "IDS_PATH": str(job.ids_json),
            "GT_PATH": str(PROJ / "evals" / "all_images.pt"),
            "EVAL_SUBSET": "shared982",
            "SIM_METRIC": "cosine",
            "BOOTSTRAP": str(int(bootstrap)),
            "SEED": str(int(job.seed)),
            "RESULT_DIR": str(tmp_dir),
            "EXP_NAME": f"eff_rsa_{job.model_name}",
        }
    )

    cmd = [sys.executable, str(PROJ / "tools" / "eval_rsa_embed.py")]
    _run(cmd, cwd=PROJ, env=env, log_path=job.eff_metrics_dir / "rsa_eval.log")

    raw = json.loads((tmp_dir / "metrics.json").read_text(encoding="utf-8"))
    s = (raw.get("rsa", {}) or {}).get("spearman", {})

    out = {
        "subj": job.subj,
        "model": job.model,
        "setting": job.setting,
        "seed": job.seed,
        "N": int(raw.get("N", 0)),
        "pairs": int(raw.get("pairs", 0)),
        "bootstrap": int(raw.get("bootstrap", bootstrap)),
        "metrics": {
            "rsa_rho": float(s.get("rho", float("nan"))),
        },
        "ci": {
            "rsa_ci95": s.get("ci95_bootstrap") or s.get("ci95_fisher") or [float("nan"), float("nan")],
            "ci_source": "bootstrap" if s.get("ci95_bootstrap") else "fisher",
        },
        "source": {
            "raw_metrics_json": str((tmp_dir / "metrics.json")),
        },
    }
    job.eff_metrics_dir.mkdir(parents=True, exist_ok=True)
    rsa_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return rsa_json


def _write_efficiency_summary(jobs: List[Job]) -> Path:
    out_csv = PROJ / "results" / "tables" / "efficiency_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subj",
        "model",
        "setting",
        "seed",
        "N",
        "ccd_acc1",
        "ccd_ci_lo",
        "ccd_ci_hi",
        "twoafc_hard",
        "twoafc_ci_lo",
        "twoafc_ci_hi",
        "rsa_rho",
        "rsa_ci_lo",
        "rsa_ci_hi",
    ]

    rows = []
    for job in jobs:
        ccd = json.loads((job.eff_metrics_dir / "ccd.json").read_text(encoding="utf-8"))
        two = json.loads((job.eff_metrics_dir / "twoafc_hard.json").read_text(encoding="utf-8"))
        rsa = json.loads((job.eff_metrics_dir / "rsa.json").read_text(encoding="utf-8"))

        ccd_ci = ccd.get("ci", {}).get("ccd_acc1_ci95") or [np.nan, np.nan]
        two_ci = two.get("ci", {}).get("twoafc_hard_ci95") or [np.nan, np.nan]
        rsa_ci = rsa.get("ci", {}).get("rsa_ci95") or [np.nan, np.nan]

        rows.append(
            {
                "subj": job.subj,
                "model": job.model,
                "setting": job.setting,
                "seed": job.seed,
                "N": int(ccd.get("N", 0) or 0),
                "ccd_acc1": float(ccd.get("metrics", {}).get("ccd_acc1", np.nan)),
                "ccd_ci_lo": float(ccd_ci[0]),
                "ccd_ci_hi": float(ccd_ci[1]),
                "twoafc_hard": float(two.get("metrics", {}).get("twoafc_hard", np.nan)),
                "twoafc_ci_lo": float(two_ci[0]),
                "twoafc_ci_hi": float(two_ci[1]),
                "rsa_rho": float(rsa.get("metrics", {}).get("rsa_rho", np.nan)),
                "rsa_ci_lo": float(rsa_ci[0]),
                "rsa_ci_hi": float(rsa_ci[1]),
            }
        )

    # deterministic order
    def _key(r: Dict) -> Tuple:
        setting_order = {"1sess": 1, "2sess": 2, "40sess": 40}
        return (int(r["subj"]), str(r["model"]), int(setting_order.get(str(r["setting"]), 999)), int(r["seed"]))

    rows.sort(key=_key)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return out_csv


def _append_main_results(jobs: List[Job]) -> None:
    path = PROJ / "results" / "tables" / "main_results.csv"
    if not path.is_file():
        return

    text = path.read_text(encoding="utf-8").splitlines()
    if not text:
        return

    header = text[0]
    cols = header.split(",")

    def _row_for(job: Job) -> Dict[str, str]:
        ccd = json.loads((job.eff_metrics_dir / "ccd.json").read_text(encoding="utf-8"))
        two = json.loads((job.eff_metrics_dir / "twoafc_hard.json").read_text(encoding="utf-8"))
        rsa = json.loads((job.eff_metrics_dir / "rsa.json").read_text(encoding="utf-8"))

        ccd_ci = ccd.get("ci", {}).get("ccd_acc1_ci95") or ["", ""]

        d = {k: "" for k in cols}
        d["group"] = "shared982_efficiency"
        d["tag"] = job.model_name
        d["subj"] = str(job.subj)
        d["ccd_N"] = str(int(ccd.get("N", 0) or 0))
        d["neg_mode"] = "hardneg"
        d["ccd_acc1"] = str(float(ccd.get("metrics", {}).get("ccd_acc1", float("nan"))))
        d["ccd_acc1_ci95"] = f"[{float(ccd_ci[0]):.4f}, {float(ccd_ci[1]):.4f}]" if ccd_ci else ""
        d["twoafc_hardest"] = str(float(two.get("metrics", {}).get("twoafc_hard", float("nan"))))
        # main_results.csv uses rsa_pearson column name; we store Spearman rho here (documented in RESULTS_INDEX)
        d["rsa_pearson"] = str(float(rsa.get("metrics", {}).get("rsa_rho", float("nan"))))
        return d

    # NOTE: main_results.csv in this repo is not strict CSV (some fields contain commas without quoting).
    # Avoid parsing; instead update rows by prefix "shared982_efficiency,<tag>," to keep the operation idempotent.
    existing_lines = text[1:]

    new_by_prefix: Dict[str, str] = {}
    for job in jobs:
        d = _row_for(job)
        line = ",".join(d.get(c, "") for c in cols)
        prefix = f"shared982_efficiency,{job.model_name},"
        new_by_prefix[prefix] = line

    out_lines: List[str] = []
    seen_prefixes = set()
    for line in existing_lines:
        replaced = False
        for prefix, new_line in new_by_prefix.items():
            if line.startswith(prefix):
                out_lines.append(new_line)
                seen_prefixes.add(prefix)
                replaced = True
                break
        if not replaced:
            out_lines.append(line)

    # Append missing prefixes (not present in file yet)
    for prefix, new_line in new_by_prefix.items():
        if prefix not in seen_prefixes:
            out_lines.append(new_line)

    path.write_text("\n".join([header] + out_lines) + "\n", encoding="utf-8")


def _append_results_index() -> None:
    path = PROJ / "results" / "RESULTS_INDEX.md"
    if not path.is_file():
        return

    md = path.read_text(encoding="utf-8")
    marker = "## Efficiency pipeline"
    if marker in md:
        return

    block = []
    block.append("\n" + marker)
    block.append("")
    block.append("Repro command:")
    block.append("")
    block.append("```bash")
    block.append("python tools/run_efficiency_pipeline.py \\")
    block.append("  --subjs 1 5 \\")
    block.append("  --models baseline textalign_llm \\")
    block.append("  --settings 1 2 40 \\")
    block.append("  --seed 0 \\")
    block.append("  --bootstrap 1000 \\")
    block.append("  --do_train --do_infer --do_eval --do_fig")
    block.append("```")
    block.append("")
    block.append("Artifacts:")
    block.append("")
    block.append("- runs/train/<subj>/<model>/<setting>/seed0/")
    block.append("- runs/infer/<subj>/<model>/<setting>/seed0/")
    block.append("- cache/model_eval_results/shared982_efficiency/embeds/")
    block.append("- cache/model_eval_results/shared982_efficiency/metrics/")
    block.append("- results/tables/efficiency_summary.csv")
    block.append("- results/figures_main/Fig_efficiency_ccd_acc1.png")
    block.append("- results/figures_main/Fig_efficiency_twoafc_hard.png")
    block.append("")
    block.append("Notes:")
    block.append("")
    block.append("- main_results.csv column `rsa_pearson` is filled with RSA Spearman rho for this efficiency block (to avoid schema changes).")

    path.write_text(md.rstrip() + "\n" + "\n".join(block) + "\n", encoding="utf-8")


def _do_fig() -> None:
    cmd = [sys.executable, str(PROJ / "tools" / "make_efficiency_figures.py")]
    subprocess.check_call(cmd, cwd=str(PROJ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjs", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--models", nargs="+", type=str, default=["baseline", "textalign_llm"])
    ap.add_argument("--settings", nargs="+", type=int, default=[1, 2, 40])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--do_infer", action="store_true")
    ap.add_argument("--do_eval", action="store_true")
    ap.add_argument("--do_fig", action="store_true")
    args = ap.parse_args()

    # Ensure HF mirror/cache settings apply to this process as well.
    _apply_hf_accel_env_inplace()

    _fix_broken_runs_symlink()

    # Epochs protocol (A2)
    epochs_by_sessions = {40: 10, 2: 15, 1: 20}

    jobs: List[Job] = []
    for subj in args.subjs:
        for model in args.models:
            for s in args.settings:
                jobs.append(Job(subj=int(subj), model=str(model), num_sessions=int(s), seed=int(args.seed)))

    # Stage: train + ckpt link
    for job in jobs:
        epochs = int(epochs_by_sessions[int(job.num_sessions)])
        _ensure_ckpt(job, epochs=epochs, batch_size=int(args.batch_size), do_train=bool(args.do_train))

    # Stage: infer + export
    for job in jobs:
        _ensure_infer(job, do_infer=bool(args.do_infer))
        _convert_and_copy_embeds(job)

    # Stage: eval
    for job in jobs:
        _eval_ccd_and_twoafc(job, bootstrap=int(args.bootstrap), do_eval=bool(args.do_eval))
        _eval_rsa(job, bootstrap=int(args.bootstrap), do_eval=bool(args.do_eval))

    # Tables + index
    _write_efficiency_summary(jobs)
    _append_main_results(jobs)
    _append_results_index()

    # Figures
    if args.do_fig:
        _do_fig()

    print("[DONE] efficiency pipeline")


if __name__ == "__main__":
    main()
