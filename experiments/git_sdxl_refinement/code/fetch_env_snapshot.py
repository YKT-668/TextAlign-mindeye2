#!/usr/bin/env python3
"""下载 ENV_SNAPSHOT git 文件 + 资产链审计文件。"""
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114")
REPO = "ykt668/textalign-mindeye2-model"
DEST = ROOT / "reports" / "_model_repo"
DEST.mkdir(parents=True, exist_ok=True)

targets = [
    "ENV_SNAPSHOT/git_rev_parse_head.txt",
    "ENV_SNAPSHOT/git_remote_v.txt",
    "ENV_SNAPSHOT/git_status_short.txt",
    "ENV_SNAPSHOT/git_status_full.txt",
    "ENV_SNAPSHOT/git_diff_stat.txt",
    "ENV_SNAPSHOT/pip_freeze.txt",
    "ENV_SNAPSHOT/python_version.txt",
    "ENV_SNAPSHOT/conda_env_export_no_builds.yml",
    "ENV_SNAPSHOT/key_dirs_tree_depth2.txt",
    "ENV_SNAPSHOT/df_mnt_work.txt",
    "ENV_SNAPSHOT/mount_mnt_work.txt",
    "audit_runs/closure_audit_20260412/s1_s2_s5_s7_asset_chain_audit.json",
    "audit_runs/closure_audit_20260412/baseline_closure_audit_report.md",
    "audit_runs/audit_s1_stage1_final_best32/audit_report.md",
]

log = []
for rp in targets:
    try:
        p = hf_hub_download(repo_id=REPO, filename=rp, repo_type="model",
                            token=True, cache_dir=os.environ["HF_HUB_CACHE"])
        safe = rp.replace("/", "__")
        shutil.copyfile(p, DEST / safe)
        log.append(f"OK {rp} ({Path(p).stat().st_size}B)")
    except Exception as e:
        log.append(f"ERR {rp} {type(e).__name__} {str(e)[:120]}")
(DEST / "_fetch2.log").write_text("\n".join(log), encoding="utf-8")
