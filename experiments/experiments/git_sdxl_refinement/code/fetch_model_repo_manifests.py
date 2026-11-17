#!/usr/bin/env python3
"""下载 ykt668/textalign-mindeye2-model 仓库中的 manifest/snapshot/audit 文件到 reports/_model_repo/。"""
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114")
REPO = "ykt668/textalign-mindeye2-model"
DEST = ROOT / "reports" / "_model_repo"
DEST.mkdir(parents=True, exist_ok=True)

targets = [
    "README.md",
    "snapshots/2026-04-12/EXPERIMENT_SNAPSHOT_2026-04-12.md",
    "snapshots/2026-04-12/HF_UPLOAD_MANIFEST.json",
    "snapshots/2026-04-12/RECOVERY_MANIFEST.json",
    "snapshots/2026-04-12/UPLOAD_PLAN.json",
    "snapshots/2026-04-12/audit_runs/s1_textalign_stage1_FINAL_BEST_32_20260115_004557/audit_report_final.md",
    "snapshots/2026-04-12/audit_runs/ours_s1_stage1_final_best32_lastpth_v1/audit_report.md",
    "snapshots/2026-04-13/EXPERIMENT_SNAPSHOT_2026-04-13.md",
    "snapshots/2026-04-13/reports/HF_UPLOAD_RESULT.json",
    "snapshots/2026-04-13/reports/RECOVERY_MANIFEST.json",
]

for rp in targets:
    try:
        p = hf_hub_download(repo_id=REPO, filename=rp, repo_type="model",
                            token=True, cache_dir=os.environ["HF_HUB_CACHE"])
        safe = rp.replace("/", "__")
        shutil.copyfile(p, DEST / safe)
        print("OK", rp, "->", safe, Path(p).stat().st_size, "bytes")
    except Exception as e:
        print("ERR", rp, type(e).__name__, str(e)[:200])
