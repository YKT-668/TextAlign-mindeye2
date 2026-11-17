#!/usr/bin/env python3
"""下载私有仓库中的关键 recovery/provenance 参考文档到 reports/provenance/。"""
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(os.environ["ROOT"])
REPO = "ykt668/textalign-mindeye2-data"
REV = "2c612c5c2ca3c5344854edbd7028769cb254ab5a"
DEST = ROOT / "reports" / "provenance"
DEST.mkdir(parents=True, exist_ok=True)

targets = [
    "final14_archive/followup_experiments/recovery_metadata/readme3.md",
    "final14_archive/followup_experiments/recovery_metadata/READMЕ3_A100_FINAL_INFERENCE_REPORT.md",
    "final14_archive/followup_experiments/fast_validation/FINAL/13_blocker_closure/model_alignment/CANONICAL_SHARED982_SOURCE.md",
    "final14_archive/followup_experiments/fast_validation/FINAL/13_blocker_closure/model_alignment/N1000_ALL_IDS_ALIGNMENT.tsv",
    ".gitattributes",
]

for rp in targets:
    try:
        p = hf_hub_download(repo_id=REPO, filename=rp, repo_type="dataset",
                            revision=REV, token=True, cache_dir=os.environ["HF_HUB_CACHE"])
        safe = rp.replace("/", "__")
        shutil.copyfile(p, DEST / safe)
        print("OK", rp, "->", safe, Path(p).stat().st_size, "bytes")
    except Exception as e:
        print("ERR", rp, type(e).__name__, str(e)[:200])
