#!/usr/bin/env python3
"""下载 ID 映射所需的小文件（<1MB），保存到 reports/_mapping/。"""
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114")
DEST = ROOT / "reports" / "_mapping"
DEST.mkdir(parents=True, exist_ok=True)

# (repo, repo_type, revision, path)
files = [
    ("pscotti/mindeyev2", "dataset", "3f20ccbe614876c83894e148cc36530c4b04caa7", "shared1000.npy"),
    ("pscotti/mindeyev2", "dataset", "3f20ccbe614876c83894e148cc36530c4b04caa7", "test_73k_images.npy"),
    ("pscotti/mindeyev2", "dataset", "3f20ccbe614876c83894e148cc36530c4b04caa7", "COCO_73k_subj_indices.hdf5"),
    ("ykt668/textalign-mindeye2-model", "model", "ceb32860e6a9e1c775e30da8f8e884e0f1926795",
     "snapshots/2026-04-12/audit_runs/s1_textalign_stage1_FINAL_BEST_32_20260115_004557/shared982_ids_manifest.csv"),
]

log = []
for repo, rtype, rev, path in files:
    try:
        p = hf_hub_download(repo_id=repo, filename=path, repo_type=rtype, revision=rev,
                            token=True, cache_dir=os.environ["HF_HUB_CACHE"])
        safe = path.replace("/", "__")
        shutil.copyfile(p, DEST / safe)
        log.append(f"OK {safe} {Path(p).stat().st_size}B")
    except Exception as e:
        log.append(f"ERR {path} {type(e).__name__} {str(e)[:120]}")

(DEST / "_fetch.log").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
