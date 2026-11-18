"""HF 补传（upload_folder 单次 commit，避开逐文件 commit 的 429 限流）。

用法: python hf_upload_folder.py <staging_root>
staging_root 内的相对路径 = 远程 PFX 下的相对路径。
"""
import os
import sys
import time

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
PFX = "supplementary/git_sdxl_refinement/2026-11-17"
REPO = "ykt668/textalign-mindeye2-data"
TOKEN = open(f"{ROOT}/hf_home/token").read().strip()
LOG = f"{ROOT}/logs/hf_upload_folder.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


stage = sys.argv[1]
assert os.path.isdir(stage)
from huggingface_hub import HfApi
api = HfApi(token=TOKEN)
t0 = time.time()
api.upload_folder(folder_path=stage, path_in_repo=PFX, repo_id=REPO, repo_type="dataset",
                  commit_message="archive: remaining small files + final manifests (2026-11-17)",
                  commit_description="single-commit pass to avoid per-file commit rate limits")
log(f"upload_folder done in {time.time()-t0:.0f}s")
print("FOLDER_UPLOAD_DONE")
