"""构建 HF 小文件补传暂存目录：/root/hf_small_stage/<PFX 下相对路径>。

只放 ≤2MB 文件（大文件 17 个已全部上传成功，避免 upload_folder 重扫 8GB）。
"""
import json
import os
import shutil

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
STAGE = "/root/hf_small_stage"
PFX = "supplementary/git_sdxl_refinement/2026-11-17"
INV = json.load(open(f"{ROOT}/reports/_archive_inventory_working.json"))

if os.path.isdir(STAGE):
    shutil.rmtree(STAGE)
os.makedirs(STAGE)

def remote_path(rel):
    rel = rel.replace(os.sep, "/")
    if rel.startswith("run/full_stage1/"):
        return f"{PFX}/stage1/{os.path.basename(rel)}"
    if rel.startswith("run/full_stage2/"):
        return f"{PFX}/stage2/{os.path.basename(rel)}"
    if rel.startswith("evaluation/precomputed_features/bigg/"):
        return f"{PFX}/evaluation/bigg_features/{os.path.basename(rel)}"
    if rel.startswith("evaluation/cpu_mindeye2_p3_clip_swav/"):
        return f"{PFX}/evaluation/features/" + rel.split("cpu_mindeye2_p3_clip_swav/")[1]
    if rel.startswith("evaluation/"):
        sub = rel.split("evaluation/")[1]
        if "per_sample" in os.path.basename(rel) and rel.count("/") == 1:
            sub = f"per_sample/{os.path.basename(rel)}"
        return f"{PFX}/evaluation/{sub}"
    if rel.startswith("reports/"):
        return f"{PFX}/reports/{os.path.basename(rel)}"
    if rel.startswith("scripts/"):
        return f"{PFX}/scripts/{os.path.basename(rel)}"
    return f"{PFX}/{rel}"

n = 0
for f in INV["files"]:
    if "hf" not in f["archive_destination"]:
        continue
    if f["file_size_bytes"] > 2 * 1024 * 1024:
        continue
    rp = remote_path(f["relative_path"]).replace(f"{PFX}/", "")
    dst = os.path.join(STAGE, rp)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, f["relative_path"]), dst)
    n += 1
print("staged small files:", n)
