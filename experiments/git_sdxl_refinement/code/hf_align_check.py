"""核对远端：169 个 HF 目标文件是否全部存在于远端，且 LFS 内容哈希一致。"""
import json, os
from huggingface_hub import HfApi

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
PFX = "supplementary/git_sdxl_refinement/2026-11-17"
TOKEN = open(f"{ROOT}/hf_home/token").read().strip()
INV = json.load(open(f"{ROOT}/reports/_archive_inventory_working.json"))


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


api = HfApi(token=TOKEN)
info = api.repo_info("ykt668/textalign-mindeye2-data", repo_type="dataset",
                     revision="main", files_metadata=True)
remote = {}
for s in info.siblings:
    if s.rfilename.startswith(PFX):
        lfs = getattr(s, "lfs", None)
        remote[s.rfilename] = (s.size, lfs["sha256"] if lfs else None)

print("remote_sha:", info.sha[:12], "remote_files:", len(remote))

missing, size_bad, lfs_bad, ok = [], [], [], 0
for f in INV["files"]:
    if "hf" not in f["archive_destination"]:
        continue
    rp = remote_path(f["relative_path"])
    if rp not in remote:
        missing.append(rp)
        continue
    rsize, rsha = remote[rp]
    if rsize != f["file_size_bytes"]:
        size_bad.append((rp, rsize, f["file_size_bytes"]))
        continue
    if rsha is not None and rsha != f["sha256"]:
        lfs_bad.append(rp)
        continue
    ok += 1

# 同路径冲突检查：两个本地文件映射到同一远程路径
from collections import Counter
paths = [remote_path(f["relative_path"]) for f in INV["files"] if "hf" in f["archive_destination"]]
dups = [p for p, c in Counter(paths).items() if c > 1]

print(json.dumps({"n_expected": len(paths), "n_ok": ok, "missing": missing,
                  "size_mismatch": size_bad, "lfs_mismatch": lfs_bad,
                  "duplicate_remote_paths": dups}, indent=1, ensure_ascii=False))
print("REMOTE_ALIGN_CHECK", "PASS" if not (missing or size_bad or lfs_bad or dups) else "FAIL")
