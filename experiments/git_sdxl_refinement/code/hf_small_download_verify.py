"""校验已回下载的小文件：SHA256 与本地 inventory 逐一比对。"""
import hashlib
import json
import os

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
DL = "/root/hf_verify_download"
PFX = "supplementary/git_sdxl_refinement/2026-11-17"
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


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


ok, mismatch, not_dl = 0, [], []
for f in INV["files"]:
    if "hf" not in f["archive_destination"] or f["file_size_bytes"] > 2 * 1024 * 1024:
        continue
    p = os.path.join(DL, remote_path(f["relative_path"]))
    if not os.path.exists(p):
        not_dl.append(remote_path(f["relative_path"]))
        continue
    if sha256(p) == f["sha256"]:
        ok += 1
    else:
        mismatch.append(remote_path(f["relative_path"]))

res = {"small_files_checked": ok + len(mismatch) + len(not_dl), "sha256_ok": ok,
       "mismatch": mismatch, "not_downloaded": not_dl}
print(json.dumps(res, indent=1, ensure_ascii=False))
json.dump(res, open(f"{ROOT}/logs/hf_small_download_verify.json", "w"), indent=1)
print("SMALL_DL_VERIFY", "PASS" if not mismatch else "FAIL")
