"""HF 归档独立回下载验证：全新 local_dir 从远程逐文件下载并比对 SHA256。

覆盖所有 archive_destination 含 hf 的文件（含 17 个大文件，约 8.2GB 重下载）。
不使用本地 HF 缓存（local_dir 模式直落盘）。
"""
import hashlib
import json
import os
import sys
import time

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
REPO = "ykt668/textalign-mindeye2-data"
PFX = "supplementary/git_sdxl_refinement/2026-11-17"
DL = "/root/hf_verify_download"
TOKEN = open(f"{ROOT}/hf_home/token").read().strip()
INV = json.load(open(f"{ROOT}/reports/_archive_inventory_working.json"))
LOG = f"{ROOT}/logs/hf_verify_download.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


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


hf_files = [f for f in INV["files"] if "hf" in f["archive_destination"]]
log(f"=== verify begin: {len(hf_files)} files ===")

from huggingface_hub import snapshot_download
snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=DL, token=TOKEN,
                  allow_patterns=[f"{PFX}/**"], max_workers=1)
log("snapshot_download finished")

ok, fail = 0, []
for f in hf_files:
    rp = remote_path(f["relative_path"])
    local = os.path.join(DL, rp)
    if not os.path.exists(local):
        fail.append((rp, "not_downloaded"))
        log(f"FAIL(not_downloaded) {rp}")
        continue
    got = sha256(local)
    if got == f["sha256"]:
        ok += 1
        if f["file_size_bytes"] > 100 * 1024 * 1024 or (i := 0):
            log(f"OK {rp} {f['file_size_bytes']/1e6:.1f}MB")
    else:
        fail.append((rp, f"sha_mismatch {got[:12]}"))
        log(f"FAIL(sha) {rp}")

result = {"n_expected": len(hf_files), "n_ok": ok, "n_fail": len(fail), "fails": fail}
log("RESULT " + json.dumps({"n_ok": ok, "n_fail": len(fail)}))
json.dump(result, open(f"{ROOT}/logs/hf_verify_result.json", "w"), indent=1)
print("HF_VERIFY_DONE", "PASS" if not fail else "FAIL")
