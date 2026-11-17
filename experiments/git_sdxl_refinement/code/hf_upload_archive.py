"""HF Dataset 上传：supplementary/git_sdxl_refinement/2026-11-17/（私有仓库）。

策略：串行上传（单 Worker）；大文件逐个 upload_file（xet 后端支持分块断点）；
每文件上传后记录日志与远端校验信息；失败重试 3 次；不暴露 token（从 hf_home/token 读）。
远程布局：
  supplementary/git_sdxl_refinement/2026-11-17/
    stage1/ stage2/ evaluation/{bigg_features,p3_features,per_sample,summaries,logs}
    reports/ scripts/ SHA256SUMS
"""
import json
import os
import sys
import time

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
PFX = "supplementary/git_sdxl_refinement/2026-11-17"
REPO = "ykt668/textalign-mindeye2-data"
INV = json.load(open(f"{ROOT}/reports/_archive_inventory_working.json"))
LOG = f"{ROOT}/logs/hf_upload.log"
TOKEN = open(f"{ROOT}/hf_home/token").read().strip()


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def remote_path(rel):
    rel = rel.replace(os.sep, "/")
    if rel.startswith("run/full_stage1/"):
        return f"{PFX}/stage1/{os.path.basename(rel)}"
    if rel.startswith("run/full_stage2/"):
        return f"{PFX}/stage2/{os.path.basename(rel)}"
    if rel.startswith("evaluation/precomputed_features/bigg/"):
        return f"{PFX}/evaluation/bigg_features/{os.path.basename(rel)}"
    if rel.startswith("evaluation/cpu_mindeye2_p3_clip_swav/"):
        sub = "features/" + rel.split("cpu_mindeye2_p3_clip_swav/")[1]
        return f"{PFX}/evaluation/{sub}"
    if rel.startswith("evaluation/"):
        sub = rel.split("evaluation/")[1]
        if "per_sample" in os.path.basename(rel) or rel.endswith("lowlevel_per_sample.csv"):
            sub = f"per_sample/{os.path.basename(rel)}"
        return f"{PFX}/evaluation/{sub}"
    if rel.startswith("reports/"):
        return f"{PFX}/reports/{os.path.basename(rel)}"
    if rel.startswith("scripts/"):
        return f"{PFX}/scripts/{os.path.basename(rel)}"
    return f"{PFX}/{rel}"


def main(only_big=None):
    from huggingface_hub import HfApi
    api = HfApi(token=TOKEN)
    files = INV["files"]
    if only_big is True:
        files = [f for f in files if f["file_size_bytes"] > 2 * 1024 * 1024
                 and f["archive_destination"] in ("hf", "github+hf", "hf+github")]
    elif only_big is False:
        files = [f for f in files if f["file_size_bytes"] <= 2 * 1024 * 1024
                 and "hf" in f["archive_destination"]]
    todo = [f for f in files if "hf" in f["archive_destination"]]
    total = sum(f["file_size_bytes"] for f in todo)
    log(f"=== upload begin: {len(todo)} files, {total/1e6:.1f} MB (only_big={only_big}) ===")
    n_ok = n_fail = 0
    for i, f in enumerate(todo):
        src = os.path.join(ROOT, f["relative_path"])
        rp = remote_path(f["relative_path"])
        for attempt in (1, 2, 3):
            t0 = time.time()
            try:
                api.upload_file(path_or_fileobj=src, path_in_repo=rp, repo_id=REPO,
                                repo_type="dataset",
                                commit_message=f"archive {f['category']} {os.path.basename(rp)}")
                dt = time.time() - t0
                speed = f["file_size_bytes"] / 1e6 / max(dt, 0.1)
                log(f"OK [{i+1}/{len(todo)}] {rp} {f['file_size_bytes']/1e6:.1f}MB "
                    f"{dt:.0f}s {speed:.1f}MB/s")
                n_ok += 1
                break
            except Exception as e:
                log(f"RETRY{attempt} {rp}: {repr(e)[:180]}")
                time.sleep(15 * attempt)
        else:
            log(f"FAIL {rp}")
            n_fail += 1
    log(f"=== upload end: ok={n_ok} fail={n_fail} ===")
    return n_fail


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    fails = main(only_big={"big": True, "small": False, "all": None}[mode])
    sys.exit(1 if fails else 0)
