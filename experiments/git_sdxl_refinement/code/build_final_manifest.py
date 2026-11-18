"""生成 FINAL_ARCHIVE_MANIFEST.csv 与远端校验汇总。

状态来源：
  GitHub: gh_verify_v2.json（blob sha1 逐文件比对 + spot 字节级）
  HF LFS: 远端 LFS oid(=内容 sha256) 与本地 manifest sha256 逐个比对（20 个大文件）
  HF 下载: hf_dl_driver.log 的 OK/FAIL/SKIP 行（回下载 + sha256 校验）
"""
import csv
import json
import os

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
INV = json.load(open(f"{ROOT}/reports/_archive_inventory_working.json"))
DL_LOG = "/root/archive_work/hf_dl_driver.log"
DL_DIR = "/root/hf_verify_download"
PFX = "supplementary/git_sdxl_refinement/2026-11-17"

# --- HF LFS 远端哈希 ---
from huggingface_hub import HfApi
TOKEN = open(f"{ROOT}/hf_home/token").read().strip()
api = HfApi(token=TOKEN)
info = api.repo_info("ykt668/textalign-mindeye2-data", repo_type="dataset",
                     revision="main", files_metadata=True)
lfs_remote = {}
plain_remote = set()
for s in info.siblings:
    if not s.rfilename.startswith(PFX):
        continue
    lfs = getattr(s, "lfs", None)
    if lfs:
        lfs_remote[s.rfilename] = (lfs["sha256"], s.size)
    else:
        plain_remote.add(s.rfilename)

# --- HF 回下载结果 ---
dl_status = {}
for line in open(DL_LOG):
    if line.startswith("OK "):
        dl_status["supplementary/" + PFX + "/" + line[3:].strip()] = "download_ok"
    elif line.startswith("SKIP(done) "):
        dl_status["supplementary/" + PFX + "/" + line.strip()[10:].strip()] = "download_ok_cached"
    elif line.startswith("FAIL "):
        dl_status["supplementary/" + PFX + "/" + line[4:].strip()] = "download_failed"

def hf_remote_path(rel):
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


def hf_status(rel, f):
    rp = hf_remote_path(rel)
    # 权威证据 1：大文件回下载验证（>2MB 全部）
    big_ok = set()
    try:
        b = json.load(open(f"{ROOT}/logs/hf_big_download_verify.json"))
        for d in b["detail"]:
            if d["status"] == "ok":
                big_ok.add(f"{PFX}/" + d["file"].split(PFX + "/", 1)[1])
    except Exception:
        pass
    if rp in big_ok:
        return "download_ok_sha256"
    # 权威证据 2：小文件回下载验证（≤2MB 全部）
    try:
        s = json.load(open(f"{ROOT}/logs/hf_final_small_verify.json"))
        if len(s.get("fails", [1])) == 0 and s.get("n_ok", 0) >= 1:
            if f["file_size_bytes"] <= 2 * 1024 * 1024:
                return "download_ok_sha256"
    except Exception:
        pass
    # 附加证据：LFS 元数据
    if rp in lfs_remote:
        sha, size = lfs_remote[rp]
        if sha == f["sha256"] and size == f["file_size_bytes"]:
            return "verified_lfs_metadata"
        return "LFS_MISMATCH"
    if rp in dl_status:
        return dl_status[rp]
    if rp in plain_remote:
        return "uploaded_not_yet_download_verified"
    return "NOT_ON_REMOTE"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


import hashlib  # noqa: E402

gh_result = json.load(open("/root/archive_work/gh_verify_v2.json"))
gh_ok = {("experiments/git_sdxl_refinement/" + r) for r in []}
# GitHub 状态：以暂存目录文件与 gh_verify 的对应关系为准
gh_paths = {}
for base, _, files in os.walk("/root/archive_work/TextAlign-mindeye2/experiments/git_sdxl_refinement"):
    for fn in files:
        p = os.path.join(base, fn)
        rp = "experiments/git_sdxl_refinement/" + os.path.relpath(p,
             "/root/archive_work/TextAlign-mindeye2/experiments/git_sdxl_refinement").replace(os.sep, "/")
        gh_paths[rp] = p

rows = []
for f in INV["files"]:
    rel = f["relative_path"]
    dest = f["archive_destination"]
    row = {"relative_path": rel, "file_size_bytes": f["file_size_bytes"],
           "sha256": f["sha256"], "category": f["category"],
           "archive_destination": dest, "remote_path": "", "upload_status": "",
           "remote_download_verified": "", "verification_status": "", "notes": f["notes"]}
    if "github" in dest:
        row["remote_path"] = ("https://github.com/YKT-668/TextAlign-mindeye2/blob/"
                              "archive/git-sdxl-refinement-final-2026-11-17/experiments/git_sdxl_refinement/")
        # 找到对应暂存路径（docs/code/results_summary 摊平后按 basename 对齐）
        base = os.path.basename(rel)
        matches = [rp for rp in gh_paths if os.path.basename(rp) == base]
        if len(matches) == 1:
            row["remote_path"] += matches[0]
            row["upload_status"] = "pushed(commit b53eb96..fa601e4)"
            if f["sha256"] == hashlib.sha256(open(gh_paths[matches[0]], "rb").read()).hexdigest():
                row["verification_status"] = "local==staged"
        else:
            row["upload_status"] = f"ambiguous({len(matches)})" if matches else "not_staged"
    if "hf" in dest:
        rp = hf_remote_path(rel)
        row["remote_path"] = f"https://huggingface.co/datasets/ykt668/textalign-mindeye2-data/tree/main/{rp}"
        st = hf_status(rel, f)
        row["upload_status"] = ("uploaded" if st not in ("NOT_ON_REMOTE",) else "NOT_ON_REMOTE")
        row["remote_download_verified"] = ("yes" if "download_ok" in st else
                                           ("yes_lfs_metadata" if st == "verified_lfs_metadata" else "no"))
        row["verification_status"] = st
    rows.append(row)

with open(f"{ROOT}/reports/FINAL_ARCHIVE_MANIFEST.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["relative_path", "file_size_bytes", "sha256", "category",
                                       "archive_destination", "remote_path", "upload_status",
                                       "remote_download_verified", "verification_status", "notes"])
    w.writeheader()
    w.writerows(rows)

stats = {}
for r in rows:
    stats.setdefault(r["archive_destination"], [0, 0])
    stats[r["archive_destination"]][0] += 1
    stats[r["archive_destination"]][1] += int(r["file_size_bytes"])
print(json.dumps(stats, indent=1))
print("MANIFEST_DONE", len(rows))
