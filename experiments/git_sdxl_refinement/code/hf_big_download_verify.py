"""校验回下载的全部大文件（>2MB）SHA256。"""
import hashlib
import json
import os

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
PFX = "supplementary/git_sdxl_refinement/2026-11-17"
DL = "/root/hf_verify_download"
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
    return f"{PFX}/{rel}"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


ok, result = 0, []
for f in INV["files"]:
    if "hf" not in f["archive_destination"] or f["file_size_bytes"] <= 2 * 1024 * 1024:
        continue
    p = os.path.join(DL, remote_path(f["relative_path"]))
    if not os.path.exists(p):
        result.append({"file": remote_path(f["relative_path"]), "status": "not_downloaded"})
        continue
    got = sha256(p)
    st = "ok" if got == f["sha256"] else "sha_mismatch"
    if st == "ok":
        ok += 1
    result.append({"file": remote_path(f["relative_path"]), "status": st,
                   "size": os.path.getsize(p)})

bad = [r for r in result if r["status"] != "ok"]
out = {"n_big": len(result), "n_ok": ok, "n_bad": len(bad), "detail": result}
print(json.dumps(out, indent=1, ensure_ascii=False)[:1500])
json.dump(out, open(f"{ROOT}/logs/hf_big_download_verify.json", "w"), indent=1)
print("BIG_DL_VERIFY", "PASS" if not bad else "FAIL")
