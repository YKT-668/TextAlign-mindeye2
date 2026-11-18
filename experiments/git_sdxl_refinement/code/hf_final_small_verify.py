"""对最后补传的脚本做独立回下载 + SHA256 比对（包含此前 7 个未下载文件）。"""
import hashlib
import json
import os
import urllib.request

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
PFX = "supplementary/git_sdxl_refinement/2026-11-17"
TOKEN = open(f"{ROOT}/hf_home/token").read().strip()
INV = json.load(open(f"{ROOT}/reports/_archive_inventory_working.json"))
DL = "/root/hf_verify_download"

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

targets = [f for f in INV["files"]
           if "hf" in f["archive_destination"] and f["file_size_bytes"] <= 2 * 1024 * 1024]
ok, fail = 0, []
for f in targets:
    rp = remote_path(f["relative_path"])
    local = os.path.join(DL, rp)
    need = (not os.path.exists(local)) or \
           hashlib.sha256(open(local, "rb").read()).hexdigest() != f["sha256"]
    if need:
        url = f"https://huggingface.co/datasets/ykt668/textalign-mindeye2-data/resolve/main/{rp}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
        data = None
        for attempt in range(4):
            try:
                data = urllib.request.urlopen(req, timeout=90).read()
                break
            except Exception:
                import time
                time.sleep(8 * (attempt + 1))
        if data is None:
            fail.append((rp, "download_failed"))
            continue
        os.makedirs(os.path.dirname(local), exist_ok=True)
        open(local, "wb").write(data)
    got = hashlib.sha256(open(local, "rb").read()).hexdigest()
    if got == f["sha256"]:
        ok += 1
    else:
        fail.append((rp, "sha_mismatch"))

print(json.dumps({"n_small": len(targets), "n_ok": ok, "n_fail": len(fail), "fails": fail},
                 indent=1, ensure_ascii=False))
json.dump({"n_small": len(targets), "n_ok": ok, "fails": fail},
          open(f"{ROOT}/logs/hf_final_small_verify.json", "w"), indent=1)
print("FINAL_SMALL_VERIFY", "PASS" if not fail else "FAIL")
