"""GitHub 归档验证 v2：git/trees API + 本地 git-blob-sha1 逐文件比对（避开 raw/git-fetch 不稳通道）。

原理：blob sha1 = sha1("blob <size>\\0" + content)，是内容的确定性指纹；
递归树一次返回分支顶端全部 (path, blob_sha, size)，与本地暂存文件逐个比对。
另抽样 3 个文件经 contents API 下载做字节级复核。
"""
import base64
import hashlib
import json
import os
import time
import urllib.request

REPO = "YKT-668/TextAlign-mindeye2"
BRANCH = "archive/git-sdxl-refinement-final-2026-11-17"
TIP = "ab8bfc98eed0dbae40191949a03f0e972eb5282d"
STAGE = "/root/archive_work/TextAlign-mindeye2/experiments/git_sdxl_refinement"
TOKEN = None
for line in open("/root/.git-credentials"):
    if line.startswith("password="):
        TOKEN = line.strip().split("=", 1)[1]


def api(url):
    last = None
    for attempt in range(1, 7):
        try:
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"token {TOKEN}")
            req.add_header("Accept", "application/vnd.github+json")
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(12 * attempt)
    raise last


def git_blob_sha1(path):
    data = open(path, "rb").read()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


tree = api(f"https://api.github.com/repos/{REPO}/git/trees/{TIP}?recursive=1")
remote = {e["path"]: (e["sha"], e.get("size")) for e in tree["tree"] if e["type"] == "blob"}
print(f"remote tree: {tree['truncated']=} blobs={len(remote)}")

ok, fail = 0, []
local_files = []
for base, _, files in os.walk(STAGE):
    for f in files:
        p = os.path.join(base, f)
        local_files.append((os.path.relpath(p, STAGE).replace(os.sep, "/"), p))
print(f"local files: {len(local_files)}")

PREFIX = "experiments/git_sdxl_refinement/"
for rel, p in sorted(local_files):
    rkey = f"{PREFIX}{rel}"
    if rkey not in remote:
        fail.append((rel, "missing_on_remote"))
        continue
    sha, size = remote[rkey]
    if sha != git_blob_sha1(p):
        fail.append((rel, "sha1_mismatch"))
        continue
    if size is not None and size != os.path.getsize(p):
        fail.append((rel, "size_mismatch"))
        continue
    ok += 1

# 抽样字节级复核（contents API base64；容错）
spot = ["README.md", "docs/FINAL_GIT_SDXL_REFINEMENT_EXPERIMENT.md", "SHA256SUMS"]
spot_results = {}
for rel in spot:
    try:
        d = api(f"https://api.github.com/repos/{REPO}/contents/experiments/git_sdxl_refinement/{rel}?ref={BRANCH}")
        remote_bytes = base64.b64decode(d["content"])
        local_bytes = open(os.path.join(STAGE, rel), "rb").read()
        spot_results[rel] = (remote_bytes == local_bytes, d["sha"], d["size"])
    except Exception as e:
        spot_results[rel] = ("api_error", repr(e)[:120], None)

refs = {"branch": api(f"https://api.github.com/repos/{REPO}/git/refs/heads/{BRANCH}")["object"],
        "tag": api(f"https://api.github.com/repos/{REPO}/git/refs/tags/git-sdxl-refinement-final-2026-11-17")["object"]}

result = {"tip": TIP, "n_local": len(local_files), "n_remote_blobs": len(remote),
          "n_ok": ok, "n_fail": len(fail), "fails": fail,
          "spot_checks": spot_results, "refs": refs}
print(json.dumps(result, indent=1, ensure_ascii=False)[:2000])
json.dump(result, open("/root/archive_work/gh_verify_v2.json", "w"), indent=1)
print("GH_VERIFY_V2_DONE", "PASS" if len(fail) == 0 and all(v[0] for v in spot_results.values()) else "FAIL")
