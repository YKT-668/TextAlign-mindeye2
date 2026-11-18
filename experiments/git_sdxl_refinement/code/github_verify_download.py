"""GitHub 归档独立验证：从 raw.githubusercontent.com 逐文件下载（全新目录，不经本地 git）
并比对 SHA256 与本地 SHA256SUMS。另通过 API 校验分支/tag 指向。
"""
import hashlib
import json
import time
import urllib.request

BRANCH = "archive/git-sdxl-refinement-final-2026-11-17"
TAG = "git-sdxl-refinement-final-2026-11-17"
REPO = "YKT-668/TextAlign-mindeye2"
SUMS = "/root/archive_work/TextAlign-mindeye2/experiments/git_sdxl_refinement/SHA256SUMS"
VERIFY_DIR = "/root/verify_gh_download"

import os
os.makedirs(VERIFY_DIR, exist_ok=True)

# 期望哈希
expect = {}
for line in open(SUMS):
    if "  " in line:
        h, p = line.strip().split("  ", 1)
        expect[p] = h

ok, fail = 0, []
for i, (rel, want) in enumerate(sorted(expect.items())):
    url = f"https://raw.githubusercontent.com/{REPO}/{TAG}/{rel}"
    data = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            break
        except Exception as e:
            time.sleep(10 * attempt)
    if data is None:
        fail.append((rel, "download_failed"))
        print(f"[{i+1}/{len(expect)}] FAIL(download) {rel}", flush=True)
        continue
    got = hashlib.sha256(data).hexdigest()
    status = "OK" if got == want else "MISMATCH"
    if status == "OK":
        ok += 1
    else:
        fail.append((rel, f"{got}!={want}"))
    with open(os.path.join(VERIFY_DIR, rel.replace("/", "__")), "wb") as fh:
        fh.write(data)
    if (i + 1) % 20 == 0 or status != "OK":
        print(f"[{i+1}/{len(expect)}] {status} {rel}", flush=True)

# API 校验分支与 tag 指向
refs = {}
for kind, ref in (("branch", BRANCH), ("tag", TAG)):
    url = f"https://api.github.com/repos/{REPO}/git/refs/{'heads' if kind=='branch' else 'tags'}/{ref}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            refs[kind] = json.load(r)["object"]
    except Exception as e:
        refs[kind] = {"error": repr(e)}

result = {"n_expected": len(expect), "n_ok": ok, "n_fail": len(fail), "fails": fail,
          "refs": refs, "expected_commit": "d08791d146d36ca57bb803e7b3ad43dde675c213"}
print(json.dumps({"n_ok": ok, "n_fail": len(fail), "refs": refs}, indent=1))
json.dump(result, open("/root/verify_gh_download/gh_verify_result.json", "w"), indent=1)
print("GH_VERIFY_DONE", "PASS" if len(fail) == 0 else "FAIL")
