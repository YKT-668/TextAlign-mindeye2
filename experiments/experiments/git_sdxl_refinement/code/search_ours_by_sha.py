#!/usr/bin/env python3
"""按 SHA256 在私有仓库 LFS 对象中搜索 Ours S1 三个文件（排除改名可能）。"""
import os
from huggingface_hub import HfApi

REPO = "ykt668/textalign-mindeye2-data"
REV = "2c612c5c2ca3c5344854edbd7028769cb254ab5a"
api = HfApi()

targets = {
    "df704fca629453c79f80b83a859694e468f1aad994257d31314e86dc7f8a85b2": "ours_enhancedrecons",
    "a6b49190a20f79ce990bc647d7cca032e9503383877c3728bec647ef8c069758": "ours_predcaptions",
    "0a837980a67886e2dfc5e0f3b08facdeab7a0d0b7d9ae6309f1f604ca7812d3c": "ours_ids",
}

info = api.dataset_info(REPO, revision=REV, files_metadata=True)
print("total siblings:", len(info.siblings))

found = {}
name_hits = []
for s in info.siblings:
    sha = s.lfs.sha256 if s.lfs else None
    if sha in targets:
        found[sha] = (s.rfilename, s.size)
    low = s.rfilename.lower()
    if any(k in low for k in ("step6d", "enhancedrecon", "predcaption", "ours_llmhard", "all_ids.pt", "full_ours")):
        name_hits.append((s.rfilename, s.size, sha))

print("=== by SHA256 ===")
for sha, v in targets.items():
    print(v, "->", found.get(sha, "NOT FOUND"))

print("=== by name (fuzzy) ===")
for rp, sz, sha in name_hits:
    print(rp, sz, sha)

# 也列出所有 .pt 及其 sha，便于核对
print("=== all .pt files with sha ===")
for s in info.siblings:
    if s.rfilename.endswith(".pt"):
        print(s.rfilename, s.size, s.lfs.sha256 if s.lfs else None)
