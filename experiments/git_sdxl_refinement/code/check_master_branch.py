#!/usr/bin/env python3
"""查询 HF Model 仓库 master 分支 (a127f9295fd5656a) 是否有 step6d/ours_llmhard/s1 输出文件。"""
import os
from huggingface_hub import HfApi

REPO = "ykt668/textalign-mindeye2-model"
REV = "a127f9295fd5656ac63ae436f07e61b80bf4efce"
api = HfApi()

info = api.model_info(REPO, revision=REV, files_metadata=True, token=True)
print("master revision sha:", info.sha, "| files:", len(info.siblings))

targets = {
    "df704fca629453c79f80b83a859694e468f1aad994257d31314e86dc7f8a85b2": "enhancedrecons",
    "a6b49190a20f79ce990bc647d7cca032e9503383877c3728bec647ef8c069758": "predcaptions",
    "0a837980a67886e2dfc5e0f3b08facdeab7a0d0b7d9ae6309f1f604ca7812d3c": "ids",
}
by_sha = {}
name_hits = []
s1_evals = []
for s in info.siblings:
    sha = s.lfs.sha256 if s.lfs else None
    if sha in targets:
        by_sha[sha] = (s.rfilename, s.size)
    low = s.rfilename.lower()
    if any(k in low for k in ("step6d", "ours_llmhard", "enhancedrecons", "full_ours")):
        name_hits.append((s.rfilename, s.size, sha))
    if "s1" in low and "evals" in low:
        s1_evals.append((s.rfilename, s.size))

print("=== by SHA256 ===")
for sha, nm in targets.items():
    print(nm, "->", by_sha.get(sha, "NOT FOUND"))
print("=== name hits (step6d/ours_llmhard/enhancedrecons) ===")
for rp, sz, sha in name_hits:
    print(" ", rp, sz, sha)
print("=== s1 evals ===")
for rp, sz in s1_evals[:30]:
    print(" ", rp, sz)
