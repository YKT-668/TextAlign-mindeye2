#!/usr/bin/env python3
"""诊断：Ours S1 文件是否在其它 revision 或其它仓库。"""
from huggingface_hub import HfApi

REPO = "ykt668/textalign-mindeye2-data"
REV = "2c612c5c2ca3c5344854edbd7028769cb254ab5a"
api = HfApi()

print("=== 当前仓库 HEAD (main) 是否有 step6d/ours 文件 ===")
try:
    info = api.dataset_info(REPO, files_metadata=True)
    print("main sha:", info.sha)
    hits = [s.rfilename for s in info.siblings
            if any(k in s.rfilename.lower() for k in ("step6d", "enhancedrecon", "predcaption", "ours_llmhard", "full_ours"))]
    print("hits:", hits if hits else "无")
except Exception as e:
    print("ERR main:", type(e).__name__, str(e)[:200])

print("\n=== ykt668 名下 datasets ===")
try:
    for d in api.list_datasets(author="ykt668", token=True):
        print("-", d.id, "| private:", getattr(d, "private", None), "| lastModified:", getattr(d, "lastModified", None))
except Exception as e:
    print("ERR list_datasets:", type(e).__name__, str(e)[:200])

print("\n=== ykt668 名下 models ===")
try:
    for m in api.list_models(author="ykt668", token=True):
        print("-", m.id, "| private:", getattr(m, "private", None))
except Exception as e:
    print("ERR list_models:", type(e).__name__, str(e)[:200])

print("\n=== 当前仓库所有 revision/tags/branches ===")
try:
    refs = api.list_repo_refs(REPO, repo_type="dataset", token=True)
    for b in getattr(refs, "branches", []) or []:
        print("branch:", b.name, b.target_commit)
    for t in getattr(refs, "tags", []) or []:
        print("tag:", t.name, t.target_commit)
    print("converts:", getattr(refs, "converts", None))
except Exception as e:
    print("ERR list_repo_refs:", type(e).__name__, str(e)[:200])
