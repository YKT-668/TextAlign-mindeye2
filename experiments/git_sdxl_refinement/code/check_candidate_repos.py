#!/usr/bin/env python3
"""检查候选仓库中是否有 Ours S1 step6d 输出文件。"""
from huggingface_hub import HfApi

api = HfApi()
targets = {
    "df704fca629453c79f80b83a859694e468f1aad994257d31314e86dc7f8a85b2": "ours_enhancedrecons",
    "a6b49190a20f79ce990bc647d7cca032e9503383877c3728bec647ef8c069758": "ours_predcaptions",
    "0a837980a67886e2dfc5e0f3b08facdeab7a0d0b7d9ae6309f1f604ca7812d3c": "ours_ids",
}

for repo, rtype in [("ykt668/textalign-mindeye2-model", "model"),
                    ("ykt668/google-cv-internship-private-archive", "dataset")]:
    print("=" * 60)
    print("REPO:", repo, rtype)
    try:
        if rtype == "model":
            info = api.model_info(repo, files_metadata=True, token=True)
        else:
            info = api.dataset_info(repo, files_metadata=True, token=True)
        print("sha:", info.sha, "| files:", len(info.siblings))
        found = {}
        name_hits = []
        for s in info.siblings:
            sha = s.lfs.sha256 if s.lfs else None
            if sha in targets:
                found[sha] = (s.rfilename, s.size)
            low = s.rfilename.lower()
            if any(k in low for k in ("step6d", "enhancedrecon", "predcaption", "ours_llmhard", "full_ours", "all_ids.pt")):
                name_hits.append((s.rfilename, s.size, sha))
        print("by-sha:", found if found else "无")
        print("by-name:", name_hits if name_hits else "无")
    except Exception as e:
        print("ERR:", type(e).__name__, str(e)[:300])
