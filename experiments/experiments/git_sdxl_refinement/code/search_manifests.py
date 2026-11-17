#!/usr/bin/env python3
"""在 model 仓库 JSON manifest 中搜索 Ours S1 / step6d 相关记录。"""
import json
from pathlib import Path

D = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/reports/_model_repo")
OUT = D / "search_results.txt"
KEYS = ["step6d", "ours_llmhard", "enhancedrecons", "aaai_ablation", "maxthroughput",
        "20260710", "all_ids", "predcaptions", "aaai_readme3"]


def walk(obj, path=""):
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(key.lower() in str(k).lower() for key in KEYS):
                hits.append(f"{path}/{k} = {str(v)[:200]}")
            hits.extend(walk(v, f"{path}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(walk(v, f"{path}[{i}]"))
    elif isinstance(obj, str):
        if any(key.lower() in obj.lower() for key in KEYS):
            hits.append(f"{path} = {obj[:300]}")
    return hits


lines = []
for f in sorted(D.glob("*.json")):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        lines.append(f"{f.name}: parse err {e}")
        continue
    hits = walk(data)
    lines.append(f"\n===== {f.name} ({len(hits)} hits) =====")
    for h in hits[:80]:
        lines.append("  " + h)

OUT.write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size)

