import os
import json
import glob
from pathlib import Path

tag = os.environ.get("NEW_TAG", "ours_s1_stage1_final_best32_lastpth_v1") # Default backup
print(f"Generating runlog for tag: {tag}")

paths=[
  f"cache/model_eval_results/shared982_latent/{tag}",
  f"cache/model_eval_results/shared982_twoafc/{tag}",
  f"cache/model_eval_results/shared982_ccd/{tag}",
  f"cache/model_eval_results/shared982_rsa/{tag}",
  f"cache/model_eval_results/shared982_isrsa/textalign_llm_s1_{tag}",
]
out=Path(f"results/tables/_runlog_{tag}.md")
out.parent.mkdir(parents=True,exist_ok=True)
lines=[f"# Runlog {tag}\n"]
for p in paths:
  pp=Path(p)
  lines.append(f"## {p}\n")
  if not pp.exists():
    lines.append("- MISSING\n"); continue
  ms=list(pp.rglob("metrics.json"))
  lines.append(f"- metrics.json: {len(ms)}\n")
  for m in ms[:10]:
    try:
      j=json.load(open(m))
      # 尽量打印关键字段（如果存在）
      keys=[]
      for k in ["ccd_acc1","twoafc","rsa_spearman","fwd_top1","bwd_top1","mean_offdiag_isrsa"]:
        if k in j: keys.append((k,j[k]))
      lines.append(f"  - {m}: {keys}\n")
    except Exception as e:
      lines.append(f"  - {m}: ERROR {e}\n")
out.write_text("".join(lines))
print("Wrote", out)
