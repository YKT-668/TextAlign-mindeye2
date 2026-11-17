#!/usr/bin/env python3
"""提取 MindEyeV2 官方 notebook 的代码/markdown cell（含编号）到文本文件。"""
import json
from pathlib import Path

ROOT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114")
SRC = ROOT / "source" / "MindEyeV2" / "src"
OUT = ROOT / "reports" / "_notebook_extract"
OUT.mkdir(parents=True, exist_ok=True)


def extract(nb_name: str):
    p = SRC / nb_name
    nb = json.loads(p.read_text(encoding="utf-8"))
    lines = [f"# Notebook: {nb_name}", f"# total cells: {len(nb['cells'])}", ""]
    for i, cell in enumerate(nb["cells"]):
        ctype = cell.get("cell_type", "?")
        src = "".join(cell.get("source", []))
        lines.append(f"\n{'='*100}\n## Cell [{i}] ({ctype})\n{'='*100}")
        lines.append(src.rstrip())
    out = OUT / (nb_name.replace(".ipynb", "") + ".md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


for name in ["enhanced_recon_inference.ipynb", "recon_inference.ipynb"]:
    extract(name)
