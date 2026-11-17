#!/usr/bin/env python3
"""grep hf_xet .so 二进制中的 HF_XET_* 环境变量，输出到 reports/_xet_envs.txt。"""
import json
import subprocess
from pathlib import Path

ROOT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114")
OUT = ROOT / "reports" / "_xet_envs.txt"
so = None
for p in (ROOT / "download_venv" / "lib" / "python3.12" / "site-packages" / "hf_xet").rglob("*"):
    if p.suffix in (".so", ".pyd"):
        so = p
        break

L = [f"so={so}"]
if so:
    try:
        out = subprocess.run(["strings", str(so)], capture_output=True, text=True, timeout=60).stdout
        envs = sorted({w for w in out.split() if w.startswith("HF_XET_")})
        L.append("HF_XET_* in binary:")
        L += ["  " + e for e in envs]
    except Exception as e:
        L.append(f"grep err {e}")

# 当前基准速度
try:
    st = json.loads((ROOT / "logs" / "status.json").read_text())
    L.append(f"baseline cur={st['speed']['current_bps']/1024:.1f} KiB/s avg={st['speed']['avg_bps']/1024:.1f} KiB/s transfer={st['current_file']['transfer_bytes']}")
except Exception as e:
    L.append(f"status err {e}")

OUT.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
