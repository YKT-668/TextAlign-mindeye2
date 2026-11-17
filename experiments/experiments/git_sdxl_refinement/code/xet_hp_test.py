#!/usr/bin/env python3
"""测试 HF_XET_HIGH_PERFORMANCE=1 下载一个 ~27MB 公开文件的耗时。输出到 reports/_xet_hp_test.txt。"""
import os
import time
from pathlib import Path

OUT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/reports/_xet_hp_test.txt")
L = []
L.append(f"HF_XET_HIGH_PERFORMANCE={os.environ.get('HF_XET_HIGH_PERFORMANCE')}")

from huggingface_hub import hf_hub_download

repo = "ykt668/textalign-mindeye2-model"
fn = "snapshots/2026-04-12/data/nsd_text/s2_train_coco_text_clip.pt"
t0 = time.time()
p = hf_hub_download(repo, fn, repo_type="model", token=True)
dt = time.time() - t0
sz = Path(p).stat().st_size
L.append(f"file={fn}")
L.append(f"size={sz} bytes time={dt:.1f}s speed={sz/dt/1024:.1f} KiB/s")

OUT.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
