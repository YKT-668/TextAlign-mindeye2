"""P3 共享工具：官方预处理、原子写、环境记录、manifest 对齐。

官方预处理链（final_evaluations.ipynb，逐字核对）：
  cell 8 : transforms.Resize((256,256))                    # 默认 BILINEAR, antialias=True
  cell 27/31 : transforms.Resize(224, BILINEAR) + Normalize  # CLIP 或 ImageNet mean/std
注意：cell 8 是 Resize((256,256)) 双参数形式，默认 BILINEAR（非 bicubic）；
torchvision>=0.15 对 tensor 输入 antialias 默认 True。本模块用 torchvision.transforms
原样复刻，不用 F.interpolate（其 antialias=False，与官方数值不同）。
"""
import json
import os
import time

import torch
from torchvision import transforms

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
P3 = f"{ROOT}/evaluation/cpu_mindeye2_p3_clip_swav"
S1 = f"{ROOT}/run/full_stage1"
S2 = f"{ROOT}/run/full_stage2"
GTD = f"{ROOT}/assets/ground_truth"
M = "s1_textalign_stage1_FINAL_BEST_32"

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_T256 = transforms.Resize((256, 256))                       # cell 8（默认 bilinear, antialias）
_T224 = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
_NORMS = {
    "imagenet": transforms.Normalize(IMNET_MEAN, IMNET_STD),      # cell 23/25/31
    "clip": transforms.Normalize(CLIP_MEAN, CLIP_STD),            # cell 27
}


def official_preprocess(x, norm="imagenet", size=224):
    """x: (B,3,H,W) [0,1] → 官方 cell8+模型 preprocess 后的 batch。"""
    if x.shape[-1] != 256:
        x = _T256(x)
    if size != 256:
        x = _T224(x)
    return _NORMS[norm](x)


def load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return torch.load(p, map_location="cpu", weights_only=False)


def load_manifest_ids():
    """REFINEMENT_EVALUATION_MANIFEST.csv 的 id 序列（配对关系的唯一依据）。"""
    import csv
    with open(f"{ROOT}/reports/REFINEMENT_EVALUATION_MANIFEST.csv", encoding="utf-8") as fh:
        return [int(r["id"]) for r in csv.DictReader(fh)]


def verify_alignment():
    """manifest id 顺序必须与张量行顺序（ids 升序）完全一致。"""
    ids_t = load_pt(f"{S1}/{M}_all_ids.pt").long()
    ids_list = [int(x) for x in ids_t]
    man_ids = load_manifest_ids()
    ok = (man_ids == ids_list) and len(set(man_ids)) == 1000
    return {"manifest_rows": len(man_ids), "ids_rows": len(ids_list),
            "manifest_equals_tensor_order": ok,
            "manifest_sorted_ascending": man_ids == sorted(man_ids)}


def atomic_save_array(arr, path):
    tmp = path + ".tmp.npy" if path.endswith(".npy") else path + ".tmp"
    import numpy as np
    np.save(tmp, arr)
    os.replace(tmp, path)


def env_snapshot(tag, log_path):
    try:
        with open("/proc/meminfo") as fh:
            mi = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in fh if ":" in l}
        line = (f"[{time.strftime('%H:%M:%S')}] {tag}: MemAvailable={mi.get('MemAvailable')}MB "
                f"SwapTotal={mi.get('SwapTotal')}MB SwapFree={mi.get('SwapFree')}MB "
                f"loadavg={open('/proc/loadavg').read().strip()}")
    except Exception as e:
        line = f"[{time.strftime('%H:%M:%S')}] {tag}: env read fail {e}"
    print(line, flush=True)
    with open(log_path, "a") as fh:
        fh.write(line + "\n")


def log_msg(log_path, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as fh:
        fh.write(line + "\n")
