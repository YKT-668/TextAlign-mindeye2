#!/usr/bin/env python3
"""数据完整性检查：beta HDF5、new_test tar、GT images/captions。输出到 reports/_data_check.txt。"""
import io
import tarfile
from pathlib import Path

import numpy as np
import h5py

ROOT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114")
OUT = ROOT / "reports" / "_data_check.txt"
L = []

# ---- A. beta HDF5 ----
L.append("=== A. beta HDF5 ===")
with h5py.File(ROOT / "assets/data/betas_all_subj01_fp32_renorm.hdf5", "r") as f:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            L.append(f"  DATASET {name}: shape={obj.shape} dtype={obj.dtype}")
    f.visititems(walk)
    # 找主 betas dataset
    for k in f.keys():
        d = f[k]
        if isinstance(d, h5py.Dataset):
            arr = d
            L.append(f"  主数据 {k}: shape={arr.shape} dtype={arr.dtype}")
            # NaN/Inf 采样检查（前 100 行 + 随机采样）
            sample = arr[:100] if arr.ndim else arr[:]
            if np.issubdtype(arr.dtype, np.floating):
                L.append(f"  NaN count(前100): {int(np.isnan(sample).sum())}, Inf: {int(np.isinf(sample).sum())}")
            L.append(f"  voxel 维度: {arr.shape[-1] if arr.ndim else 'scalar'}")
            break

# ---- B. new_test tar ----
L.append("\n=== B. new_test tar ===")
tar = tarfile.open(ROOT / "assets/data/wds_subj01_new_test_0.tar")
names = tar.getnames()
L.append(f"  成员数: {len(names)}")
behavs = []
with tarfile.open(ROOT / "assets/data/wds_subj01_new_test_0.tar") as t:
    for m in t.getmembers():
        if m.name.endswith("behav.npy"):
            f = t.extractfile(m)
            behavs.append(np.load(io.BytesIO(f.read())))
if behavs:
    b = np.concatenate(behavs, axis=0) if len(behavs) > 1 else behavs[0]
    L.append(f"  behav 总记录: {b.shape}")
    img_idx = b[:, 0, 0].astype(int)
    vox_idx = b[:, 0, 5].astype(int)
    L.append(f"  唯一图像 ID: {len(np.unique(img_idx))}")
    # 每个唯一 ID 重复次数
    uniq, cnts = np.unique(img_idx, return_counts=True)
    L.append(f"  重复次数分布: min={cnts.min()} max={cnts.max()} 唯一值={sorted(set(cnts.tolist()))}")
    L.append(f"  图像 ID 范围: {img_idx.min()} - {img_idx.max()}")
    L.append(f"  voxel 索引范围: {vox_idx.min()} - {vox_idx.max()}")

# ---- C. GT images + captions ----
L.append("\n=== C. GT images/captions ===")
import torch
img = torch.load(ROOT / "assets/ground_truth/all_images.pt", map_location="cpu", mmap=True, weights_only=False)
L.append(f"  all_images.pt: shape={tuple(img.shape)} dtype={img.dtype}")
caps = torch.load(ROOT / "assets/ground_truth/all_captions.pt", map_location="cpu", weights_only=False)
if isinstance(caps, (list, tuple)):
    L.append(f"  all_captions.pt: list len={len(caps)}")
    L.append(f"  前3条: {[str(c)[:80] for c in caps[:3]]}")
    empties = [i for i, c in enumerate(caps) if not str(c).strip()]
    L.append(f"  空 caption: {len(empties)}")
elif torch.is_tensor(caps):
    L.append(f"  all_captions.pt: tensor shape={tuple(caps.shape)} dtype={caps.dtype}")
else:
    L.append(f"  all_captions.pt: type={type(caps)}")

OUT.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
