#!/usr/bin/env python3
"""数据完整性检查 v3：精确定位 behav.npy，处理 numpy captions。"""
import io
import tarfile
from pathlib import Path

import numpy as np
import h5py

ROOT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114")
OUT = ROOT / "reports" / "_data_check.txt"
L = []


def flush():
    OUT.write_text("\n".join(L), encoding="utf-8")


L.append("=== A. beta HDF5 ===")
with h5py.File(ROOT / "assets/data/betas_all_subj01_fp32_renorm.hdf5", "r") as f:
    ds = f["betas"]
    L.append(f"  betas shape={ds.shape} dtype={ds.dtype} voxel_dim={ds.shape[-1]}")
    s = ds[:200]
    L.append(f"  NaN(前200)={int(np.isnan(s).sum())} Inf={int(np.isinf(s).sum())}")
flush()

L.append("\n=== B. new_test tar ===")
behavs = []
with tarfile.open(ROOT / "assets/data/wds_subj01_new_test_0.tar") as t:
    members = t.getmembers()
    L.append(f"  成员数={len(members)}")
    for m in members:
        base = m.name.rstrip("/").split("/")[-1]
        if base == "behav.npy":  # 精确匹配，排除 past/future/olds_behav
            behavs.append(np.load(io.BytesIO(t.extractfile(m).read())))
L.append(f"  behav.npy 文件数={len(behavs)}")
img_idx = None
if behavs:
    shapes = sorted({th.shape for th in behavs})
    L.append(f"  单条 behav shape 集合={shapes[:5]}")
    b = np.concatenate([th.reshape(-1, th.shape[-1]) for th in behavs], axis=0)
    L.append(f"  拼接 shape={b.shape}")
    img_idx = b[:, 0].astype(int); vox_idx = b[:, 5].astype(int)
    uniq, cnts = np.unique(img_idx, return_counts=True)
    L.append(f"  记录数={len(img_idx)} 唯一图像ID={len(uniq)}")
    L.append(f"  重复次数分布={sorted(set(cnts.tolist()))[:6]} (期望含3)")
    L.append(f"  图像ID范围={img_idx.min()}-{img_idx.max()} voxel索引范围={vox_idx.min()}-{vox_idx.max()}")
flush()

L.append("\n=== C. GT images/captions ===")
import torch
img = torch.load(ROOT / "assets/ground_truth/all_images.pt", map_location="cpu", mmap=True, weights_only=False)
L.append(f"  all_images.pt shape={tuple(img.shape)} dtype={img.dtype}")
caps = torch.load(ROOT / "assets/ground_truth/all_captions.pt", map_location="cpu", weights_only=False)
if isinstance(caps, np.ndarray):
    L.append(f"  all_captions.pt numpy shape={caps.shape} dtype={caps.dtype}")
    L.append(f"  前3条={[str(c)[:70] for c in caps[:3]]}")
    L.append(f"  空caption={sum(1 for c in caps.ravel() if not str(c).strip())}")
elif isinstance(caps, (list, tuple)):
    L.append(f"  all_captions.pt list len={len(caps)}")
    L.append(f"  前3条={[str(c)[:70] for c in caps[:3]]}")
else:
    L.append(f"  all_captions.pt type={type(caps)}")
flush()

L.append("\n=== D. 交叉映射 ===")
t73 = np.load(ROOT / "reports/_mapping/test_73k_images.npy").astype(np.int64)
L.append(f"  test_73k_images shape={t73.shape} unique={len(np.unique(t73))}")
if img_idx is not None:
    set_new = set(img_idx.tolist()); set_gt = set(t73.tolist())
    L.append(f"  new_test唯一ID == test_73k_images(集合): {set_new == set_gt}")
flush()
print("DONE")
