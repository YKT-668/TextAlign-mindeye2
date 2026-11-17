#!/usr/bin/env python3
"""数据完整性检查 v4：正确匹配 {key}.behav.npy。"""
import io
import tarfile
from pathlib import Path

import numpy as np
import h5py

ROOT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114")
OUT = ROOT / "reports" / "_data_check.txt"
L = []
def flush(): OUT.write_text("\n".join(L), encoding="utf-8")

L.append("=== A. beta ===")
with h5py.File(ROOT / "assets/data/betas_all_subj01_fp32_renorm.hdf5", "r") as f:
    ds = f["betas"]; L.append(f"  shape={ds.shape} dtype={ds.dtype} voxel_dim={ds.shape[-1]}")
    s = ds[:200]; L.append(f"  NaN={int(np.isnan(s).sum())} Inf={int(np.isinf(s).sum())}")
flush()

L.append("\n=== B. new_test tar ===")
behavs = []
with tarfile.open(ROOT / "assets/data/wds_subj01_new_test_0.tar") as t:
    members = t.getmembers(); L.append(f"  成员数={len(members)}")
    for m in members:
        parts = m.name.rstrip("/").split("/")[-1].split(".")
        if len(parts) >= 2 and parts[-2] == "behav" and parts[-1] == "npy":
            behavs.append(np.load(io.BytesIO(t.extractfile(m).read())))
L.append(f"  {key}.behav.npy 文件数={len(behavs)}")
img_idx = None
if behavs:
    L.append(f"  单条 shape 集合={sorted({tuple(x.shape) for x in behavs})[:5]}")
    b = np.concatenate([x.reshape(-1, x.shape[-1]) for x in behavs], axis=0)
    L.append(f"  拼接 shape={b.shape}")
    img_idx = b[:, 0].astype(int); vox_idx = b[:, 5].astype(int)
    u, c = np.unique(img_idx, return_counts=True)
    L.append(f"  记录数={len(img_idx)} 唯一图像ID={len(u)} 重复分布={sorted(set(c.tolist()))[:6]}")
    L.append(f"  ID范围={img_idx.min()}-{img_idx.max()} voxel范围={vox_idx.min()}-{vox_idx.max()}")
flush()

L.append("\n=== C. GT ===")
import torch
img = torch.load(ROOT / "assets/ground_truth/all_images.pt", map_location="cpu", mmap=True, weights_only=False)
L.append(f"  all_images.pt shape={tuple(img.shape)} dtype={img.dtype}")
caps = torch.load(ROOT / "assets/ground_truth/all_captions.pt", map_location="cpu", weights_only=False)
if isinstance(caps, np.ndarray):
    L.append(f"  all_captions.pt numpy shape={caps.shape} dtype={caps.dtype} 空={sum(1 for x in caps.ravel() if not str(x).strip())}")
flush()

L.append("\n=== D. 交叉映射 ===")
t73 = np.load(ROOT / "reports/_mapping/test_73k_images.npy").astype(np.int64)
L.append(f"  test_73k_images shape={t73.shape} unique={len(np.unique(t73))}")
if img_idx is not None:
    sn = set(img_idx.tolist()); sg = set(t73.tolist())
    L.append(f"  new_test唯一ID == test_73k (集合): {sn == sg}")
    L.append(f"  new_test - test73k: {sorted(sn-sg)[:5]}; test73k - new_test: {sorted(sg-sn)[:5]}")
flush()
print("DONE")
