#!/usr/bin/env python3
"""数据完整性检查：beta HDF5、new_test tar、GT、交叉映射。分节增量写出。"""
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
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            L.append(f"  DATASET {name}: shape={obj.shape} dtype={obj.dtype}")
    f.visititems(walk)
    main_ds = None
    for k in f.keys():
        if isinstance(f[k], h5py.Dataset):
            main_ds = f[k]; break
    if main_ds is not None:
        L.append(f"  main={main_ds.name} shape={main_ds.shape} dtype={main_ds.dtype}")
        L.append(f"  voxel 维度={main_ds.shape[-1]} (subject1 应为 15724)")
        s = main_ds[:200]
        if np.issubdtype(main_ds.dtype, np.floating):
            L.append(f"  NaN(前200行)={int(np.isnan(s).sum())} Inf={int(np.isinf(s).sum())}")
flush()

L.append("\n=== B. new_test tar ===")
behavs = []
with tarfile.open(ROOT / "assets/data/wds_subj01_new_test_0.tar") as t:
    members = t.getmembers()
    L.append(f"  成员数={len(members)}")
    for m in members:
        if m.name.endswith("behav.npy"):
            behavs.append(np.load(io.BytesIO(t.extractfile(m).read())))
img_idx = None
if behavs:
    b = np.concatenate(behavs, axis=0)
    L.append(f"  behav 拼接 shape={b.shape} ndim={b.ndim}")
    if b.ndim == 3:
        img_idx = b[:, 0, 0].astype(int); vox_idx = b[:, 0, 5].astype(int)
    else:
        img_idx = b[:, 0].astype(int); vox_idx = b[:, 5].astype(int)
    uniq, cnts = np.unique(img_idx, return_counts=True)
    L.append(f"  记录数={len(img_idx)} 唯一图像ID={len(uniq)}")
    L.append(f"  重复次数分布={sorted(set(cnts.tolist()))} (期望[3])")
    L.append(f"  图像ID范围={img_idx.min()}-{img_idx.max()} voxel索引范围={vox_idx.min()}-{vox_idx.max()}")
flush()

L.append("\n=== C. GT images/captions ===")
import torch
img = torch.load(ROOT / "assets/ground_truth/all_images.pt", map_location="cpu", mmap=True, weights_only=False)
L.append(f"  all_images.pt shape={tuple(img.shape)} dtype={img.dtype}")
caps = torch.load(ROOT / "assets/ground_truth/all_captions.pt", map_location="cpu", weights_only=False)
if isinstance(caps, (list, tuple)):
    L.append(f"  all_captions.pt type=list len={len(caps)}")
    L.append(f"  前3条={[str(c)[:70] for c in caps[:3]]}")
    L.append(f"  空caption={sum(1 for c in caps if not str(c).strip())}")
else:
    L.append(f"  all_captions.pt type={type(caps)}")
flush()

L.append("\n=== D. 交叉映射 ===")
t73 = np.load(ROOT / "reports/_mapping/test_73k_images.npy").astype(np.int64)
L.append(f"  test_73k_images.npy shape={t73.shape} unique={len(np.unique(t73))}")
if img_idx is not None:
    set_new = set(img_idx.tolist()); set_gt = set(t73.tolist())
    L.append(f"  new_test唯一ID == test_73k_images(集合): {set_new == set_gt}")
flush()
print("DONE")
