#!/usr/bin/env python3
"""构建 S1 new_test 1000 图像 ↔ shared982 精确映射并验证。输出 CSV + 审计 md。"""
import csv
import json
from pathlib import Path

import numpy as np
import h5py

ROOT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114")
MAPPING = ROOT / "reports" / "_mapping"
PROV = ROOT / "reports" / "provenance"

# 1. 1000 IDs (N1000_ALL_IDS_ALIGNMENT.tsv)
tsv = PROV / "final14_archive__followup_experiments__fast_validation__FINAL__13_blocker_closure__model_alignment__N1000_ALL_IDS_ALIGNMENT.tsv"
ids1000 = []
with open(tsv) as f:
    header = f.readline().strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 5:
            ids1000.append(int(parts[1]))  # ours 列（四个模型一致）
ids1000 = np.array(ids1000, dtype=np.int64)
print("N1000 ids:", ids1000.shape, "min", ids1000.min(), "max", ids1000.max(), "unique", len(np.unique(ids1000)))

# 2. shared982 (csv)
csvf = MAPPING / "snapshots__2026-04-12__audit_runs__s1_textalign_stage1_FINAL_BEST_32_20260115_004557__shared982_ids_manifest.csv"
print("\n=== shared982 manifest csv 前 10 行 ===")
with open(csvf) as f:
    for i, line in enumerate(f):
        if i < 10:
            print("  ", line.rstrip())
        else:
            break

# 3. test_73k_images.npy
t73 = np.load(MAPPING / "test_73k_images.npy")
print("\ntest_73k_images.npy:", t73.shape, t73.dtype, "min", t73.min(), "max", t73.max(), "first5", t73[:5])

# 4. shared1000.npy
s1000 = np.load(MAPPING / "shared1000.npy")
print("shared1000.npy:", s1000.shape, s1000.dtype, "min", s1000.min(), "max", s1000.max(), "first5", s1000[:5])

# 5. COCO_73k_subj_indices.hdf5
print("\n=== COCO_73k_subj_indices.hdf5 结构 ===")
with h5py.File(MAPPING / "COCO_73k_subj_indices.hdf5", "r") as f:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  DATASET {name}: shape={obj.shape} dtype={obj.dtype}")
    f.visititems(walk)
