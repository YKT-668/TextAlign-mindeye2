#!/usr/bin/env python3
"""用 numpy 构建并验证 S1 new_test 1000 ↔ shared982 映射。输出 CSV + 验证结果。"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114")
MAPPING = ROOT / "reports" / "_mapping"
PROV = ROOT / "reports" / "provenance"
OUT_CSV = ROOT / "reports" / "S1_NEW_TEST_TO_SHARED982_MAPPING.csv"
OUT_LOG = ROOT / "reports" / "_mapping" / "_verify.txt"

L = []

# 1. 1000 IDs from TSV
tsv = PROV / "final14_archive__followup_experiments__fast_validation__FINAL__13_blocker_closure__model_alignment__N1000_ALL_IDS_ALIGNMENT.tsv"
ids1000 = []
with open(tsv) as f:
    next(f)
    for line in f:
        p = line.strip().split("\t")
        if len(p) >= 5:
            ids1000.append(int(p[1]))
ids1000 = np.array(ids1000, dtype=np.int64)
L.append(f"N1000 ids: {ids1000.shape}, unique={len(np.unique(ids1000))}, min={ids1000.min()}, max={ids1000.max()}")

# 2. shared1000.npy
s1000 = np.load(MAPPING / "shared1000.npy").astype(np.int64)
L.append(f"shared1000.npy: {s1000.shape}, unique={len(np.unique(s1000))}, min={s1000.min()}, max={s1000.max()}")

# 3. test_73k_images.npy
t73 = np.load(MAPPING / "test_73k_images.npy").astype(np.int64)
L.append(f"test_73k_images.npy: {t73.shape}, unique={len(np.unique(t73))}, min={t73.min()}, max={t73.max()}")

# 4. shared982 CSV
csvf = MAPPING / "snapshots__2026-04-12__audit_runs__s1_textalign_stage1_FINAL_BEST_32_20260115_004557__shared982_ids_manifest.csv"
s982_ids = []
s982_src_idx = []
with open(csvf) as f:
    reader = csv.DictReader(f)
    for row in reader:
        s982_ids.append(int(row["id"]))
        s982_src_idx.append(int(row["source_shared1000_index"]))
s982_ids = np.array(s982_ids, dtype=np.int64)
s982_src_idx = np.array(s982_src_idx, dtype=np.int64)
L.append(f"shared982: {s982_ids.shape}, unique={len(np.unique(s982_ids))}, src_idx range {s982_src_idx.min()}-{s982_src_idx.max()}")

# ============ 验证 ============
L.append("\n=== 验证 ===")
# a. 1000 TSV ids 是否等于 shared1000（集合）
set1000 = set(ids1000.tolist())
set_s1000 = set(s1000.tolist())
L.append(f"[a] ids1000 == shared1000 (set): {set1000 == set_s1000}")
L.append(f"    ids1000 不在 shared1000 的: {sorted(set1000 - set_s1000)[:10]}")
L.append(f"    shared1000 不在 ids1000 的: {sorted(set_s1000 - set1000)[:10]}")

# b. shared982 == 前 982 个 shared1000?
L.append(f"[b] shared982 == shared1000[:982] (顺序): {bool(np.array_equal(s982_ids, s1000[:982]))}")

# c. shared982 ⊂ ids1000?
set982 = set(s982_ids.tolist())
L.append(f"[c] shared982 ⊂ ids1000: {set982 <= set1000}  (交集 {len(set982 & set1000)}/982)")
L.append(f"    shared982 不在 ids1000 的: {sorted(set982 - set1000)[:10]}")

# d. test_73k_images == shared1000 (集合)?
set_t73 = set(t73.tolist())
L.append(f"[d] test_73k_images == shared1000 (set): {set_t73 == set_s1000}")
L.append(f"    test_73k 不在 shared1000 的: {sorted(set_t73 - set_s1000)[:10]}")

# e. ids1000 是否严格递增（sorted）?
L.append(f"[e] ids1000 严格递增: {bool(np.all(np.diff(ids1000) > 0))}")

# f. 1000 个唯一, 无重复
L.append(f"[f] ids1000 无重复: {len(np.unique(ids1000)) == 1000}")

# g. 18 个非 shared982 的 ids1000
extra18 = sorted(set1000 - set982)
L.append(f"[g] 非 shared982 的 1000 内 ID（{len(extra18)} 个）: {extra18}")

OUT_LOG.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
