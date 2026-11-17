#!/usr/bin/env python3
"""生成 S1 new_test ↔ shared982 映射 CSV + 审计报告。"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114")
MAPPING = ROOT / "reports" / "_mapping"
PROV = ROOT / "reports" / "provenance"
OUT_CSV = ROOT / "reports" / "S1_NEW_TEST_TO_SHARED982_MAPPING.csv"

tsv = PROV / "final14_archive__followup_experiments__fast_validation__FINAL__13_blocker_closure__model_alignment__N1000_ALL_IDS_ALIGNMENT.tsv"
ids = []
with open(tsv) as f:
    next(f)
    for line in f:
        p = line.rstrip().split("\t")
        if len(p) >= 5:
            ids.append(int(p[1]))
ids = np.array(ids, dtype=np.int64)

csvf = MAPPING / "snapshots__2026-04-12__audit_runs__s1_textalign_stage1_FINAL_BEST_32_20260115_004557__shared982_ids_manifest.csv"
s982 = set(int(r["id"]) for r in csv.DictReader(open(csvf)))

t73 = np.load(MAPPING / "test_73k_images.npy").astype(np.int64)
assert np.array_equal(t73, ids), "test_73k_images 与 ids1000 顺序应一致"

rows = []
for i, sid in enumerate(ids.tolist()):
    in982 = sid in s982
    rows.append([
        i,                          # new_test_index
        sid,                        # nsd_stimulus_id
        "",                         # coco_image_id（可选，需 h5py 读 COCO_73k_subj_indices.hdf5）
        i,                          # local_gt_index（== new_test_index，顺序已核）
        "yes" if in982 else "no",   # shared982
        i,                          # caption_index（== new_test_index）
        "N1000_ALL_IDS_ALIGNMENT.tsv+test_73k_images.npy+shared982_ids_manifest.csv",
        "VERIFIED",                 # verification_status
    ])

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["new_test_index", "nsd_stimulus_id", "coco_image_id", "local_gt_index",
                "shared982", "caption_index", "mapping_source", "verification_status"])
    w.writerows(rows)

print("wrote", OUT_CSV, "rows=", len(rows), "shared982_yes=", sum(1 for r in rows if r[4] == "yes"))
