#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate training captions for TextAlign using 73k semantic cluster labels.

This is a fallback path when we do not have the official 73k-index -> COCO image_id
mapping (and thus cannot look up true COCO captions).

Outputs:
  data/nsd_text/train_coco_captions.json
with keys:
  - nsd_ids: list[int]  (here: cocoidx in [0, 72999])
  - captions: list[str]

Captions are taken from COCO_73k_semantic_cluster.npy (length 73000), which contains
short text prompts like "photo of surfer".
"""

import argparse
import glob
import json
import os
from typing import List

import numpy as np
import webdataset as wds


def collect_unique_cocoidx_from_wds(subj: int) -> List[int]:
    subj_str = f"subj{subj:02d}"
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    shard_dir = os.path.join(root, "src", "wds", subj_str, "train")

    shards = sorted(glob.glob(os.path.join(shard_dir, "*.tar")))
    if not shards:
        raise FileNotFoundError(f"No train shards found under: {shard_dir}")

    print(f"[WDS] {subj_str} train shards = {len(shards)}")

    unique_ids = set()

    # Each shard contains behav.npy with shape [B, 1, 17] after batching.
    # Index 0 is cocoidx (73KID-1) per dataset_creation.ipynb.
    for p in shards:
        print(f"[WDS] reading {p}")
        ds = wds.WebDataset(p).decode().to_tuple("behav.npy")
        for (behav,) in ds:
            # behav: [1, 17] per sample
            cocoidx = int(behav[0, 0])
            unique_ids.add(cocoidx)

    unique_ids = sorted(unique_ids)
    print(f"[WDS] unique train cocoidx: {len(unique_ids)}")
    if unique_ids:
        print(f"[WDS] id range: min={min(unique_ids)} max={max(unique_ids)}")

    return unique_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument(
        "--semantic_cluster_npy",
        type=str,
        default="COCO_73k_semantic_cluster.npy",
        help="Path to COCO_73k_semantic_cluster.npy (length 73000).",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default="data/nsd_text/train_coco_captions.json",
        help="Output json path.",
    )
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    semantic_path = args.semantic_cluster_npy
    if not os.path.isabs(semantic_path):
        semantic_path = os.path.join(root, semantic_path)

    out_path = args.out_json
    if not os.path.isabs(out_path):
        out_path = os.path.join(root, out_path)

    cocoidxs = collect_unique_cocoidx_from_wds(args.subj)

    print(f"[LOAD] semantic cluster labels: {semantic_path}")
    labels = np.load(semantic_path)
    if labels.shape[0] < (max(cocoidxs) + 1):
        raise RuntimeError(
            f"semantic labels length {labels.shape[0]} < max_id+1 {max(cocoidxs)+1}"
        )

    captions = [str(labels[i]) for i in cocoidxs]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"nsd_ids": cocoidxs, "captions": captions}, f, ensure_ascii=False)

    print(f"[SAVE] wrote {out_path}")


if __name__ == "__main__":
    main()
