#!/usr/bin/env python3
"""Audit overlap between subject training sets and a shared evaluation split."""

import argparse
import glob
import os
import pickle

import numpy as np
import torch
import webdataset as wds


def get_train_ids(wds_root: str, subject: int) -> set[int]:
    shards = sorted(glob.glob(os.path.join(wds_root, f"subj0{subject}", "train", "*.tar")))
    ids: set[int] = set()
    print(f"subj{subject}: scanning {len(shards)} shards")
    for shard in shards:
        dataset = (
            wds.WebDataset(shard, shardshuffle=False)
            .decode("torch")
            .rename(behav="behav.npy")
            .to_tuple("behav")
        )
        for (behav,) in torch.utils.data.DataLoader(dataset, batch_size=2048):
            ids.update(behav[:, 0, 0].int().numpy().tolist())
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wds-root", default=os.environ.get("NSD_ROOT"))
    parser.add_argument("--stim-info", required=True)
    parser.add_argument("--shared-mask", required=True)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 5, 7])
    args = parser.parse_args()
    if not args.wds_root:
        parser.error("--wds-root or NSD_ROOT is required")

    with open(args.stim_info, "rb") as handle:
        pickle.load(handle, encoding="latin1")

    train_sets = {subject: get_train_ids(args.wds_root, subject) for subject in args.subjects}
    reference = train_sets[args.subjects[0]]
    for subject, ids in train_sets.items():
        print(f"subj{subject}: {len(ids)} unique; reference overlap={len(reference & ids)}")

    shared_mask = np.load(args.shared_mask)
    shared_ids = set(np.flatnonzero(shared_mask).tolist())
    for subject, ids in train_sets.items():
        print(f"subj{subject} vs shared split: {len(ids & shared_ids)}")

    if args.teacher:
        teacher = torch.load(args.teacher, map_location="cpu", weights_only=False)
        teacher_ids = set(teacher["image_ids"].tolist())
        print(f"teacher ids: {len(teacher_ids)}")
        for subject, ids in train_sets.items():
            print(f"teacher vs subj{subject}: {len(teacher_ids & ids)}")


if __name__ == "__main__":
    main()
