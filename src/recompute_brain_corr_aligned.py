#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from torchmetrics import PearsonCorrCoef
from torchvision import transforms

from models import GNet8_Encoder


REGIONS = ["nsd_general", "V1", "V2", "V3", "V4", "higher_vis"]
RUN_DEBUG_METRIC_ORDER = [
    "PixCorr",
    "SSIM",
    "AlexNet(2)",
    "AlexNet(5)",
    "InceptionV3",
    "CLIP",
    "EffNet-B",
    "SwAV",
    "FwdRetrieval",
    "BwdRetrieval",
    "Brain Corr. nsd_general",
    "Brain Corr. V1",
    "Brain Corr. V2",
    "Brain Corr. V3",
    "Brain Corr. V4",
    "Brain Corr. higher_vis",
]


def get_num_test(subj: int, new_test: bool) -> int:
    if new_test:
        if subj in (3, 6):
            return 2371
        if subj in (4, 8):
            return 2188
        return 3000
    if subj in (3, 6):
        return 2113
    if subj in (4, 8):
        return 1985
    return 2770


def get_test_url(data_path: str, subj: int, new_test: bool) -> str:
    split = "new_test" if new_test else "test"
    return f"{data_path}/wds/subj0{subj}/{split}/0.tar"


def my_split_by_node(urls):
    return urls


def load_test_indices(data_path: str, subj: int, new_test: bool):
    num_test = get_num_test(subj, new_test)
    test_url = get_test_url(data_path, subj, new_test)
    test_data = (
        wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    test_dl = torch.utils.data.DataLoader(
        test_data,
        batch_size=num_test,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    test_images_idx = []
    test_voxels_idx = []
    for _, (behav, _, _, _) in enumerate(test_dl):
        test_images_idx = np.append(test_images_idx, behav[:, 0, 0].cpu().numpy())
        test_voxels_idx = np.append(test_voxels_idx, behav[:, 0, 5].cpu().numpy())

    return test_images_idx.astype(np.int64), test_voxels_idx.astype(np.int64), test_url


def build_aligned_target_voxels(
    data_path: str,
    subj: int,
    all_ids: np.ndarray,
    test_images_idx: np.ndarray,
    test_voxels_idx: np.ndarray,
):
    betas_path = f"{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5"
    with h5py.File(betas_path, "r") as f:
        betas = f["betas"]
        test_voxels = torch.from_numpy(betas[test_voxels_idx]).float()

    uniq_imgs = np.unique(test_images_idx)
    averaged_by_img = {}
    rep_hist = {1: 0, 2: 0, 3: 0, "other": 0}

    for uniq_img in uniq_imgs:
        locs = np.where(test_images_idx == uniq_img)[0]
        n_rep = len(locs)
        if n_rep == 1:
            rep_hist[1] += 1
            locs = locs.repeat(3)
        elif n_rep == 2:
            rep_hist[2] += 1
            locs = np.concatenate((locs, locs[:1]))
        elif n_rep == 3:
            rep_hist[3] += 1
        else:
            rep_hist["other"] += 1
            raise ValueError(f"Unexpected repetitions for image {uniq_img}: {n_rep}")

        averaged_by_img[int(uniq_img)] = torch.mean(test_voxels[locs], dim=0)

    missing = [int(x) for x in all_ids.tolist() if int(x) not in averaged_by_img]
    if missing:
        raise KeyError(f"Found {len(missing)} all_ids not present in test set. Example: {missing[:10]}")

    target_voxels = torch.stack([averaged_by_img[int(x)] for x in all_ids.tolist()], dim=0)
    return target_voxels, rep_hist, len(uniq_imgs)


def load_subject_masks(data_path: str, subj: int):
    candidates = [
        os.path.join(data_path, "brain_region_masks.hdf5"),
        "brain_region_masks.hdf5",
    ]
    masks_path = None
    for p in candidates:
        if os.path.exists(p):
            masks_path = p
            break
    if masks_path is None:
        raise FileNotFoundError("brain_region_masks.hdf5 not found in data_path or cwd")

    key = f"subj0{subj}"
    with h5py.File(masks_path, "r") as f:
        g = f[key]
        masks = {k: g[k][:] for k in REGIONS}
    return masks, masks_path


def to_bool_or_index(mask_arr: np.ndarray):
    if mask_arr.dtype == np.bool_:
        return mask_arr
    if np.issubdtype(mask_arr.dtype, np.integer):
        return mask_arr.astype(np.int64)
    raise TypeError(f"Unsupported mask dtype: {mask_arr.dtype}")


def parse_old_brain_corr(old_table_path: str):
    if not old_table_path or (not os.path.exists(old_table_path)):
        return {}
    df = pd.read_csv(old_table_path, sep="\t")
    if "Value" in df.columns:
        values = df["Value"].tolist()
    else:
        values = df.iloc[:, 0].tolist()
    metric_to_val = dict(zip(RUN_DEBUG_METRIC_ORDER, [float(v) for v in values]))
    return {r: metric_to_val.get(f"Brain Corr. {r}") for r in REGIONS}


def main():
    parser = argparse.ArgumentParser(description="Recompute aligned Brain Corr from existing eval exports")
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--cache_dir", type=str, default=".")
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--subj", type=int, required=True)
    parser.add_argument("--new_test", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_weighted_enhanced", action="store_true")
    parser.add_argument("--old_table", type=str, default=None)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--out_tsv", type=str, default=None)
    args = parser.parse_args()

    data_path = str(Path(args.data_path).resolve())
    cache_dir = str(Path(args.cache_dir).resolve())
    eval_dir = str(Path(args.eval_dir).resolve())

    recons_path = os.path.join(eval_dir, f"{args.model_name}_all_enhancedrecons.pt")
    ids_path = os.path.join(eval_dir, f"{args.model_name}_all_ids.pt")
    blurry_path = os.path.join(eval_dir, f"{args.model_name}_all_blurryrecons.pt")
    gnet_path = os.path.join(cache_dir, "gnet_multisubject.pt")

    all_recons = torch.load(recons_path, map_location="cpu").float()
    all_ids = torch.load(ids_path, map_location="cpu")
    all_ids = torch.as_tensor(all_ids).view(-1).cpu().numpy().astype(np.int64)

    used_weighted_enhanced = False
    if (not args.no_weighted_enhanced) and os.path.exists(blurry_path):
        all_blurry = torch.load(blurry_path, map_location="cpu").float()
        all_recons = all_recons * 0.75 + all_blurry * 0.25
        used_weighted_enhanced = True

    test_images_idx, test_voxels_idx, test_url = load_test_indices(data_path, args.subj, args.new_test)

    target_voxels, rep_hist, uniq_img_count = build_aligned_target_voxels(
        data_path,
        args.subj,
        all_ids,
        test_images_idx,
        test_voxels_idx,
    )

    recon_list = [
        transforms.ToPILImage()(all_recons[i].detach().cpu().clamp(0, 1))
        for i in range(all_recons.shape[0])
    ]

    gnet = GNet8_Encoder(device=args.device, subject=args.subj, model_path=gnet_path)
    beta_primes = gnet.predict(recon_list).float().cpu()

    if beta_primes.shape != target_voxels.shape:
        raise RuntimeError(
            f"Shape mismatch: target_voxels={tuple(target_voxels.shape)} vs beta_primes={tuple(beta_primes.shape)}"
        )

    masks, masks_path = load_subject_masks(data_path, args.subj)

    pec = PearsonCorrCoef(num_outputs=len(recon_list))
    corrected = {}
    vox_counts = {}

    for region, raw_mask in masks.items():
        mask = to_bool_or_index(np.asarray(raw_mask))
        t = target_voxels[:, mask]
        p = beta_primes[:, mask]
        score = pec(t.moveaxis(0, 1), p.moveaxis(0, 1))
        corrected[region] = float(torch.mean(score).item())
        vox_counts[region] = int(t.shape[1])

    if args.old_table is None:
        args.old_table = os.path.join(data_path, "tables", f"{args.model_name}_all_enhancedrecons.csv")
    old = parse_old_brain_corr(args.old_table)

    if args.out_json is None:
        args.out_json = os.path.join(data_path, "tables", f"{args.model_name}_brain_corr_aligned.json")
    if args.out_tsv is None:
        args.out_tsv = os.path.join(data_path, "tables", f"{args.model_name}_brain_corr_aligned.tsv")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)

    comparison_rows = []
    for r in REGIONS:
        ov = old.get(r)
        cv = corrected[r]
        delta = None if ov is None else (cv - ov)
        comparison_rows.append({
            "region": r,
            "old_run_debug": ov,
            "corrected_aligned": cv,
            "delta": delta,
            "n_voxels": vox_counts[r],
        })

    summary = {
        "model_name": args.model_name,
        "subj": int(args.subj),
        "eval_dir": eval_dir,
        "data_path": data_path,
        "cache_dir": cache_dir,
        "n_recons": int(all_recons.shape[0]),
        "n_all_ids": int(len(all_ids)),
        "new_test": bool(args.new_test),
        "test_url": test_url,
        "n_test_samples": int(len(test_images_idx)),
        "n_unique_test_images": int(uniq_img_count),
        "rep_histogram": rep_hist,
        "used_weighted_enhanced": used_weighted_enhanced,
        "gnet_path": gnet_path,
        "masks_path": masks_path,
        "old_table": args.old_table,
        "regions": comparison_rows,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(comparison_rows).to_csv(args.out_tsv, sep="\t", index=False)

    print(json.dumps({
        "model_name": args.model_name,
        "subj": args.subj,
        "out_json": args.out_json,
        "out_tsv": args.out_tsv,
        "corrected": corrected,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
