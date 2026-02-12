import os, json, glob
import numpy as np
import webdataset as wds
import torch

PROJ = "/home/vipuser/MindEyeV2_Project"

def main():
    # 1) 读 ids.json（shared1000 的全局 COCO image id）
    ids_path = os.path.join(PROJ, "data/nsd_text/ids.json")
    with open(ids_path, "r") as f:
        ids = [int(x) for x in json.load(f)]
    idset = set(ids)
    print(f"[ids.json] len={len(ids)} min={min(ids)} max={max(ids)}")

    # 2) subj01 的 train 分片，和训练脚本保持一致
    shards = sorted(glob.glob(os.path.join(PROJ, "src/wds/subj01/train/*.tar")))
    print(f"[train] shards={len(shards)}")

    dataset = (
        wds.WebDataset(shards, resampled=False)
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )

    # 用 DataLoader 来 collate，behav 的形状就会和 Train_textalign 里一样
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, drop_last=False
    )

    all_ids = []

    # 3) 扫前 200 个 batch 的 image id（够估计交集情况了）
    for i, (behav, past_behav, future_behav, olds_behav) in enumerate(loader):
        # 和 Train_textalign.py 里一致：image_idx = behav[:, 0, 0]
        if behav.ndim == 3:
            img_ids = behav[:, 0, 0].cpu().numpy().astype(int)
        elif behav.ndim == 2:
            # 保险起见，如果是 (B, 6) 这种，就退化成 [:,0]
            img_ids = behav[:, 0].cpu().numpy().astype(int)
        else:
            raise RuntimeError(f"Unexpected behav shape: {behav.shape}")

        all_ids.append(img_ids)

        if i >= 200:  # 最多看 200 个 batch
            break

    if not all_ids:
        print("[train] 没读到任何样本")
        return

    all_ids = np.concatenate(all_ids)
    uniq = np.unique(all_ids)
    inter = sorted(idset.intersection(uniq.tolist()))

    print(f"[train] sampled total ids: {all_ids.shape[0]}")
    print(f"[train] unique ids: {uniq.shape[0]} min={int(uniq.min())} max={int(uniq.max())}")
    print(f"[intersection] size: {len(inter)}")
    print(f"[intersection] first 20: {inter[:20]}")

if __name__ == "__main__":
    main()
