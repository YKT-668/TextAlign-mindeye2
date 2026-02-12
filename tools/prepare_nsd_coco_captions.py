#!/usr/bin/env python
# coding: utf-8
"""
prepare_nsd_coco_captions.py

功能：
- 从 subj01 的 WebDataset 训练集 (src/wds/subj01/train/*.tar) 里收集所有出现过的 image_id
- 从 COCO captions (train2017 + val2017) 中抽取对应的 caption
- 为每个 image_id 选一条 caption（简单起见取第一条）
- 输出到 data/nsd_text/train_coco_captions.json:
  {
    "image_ids": [...],
    "captions": [...]
  }
"""

import os
import json
import glob
from collections import defaultdict

import torch
import webdataset as wds


def find_project_root():
    """根据当前脚本位置推断项目根目录"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


def find_coco_caption_jsons(data_dir):
    """
    在你的 data 目录下，尝试自动寻找 COCO caption json。
    优先使用 data/coco_annotations/annotations/...
    其次用 data/coco_ann/...
    """
    train_candidates = [
        os.path.join(data_dir, "coco_annotations/annotations/captions_train2017.json"),
        os.path.join(data_dir, "coco_ann/captions_train2017.json"),
    ]
    val_candidates = [
        os.path.join(data_dir, "coco_annotations/annotations/captions_val2017.json"),
        os.path.join(data_dir, "coco_ann/captions_val2017.json"),
    ]

    train_json = next((p for p in train_candidates if os.path.exists(p)), None)
    val_json = next((p for p in val_candidates if os.path.exists(p)), None)

    if train_json is None or val_json is None:
        raise FileNotFoundError(
            f"找不到 COCO captions json，请检查：\n"
            f"  试过的 train 路径: {train_candidates}\n"
            f"  试过的 val   路径: {val_candidates}"
        )

    return train_json, val_json


def collect_train_image_ids(data_path, subj=1):
    """
    遍历 subj0{subj} 的 train WebDataset，收集所有出现过的 image_id。
    data_path: 训练时传入的 --data_path（你用的是 PROJ_ROOT/src）
    返回：排序后的去重 image_id 列表
    """
    import numpy as np

    wds_root = os.path.join(data_path, f"wds/subj0{subj}/train")
    shards = sorted(glob.glob(os.path.join(wds_root, "*.tar")))
    if not shards:
        raise FileNotFoundError(f"在 {wds_root} 下没有找到 .tar 分片，请确认路径是否正确。")

    print(f"[WDS] subj0{subj} train shards = {len(shards)}")
    ids_set = set()

    first = True

    for shard in shards:
        print(f"[WDS] reading {shard}")
        ds = (
            wds.WebDataset(shard)
            .decode("torch")
            .rename(behav="behav.npy")
            .to_tuple("behav")
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=512)

        for behav, in dl:
            # behav: 可能是 [B, K] 或 [B, 1, K] 等，我们只想要“第 0 列”的 image_id
            arr = behav.numpy()

            if first:
                print(f"[DEBUG] behav.shape = {arr.shape}, dtype = {arr.dtype}")
                first = False

            # 取出 image_id 那一维
            if arr.ndim == 2:
                col0 = arr[:, 0]
            elif arr.ndim == 3:
                # 跟你训练脚本里 behav[:,0,0] 一致
                col0 = arr[:, 0, 0]
            else:
                raise RuntimeError(f"Unexpected behav.ndim = {arr.ndim}")

            # col0 里的元素可能是标量，也可能是长度为1的小数组/list，这里逐个强转成 int
            for v in col0:
                # 先转成 numpy 数组方便处理
                vv = np.array(v)
                # 如果还有维度，就取第一个元素
                while vv.ndim > 0:
                    vv = vv.flat[0]
                try:
                    iid = int(vv)
                except Exception as e:
                    raise RuntimeError(f"无法将值 {v} 转成 int: {e}")
                ids_set.add(iid)

    ids_list = sorted(ids_set)
    print(f"[WDS] unique train image ids: {len(ids_list)}")
    if ids_list:
        print(f"[WDS] id range: min={ids_list[0]} max={ids_list[-1]}")
    return ids_list


def main():
    proj_root = find_project_root()
    data_dir = os.path.join(proj_root, "data")
    # 这里默认 data_path = PROJ_ROOT/src，与你训练脚本里 --data_path 一致
    data_path = os.path.join(proj_root, "src")

    os.makedirs(os.path.join(data_dir, "nsd_text"), exist_ok=True)

    # 1. 找 COCO caption json
    train_json, val_json = find_coco_caption_jsons(data_dir)
    print(f"[COCO] train json: {train_json}")
    print(f"[COCO] val   json: {val_json}")

    # 2. 收集 subj01 训练集中出现的 image_id
    train_ids = collect_train_image_ids(data_path, subj=1)

    # 3. 加载 COCO captions，建立 image_id -> [captions]
    coco_to_caps = defaultdict(list)
    for path in [train_json, val_json]:
        print(f"[COCO] loading {path}")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        anns = obj["annotations"]
        for ann in anns:
            coco_to_caps[int(ann["image_id"])].append(ann["caption"])

    print(f"[COCO] total image_ids in caption json: {len(coco_to_caps)}")

    # 4. 按 train_ids 过滤，给每个 image_id 选一条 caption（取第一条）
    image_ids_out = []
    captions_out = []
    missed = []

    for iid in train_ids:
        caps = coco_to_caps.get(int(iid), [])
        if not caps:
            missed.append(iid)
            continue
        image_ids_out.append(int(iid))
        captions_out.append(caps[0])  # 简单起见取第一条

    print(f"[MAP] matched {len(image_ids_out)}/{len(train_ids)} train image ids 有 caption")
    print(f"[MAP] missed {len(missed)} ids")
    if missed:
        print("[MAP] first 20 missed ids:", missed[:20])

    # 5. 保存到 data/nsd_text/train_coco_captions.json
    out_path = os.path.join(data_dir, "nsd_text", "train_coco_captions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"image_ids": image_ids_out, "captions": captions_out},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[SAVE] wrote {out_path}")


if __name__ == "__main__":
    main()
