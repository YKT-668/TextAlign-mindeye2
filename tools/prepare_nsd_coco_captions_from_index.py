#!/usr/bin/env python
# coding: utf-8

import os
import json
import glob
import numpy as np
import torch
import webdataset as wds
import torch.utils.data as data


def collect_train_image_ids(data_path, subj=1):
    """
    遍历 subj0{subj} 的 train WebDataset，收集所有出现过的 image_id（behav[:,0,0]）。
    data_path: 训练时传入的 --data_path（你用的是 PROJ_ROOT/src）
    返回：排序后的去重 image_id 列表（本地索引 0..72999）
    """
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
        dl = data.DataLoader(ds, batch_size=512)

        for behav, in dl:
            arr = behav.numpy()  # [B,1,17]
            if first:
                print(f"[DEBUG] behav.shape = {arr.shape}, dtype = {arr.dtype}")
                first = False

            # 和 Train_textalign.py 一样，用 behav[:,0,0] 当 image_id
            if arr.ndim == 3:
                col0 = arr[:, 0, 0]
            elif arr.ndim == 2:
                col0 = arr[:, 0]
            else:
                raise RuntimeError(f"Unexpected behav.ndim = {arr.ndim}")

            for v in col0:
                vv = np.array(v)
                while vv.ndim > 0:
                    vv = vv.flat[0]
                ids_set.add(int(vv))

    ids_list = sorted(ids_set)
    print(f"[WDS] unique train image ids: {len(ids_list)}")
    if ids_list:
        print(f"[WDS] id range: min={ids_list[0]} max={ids_list[-1]}")
    return ids_list


def main():
    proj_root = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.abspath(os.path.join(proj_root, os.pardir))
    data_root = os.path.join(proj_root, "data")

    # 1. 收集 subj01 train 中出现过的所有 image_id（本地索引）
    data_path = os.path.join(proj_root, "src")
    train_ids = collect_train_image_ids(data_path, subj=1)

    # 2. 读取 coco_full_index.pt + coco_full_captions.txt
    idx_path = os.path.join(data_root, "coco_full_index.pt")
    caps_path = os.path.join(data_root, "coco_full_captions.txt")

    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"找不到 {idx_path}")
    if not os.path.isfile(caps_path):
        raise FileNotFoundError(f"找不到 {caps_path}")

    print(f"[COCO_FULL] loading index: {idx_path}")
    idx_obj = torch.load(idx_path, map_location="cpu")
    if isinstance(idx_obj, dict):
        # 尽量兼容几种可能的 key
        if "index" in idx_obj:
            idx = idx_obj["index"]
        elif "image_ids" in idx_obj:
            idx = idx_obj["image_ids"]
        else:
            # 兜底：取第一个 tensor
            idx = None
            for v in idx_obj.values():
                if torch.is_tensor(v):
                    idx = v
                    break
            if idx is None:
                raise RuntimeError(f"无法从 {idx_path} 中解析 index（dict 无可用 tensor）")
    elif torch.is_tensor(idx_obj):
        idx = idx_obj
    else:
        raise RuntimeError(f"{idx_path} 类型异常: {type(idx_obj)}")

    idx = idx.view(-1).cpu().numpy().astype(int)
    print(f"[COCO_FULL] index shape = {idx.shape}, min={idx.min()}, max={idx.max()}")

    print(f"[COCO_FULL] loading captions: {caps_path}")
    with open(caps_path, "r", encoding="utf-8") as f:
        caps = [line.rstrip("\n") for line in f]
    print(f"[COCO_FULL] captions lines: {len(caps)}")

    if len(caps) != len(idx):
        raise RuntimeError(
            f"index 长度 {len(idx)} 与 captions 行数 {len(caps)} 不一致，请检查数据。"
        )

    # 3. 构建 “本地 image_idx -> 任意一条 caption” 的映射
    img2cap = {}
    for gid, cap in zip(idx, caps):
        # 同一张图有多条 caption，只取第一条即可；如果你以后想平均文本特征再改
        if gid not in img2cap:
            img2cap[gid] = cap
    print(f"[COCO_FULL] unique image ids with ≥1 caption: {len(img2cap)}")

    # 4. 只抽取“训练里实际用到的那 9000 张图”
    nsd_ids = []
    cap_texts = []
    missed = []

    for iid in train_ids:
        cap = img2cap.get(iid, None)
        if cap is None:
            missed.append(iid)
        else:
            nsd_ids.append(int(iid))
            cap_texts.append(cap)

    print(f"[MAP2] matched {len(nsd_ids)}/{len(train_ids)} train image ids")
    if missed:
        print(f"[MAP2] missed {len(missed)} ids; first 20: {missed[:20]}")

    # 5. 保存成 JSON，后面 encode_nsd_coco_text_clip.py 会用
    out_dir = os.path.join(data_root, "nsd_text")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "train_coco_captions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"nsd_ids": nsd_ids, "captions": cap_texts},
                  f, ensure_ascii=False, indent=2)

    print(f"[SAVE] wrote {out_path}")


if __name__ == "__main__":
    main()
