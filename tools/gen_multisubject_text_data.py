#!/usr/bin/env python
# coding: utf-8
"""
tools/gen_multisubject_text_data.py

功能：
1. 指定被试 (subj1, subj2, ... subj8)
2. 扫描该被试的 WDS 训练数据，收集真实使用的 valid image ids (NSD 73k Indices)
3. 使用 nsd_stim_info_merged.pkl 将 Index 映射到 COCO ID
4. 从 COCO JSON 中提取对应的 Caption
5. 使用 CLIP (ViT-L/14) 提取文本特征
6. 保存为 data/nsd_text/train_coco_text_clip_subjXX.pt

用法：
  python tools/gen_multisubject_text_data.py --subj 2
  python tools/gen_multisubject_text_data.py --subj 1 2 5 7
"""

import os
import sys
import glob
import json
import pickle
import argparse
import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel

# 确保能找到项目根目录
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJ_ROOT, "data")
SRC_DIR = os.path.join(_PROJ_ROOT, "src")
TEXT_OUT_DIR = os.path.join(DATA_DIR, "nsd_text")
STIM_INFO_PATH = os.path.join(_PROJ_ROOT, "nsd_stim_info_merged.pkl")

def get_wds_shards(subj):
    wds_root = os.path.join(SRC_DIR, f"wds/subj0{subj}/train")
    shards = sorted(glob.glob(os.path.join(wds_root, "*.tar")))
    if not shards:
        pass
    return shards

def load_stim_info():
    """加载 NSD ID -> COCO ID 映射"""
    print(f"[INFO] Loading stim info from {STIM_INFO_PATH}...")
    if not os.path.exists(STIM_INFO_PATH):
        raise FileNotFoundError(f"Missing {STIM_INFO_PATH}")
    
    with open(STIM_INFO_PATH, "rb") as f:
        obj = pickle.load(f, encoding='latin1')
    
    # 支持 DataFrame 或 dict
    if hasattr(obj, "columns") and "cocoId" in obj.columns:
        coco_ids = obj["cocoId"].to_numpy().astype(int)
    elif isinstance(obj, dict) and "cocoId" in obj:
        coco_ids = np.asarray(obj["cocoId"]).astype(int)
    else:
        # 尝试直接属性访问
        coco_ids = np.asarray(getattr(obj, "cocoId")).astype(int)
        
    print(f"[INFO] Stim info loaded. Count: {len(coco_ids)}")
    return coco_ids

def collect_train_ids(subj):
    """从 WebDataset 扫描真实的训练 Image IDs (NSD Local IDs)"""
    shards = get_wds_shards(subj)
    if not shards:
        print(f"[ERROR] Subj0{subj} 没有找到训练 shards. 跳过.")
        return None

    print(f"[Subj0{subj}] 正在扫描 {len(shards)} 个 shards 获取 Image IDs...")
    ids_set = set()
    count_samples = 0
    
    for shard in tqdm(shards, desc=f"Scanning S{subj}"):
        ds = wds.WebDataset(shard).decode("torch").to_tuple("behav.npy")
        dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
        
        for behav, in dl:
            arr = behav.numpy()
            flat = arr.flatten()
            target_id = int(flat[0])
            ids_set.add(target_id)
            count_samples += 1

    unique_ids = sorted(list(ids_set))
    print(f"[Subj0{subj}] 扫描完成. Samples: {count_samples}, Unique IDs: {len(unique_ids)}")
    if len(unique_ids) > 0:
        print(f"  Range: {min(unique_ids)} - {max(unique_ids)}")
    return unique_ids

def load_coco_mapping(data_dir):
    """加载 COCO 标注 (Train + Val) -> {coco_id: [captions]}"""
    train_path = os.path.join(data_dir, "coco_annotations/annotations/captions_train2017.json")
    val_path = os.path.join(data_dir, "coco_annotations/annotations/captions_val2017.json")
    
    # 备用路径兼容
    if not os.path.exists(train_path):
        train_path = os.path.join(data_dir, "coco_ann/captions_train2017.json")
        val_path = os.path.join(data_dir, "coco_ann/captions_val2017.json")

    coco_map = {}
    
    for p in [train_path, val_path]:
        print(f"[COCO] Loading {p} ...")
        if not os.path.exists(p):
            print(f"[WARN] 文件不存在: {p}")
            continue
            
        with open(p, "r") as f:
            data = json.load(f)
            for item in data["annotations"]:
                iid = item["image_id"]
                cap = item["caption"]
                if iid not in coco_map:
                    coco_map[iid] = []
                coco_map[iid].append(cap)
                
    print(f"[COCO] Loaded {len(coco_map)} images with captions.")
    return coco_map

def encode_and_save(subj, unique_ids, coco_map, stim_coco_ids, model, tokenizer, device):
    """为指定 ID 列表生成 CLIP 特征并未保存"""
    
    # 1. 准备文本
    captions_list = []
    valid_local_ids = []
    missing_count = 0
    
    print(f"[Subj0{subj}] Mapping Local ID -> COCO ID -> Caption...")
    
    for local_id in unique_ids:
        # 范围检查
        if local_id < 0 or local_id >= len(stim_coco_ids):
            missing_count += 1
            continue
            
        coco_id = stim_coco_ids[local_id]
        
        if coco_id in coco_map:
            # 默认取第一条 caption
            captions_list.append(coco_map[coco_id][0])
            valid_local_ids.append(local_id)
        else:
            missing_count += 1
            
    print(f"[Subj0{subj}] Matching: {len(valid_local_ids)} found, {missing_count} missing.")
    
    if len(valid_local_ids) == 0:
        print("[ERROR] No valid captions found. Skip encoding.")
        return

    # 2. 编码
    print(f"[Subj0{subj}] Encoding {len(captions_list)} captions with CLIP ViT-L/14...")
    
    batch_size = 128
    all_feats = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(captions_list), batch_size)):
            batch_caps = captions_list[i : i + batch_size]
            inputs = tokenizer(
                batch_caps, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            
            feats = model.get_text_features(**inputs) # [B, 768]
            all_feats.append(feats.cpu())
            
    all_feats = torch.cat(all_feats, dim=0) # [N, 768]
    print(f"[Subj0{subj}] Encoded shape: {all_feats.shape}")
    
    # 3. 保存
    os.makedirs(TEXT_OUT_DIR, exist_ok=True)
    save_path = os.path.join(TEXT_OUT_DIR, f"train_coco_text_clip_subj0{subj}.pt")
    
    result = {
        "image_ids": torch.tensor(valid_local_ids, dtype=torch.long),
        "text_feats": all_feats
    }
    
    torch.save(result, save_path)
    print(f"[SUCCESS] Saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate CLIP text features for specific subjects")
    parser.add_argument("--subj", type=int, nargs="+", default=[1], help="List of subjects to process (e.g. 1 2 5 7)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # 0. 加载映射
    stim_coco_ids = load_stim_info()
    
    # 1. 加载 COCO 字典 (一次性)
    coco_map = load_coco_mapping(DATA_DIR)
    
    # 2. 加载模型 (一次性)
    print(f"[CLIP] Loading model to {args.device}...")
    try:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device)
    except Exception as e:
        print(f"[ERROR] Failed to load CLIP model: {e}")
        return

    # 3. 循环处理
    for s in args.subj:
        print(f"\n{'='*40}\nProcessing Subject {s}\n{'='*40}")
        unique_ids = collect_train_ids(s)
        if unique_ids:
            encode_and_save(s, unique_ids, coco_map, stim_coco_ids, model, tokenizer, args.device)
            
    print("\n[ALL DONE]")

if __name__ == "__main__":
    main()
