#!/usr/bin/env python
# coding: utf-8
"""
extract_all_features.py

功能:
  加载一个预训练的 MindEye 模型，为多个被试提取并合并以下两种特征：
  1. 大脑解码向量 (fMRI -> 1664D)
  2. 对应的真值图像的 ViT-H 特征 (Image -> 1024D)
  
  这两个合并后的特征文件，将作为训练通用投影矩阵的输入。

用法:
  python tools/extract_all_features.py \
    --mindeye_model_dir /path/to/pretrain_model \
    --out_dir /path/to/output_data_dir
"""
import glob
import os
import sys
import json
import argparse
import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import webdataset as wds
from PIL import Image

# --- 一些常量：官方 MindEye2 的隐藏维度 & CLIP 维度 ---
OFFICIAL_H = 4096          # 官方 final_subj01_pretrained_40sess_24bs 用的是 4096
DEFAULT_H = 1024           # 你自己训练的小号模型用的是 1024，可以保留兼容
DEFAULT_N_BLOCKS = 4
CLIP_EMB_DIM = 1664
CLIP_SEQ_DIM = 256

# --- 添加必要的项目路径 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.dirname(script_dir)
src_path = os.path.join(proj_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)
if proj_root not in sys.path:
    sys.path.append(proj_root)

# --- 导入项目模块 ---
import importlib
utils = None
for mod_name in ("utils", "src.utils"):
    try:
        utils = importlib.import_module(mod_name)
        break
    except ModuleNotFoundError:
        continue
if utils is None:
    raise ImportError(
        f"无法导入 utils。已尝试模块名 ['utils','src.utils']，并添加路径: {src_path} 和 {proj_root}. 请确认项目结构。"
    )
from models import BrainNetwork, PriorNetwork, BrainDiffusionPrior
try:
    import open_clip
except ImportError:
    raise ImportError("open_clip not found. Please run `pip install open-clip-torch`.")

# ==============================================================================
# §1. 参数解析
# ==============================================================================
parser = argparse.ArgumentParser(description="为训练通用投影矩阵批量提取多被试特征")
parser.add_argument(
    "--mindeye_model_dir", type=str, required=True,
    help="指向预训练MindEye模型目录的路径 (包含 last.pth 和 可选 args.json)"
)
parser.add_argument(
    "--out_dir", type=str, required=True,
    help="输出合并后的特征文件 (.pt) 的目录"
)
parser.add_argument(
    "--data_path", type=str, default="/home/vipuser/MindEyeV2_Project/src",
    help="NSD数据集的根目录"
)
parser.add_argument(
    "--subjects", type=int, nargs='+', default=list(range(1, 9)),
    help="要处理的被试ID列表，默认为 1 到 8"
)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    help="运行设备"
)
parser.add_argument(
    "--split", type=str, default="test",
    choices=["train", "test"],
    help="使用哪个数据划分：'train' 或 'test'（默认 test）"
)
args = parser.parse_args()

# ==============================================================================
# §2. 模型加载
# ==============================================================================

print(f"🧠 加载预训练的 MindEye 模型从: {args.mindeye_model_dir}")

# --- 2.1 读取 / 推断模型配置 ---
ckpt_path = os.path.join(args.mindeye_model_dir, 'last.pth')
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"错误: 在指定目录中找不到模型权重文件 'last.pth': {args.mindeye_model_dir}")

# 默认先假设是你自己训练的小模型
hidden_dim = DEFAULT_H
n_blocks = DEFAULT_N_BLOCKS

args_json_path = os.path.join(args.mindeye_model_dir, 'args.json')
if os.path.exists(args_json_path):
    # 如果目录里有 args.json，就按里面的信息来（兼容你自己训练的模型）
    with open(args_json_path, 'r') as f:
        model_args = json.load(f)
    hidden_dim = model_args.get('hidden_dim', DEFAULT_H)
    n_blocks = model_args.get('n_blocks', DEFAULT_N_BLOCKS)
    print(f"   - 模型配置来自 args.json: hidden_dim={hidden_dim}, n_blocks={n_blocks}")
else:
    # 没有 args.json 的情况：很大概率是官方 final_subj01_pretrained_40sess_24bs
    # 这里我们根据目录名做一个简单的 heuristics
    if "final_subj01_pretrained_40sess_24bs" in os.path.basename(args.mindeye_model_dir):
        hidden_dim = OFFICIAL_H
        n_blocks = DEFAULT_N_BLOCKS
        print("   - 警告: 找不到 args.json，检测到是官方 subj01 40sess 模型，使用 OFFICIAL_H=4096.")
    else:
        hidden_dim = DEFAULT_H
        n_blocks = DEFAULT_N_BLOCKS
        print("   - 警告: 找不到 args.json，使用默认配置: hidden_dim=1024, n_blocks=4.")

# --- 2.2 加载所有被试体素数，用于构建 ridge 头 ---
num_voxels_list = []
for s in range(1, 9):
    try:
        f = h5py.File(f'{args.data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
        num_voxels_list.append(f['betas'].shape[1])
    except FileNotFoundError:
        # 某个被试缺数据时用占位符填充，保证长度为8
        num_voxels_list.append(10000)

# --- 2.3 构建 MindEye 模块 (只要 ridge + backbone 就够提 brain_clip) ---
class RidgeRegression(nn.Module):
    def __init__(self, input_sizes, out_features):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(s, out_features) for s in input_sizes])

    def forward(self, x, subj_idx: int):
        """
        x 期望形状:
          - [B, n_vox]
          - 或 [B, 1, n_vox]（多一维也兼容）
        返回:
          - [B, 1, hidden_dim]
        """
        # 如果是 [B, 1, n_vox]，压掉中间那一维
        if x.ndim == 3:
            x = x[:, 0, :]          # -> [B, n_vox]
        elif x.ndim == 1:
            x = x.unsqueeze(0)      # -> [1, n_vox]

        out = self.linears[subj_idx](x)  # [B, hidden_dim]
        return out.unsqueeze(1)          # [B, 1, hidden_dim]


class MindEyeModule(nn.Module):
    def __init__(self):
        super().__init__()

model = MindEyeModule()
model.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)
model.backbone = BrainNetwork(
    h=hidden_dim,
    in_dim=hidden_dim,
    seq_len=1,
    n_blocks=n_blocks,
    clip_size=CLIP_EMB_DIM,
    out_dim=CLIP_EMB_DIM * CLIP_SEQ_DIM,
    blurry_recon=False,   # 这里不开 blurry recon，只用语义主干提 brain_clip 即可
    clip_scale=1,
)
model.to(args.device)

# --- 2.4 加载权重：用 strict=False 允许多出来的模块 (diffusion_prior, blurry 分支等) ---
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint["model_state_dict"]

# 直接用 strict=False 加载，跳过未用到的模块键
load_msg = model.load_state_dict(state_dict, strict=False)
print("   - load_state_dict 结果:", load_msg)
model.eval()
print("   - ✅ MindEye 模型权重加载成功。")

# --- 2.5 加载 ViT-H 图像编码器 ---
print("\n🖼️  加载 ViT-H/14 图像编码器 (用于生成目标特征)...")
vith_model, _, vith_preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained="laion2b_s32b_b79k", device=args.device
)
vith_model.eval()
print("   - ✅ ViT-H/14 加载成功。")

# ==============================================================================
# §3. 特征提取
# ==============================================================================
all_brain_vectors = []
all_image_vectors = []

print("\n💾 打开 COCO 图像数据库...")
image_db = h5py.File(f'{args.data_path}/coco_images_224_float16.hdf5', 'r')
images_dataset = image_db['images']

def tensor_to_pil(t):
    if t.max() <= 1.0:
        t = t * 255.0
    return Image.fromarray(t.permute(1, 2, 0).to(torch.uint8).numpy())

# 遍历指定的每个被试
for subj_id in args.subjects:
    print(f"\n--- 开始处理被试: subj0{subj_id} ---")
    
    # 3.1 加载 fMRI 和测试集数据
    try:
        f = h5py.File(f'{args.data_path}/betas_all_subj0{subj_id}_fp32_renorm.hdf5', 'r')
        voxels = torch.Tensor(f['betas'][:]).to('cpu')
    except FileNotFoundError:
        print(f"   - 警告: 找不到 subj0{subj_id} 的fMRI数据，跳过该被试。")
        continue

        # --- 3.1 加载被试的 fMRI 和行为文件（train/test 可选） ---
    wds_root = os.path.join(args.data_path, "wds", f"subj0{subj_id}")

    # 根据 split 选择要读的 tar 文件
    urls = []
    if args.split == "test":
        # 优先用 test/，没有就用 new_test/
        for subdir in ["test", "new_test"]:
            pattern = os.path.join(wds_root, subdir, "*.tar")
            found = sorted(glob.glob(pattern))
            if found:
                urls = found
                print(f"   - 使用 {subdir} split，下有 {len(urls)} 个 shard")
                break
    else:  # train
        pattern = os.path.join(wds_root, "train", "*.tar")
        urls = sorted(glob.glob(pattern))
        if urls:
            print(f"   - 使用 train split，下有 {len(urls)} 个 shard")

    if not urls:
        print(f"   - 警告: 在 {wds_root} 下找不到 split='{args.split}' 的 WebDataset，跳过该被试。")
        continue

    # 先跑一遍统计样本数，再重新构建一次 dataset 给 DataLoader 用
    dataset = wds.WebDataset(urls).decode("torch").to_tuple("behav.npy")
    num_samples = sum(1 for _ in dataset)
    print(f"   - 该 split 总样本数: {num_samples}")

    dataset = wds.WebDataset(urls).decode("torch").to_tuple("behav.npy")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)

    # 获取该 split 对应的 fMRI 体素和图像索引
    behav = next(iter(dataloader))[0]   # [N, 1, 6] 类似
    subj_voxels = voxels[behav[:, 0, 5].long()]   # trial idx -> voxel 行
    subj_image_indices = behav[:, 0, 0].long()    # trial 对应的图像索引

    # 对重复图像取平均
    unique_indices, inverse_indices = torch.unique(subj_image_indices, return_inverse=True)
    print(f"   - 找到 {len(unique_indices)} 个唯一图像样本。")
    
    # 3.2 提取大脑 & 图像特征
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(args.device == "cuda")):
        for i in tqdm(range(len(unique_indices)), desc=f"   - 提取特征 (Subj {subj_id})"):
            img_idx = unique_indices[i]
            
            # 所有对应的 fMRI 重复
            # 找到该图像对应的所有 fMRI 重复
            fmri_locs = (subj_image_indices == img_idx).nonzero(as_tuple=False).view(-1)
            if fmri_locs.numel() == 0:
                continue

            # 取出这些重复的体素数据
            fmri_samples = subj_voxels[fmri_locs]  # 预期形状: [K, n_vox]，但也兼容 [K, 1, n_vox]
            if fmri_samples.ndim == 3:
                # 如果多了一维，压掉中间那一维 -> [K, n_vox]
                fmri_samples = fmri_samples[:, 0, :]

            # 在“重复维度 K”上做平均，不要动 voxel 维
            avg_voxel = fmri_samples.mean(dim=0, keepdim=True)   # [1, n_vox]
            avg_voxel = avg_voxel.to(args.device)

            # 1. 提取大脑解码向量 (fMRI -> 1664D)
            ridge_out = model.ridge(avg_voxel, subj_id - 1)      # [1, 1, H]
            _, brain_vec, _ = model.backbone(ridge_out)          # [1, 256, 1664]
            brain_vec_1664 = brain_vec.mean(dim=1)               # [1, 1664]
            all_brain_vectors.append(brain_vec_1664.cpu())

            
            # 2) 真值图像 -> ViT-H 特征 (shape [1, 1024])
            gt_image_data = torch.from_numpy(images_dataset[img_idx.item()]).float()
            pil_img = tensor_to_pil(gt_image_data)
            vith_image_input = vith_preprocess(pil_img).unsqueeze(0).to(args.device)
            image_vec_1024 = vith_model.encode_image(vith_image_input)
            all_image_vectors.append(image_vec_1024.cpu())

image_db.close()

# ==============================================================================
# §4. 合并并保存
# ==============================================================================
if not all_brain_vectors or not all_image_vectors:
    print("\n❌ 错误: 未能成功提取任何特征。请检查模型路径和数据路径是否正确。")
    sys.exit(1)

print("\n--- 特征提取完成，正在合并和保存... ---")

final_brain_vectors = torch.cat(all_brain_vectors, dim=0)
final_image_vectors = torch.cat(all_image_vectors, dim=0)

final_brain_vectors = nn.functional.normalize(final_brain_vectors, dim=1)
final_image_vectors = nn.functional.normalize(final_image_vectors, dim=1)

os.makedirs(args.out_dir, exist_ok=True)

out_brain_path = os.path.join(args.out_dir, "all_subjects_brain_vectors.pt")
out_image_path = os.path.join(args.out_dir, "all_subjects_gt_vith.pt")

torch.save(final_brain_vectors, out_brain_path)
torch.save(final_image_vectors, out_image_path)

print("\n🎉 全部完成！")
print(f"✅ 通用大脑解码向量已保存: {out_brain_path}")
print(f"   - 形状: {final_brain_vectors.shape}")
print(f"✅ 通用图像目标向量已保存: {out_image_path}")
print(f"   - 形状: {final_image_vectors.shape}")
print("\n现在您可以使用这些文件来训练一个通用的投影矩阵了。")
