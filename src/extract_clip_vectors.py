#!/usr/bin/env python
# coding: utf-8
"""
精简版推理脚本：只提取 brain->CLIP 向量，不做图像生成
适用于后续 RAG 检索和 SD1.5/SDXL 生成流程
"""

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

# 添加本地路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import utils
from models import BrainNetwork, PriorNetwork, BrainDiffusionPrior

# 禁用 xformers 避免兼容性问题
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["FLASH_ATTENTION_DISABLE"] = "1"

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  使用设备: {device}")

# ==================== 参数解析 ====================
parser = argparse.ArgumentParser(description="提取 brain->CLIP 向量（精简版）")
parser.add_argument("--model_name", type=str, required=True, help="模型名称")
parser.add_argument("--data_path", type=str, required=True, help="NSD 数据路径")
parser.add_argument("--subj", type=int, required=True, choices=[1,2,3,4,5,6,7,8], help="被试编号")
parser.add_argument("--hidden_dim", type=int, default=1024, help="隐藏层维度")
parser.add_argument("--n_blocks", type=int, default=4, help="Backbone 块数")
parser.add_argument("--new_test", action="store_true", help="使用新测试集")
parser.add_argument("--clip_out", type=str, required=True, help="CLIP 向量输出路径 (.pt)")
parser.add_argument("--ids_out", type=str, default=None, help="图像 ID 输出路径 (.json)")
parser.add_argument("--seed", type=int, default=42, help="随机种子")

args = parser.parse_args()

# ==================== 根据 checkpoint 推断 hidden_dim（兼容官方大模型） ====================
proj_root = os.path.dirname(script_dir)
candidate_model_dirs = [
    f"/home/vipuser/train_logs/{args.model_name}",
    f"/home/train_logs/{args.model_name}",
    os.path.join(proj_root, "train_logs", args.model_name),
]

model_dir_for_cfg = None
for od in candidate_model_dirs:
    if os.path.isdir(od):
        model_dir_for_cfg = od
        break
if model_dir_for_cfg is None:
    # 如果一个都没找到，就用最后一个候选路径作为兜底（方便本地/其它机器）
    model_dir_for_cfg = candidate_model_dirs[-1]

def infer_hidden_dim(model_dir: str, default_h: int = 1024) -> int:
    """
    尝试从 args.json 或模型目录名推断 hidden_dim：
      1) 若存在 args.json，则优先读取 hidden_dim/h/H 字段；
      2) 若检测到是官方 subj01 40sess 模型，则使用 OFFICIAL_H=4096；
      3) 否则退回默认值 default_h。
    """
    args_path = os.path.join(model_dir, "args.json")
    if os.path.exists(args_path):
        try:
            with open(args_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k in ("hidden_dim", "h", "H"):
                if k in cfg:
                    h = int(cfg[k])
                    print(f"   - 从 args.json 读取 hidden_dim={h}")
                    return h
            print(f"   - args.json 中未找到 hidden_dim 字段，使用默认 hidden_dim={default_h}")
            return default_h
        except Exception as e:
            print(f"   - 警告: 读取 args.json 失败({e})，使用默认 hidden_dim={default_h}")
            return default_h

    base = os.path.basename(os.path.normpath(model_dir))
    if "final_subj01_pretrained_40sess_24bs" in base:
        OFFICIAL_H = 4096
        print("   - 警告: 找不到 args.json，检测到是官方 subj01 40sess 模型，使用 OFFICIAL_H=4096.")
        return OFFICIAL_H

    print(f"   - 警告: 找不到 args.json，使用默认 hidden_dim={default_h}")
    return default_h

# 覆盖命令行里的 hidden_dim，确保官方大模型用 4096
args.hidden_dim = infer_hidden_dim(model_dir_for_cfg, args.hidden_dim)

# 设置随机种子
utils.seed_everything(args.seed)

# 创建输出目录
os.makedirs(os.path.dirname(args.clip_out), exist_ok=True)
if args.ids_out:
    os.makedirs(os.path.dirname(args.ids_out), exist_ok=True)

print(f"\n{'='*60}")
print(f"📋 配置信息")
print(f"{'='*60}")
print(f"模型名称: {args.model_name}")
print(f"被试: subj0{args.subj}")
print(f"隐藏层维度: {args.hidden_dim}")
print(f"CLIP 向量输出: {args.clip_out}")
if args.ids_out:
    print(f"图像 ID 输出: {args.ids_out}")
print(f"{'='*60}\n")

# ==================== 加载 fMRI 数据 ====================
print("📦 加载 fMRI 体素数据...")
voxels = {}
f = h5py.File(f'{args.data_path}/betas_all_subj0{args.subj}_fp32_renorm.hdf5', 'r')
betas = f['betas'][:]
betas = torch.Tensor(betas).to("cpu")
num_voxels = betas[0].shape[-1]
voxels[f'subj0{args.subj}'] = betas
print(f"✅ 加载完成，体素数: {num_voxels}")

# ==================== 加载测试集 ====================
print("\n📊 加载测试集...")
if not args.new_test:
    if args.subj in [3, 6]:
        num_test = 2113
    elif args.subj in [4, 8]:
        num_test = 1985
    else:
        num_test = 2770
    test_url = f"{args.data_path}/wds/subj0{args.subj}/test/0.tar"
else:
    if args.subj in [3, 6]:
        num_test = 2371
    elif args.subj in [4, 8]:
        num_test = 2188
    else:
        num_test = 3000
    test_url = f"{args.data_path}/wds/subj0{args.subj}/new_test/0.tar"

print(f"测试集路径: {test_url}")

def my_split_by_node(urls):
    return urls

test_data = (
    wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)
    .decode("torch")
    .rename(
        behav="behav.npy",
        past_behav="past_behav.npy",
        future_behav="future_behav.npy",
        olds_behav="olds_behav.npy",
    )
    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
)

test_dl = torch.utils.data.DataLoader(
    test_data,
    batch_size=num_test,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
)
print(f"✅ 测试集加载完成，样本数: {num_test}")

# ==================== 准备测试数据索引 ====================
print("\n🔍 准备测试数据索引...")
test_images_idx = []
test_voxels_idx = []

for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = voxels[f'subj0{args.subj}'][behav[:, 0, 5].cpu().long()]
    test_voxels_idx = np.append(test_voxels_idx, behav[:, 0, 5].cpu().numpy())
    test_images_idx = np.append(test_images_idx, behav[:, 0, 0].cpu().numpy())

test_images_idx = test_images_idx.astype(int)
test_voxels_idx = test_voxels_idx.astype(int)

unique_images = np.unique(test_images_idx)
print(f"✅ 索引准备完成")
print(f"   - 总体素样本: {len(test_voxels)}")
print(f"   - 唯一图像数: {len(unique_images)}")

# ==================== 构建 MindEye 模型 ====================
print(f"\n🧠 构建 MindEye 模型...")

# CLIP 参数（对应 ViT-bigG-14）
clip_seq_dim = 256
clip_emb_dim = 1664

class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()

    def forward(self, x):
        return x

model = MindEyeModule()

# Ridge Regression
class RidgeRegression(torch.nn.Module):
    def __init__(self, input_sizes, out_features):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_size, out_features) for input_size in input_sizes]
        )

    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
        return out

model.ridge = RidgeRegression([num_voxels], out_features=args.hidden_dim)

# Backbone Network
model.backbone = BrainNetwork(
    h=args.hidden_dim,
    in_dim=args.hidden_dim,
    seq_len=1,
    clip_size=clip_emb_dim,
    out_dim=clip_emb_dim * clip_seq_dim,
)

# Diffusion Prior
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim // 52
timesteps = 100

prior_network = PriorNetwork(
    dim=out_dim,
    depth=depth,
    dim_head=dim_head,
    heads=heads,
    causal=False,
    num_tokens=clip_seq_dim,
    learned_query_mode="pos_emb",
)

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)

model.to(device)

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ 模型构建完成")
print(f"   - 总参数: {total_params:,}")
print(f"   - 可训练参数: {trainable_params:,}")

# ==================== 加载预训练权重 ====================
print(f"\n📥 加载预训练权重...")

# 优先在常见位置查找 last.pth，按顺序尝试，便于在不同部署下直接运行
proj_root = os.path.dirname(script_dir)
candidate_outdirs = [
    f"/home/vipuser/train_logs/{args.model_name}",
    f"/home/train_logs/{args.model_name}",
    os.path.join(proj_root, "train_logs", args.model_name),
]

pth_path = None
for od in candidate_outdirs:
    # 支持 .pth 或 .pt 扩展名
    for fname in ("last.pth", "last.pt"):
        pp = os.path.join(od, fname)
        if os.path.exists(pp):
            pth_path = pp
            outdir = od
            break
    if pth_path is not None:
        break

if pth_path is None:
    tried = "\n  - ".join(candidate_outdirs)
    raise FileNotFoundError(
        f"未找到 {args.model_name} 的 last.pth/.pt。已尝试位置:\n  - {tried}\n"
        f"请确认训练输出目录或将 checkpoint 放置到上述任一位置。"
    )

print(f"🔎  使用检查点目录: {outdir}")
checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

# 尝试找到 state_dict
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# 严格模式关闭，允许官方模型里多一些/少一些模块
model.load_state_dict(state_dict, strict=False)
del checkpoint, state_dict

print(f"✅ 权重加载成功: {pth_path}")

# ==================== 开始提取 CLIP 向量 ====================
print(f"\n{'='*60}")
print(f"🚀 开始提取 brain->CLIP 向量")
print(f"{'='*60}\n")

model.eval()
saved_vecs = []
saved_ids = []

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    for uniq_img in tqdm(unique_images, desc="提取 CLIP 向量"):
        # 找到该图像的所有重复
        locs = np.where(test_images_idx == uniq_img)[0]

        # 确保有 3 个重复（MindEye2 的标准做法）
        if len(locs) == 1:
            locs = locs.repeat(3)
        elif len(locs) == 2:
            locs = np.concatenate((locs, locs[:1]))

        # 获取对应的体素数据
        voxel = test_voxels[None, locs].to(device)  # [1, 3, num_voxels]

        # 对 3 个重复求平均
        accum_clip_voxels = None

        for rep in range(3):
            voxel_ridge = model.ridge(voxel[:, [rep]], 0)
            backbone_out, clip_voxels_out, blurry_image_enc_out = model.backbone(voxel_ridge)

            if rep == 0:
                accum_clip_voxels = clip_voxels_out
            else:
                accum_clip_voxels += clip_voxels_out

        # 平均
        clip_voxels = accum_clip_voxels / 3  # [1, 256, 1664]

        # 如果是序列，取平均池化得到单个向量
        if clip_voxels.dim() == 3:
            vec = clip_voxels.mean(dim=1)  # [1, 1664]
        else:
            vec = clip_voxels

        # 转为 CPU 并保存
        vec = vec.squeeze(0).detach().float().cpu()  # [1664]
        saved_vecs.append(vec)
        saved_ids.append(int(uniq_img))

# ==================== 保存结果 ====================
print(f"\n{'='*60}")
print(f"💾 保存结果")
print(f"{'='*60}")

# 保存 CLIP 向量
V = torch.stack(saved_vecs, dim=0)  # [N, 1664]
torch.save(V, args.clip_out)
print(f"✅ CLIP 向量已保存")
print(f"   - 路径: {args.clip_out}")
print(f"   - 形状: {tuple(V.shape)}")

# 保存图像 ID
if args.ids_out:
    with open(args.ids_out, "w", encoding="utf-8") as f:
        json.dump(saved_ids, f, indent=2)
    print(f"✅ 图像 ID 已保存")
    print(f"   - 路径: {args.ids_out}")
    print(f"   - 数量: {len(saved_ids)}")
else:
    # 默认保存到与 clip_out 同目录
    default_ids_path = args.clip_out.replace(".pt", "_ids.json")
    with open(default_ids_path, "w", encoding="utf-8") as f:
        json.dump(saved_ids, f, indent=2)
    print(f"✅ 图像 ID 已保存（默认路径）")
    print(f"   - 路径: {default_ids_path}")
    print(f"   - 数量: {len(saved_ids)}")

print(f"\n{'='*60}")
print(f"🎉 提取完成！")
print(f"{'='*60}\n")

print("📊 后续使用建议：")
print("1. 使用 brain_clip.pt 作为 RAG 检索的 query 向量")
print("2. 检索 Top-K 最相似的 COCO 图像及其 captions")
print("3. 将检索到的 captions 输入 LLM 生成结构化提示")
print("4. 使用生成的提示 + IP-Adapter 进行 SD1.5/SDXL 生成")
