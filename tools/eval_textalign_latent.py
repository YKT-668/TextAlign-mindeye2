#!/usr/bin/env python
# coding: utf-8
import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F

# ----------------------------------
# 基本路径
# ----------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

# generative-models 里的 FrozenOpenCLIPImageEmbedder
_GEN_MODELS_DIR = os.path.join(_PROJ_ROOT, "generative-models")
if _GEN_MODELS_DIR not in sys.path:
    sys.path.append(_GEN_MODELS_DIR)
from sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

device = "cuda" if torch.cuda.is_available() else "cpu"

# 你当前实验的名字
exp_name   = "s7_textalign_coco_train_long_v10"
infer_dir  = os.path.join(_PROJ_ROOT, "train_logs", exp_name, "inference")
brain_path = os.path.join(infer_dir, "brain_clip.pt")
ids_path   = os.path.join(infer_dir, "ids.json")

# 之前评估 pipeline 时生成的 all_images.pt
gt_path    = os.path.join(_PROJ_ROOT, "evals", "all_images.pt")

print("[PATH] brain:", brain_path)
print("[PATH] ids  :", ids_path)
print("[PATH] gt   :", gt_path)

assert os.path.isfile(brain_path), f"brain_clip.pt 不存在: {brain_path}"
assert os.path.isfile(ids_path),   f"ids.json 不存在: {ids_path}"
assert os.path.isfile(gt_path),    f"GT 特征文件不存在: {gt_path}"

# ----------------------------------
# 1) 读取 brain_clip
# ----------------------------------
obj = torch.load(brain_path, map_location="cpu")
if isinstance(obj, torch.Tensor):
    brain_feats = obj
elif isinstance(obj, dict):
    cand = None
    for k, v in obj.items():
        if torch.is_tensor(v) and v.dim() == 2:
            cand = v
            print(f"[LOAD] brain_clip: 使用 dict['{k}'] 作为特征")
            break
    assert cand is not None, f"brain_clip.pt 中没有找到合适的 2D tensor: keys={list(obj.keys())}"
    brain_feats = cand
else:
    raise RuntimeError(f"brain_clip 格式不支持: {type(obj)}")

print("[INFO] brain_feats:", brain_feats.shape)

# ----------------------------------
# 2) 读取 ids.json（1000 个 global image id）
# ----------------------------------
with open(ids_path, "r") as f:
    brain_ids = json.load(f)
brain_ids = np.asarray(brain_ids, dtype=np.int64)
print("[INFO] brain_ids:", brain_ids.shape, "min=", brain_ids.min(), "max=", brain_ids.max())

# ----------------------------------
# 3) 读取 GT（all_images.pt），如有必要先用 CLIP 编成特征
# ----------------------------------
gt_obj = torch.load(gt_path, map_location="cpu")

gt_feats = None
gt_ids   = None

if isinstance(gt_obj, torch.Tensor):
    # 情况 A：已经是 [N, D] 的特征
    if gt_obj.dim() == 2:
        gt_feats = gt_obj
        print("[GT] 直接使用 tensor 作为特征，shape =", gt_feats.shape)

    # 情况 B：是 [N, 3, 224, 224] 的图像，需要过一遍 CLIP
    elif gt_obj.dim() == 4 and gt_obj.shape[1] == 3:
        imgs = gt_obj.float()   # [N,3,224,224]
        print("[GT] 检测到原始图像，shape =", imgs.shape, " -> 使用 CLIP 编码为 1664 维特征")

        # 建立与训练/推理一致的 CLIP image encoder
        clip_img_embedder = FrozenOpenCLIPImageEmbedder(
            arch="ViT-bigG-14",
            version="laion2b_s39b_b160k",
            output_tokens=True,
        ).to(device)
        clip_img_embedder.eval()
        for p in clip_img_embedder.parameters():
            p.requires_grad_(False)

        feats_list = []
        bs = 64
        with torch.no_grad():
            for i in range(0, imgs.size(0), bs):
                x = imgs[i:i + bs].to(device)
                out = clip_img_embedder(x)
                # 和训练里一样：优先拿 out[0]
                if isinstance(out, tuple):
                    z = out[0]
                else:
                    z = out
                # 若是 [B, 256, 1664]，取 CLS token 或平均；这里采用 CLS(第 0 个)
                if z.dim() == 3:
                    z = z[:, 0, :]   # [B, 1664]
                elif z.dim() == 2:
                    pass
                else:
                    raise RuntimeError(f"CLIP 输出维度异常: {z.shape}")
                feats_list.append(z.detach().cpu())
        gt_feats = torch.cat(feats_list, dim=0)
        print("[GT] CLIP 特征编码完成, shape =", gt_feats.shape)

    else:
        raise RuntimeError(f"GT tensor 形状不支持: {gt_obj.shape}")

elif isinstance(gt_obj, dict):
    # dict 情况：先找特征
    for key in ["features", "feats", "clip_feats", "img_feats"]:
        if key in gt_obj and torch.is_tensor(gt_obj[key]):
            gt_feats = gt_obj[key]
            print(f"[GT] 使用 dict['{key}'] 作为特征，shape =", gt_feats.shape)
            break
    if gt_feats is None:
        for k, v in gt_obj.items():
            if torch.is_tensor(v) and v.dim() == 2:
                gt_feats = v
                print(f"[GT] 兜底使用 dict['{k}'] 作为特征，shape =", gt_feats.shape)
                break
    assert gt_feats is not None, f"GT 文件中没有找到 2D 特征 tensor: keys={list(gt_obj.keys())}"

    # 再找 ids
    for key in ["ids", "image_ids", "img_ids", "nsd_ids"]:
        if key in gt_obj:
            gt_ids = np.asarray(gt_obj[key], dtype=np.int64)
            print(f"[GT] 使用 dict['{key}'] 作为 image_ids，shape =", gt_ids.shape)
            break
else:
    raise RuntimeError(f"GT 文件格式不支持: {type(gt_obj)}")

# 如果没有单独的 gt_ids，就假设顺序与 brain_ids 一致
if gt_ids is None:
    gt_ids = brain_ids.copy()
    print("[GT] 未检测到单独的 ids 字段，假设顺序与 brain_ids 一致")

# 维度检查
# assert gt_feats.shape[0] == len(gt_ids), f"gt_feats N={gt_feats.shape[0]} 与 gt_ids N={len(gt_ids)} 不一致"
# assert gt_feats.shape[1] == brain_feats.shape[1], (
 #   f"特征维度不一致: brain D={brain_feats.shape[1]} vs gt D={gt_feats.shape[1]}"
#)

# ----------------------------------
# 4) 按 ids 对齐顺序（保证 brain_feats[i] 和 gt_feats[i] 是同一张图）
# ----------------------------------
id2row = {int(gid): i for i, gid in enumerate(gt_ids)}
rows = []
for gid in brain_ids:
    if int(gid) not in id2row:
        raise KeyError(f"GT 中找不到 image_id={int(gid)}，请检查 GT 特征文件的 ids 是否完整")
    rows.append(id2row[int(gid)])
rows = np.asarray(rows, dtype=np.int64)

gt_sel = gt_feats[rows]
print("[INFO] gt_sel:", gt_sel.shape)

assert brain_feats.shape == gt_sel.shape, f"特征维度不一致: brain {brain_feats.shape} vs gt {gt_sel.shape}"

# ----------------------------------
# 5) 归一化并计算相似度矩阵
# ----------------------------------
# 统一到同一个设备 & float32，避免 float vs half 报错
brain_feats = brain_feats.to(device=device, dtype=torch.float32)
gt_sel      = gt_sel.to(device=device, dtype=torch.float32)

brain_n = F.normalize(brain_feats, dim=-1)
gt_n    = F.normalize(gt_sel,     dim=-1)

sim = brain_n @ gt_n.t()      # [N, N]
labels = torch.arange(sim.size(0), device=sim.device)


def topk_acc(sim_mat, labels, k=1):
    topk = sim_mat.topk(k, dim=-1).indices   # [N, k]
    correct = (topk == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return correct

# FWD: brain -> image
top1 = topk_acc(sim, labels, k=1)
top5 = topk_acc(sim, labels, k=5)

# BWD: image -> brain（对称性检查）
sim_T = sim.t()
top1_b = topk_acc(sim_T, labels, k=1)
top5_b = topk_acc(sim_T, labels, k=5)

print("====================================================")
print("TextAlign latent eval (subj01 new_test 1000):")
print(f"  FWD  Top-1: {top1*100:.2f}%   Top-5: {top5*100:.2f}%")
print(f"  BWD  Top-1: {top1_b*100:.2f}%   Top-5: {top5_b*100:.2f}%")
print("====================================================")
