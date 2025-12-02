#!/usr/bin/env python 
# coding: utf-8
import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------------------------
# 基本路径
# ----------------------------------
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

# 结果根目录：统一放到 cache/model_eval_results
RESULT_ROOT = os.path.join(_PROJ_ROOT, "cache", "model_eval_results")

# generative-models 里的 FrozenOpenCLIPImageEmbedder
_GEN_MODELS_DIR = os.path.join(_PROJ_ROOT, "generative-models")
if _GEN_MODELS_DIR not in sys.path:
    sys.path.append(_GEN_MODELS_DIR)
from sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

device = "cuda" if torch.cuda.is_available() else "cpu"

# 你当前实验的名字（改这里就行）
exp_name   = os.environ.get("EXP_NAME", "s1_textalign_coco_train_long_v1")

# 指定本地 bigG 权重（避免联网下载）。如不存在则回退到默认 tag。
LOCAL_BIGG_BIN = os.path.join(_PROJ_ROOT, "cache", "hf_bigG", "open_clip_pytorch_model.bin")

# ----------------------------------
# 路径设置
# ----------------------------------
exp_dir    = os.path.join(_PROJ_ROOT, "train_logs", exp_name)
infer_dir  = os.path.join(exp_dir, "inference")
brain_path = os.path.join(infer_dir, "brain_clip.pt") if os.path.isfile(os.path.join(infer_dir, "brain_clip.pt")) else (
    os.path.join(exp_dir, "brain_clip.pt") if os.path.isfile(os.path.join(exp_dir, "brain_clip.pt")) else None
)
ids_path   = os.path.join(infer_dir, "ids.json") if os.path.isfile(os.path.join(infer_dir, "ids.json")) else (
    os.path.join(exp_dir, "ids.json") if os.path.isfile(os.path.join(exp_dir, "ids.json")) else None
)

# 之前评估 pipeline 时生成的 all_images.pt
gt_path    = os.path.join(_PROJ_ROOT, "evals", "all_images.pt")

print("[PATH] brain:", brain_path)
print("[PATH] ids  :", ids_path)
print("[PATH] gt   :", gt_path)

assert brain_path is not None and os.path.isfile(brain_path), f"brain_clip.pt 不存在于 {exp_dir} 或其 inference 子目录"
assert ids_path is not None and os.path.isfile(ids_path),   f"ids.json 不存在于 {exp_dir} 或其 inference 子目录"
assert os.path.isfile(gt_path),    f"GT 特征文件不存在: {gt_path}"

# 本次评估结果输出目录
result_dir = os.path.join(RESULT_ROOT, exp_name)
os.makedirs(result_dir, exist_ok=True)
print("[OUT] result_dir:", result_dir)

# ======================================================================
# 1) 读取 brain_clip（[N, D]）
# ======================================================================
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

# ======================================================================
# 2) 读取 ids.json（1000 个 global image id）
# ======================================================================
with open(ids_path, "r") as f:
    brain_ids = json.load(f)
brain_ids = np.asarray(brain_ids, dtype=np.int64)
print("[INFO] brain_ids:", brain_ids.shape, "min=", brain_ids.min(), "max=", brain_ids.max())

# ======================================================================
# 3) 读取 GT（all_images.pt），如有必要先用 CLIP 编成特征
# ======================================================================
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
        # 如果是原始图像，优先尝试离线缓存特征文件，避免再次下载大权重
        offline_feat_path = os.path.join(_PROJ_ROOT, "evals", "all_images_bigG_1664.pt")
        imgs = gt_obj.float()   # [N,3,224,224]
        if os.path.isfile(offline_feat_path):
            gt_feats = torch.load(offline_feat_path, map_location="cpu")
            assert gt_feats.dim() == 2 and gt_feats.shape[0] == imgs.shape[0], (
                f"缓存特征形状不匹配: {gt_feats.shape} vs {imgs.shape[0]}"
            )
            print(f"[GT] 发现离线特征文件 {offline_feat_path}，直接加载，shape = {gt_feats.shape}")
        else:
            print("[GT] 检测到原始图像，shape =", imgs.shape, " -> 使用 CLIP 编码为 1664 维特征 (首次生成缓存)")
            # 若存在本地权重，直接用路径；否则回退到官方标识（可能触发下载）。
            version_str = LOCAL_BIGG_BIN if os.path.isfile(LOCAL_BIGG_BIN) else "laion2b_s39b_b160k"
            if os.path.isfile(LOCAL_BIGG_BIN):
                print(f"[CLIP] 使用本地权重: {LOCAL_BIGG_BIN}")
            else:
                print("[CLIP] 未找到本地权重，回退为 laion2b_s39b_b160k（可能需要联网下载）")
            clip_img_embedder = FrozenOpenCLIPImageEmbedder(
                arch="ViT-bigG-14",
                version=version_str,
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
                    if isinstance(out, tuple):
                        z = out[0]
                    else:
                        z = out
                    if z.dim() == 3:
                        z = z[:, 0, :]   # [B, 1664]
                    elif z.dim() == 2:
                        pass
                    else:
                        raise RuntimeError(f"CLIP 输出维度异常: {z.shape}")
                    feats_list.append(z.detach().cpu())
            gt_feats = torch.cat(feats_list, dim=0)
            print("[GT] CLIP 特征编码完成, shape =", gt_feats.shape, " -> 保存到离线文件")
            try:
                torch.save(gt_feats, offline_feat_path)
                print(f"[GT] 离线特征已保存: {offline_feat_path}")
            except Exception as e:
                print(f"[WARN] 保存离线特征失败: {e}")

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

# ======================================================================
# 4) 按 ids 对齐顺序（保证 brain_feats[i] 和 gt_feats[i] 是同一张图）
# ======================================================================
id2row = {int(gid): i for i, gid in enumerate(gt_ids)}
rows = []
for gid in brain_ids:
    if int(gid) not in id2row:
        raise KeyError(f"GT 中找不到 image_id={int(gid)}，请检查 GT 特征文件的 ids 是否完整")
    rows.append(id2row[int(gid)])
rows = np.asarray(rows, dtype=np.int64)

gt_sel = gt_feats[rows]
print("[INFO] gt_sel:", gt_sel.shape)

assert brain_feats.shape == gt_sel.shape, \
    f"特征维度不一致: brain {brain_feats.shape} vs gt {gt_sel.shape}"

# ======================================================================
# 5) 归一化并计算相似度矩阵
# ======================================================================
brain_feats = brain_feats.to(device=device, dtype=torch.float32)
gt_sel      = gt_sel.to(device=device, dtype=torch.float32)

brain_n = F.normalize(brain_feats, dim=-1)
gt_n    = F.normalize(gt_sel,     dim=-1)

# sim[i, j] = cos(z_brain[i], z_img[j])
sim = brain_n @ gt_n.t()      # [N, N]
labels = torch.arange(sim.size(0), device=sim.device)

# 对角线相似度（CLIP latent 对齐程度）
sim_diag = torch.diag(sim)    # [N]

# ======================================================================
# 6) 各种指标计算
# ======================================================================
def topk_acc(sim_mat, labels, k=1):
    topk = sim_mat.topk(k, dim=-1).indices   # [N, k]
    correct = (topk == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return correct

def rank_stats(sim_mat, labels):
    """
    返回：
      ranks: 每个样本的 rank（1=最优）
      recall_at: dict{k: recall@k}
      mean_rank, median_rank
    """
    # sim_mat: [N, N]，越大越相似
    sorted_idx = sim_mat.argsort(dim=1, descending=True)  # [N, N]
    # sorted_idx[i] 中找 label i 所在的位置
    matches = (sorted_idx == labels.unsqueeze(1))         # [N, N] bool
    # 每行恰好有一个 True，argmax 给出索引
    ranks = matches.float().argmax(dim=1) + 1            # [N] in [1..N]

    ranks_np = ranks.detach().cpu().numpy()
    mean_rank   = float(ranks_np.mean())
    median_rank = float(np.median(ranks_np))

    recall_at = {}
    for k in [1, 5, 10, 20, 50]:
        recall_at[k] = float((ranks_np <= k).mean())

    return ranks, recall_at, mean_rank, median_rank

# FWD: brain -> image
top1_fwd = topk_acc(sim, labels, k=1)
top5_fwd = topk_acc(sim, labels, k=5)
ranks_fwd, recall_fwd, mean_rank_fwd, median_rank_fwd = rank_stats(sim, labels)

# BWD: image -> brain
sim_T = sim.t()
top1_bwd = topk_acc(sim_T, labels, k=1)
top5_bwd = topk_acc(sim_T, labels, k=5)
ranks_bwd, recall_bwd, mean_rank_bwd, median_rank_bwd = rank_stats(sim_T, labels)

# CLIP(latent) 相似度统计
sim_diag_np = sim_diag.detach().cpu().numpy()
clip_latent_mean   = float(sim_diag_np.mean())
clip_latent_std    = float(sim_diag_np.std())
clip_latent_p25    = float(np.percentile(sim_diag_np, 25))
clip_latent_p50    = float(np.percentile(sim_diag_np, 50))
clip_latent_p75    = float(np.percentile(sim_diag_np, 75))

# ======================================================================
# 7) 打印 & 保存数值结果
# ======================================================================
print("====================================================")
print("TextAlign latent eval (单被试 latent 检索 + CLIP(latent))")
print(f"[Retrieval FWD] Top-1: {top1_fwd*100:.2f}%   Top-5: {top5_fwd*100:.2f}%")
print(f"[Retrieval BWD] Top-1: {top1_bwd*100:.2f}%   Top-5: {top5_bwd*100:.2f}%")
print(f"[FWD]  mean rank = {mean_rank_fwd:.2f}, median rank = {median_rank_fwd:.2f}")
print(f"[BWD]  mean rank = {mean_rank_bwd:.2f}, median rank = {median_rank_bwd:.2f}")
print("[FWD]  Recall@K:", {k: f"{v*100:.2f}%" for k, v in recall_fwd.items()})
print("[BWD]  Recall@K:", {k: f"{v*100:.2f}%" for k, v in recall_bwd.items()})
print(f"[CLIP(latent)] mean={clip_latent_mean:.4f}, std={clip_latent_std:.4f}, "
      f"p25={clip_latent_p25:.4f}, p50={clip_latent_p50:.4f}, p75={clip_latent_p75:.4f}")
print("====================================================")

# 保存 metrics.json
metrics = {
    "exp_name": exp_name,
    "N": int(sim.size(0)),
    "retrieval": {
        "fwd": {
            "top1": top1_fwd,
            "top5": top5_fwd,
            "mean_rank": mean_rank_fwd,
            "median_rank": median_rank_fwd,
            "recall_at": recall_fwd,
        },
        "bwd": {
            "top1": top1_bwd,
            "top5": top5_bwd,
            "mean_rank": mean_rank_bwd,
            "median_rank": median_rank_bwd,
            "recall_at": recall_bwd,
        },
    },
    "clip_latent": {
        "mean": clip_latent_mean,
        "std": clip_latent_std,
        "p25": clip_latent_p25,
        "p50": clip_latent_p50,
        "p75": clip_latent_p75,
    },
}

with open(os.path.join(result_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(result_dir, "metrics.txt"), "w") as f:
    f.write("=== TextAlign latent eval ===\n")
    f.write(json.dumps(metrics, indent=2))
    f.write("\n")

# ======================================================================
# 8) 可视化：直方图 & 检索曲线 & 热力图
# ======================================================================

# 8.1 CLIP(latent) 对角线相似度直方图
plt.figure(figsize=(5,4))
plt.hist(sim_diag_np, bins=40, alpha=0.8)
plt.xlabel("cosine(sim_brain, sim_image)  (diagonal)")
plt.ylabel("count")
plt.title(f"CLIP(latent) diag cosine\nexp={exp_name}")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "sim_diag_hist.png"), dpi=200)
plt.close()

# 8.2 FWD/BWD rank 直方图（log 纵轴方便看长尾）
def save_rank_hist(ranks, title, fname):
    ranks_np = ranks.detach().cpu().numpy()
    plt.figure(figsize=(5,4))
    plt.hist(ranks_np, bins=50, log=True)
    plt.xlabel("rank (1 = best)")
    plt.ylabel("count (log)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, fname), dpi=200)
    plt.close()

save_rank_hist(ranks_fwd, f"FWD rank distribution\nexp={exp_name}", "rank_hist_fwd.png")
save_rank_hist(ranks_bwd, f"BWD rank distribution\nexp={exp_name}", "rank_hist_bwd.png")

# 8.3 Retrieval Recall@K 曲线（FWD/BWD 各一条）
Ks = [1, 5, 10, 20, 50]
fwd_vals = [recall_fwd[k]*100 for k in Ks]
bwd_vals = [recall_bwd[k]*100 for k in Ks]

plt.figure(figsize=(5,4))
plt.plot(Ks, fwd_vals, marker="o", label="FWD")
plt.plot(Ks, bwd_vals, marker="s", label="BWD")
plt.xlabel("K")
plt.ylabel("Recall@K (%)")
plt.title(f"Retrieval Recall@K\nexp={exp_name}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "retrieval_curve.png"), dpi=200)
plt.close()

# 8.4 相似度矩阵热力图（抽一个 64×64 子块）
N = sim.size(0)
if N >= 64:
    # 为了视觉效果，随机取 64 个样本（也可以取前 64 个）
    idx = torch.randperm(N)[:64]
    sim_sub = sim[idx][:, idx].detach().cpu().numpy()
    plt.figure(figsize=(5,4))
    plt.imshow(sim_sub, vmin=-1, vmax=1, cmap="viridis")
    plt.colorbar(label="cosine similarity")
    plt.title(f"Similarity heatmap (64x64 subset)\nexp={exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "sim_heatmap_64x64.png"), dpi=200)
    plt.close()

print(f"[DONE] 所有指标与图像已保存到: {result_dir}")
