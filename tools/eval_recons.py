#!/usr/bin/env python
import argparse, os, json, math, glob

import torch
import torch.nn.functional as F


def cosine_sim(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()  # [-1,1]


def topk_acc(scores, gt_indices, k=1):
    if len(gt_indices) == 0:
        return float("nan")
    correct = 0
    total = 0
    topk = torch.topk(scores, k, dim=1).indices  # [N,k]
    for i, gi in enumerate(gt_indices):
        if gi < 0:
            continue
        total += 1
        if gi in topk[i].tolist():
            correct += 1
    return (correct / total) if total > 0 else float("nan")


def two_way_identification(scores, gt_indices, seed: int = 42):
    """
    CLIP two-way identification (chance = 0.5):

    对每个有效样本 i（有 gt_indices[i]）:
      - 取正例索引 gi = gt_indices[i]
      - 随机采样一个负例索引 j != gi
      - 若 scores[i, gi] > scores[i, j] 记为一次“正确”

    返回所有有效样本上的平均正确率 ∈ [0,1]。
    """
    if len(gt_indices) == 0:
        return float("nan")

    N, M = scores.shape
    g = torch.tensor(gt_indices, dtype=torch.long)
    valid_mask = g.ge(0) & g.lt(M)
    if not valid_mask.any():
        return float("nan")

    g = g[valid_mask]                  # [Nv]
    S_valid = scores[valid_mask]       # [Nv, M]
    Nv = S_valid.shape[0]

    gen = torch.Generator(device=S_valid.device)
    gen.manual_seed(seed)

    # 为每个样本采一个负例索引 j != gi
    neg_idx = torch.randint(low=0, high=M - 1, size=(Nv,), generator=gen, device=S_valid.device)
    # 把 >= gi 的索引整体 +1，避免等于 gi
    neg_idx = neg_idx + (neg_idx >= g.to(S_valid.device))

    pos_sim = S_valid[torch.arange(Nv, device=S_valid.device), g.to(S_valid.device)]
    neg_sim = S_valid[torch.arange(Nv, device=S_valid.device), neg_idx]

    correct = (pos_sim > neg_sim).float().mean().item()
    return correct


def safe_load(path):
    """兼容老的 numpy pickling 的 pt 文件."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        import numpy as np
        from torch.serialization import add_safe_globals
        add_safe_globals([np.core.multiarray._reconstruct])
        return torch.load(path, map_location="cpu", weights_only=False)


def _build_clip_encoder(device: str = "cuda"):
    """
    构建 CLIP 图像编码器，用于将 [N,3,H,W] 原图转成 [N,768] 特征。
    只在需要时调用。
    """
    from transformers import CLIPModel, CLIPImageProcessor

    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPImageProcessor.from_pretrained(model_name)
    return model, processor


def ensure_clip_features(gt_path: str, device: str = "cuda"):
    """
    保证返回的是 2D 的 CLIP 图像特征矩阵：
      - 若 gt_path 本身是 [N,D] tensor，直接返回；
      - 若是 [N,3,H,W] 图像 tensor，则自动用 CLIP 提特征并保存 *_clip.pt；
      - 其它情况抛错。
    """
    print(f"[eval] 加载 Ground Truth: {gt_path}")
    obj = safe_load(gt_path)

    # 已经是特征矩阵
    if isinstance(obj, torch.Tensor) and obj.ndim == 2:
        print(f"[eval] 发现已是特征矩阵，shape={tuple(obj.shape)}")
        return obj, gt_path

    # 原始图像 → CLIP 特征
    if isinstance(obj, torch.Tensor) and obj.ndim == 4 and obj.shape[1] == 3:
        print(f"[eval] 检测到原始图像，shape={tuple(obj.shape)}，开始编码为 CLIP 特征...")
        imgs = obj  # [N,3,H,W]
        device = device if torch.cuda.is_available() else "cpu"

        from torchvision.transforms.functional import to_pil_image
        from tqdm.auto import tqdm

        model, processor = _build_clip_encoder(device=device)

        all_feats = []
        bs = 32
        for i in tqdm(range(0, imgs.shape[0], bs), desc="encode GT images"):
            batch = imgs[i:i + bs]  # [B,3,H,W]
            pil_batch = [to_pil_image(img) for img in batch]
            inputs = processor(images=pil_batch, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)  # [B,768]
            all_feats.append(feats.cpu())

        all_feats = torch.cat(all_feats, dim=0)  # [N,768]

        out_path = gt_path.replace(".pt", "_clip.pt")
        torch.save(all_feats, out_path)
        print(f"[eval] 已保存 CLIP GT 特征到: {out_path}, shape={tuple(all_feats.shape)}")
        return all_feats, out_path

    raise TypeError(
        f"[eval] 不支持的 GT 文件类型: {type(obj)} (ndim={getattr(obj, 'ndim', None)})，"
        f"请确认 {gt_path} 是 2D 特征矩阵或 4D 图像张量。"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--gt_images", required=True)     # [M,D] 或 [M,3,H,W]
    ap.add_argument("--gt_captions", default="")      # 保留参数，暂时不用
    args = ap.parse_args()

    eval_results = os.path.join(args.model_dir, "eval_results")
    recons_pt = os.path.join(eval_results, "recons_features.pt")
    ids_json  = os.path.join(eval_results, "recons_ids.json")
    metrics_json = os.path.join(eval_results, "metrics.json")

    # 载入重建特征
    R = safe_load(recons_pt)      # [N,D]
    assert isinstance(R, torch.Tensor) and R.ndim == 2, f"重建特征格式异常: {type(R)}, ndim={getattr(R,'ndim',None)}"

    # 先确保 GT 是 CLIP 特征，如果传进来的是图像，会自动编码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, used_gt_path = ensure_clip_features(args.gt_images, device=device)

    # 若维度仍不一致，再按你原来的逻辑在同目录下“换一个 pt”
    if R.shape[1] != G.shape[1]:
        gt_dir = os.path.dirname(args.gt_images)
        candidates = glob.glob(os.path.join(gt_dir, "*.pt"))
        found = False
        for cand in candidates:
            if cand == args.gt_images or cand == used_gt_path:
                continue
            try:
                G2 = safe_load(cand)
                if isinstance(G2, torch.Tensor) and G2.ndim == 2 and G2.shape[1] == R.shape[1]:
                    print(f"[eval] 自动切换 ground truth: {cand} (shape={G2.shape})")
                    G = G2
                    found = True
                    break
            except Exception as e:
                print(f"[eval] 跳过 {cand}: {e}")
        if not found:
            raise AssertionError(
                f"dim mismatch: {R.shape} vs {G.shape}，且未找到可用的 ground truth 特征。"
                f"请检查特征提取和评测配置。"
            )

    assert R.ndim == 2 and G.ndim == 2 and R.shape[1] == G.shape[1], \
        f"dim mismatch: {R.shape} vs {G.shape}"

    # 相似度矩阵 [N,M]
    S = cosine_sim(R, G)
    print(f"[eval] sim min/max/mean: {float(S.min()):.6f}/{float(S.max()):.6f}/{float(S.mean()):.6f}")

    # ids
    if os.path.isfile(ids_json):
        raw = json.load(open(ids_json, "r", encoding="utf-8"))
        # 处理 None / 越界
        idx = []
        for i in raw:
            if isinstance(i, int) and 0 <= i < G.shape[0]:
                idx.append(i)
            else:
                idx.append(-1)
        if len(idx) != R.shape[0]:
            n = min(len(idx), R.shape[0])
            idx = idx[:n]
            S = S[:n]
    else:
        idx = [-1] * R.shape[0]

    # clip cosine：优先真实对齐；无对齐 → 最近邻均值
    if len(idx) > 0 and all(i >= 0 for i in idx):
        ar = torch.arange(len(idx))
        gather = S[ar, torch.tensor(idx)]
        mean_cos = float(gather.mean().item())       # ∈[-1,1]
    else:
        mean_cos = float(S.max(dim=1).values.mean().item())

    top1 = topk_acc(S, idx, k=1)
    top5 = topk_acc(S, idx, k=5)
    two_way = two_way_identification(S, idx, seed=42)

    out = {
        "model_dir": args.model_dir,
        "top1_new": None if math.isnan(top1) else top1,
        "top5_new": None if math.isnan(top5) else top5,
        "clip_cosine": mean_cos,
        "clip_two_way": None if math.isnan(two_way) else two_way,
        "mse": None,
        "lpips": None,
        "set": "new_test",
    }
    os.makedirs(eval_results, exist_ok=True)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[eval] wrote:", metrics_json, out)


if __name__ == "__main__":
    main()
