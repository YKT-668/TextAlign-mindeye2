#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse

import torch
import matplotlib.pyplot as plt


def load_gt_images(gt_path):
    print(f"[vis] 加载 GT 图像: {gt_path}")
    imgs = torch.load(gt_path, map_location="cpu")  # [N, 3, H, W]
    assert isinstance(imgs, torch.Tensor) and imgs.ndim == 4 and imgs.shape[1] == 3, \
        f"GT 格式不对, 期望 [N,3,H,W]，实际: {type(imgs)}, shape={getattr(imgs,'shape',None)}"
    print(f"[vis] GT 形状: {tuple(imgs.shape)}")
    return imgs


def load_ids(ids_path):
    print(f"[vis] 加载重建对应的 GT 索引: {ids_path}")
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    # 过滤 None / 非法
    cleaned = []
    for i in ids:
        if isinstance(i, int) and i >= 0:
            cleaned.append(i)
        else:
            cleaned.append(-1)
    print(f"[vis] 共 {len(cleaned)} 个重建样本")
    return cleaned


def load_recon_image(recons_dir, gt_idx):
    # 生成图像文件名按 "<gt_idx>.png"
    fname = os.path.join(recons_dir, f"{gt_idx}.png")
    if not os.path.isfile(fname):
        return None
    import PIL.Image as Image
    return Image.open(fname).convert("RGB")


def tensor_to_pil(t):
    # t: [3,H,W], [0,1] or [-1,1] 或原始
    import PIL.Image as Image
    t = t.detach().cpu()
    # 简单归一化到 [0,1]
    t_min = float(t.min())
    t_max = float(t.max())
    if t_max > t_min:
        t = (t - t_min) / (t_max - t_min)
    t = (t * 255.0).clamp(0, 255).byte()
    t = t.permute(1, 2, 0).numpy()  # [H,W,3]
    return Image.fromarray(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="evals/all_images.pt",
                        help="GT 图像 pt 文件（[N,3,H,W]）")
    parser.add_argument("--ids", default="runs/subj01_inference_run_final/eval_results/recons_ids.json",
                        help="重建对应的 GT 索引 json")
    parser.add_argument("--recons_dir", default="runs/subj01_inference_run_final/generated_images",
                        help="生成图像所在目录")
    parser.add_argument("--num", type=int, default=12,
                        help="可视化多少个样本（从前面开始）")
    parser.add_argument("--out", default="debug_vis/recons_pairs.png",
                        help="输出保存路径")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    gt_imgs = load_gt_images(args.gt)                         # [N,3,H,W]
    ids = load_ids(args.ids)                                  # len = N_recons
    N = min(args.num, len(ids))

    import math
    import PIL.Image as Image

    # 每个样本 2 张图（左 GT, 右 Recon）
    cols = 2
    rows = N

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
    if rows == 1:
        axes = [axes]  # 统一成 list

    for row in range(rows):
        gt_idx = ids[row]
        ax_gt, ax_rec = axes[row]

        if gt_idx < 0 or gt_idx >= gt_imgs.shape[0]:
            ax_gt.set_title(f"GT idx invalid ({gt_idx})")
            ax_gt.axis("off")
            ax_rec.axis("off")
            continue

        # GT 图
        gt_pil = tensor_to_pil(gt_imgs[gt_idx])
        ax_gt.imshow(gt_pil)
        ax_gt.set_title(f"GT (idx={gt_idx})")
        ax_gt.axis("off")

        # 重建图
        rec_img = load_recon_image(args.recons_dir, gt_idx)
        if rec_img is None:
            ax_rec.set_title(f"Recon NOT FOUND ({gt_idx}.png)")
            ax_rec.axis("off")
        else:
            ax_rec.imshow(rec_img)
            ax_rec.set_title(f"Recon ({gt_idx}.png)")
            ax_rec.axis("off")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[vis] 已保存对比图到: {args.out}")


if __name__ == "__main__":
    main()
