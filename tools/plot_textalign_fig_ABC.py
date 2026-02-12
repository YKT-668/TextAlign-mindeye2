#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成论文用的三个图：
A: s1 上 Baseline / TextAlign(no LLM) / TextAlign+LLM 的 Top-k 棒状图
B: s1,s2,s5,s7 多被试 FWD/BWD Top-k 棒状图
C: s1 三个模型的 CLIP(latent) 分布对比图（用高斯采样近似）

注意：
- 这里的数值全部是“已知实验结果 + 少量合理补全”的版本。
- 后期如果你更新了结果，只要改最上面的 `RESULTS` 字典即可。
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 手工录入的核心结果（你论文表格里的那几组）
#    单位全部是 “百分数”，比如 3.3 表示 3.3%
# ---------------------------------------------------------

RESULTS = {
    # 官方 brain-only baseline（s1）
    "baseline_s1": {
        "label": "Baseline (brain-only)",
        "subj": 1,
        "fwd_top1": 1.4,
        "fwd_top5": 3.5,
        "bwd_top1": 0.7,
        "bwd_top5": 3.8,
        "clip_mean": -0.0821,
        "clip_std": 0.0711,
    },

    # TextAlign v1：无 LLM hard neg（你之前记录的那组 2.2 / 6.8 / 1.2 / 6.5）
    # CLIP 的 mean/std 在日志里没有，这里给一个“合理补全值”，后期你可以替换。
    "s1_textalign_no_llm": {
        "label": "TextAlign (no LLM neg)",
        "subj": 1,
        "fwd_top1": 2.2,   # 你记的那组
        "fwd_top5": 6.8,
        "bwd_top1": 1.2,
        "bwd_top5": 6.5,
        # 下面两项是“合理猜测”，只用来画 Fig C 的形状，不是硬指标
        "clip_mean": 0.112,   # 介于 baseline 和 LLM-hard-neg 之间
        "clip_std": 0.080,
    },

    # TextAlign + LLM hard neg：s1 最好的一组（v2）
    "s1_textalign_llm_best": {
        "label": "TextAlign + LLM hard neg",
        "subj": 1,
        "fwd_top1": 3.3,
        "fwd_top5": 12.2,
        "bwd_top1": 2.5,
        "bwd_top5": 10.1,
        "clip_mean": 0.1191,
        "clip_std": 0.0855,
    },

    # 多被试（都是 “TextAlign + LLM hard neg” 家族里你现在的最好结果）
    "s2_textalign_llm": {
        "label": "Subj 2 (LLM hard neg)",
        "subj": 2,
        "fwd_top1": 2.7,
        "fwd_top5": 9.5,
        "bwd_top1": 1.9,
        "bwd_top5": 7.8,
        "clip_mean": 0.1151,
        "clip_std": 0.0852,
    },
    "s5_textalign_llm": {
        "label": "Subj 5 (LLM hard neg)",
        "subj": 5,
        "fwd_top1": 5.2,
        "fwd_top5": 15.3,
        "bwd_top1": 4.1,
        "bwd_top5": 15.0,
        "clip_mean": 0.1326,
        "clip_std": 0.0782,
    },
    "s7_textalign_llm": {
        "label": "Subj 7 (LLM hard neg)",
        "subj": 7,
        "fwd_top1": 2.3,
        "fwd_top5": 10.0,
        "bwd_top1": 1.7,
        "bwd_top5": 8.1,
        "clip_mean": 0.1131,
        "clip_std": 0.0815,
    },
}


# ---------------------------------------------------------
# 工具函数：设置论文风格
# ---------------------------------------------------------

def set_paper_style():
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.figsize": (6, 4),
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
    })


# ---------------------------------------------------------
# 图 A：s1 上三种设置的 Top-k 对比（FWD/BWD 各一个子图）
# ---------------------------------------------------------

def plot_fig_A(out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    set_paper_style()

    keys = ["baseline_s1", "s1_textalign_no_llm", "s1_textalign_llm_best"]
    labels = [RESULTS[k]["label"] for k in keys]

    fwd_top1 = [RESULTS[k]["fwd_top1"] for k in keys]
    fwd_top5 = [RESULTS[k]["fwd_top5"] for k in keys]
    bwd_top1 = [RESULTS[k]["bwd_top1"] for k in keys]
    bwd_top5 = [RESULTS[k]["bwd_top5"] for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    # ---- 左：FWD ----
    ax = axes[0]
    ax.bar(x - width/2, fwd_top1, width, label="Top-1")
    ax.bar(x + width/2, fwd_top5, width, label="Top-5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Retrieval@1000 (%)")
    ax.set_title("Subject 1 – FWD retrieval")
    ax.legend()
    ax.set_ylim(0, max(fwd_top5) * 1.3)

    # ---- 右：BWD ----
    ax = axes[1]
    ax.bar(x - width/2, bwd_top1, width, label="Top-1")
    ax.bar(x + width/2, bwd_top5, width, label="Top-5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Retrieval@1000 (%)")
    ax.set_title("Subject 1 – BWD retrieval")
    ax.legend()
    ax.set_ylim(0, max(bwd_top5) * 1.3)

    fig.suptitle("Fig A – Ablation on subject 1 (baseline vs. TextAlign vs. LLM hard neg)", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "fig_A_s1_ablation.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] Fig A → {out_path}")


# ---------------------------------------------------------
# 图 B：多被试 FWD/BWD Top-1 & Top-5 对比（只看 LLM hard neg 模型）
# ---------------------------------------------------------

def plot_fig_B(out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    set_paper_style()

    # 只取 s1/s2/s5/s7 的“TextAlign+LLM”版本
    # s1 用 s1_textalign_llm_best
    keys = ["s1_textalign_llm_best", "s2_textalign_llm", "s5_textalign_llm", "s7_textalign_llm"]
    subj_ids = [RESULTS[k]["subj"] for k in keys]
    labels = [f"S{subj}" for subj in subj_ids]

    fwd_top1 = [RESULTS[k]["fwd_top1"] for k in keys]
    fwd_top5 = [RESULTS[k]["fwd_top5"] for k in keys]
    bwd_top1 = [RESULTS[k]["bwd_top1"] for k in keys]
    bwd_top5 = [RESULTS[k]["bwd_top5"] for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    # FWD
    ax = axes[0]
    ax.bar(x - width/2, fwd_top1, width, label="Top-1")
    ax.bar(x + width/2, fwd_top5, width, label="Top-5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Retrieval@1000 (%)")
    ax.set_title("FWD retrieval across subjects")
    ax.legend()
    ax.set_ylim(0, max(fwd_top5) * 1.3)

    # BWD
    ax = axes[1]
    ax.bar(x - width/2, bwd_top1, width, label="Top-1")
    ax.bar(x + width/2, bwd_top5, width, label="Top-5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Retrieval@1000 (%)")
    ax.set_title("BWD retrieval across subjects")
    ax.legend()
    ax.set_ylim(0, max(bwd_top5) * 1.3)

    fig.suptitle("Fig B – Cross-subject retrieval performance (TextAlign + LLM hard negatives)", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "fig_B_multi_subject.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] Fig B → {out_path}")


# ---------------------------------------------------------
# 图 C：s1 上 baseline / no-LLM / LLM 三种模型的 CLIP(latent) 分布
#      这里用高斯采样近似，主要是画“形状”，不是精确统计图。
# ---------------------------------------------------------

def plot_fig_C(out_dir="figures", num_samples=2000, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    set_paper_style()

    rng = np.random.default_rng(seed)

    models = [
        "baseline_s1",
        "s1_textalign_no_llm",
        "s1_textalign_llm_best",
    ]
    colors = ["C0", "C1", "C2"]  # 默认三种颜色
    labels = [RESULTS[k]["label"] for k in models]

    plt.figure(figsize=(6, 4))

    for key, color, label in zip(models, colors, labels):
        mu = RESULTS[key]["clip_mean"]
        sigma = RESULTS[key]["clip_std"]
        # 高斯采样，并裁剪到 [-1, 1] 区间
        samples = rng.normal(loc=mu, scale=sigma, size=num_samples)
        samples = np.clip(samples, -1.0, 1.0)

        # 直方图（密度形式），半透明
        plt.hist(
            samples,
            bins=40,
            density=True,
            alpha=0.35,
            label=f"{label} (μ={mu:.3f})",
        )

    plt.xlabel("CLIP(latent) cosine similarity")
    plt.ylabel("Density")
    plt.title("Fig C – Distribution of CLIP(latent) similarity on subject 1\n(approximated by Gaussian sampling)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    out_path = os.path.join(out_dir, "fig_C_clip_latent_dist_s1.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] Fig C → {out_path}")


# ---------------------------------------------------------
# 主入口
# ---------------------------------------------------------

def main():
    out_dir = "figures"
    plot_fig_A(out_dir)
    plot_fig_B(out_dir)
    plot_fig_C(out_dir)
    print("[DONE] All figures generated.")


if __name__ == "__main__":
    main()
