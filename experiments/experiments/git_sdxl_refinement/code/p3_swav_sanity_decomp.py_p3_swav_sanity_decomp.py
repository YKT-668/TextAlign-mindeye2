"""SwAV sanity 差异来源分解：预检(48张, 0.3547) vs 全量官方协议值。

预检与全量的差异来源假设：
  H1 子集效应：前 48 张 ≠ 全 1000 张均值（sd≈0.11 → 48 张 SE≈0.016）；
  H2 预处理差异：预检用 F.interpolate 直达 224（antialias=False、缺 cell8 的 256 段），
     全量用官方 transforms.Resize 两段式（bilinear, antialias=True）；
  H3 参考值口径：MindEye2 论文 ≈0.32 是 THEIR 重建图上的值，不是本项目 A/B 的期望值。
输出：三组距离全量均值 + 前48子集均值 + H1/H2 分解。只读特征，不跑模型。
"""
import json

import numpy as np
from scipy.spatial.distance import correlation as corr_dist

P3 = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/evaluation/cpu_mindeye2_p3_clip_swav"
F = f"{P3}/swav"
GT = np.load(f"{F}/GT_features.npy").astype(np.float64)
A = np.load(f"{F}/A_raw_features.npy").astype(np.float64)

d_full = np.array([corr_dist(GT[i], A[i]) for i in range(1000)])
d_48 = d_full[:48]
out = {
    "preflight_48img_value": 0.354698,
    "preflight_protocol": "F.interpolate 直达 224 (antialias=False, 无 cell8 段)",
    "full_official_protocol": {"n": 1000, "mean": round(float(d_full.mean()), 6),
                               "sd": round(float(d_full.std(ddof=1)), 6),
                               "first48_mean": round(float(d_48.mean()), 6)},
    "H1_subset_effect": round(float(d_48.mean() - d_full.mean()), 6),
    "H2_note": "H2（预处理差异）无法从本分解中单独分离（预检特征未保留）；"
               "全量结果才是官方协议值，预检值仅作加载合理性检查",
    "H3_note": "MindEye2 论文 SwAV≈0.32 是其自身重建的均值；本项目 A/B 是不同生成图，"
               "无必须等于 0.32 的约束",
}
with open(f"{P3}/swav/sanity_gap_decomposition.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False))
print("SWAV_SANITY_DONE")
