"""P1 低层指标：PixCorr / SSIM（严格按 MindEye2 final_evaluations.ipynb cell 19/21）。

官方：
  preprocess = Resize(425, BILINEAR)  # 对 [0,1] 张量
  PixCorr: np.corrcoef(gt_flat, recon_flat)[0,1] 逐图 -> 平均
  SSIM   : rgb2gray -> ssim(rec, img, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, data_range=1.0) 逐图 -> 平均

内存策略：mmap 读入、按 chunk 处理（本机仅 ~1GB 可用 RAM）。
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_fn

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
S1 = f"{ROOT}/run/full_stage1"
S2 = f"{ROOT}/run/full_stage2"
GTD = f"{ROOT}/assets/ground_truth"
M = "s1_textalign_stage1_FINAL_BEST_32"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{ROOT}/evaluation/cpu_mindeye2_p1")
ap.add_argument("--include_blend", action="store_true")
ap.add_argument("--chunk", type=int, default=16)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

N982 = 982


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


gt = load_pt(f"{GTD}/all_images.pt")            # 保持 fp16，按块转 float
blurry = load_pt(f"{S1}/{M}_all_blurryrecons.pt")
ids = torch.load(f"{S1}/{M}_all_ids.pt", map_location="cpu", weights_only=False).long()
groups = {"A_raw": load_pt(f"{S1}/{M}_all_enhancedrecons.pt"),
          "B_refined": load_pt(f"{S2}/{M}_stage2_img2img.pt")}

n = int(ids.numel())
print(f"n={n} groups={list(groups)} include_blend={args.include_blend}", flush=True)


def get_blend(name, idx):
    """按块生成 .75/.25 融合图（诊断组），避免整组 fp32 常驻。"""
    b = F.interpolate(blurry[idx].float(), size=(256, 256), mode="bicubic", align_corners=False)
    if name.startswith("A_"):
        a = F.interpolate(groups["A_raw"][idx].float(), size=(256, 256), mode="bicubic", align_corners=False)
    else:
        a = groups["B_refined"][idx].float()
    return (a * 0.75 + b * 0.25).clamp(0, 1)


def get_group(name, idx):
    if "DIAGNOSTIC" in name:
        return get_blend(name, idx)
    return groups[name][idx].float()


def to425(x):
    return F.interpolate(x.float(), size=(425, 425), mode="bilinear", align_corners=False)


group_names = list(groups)
if args.include_blend:
    group_names += ["A_blend075_DIAGNOSTIC", "B_blend075_DIAGNOSTIC"]

rows = []
for gname in group_names:
    for start in range(0, n, args.chunk):
        idx = list(range(start, min(start + args.chunk, n)))
        a = to425(get_group(gname, idx))
        b = to425(gt[idx].float())
        ag = rgb2gray(a.permute(0, 2, 3, 1).numpy())
        bg = rgb2gray(b.permute(0, 2, 3, 1).numpy())
        af = a.reshape(len(idx), -1).numpy()
        bf = b.reshape(len(idx), -1).numpy()
        for k, i in enumerate(idx):
            rows.append({"group": gname, "idx": i, "id": int(ids[i]),
                         "pixcorr": float(np.corrcoef(bf[k], af[k])[0, 1]),
                         "ssim": float(ssim_fn(ag[k], bg[k], data_range=1.0,
                                               gaussian_weights=True, sigma=1.5,
                                               use_sample_covariance=False))})
        if start % (args.chunk * 10) == 0:
            print(f"  {gname} {start + len(idx)}/{n}", flush=True)

with open(f"{args.out}/lowlevel_per_sample.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["group", "idx", "id", "pixcorr", "ssim"])
    w.writeheader()
    w.writerows(rows)

summary = {}
for gname in group_names:
    for scope, s in (("1000", 0), ("982", N982)):
        sel = [r for r in rows if r["group"] == gname and r["idx"] < s]
        for metric in ("pixcorr", "ssim"):
            v = np.array([r[metric] for r in sel], dtype=np.float64)
            summary[f"{gname}|{metric}|{scope}"] = {
                "n": int(len(v)), "mean": round(float(v.mean()), 6),
                "std": round(float(v.std(ddof=1)), 6), "median": round(float(np.median(v)), 6)}
with open(f"{args.out}/lowlevel_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# A/B 配对差（同一 idx）
for scope, s in (("1000", n), ("982", N982)):
    d_pix = []
    d_ssi = []
    for gname in groups:
        if gname.startswith("A_") and "DIAGNOSTIC" in gname:
            continue
    a_map = {r["idx"]: r for r in rows if r["group"] == "A_raw" and r["idx"] < s}
    b_map = {r["idx"]: r for r in rows if r["group"] == "B_refined" and r["idx"] < s}
    for i in a_map:
        d_pix.append(b_map[i]["pixcorr"] - a_map[i]["pixcorr"])
        d_ssi.append(b_map[i]["ssim"] - a_map[i]["ssim"])
    for name, d in (("pixcorr", d_pix), ("ssim", d_ssi)):
        arr = np.array(d)
        summary[f"DELTA_B_minus_A|{name}|{scope}"] = {
            "n": int(len(arr)), "mean": round(float(arr.mean()), 6),
            "median": round(float(np.median(arr)), 6),
            "improved": int((arr > 0).sum()), "worsened": int((arr < 0).sum()),
            "tied": int((arr == 0).sum())}
with open(f"{args.out}/lowlevel_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps({k: v for k, v in summary.items() if k.startswith("DELTA")}, indent=2, ensure_ascii=False), flush=True)
print("LOWLEVEL_DONE", flush=True)
