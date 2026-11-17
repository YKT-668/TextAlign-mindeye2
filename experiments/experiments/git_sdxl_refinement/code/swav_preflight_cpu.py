"""SwAV 模型加载验证（CPU，≤50 张，不做全量）。

官方来源：MindEye2 notebook cell 31 → torch.hub.load('facebookresearch/swav:main','resnet50')
本地权重：/root/.cache/torch/hub/checkpoints/swav_800ep_pretrain.pth.tar
         SHA256 029d0b8c2e70bcee3f8beb70cc104ef51585c090224d33fb9c6f51d146cfd1eb
策略：checkpoint 键名逐条核对 → 显式映射（module.* 前缀剥离）→ 任何 missing/unexpected/
shape-mismatch 必须记录，不静默忽略；fc（分类头）允许不匹配，但必须逐键列出并证明
不在 avgpool 特征路径上。
"""
import json
import os
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet50

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
CKPT = "/root/.cache/torch/hub/checkpoints/swav_800ep_pretrain.pth.tar"
S1 = f"{ROOT}/run/full_stage1"
GTD = f"{ROOT}/assets/ground_truth"
M = "s1_textalign_stage1_FINAL_BEST_32"
OUTD = f"{ROOT}/evaluation/gpu_missing_metrics"
os.makedirs(OUTD, exist_ok=True)
N_IMG = 48          # ≤50 张约束
BATCH = 8
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

rep = {"checkpoint": {"path": CKPT, "sha256": "029d0b8c2e70bcee3f8beb70cc104ef51585c090224d33fb9c6f51d146cfd1eb",
                      "official_url": "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"}}

# ---- 1) 检查 checkpoint 键名 ----
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
keys = list(sd.keys())
prefixes = sorted({k.split(".")[0] for k in keys})
rep["checkpoint"]["n_keys"] = len(keys)
rep["checkpoint"]["top_level_prefixes"] = prefixes
rep["checkpoint"]["embedded_queue_keys"] = [k for k in keys if "queue" in k.lower() or "prototypes" in k.lower()]

# ---- 2) 目标架构：torchvision ResNet50（与 SwAV hub 的 resnet50 同构，官方按 avgpool 取特征）----
model = resnet50(weights=None)
tv_keys = set(model.state_dict().keys())

mapped = {}
for k, v in sd.items():
    nk = k[len("module."):] if k.startswith("module.") else k
    mapped[nk] = v

missing = sorted(set(model.state_dict().keys()) - set(mapped.keys()))          # 模型要、ckpt 无
unexpected = sorted(set(mapped.keys()) - tv_keys)                               # ckpt 有、模型无
shape_mismatch = []
for k in sorted(set(mapped.keys()) & tv_keys):
    if tuple(mapped[k].shape) != tuple(model.state_dict()[k].shape):
        shape_mismatch.append({"key": k, "ckpt_shape": list(mapped[k].shape),
                               "model_shape": list(model.state_dict()[k].shape)})

rep["key_mapping"] = {
    "rule": "strip 'module.' prefix；其余键名与 torchvision resnet50 完全一致" 
            if not unexpected and not shape_mismatch else "strip 'module.' + 下方逐键差异清单",
    "n_mapped": len(mapped),
    "missing_in_ckpt": missing,
    "unexpected_in_ckpt": unexpected,
    "shape_mismatch": shape_mismatch,
    "note": "fc.* 若不匹配，仅影响分类头；avgpool 特征路径为 conv1..layer4+avgpool，与 fc 无关",
}

# fc 类键允许不匹配（记录后从加载集合剔除）；conv/bn/layer 必须严格匹配
skip = {e["key"] for e in shape_mismatch}
load_sd = {k: v for k, v in mapped.items() if k not in skip}
info = model.load_state_dict(load_sd, strict=False)
# 硬校验（显式记录，不静默忽略）：
#   unexpected 只允许 SwAV 投影头/原型（官方 ResNet 自带，torchvision 无；均不在 avgpool 特征路径上）
#   missing 只允许 fc.*（SwAV backbone 无分类头 fc；fc 在 avgpool 之后，不影响特征）
head_extras = [k for k in info.unexpected_keys
               if k.startswith("projection_head.") or k.startswith("prototypes.")]
other_unexpected = [k for k in info.unexpected_keys if k not in head_extras]
fc_missing = [k for k in info.missing_keys if k.startswith("fc.")]
other_missing = [k for k in info.missing_keys if k not in fc_missing]
assert not other_unexpected, f"unexpected keys: {other_unexpected}"
assert not other_missing, f"non-fc missing keys: {other_missing}"

# 逐键核对 backbone（conv1/bn1/layer1-4）参数量与形状完全一致
bb_model_keys = [k for k in model.state_dict() if k.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4."))]
bb_matched = [k for k in bb_model_keys if k in mapped
              and tuple(mapped[k].shape) == tuple(model.state_dict()[k].shape)]
backbone_parity = len(bb_matched) == len(bb_model_keys)

rep["load_state_dict"] = {
    "strict": False,
    "skipped_keys_due_to_shape": sorted(skip),
    "missing_after_load": list(info.missing_keys),
    "unexpected_after_load": list(info.unexpected_keys),
    "swav_head_extras_excluded": head_extras,
    "fc_missing_excluded": fc_missing,
    "feature_path_keys_verified": backbone_parity,
    "backbone_keys_matched": f"{len(bb_matched)}/{len(bb_model_keys)}",
}

model.eval().requires_grad_(False)
from torchvision.models.feature_extraction import create_feature_extractor
feat_model = create_feature_extractor(model, return_nodes=["avgpool"])   # 官方 cell 31 同款

# ---- 3) 少量样本 CPU 前向（官方预处理：Resize(224,BILINEAR)+ImageNet norm；cell 8 的 256 resize 之后）----
def load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return torch.load(p, map_location="cpu", weights_only=False)

gt = load_pt(f"{GTD}/all_images.pt").float()
A = load_pt(f"{S1}/{M}_all_enhancedrecons.pt").float()
ids = torch.load(f"{S1}/{M}_all_ids.pt", map_location="cpu", weights_only=False).long()

gt_s, a_s = gt[:N_IMG], A[:N_IMG]

@torch.no_grad()
def feats(x):
    outs = []
    for s in range(0, len(x), BATCH):
        b = x[s:s + BATCH]
        b = F.interpolate(b, size=(224, 224), mode="bilinear", align_corners=False) if b.shape[-1] != 224 else b
        b = (b - IMNET_MEAN) / IMNET_STD
        t0 = time.time()
        y = feat_model(b)["avgpool"]
        dt = time.time() - t0
        outs.append((y.flatten(1), dt))
    feat = torch.cat([o[0] for o in outs])
    return feat, [o[1] for o in outs]

f_gt, t_gt = feats(gt_s)
f_a, t_a = feats(a_s)

# 官方距离：scipy.spatial.distance.correlation（1 - Pearson r），越低越好
from scipy.spatial.distance import correlation
dists = [correlation(f_gt[i].numpy(), f_a[i].numpy()) for i in range(N_IMG)]

rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
rep["forward_test"] = {
    "n_images": N_IMG, "batch": BATCH, "device": "cpu",
    "feature_dim": int(f_gt.shape[1]), "feature_source": "avgpool (official cell 31)",
    "input_preproc": "Resize(224, BILINEAR) + ImageNet mean/std（官方 cell 31；cell 8 的 256 融合口径不变）",
    "time_gt_s": round(sum(t_gt), 2), "time_A_s": round(sum(t_a), 2),
    "s_per_image_mean": round((sum(t_gt) + sum(t_a)) / (2 * N_IMG), 3),
    "peak_rss_mb": round(rss_mb, 1),
    "swav_correlation_distance_mean_first48_A": round(float(np.mean(dists)), 6),
    "swav_correlation_distance_sd_first48_A": round(float(np.std(dists)), 6),
    "note": "官方 MindEye2 表中 SwAV 距离量级 ≈0.32；此子集值仅作加载合理性 sanity check，非正式结果",
}

with open(f"{OUTD}/swav_preflight_cpu.json", "w", encoding="utf-8") as fh:
    json.dump(rep, fh, indent=2, ensure_ascii=False)
print(json.dumps(rep["forward_test"], ensure_ascii=False))
print(json.dumps(rep["key_mapping"], ensure_ascii=False)[:600])
print("SWAV_PREFLIGHT_DONE")
