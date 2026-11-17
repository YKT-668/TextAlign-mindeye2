"""P1 高层/特征指标（严格按 MindEye2 final_evaluations.ipynb）。

识别类（官方 two_way_identification，实为 N-way Pearson 识别）：
  AlexNet-2 (features.4)、AlexNet-5 (features.11)、InceptionV3 (avgpool)、CLIP ViT-L/14（HF 代理）
距离类（scipy.spatial.distance.correlation，越低越好）：
  EfficientNet-B1 (avgpool)、SwAV-ResNet50 (avgpool)

内存策略：图像 mmap + 分块前向；特征写 fp16 memmap；识别矩阵按维度分块累加。
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
S1 = f"{ROOT}/run/full_stage1"
S2 = f"{ROOT}/run/full_stage2"
GTD = f"{ROOT}/assets/ground_truth"
M = "s1_textalign_stage1_FINAL_BEST_32"
N982 = 982
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["alexnet", "inception", "effnet", "swav", "clipl"])
ap.add_argument("--out", default=f"{ROOT}/evaluation/cpu_mindeye2_p1")
ap.add_argument("--include_blend", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
FEAT = f"{args.out}/features/{args.model}"
os.makedirs(FEAT, exist_ok=True)


def load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return torch.load(p, map_location="cpu", weights_only=False)


ids = torch.load(f"{S1}/{M}_all_ids.pt", map_location="cpu", weights_only=False).long()
n = int(ids.numel())
gt = load_pt(f"{GTD}/all_images.pt").float()
groups = {"A_raw": load_pt(f"{S1}/{M}_all_enhancedrecons.pt").float(),
          "B_refined": load_pt(f"{S2}/{M}_stage2_img2img.pt").float()}
if args.include_blend:
    blurry = load_pt(f"{S1}/{M}_all_blurryrecons.pt").float()
    b256 = F.interpolate(blurry, size=(256, 256), mode="bicubic", align_corners=False)
    a256 = F.interpolate(groups["A_raw"], size=(256, 256), mode="bicubic", align_corners=False)
    groups["A_blend075_DIAGNOSTIC"] = (a256 * 0.75 + b256 * 0.25).clamp(0, 1)
    groups["B_blend075_DIAGNOSTIC"] = (groups["B_refined"] * 0.75 + b256 * 0.25).clamp(0, 1)

SPEC = {
    "alexnet": dict(size=256, mean=IMNET_MEAN, std=IMNET_STD, batch=8, kind="ident",
                    nodes=["features.4", "features.11"], names=["AlexNet(2)", "AlexNet(5)"]),
    "inception": dict(size=342, mean=IMNET_MEAN, std=IMNET_STD, batch=6, kind="ident",
                      nodes=["avgpool"], names=["InceptionV3"]),
    "effnet": dict(size=255, mean=IMNET_MEAN, std=IMNET_STD, batch=10, kind="distance",
                   nodes=["avgpool"], names=["EfficientNet-B1"]),
    "swav": dict(size=224, mean=IMNET_MEAN, std=IMNET_STD, batch=8, kind="distance",
                 nodes=["avgpool"], names=["SwAV-ResNet50"]),
    "clipl": dict(size=224, mean=CLIP_MEAN, std=CLIP_STD, batch=8, kind="ident",
                  nodes=None, names=["CLIP-ViT-L14_HF_PROXY"]),
}
spec = SPEC[args.model]

if args.model == "alexnet":
    from torchvision.models import AlexNet_Weights, alexnet
    from torchvision.models.feature_extraction import create_feature_extractor
    model = create_feature_extractor(alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), return_nodes=spec["nodes"])
elif args.model == "inception":
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchvision.models.feature_extraction import create_feature_extractor
    model = create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes=spec["nodes"])
elif args.model == "effnet":
    from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1
    from torchvision.models.feature_extraction import create_feature_extractor
    model = create_feature_extractor(efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
                                     return_nodes=spec["nodes"])
elif args.model == "swav":
    from torchvision.models.feature_extraction import create_feature_extractor
    model = create_feature_extractor(torch.hub.load("facebookresearch/swav:main", "resnet50"),
                                     return_nodes=spec["nodes"])
else:
    from transformers import CLIPVisionModelWithProjection
    import os as _os
    _dt = torch.float32 if _os.environ.get("CLIP_FP16", "0") != "1" else torch.float16
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=_dt)
    print(f"[clipl] CLIPVisionModelWithProjection dtype={_dt}", flush=True)
model.eval().requires_grad_(False)
print(f"[{args.model}] model loaded", flush=True)


@torch.no_grad()
def forward_batch(x):
    c = (x.float() - spec["mean"]) / spec["std"]
    if args.model == "clipl":
        return [F.normalize(model(pixel_values=c).image_embeds, dim=-1)]
    out = model(c)
    return [out[k].flatten(1) for k in spec["nodes"]]


@torch.no_grad()
def extract_all(x, paths):
    """一次前向取出所有层，分别写入 fp16 memmap（AlexNet 双层只跑一次）。"""
    probe = forward_batch(x[:2])
    Ds = [int(np.prod(p.shape[1:])) for p in probe]
    mms = [np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(n, D))
           for p, D in zip(paths, Ds)]
    for s in range(0, n, spec["batch"]):
        idx = list(range(s, min(s + spec["batch"], n)))
        feats = forward_batch(x[idx])
        for mm, f in zip(mms, feats):
            mm[idx] = f.numpy().astype(np.float16)
        if s % (spec["batch"] * 15) == 0:
            print(f"   {os.path.basename(paths[0])} {min(s+spec['batch'],n)}/{n} D={Ds}", flush=True)
    for mm in mms:
        mm.flush()
    del mms
    for p, D in zip(paths, Ds):
        open(p + ".ok", "w").write(str(D))
        print(f"   saved {p} D={D}", flush=True)


def prep(x):
    return x if x.shape[-1] == spec["size"] else F.interpolate(
        x, size=(spec["size"], spec["size"]), mode="bilinear", align_corners=False)


all_groups = dict(groups)
all_groups["GT"] = gt
for gname, g in all_groups.items():
    paths = [f"{FEAT}/{gname}__{li}.npy" for li in range(len(spec["names"]))]
    if all(os.path.exists(p) and os.path.exists(p + ".ok") for p in paths):
        print(f"   reuse {gname}", flush=True)
        continue
    for p in paths:
        for q in (p, p + ".ok"):
            if os.path.exists(q):
                os.remove(q)
    extract_all(prep(g), paths)
print("FEATURES_DONE", flush=True)


# ---------------- 指标计算 ----------------
def row_center_normalize(src, dst):
    D = np.load(src, mmap_mode="r").shape[1]
    a = np.load(src, mmap_mode="r")
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float16, shape=(n, D))
    for s in range(0, n, 128):
        blk = np.asarray(a[s:s + 128], dtype=np.float32)
        blk -= blk.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(blk, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1e-12
        out[s:s + 128] = (blk / nrm).astype(np.float16)
    out.flush()
    del out
    return D


def pearson_matrix(xp, yp, D, chunk=8192):
    X = np.load(xp, mmap_mode="r")
    Y = np.load(yp, mmap_mode="r")
    R = np.zeros((n, n), dtype=np.float32)
    for d0 in range(0, D, chunk):
        d1 = min(d0 + chunk, D)
        R += np.asarray(X[:, d0:d1], dtype=np.float32) @ np.asarray(Y[:, d0:d1], dtype=np.float32).T
    return R


def ident_scores(R, scope):
    """官方 two_way_identification：
    success[i,j] = r[i,j] < r[j,j];  acc = mean_j(#{i: success})/(N-1)
    返回 (acc, 每列 1-based 排名，1=最好)
    """
    r = R[:scope, :scope].astype(np.float64)
    diag = np.diag(r)
    cnt = (r < diag[None, :]).sum(axis=0)          # = success.sum(0)
    acc = float(cnt.mean() / (scope - 1))
    ranks = 1 + (r >= diag[None, :]).sum(axis=0)   # 含自身
    return acc, ranks


def corr_distance(g_path, r_path):
    G = np.asarray(np.load(g_path, mmap_mode="r")[:n], dtype=np.float64)
    Rp = np.asarray(np.load(r_path, mmap_mode="r")[:n], dtype=np.float64)
    Gc = G - G.mean(axis=1, keepdims=True)
    Rc = Rp - Rp.mean(axis=1, keepdims=True)
    num = (Gc * Rc).sum(axis=1)
    den = np.linalg.norm(Gc, axis=1) * np.linalg.norm(Rc, axis=1)
    corr = num / np.maximum(den, 1e-12)
    return 1.0 - corr      # 官方 scipy correlation distance


summary = {}
per_sample_rows = []
gcount = len(all_groups) - 1     # 除 GT 外的组
for li, metric_name in enumerate(spec["names"]):
    gtp = f"{FEAT}/GT__{li}.npy"
    if spec["kind"] == "ident":
        D = row_center_normalize(gtp, f"{FEAT}/GT__{li}__norm.npy")
        for gname in all_groups:
            if gname == "GT":
                continue
            gp = f"{FEAT}/{gname}__{li}.npy"
            row_center_normalize(gp, f"{FEAT}/{gname}__{li}__norm.npy")
            R = pearson_matrix(f"{FEAT}/GT__{li}__norm.npy", f"{FEAT}/{gname}__{li}__norm.npy", D)
            np.save(f"{FEAT}/R_{gname}__{li}.npy", R.astype(np.float32))
            for scope in (1000, N982):
                acc, ranks = ident_scores(R, scope)
                summary[f"{metric_name}|{gname}|acc_N{scope}"] = round(acc, 6)
                if scope == 1000:
                    for i in range(n):
                        per_sample_rows.append({"metric": metric_name, "kind": "ident",
                                                "group": gname, "idx": i, "id": int(ids[i]),
                                                "value": float(ranks[i]), "extra": "rank_1_is_best"})
                    summary[f"{metric_name}|{gname}|hit1"] = round(float((ranks == 1).mean()), 6)
                    summary[f"{metric_name}|{gname}|median_rank"] = float(np.median(ranks))
            print(f"[{metric_name}] {gname} acc_1000={summary[f'{metric_name}|{gname}|acc_N1000']:.4f} "
                  f"acc_982={summary[f'{metric_name}|{gname}|acc_N982']:.4f}", flush=True)
    else:
        for gname in all_groups:
            if gname == "GT":
                continue
            d = corr_distance(gtp, f"{FEAT}/{gname}__{li}.npy")
            for scope, s in (("1000", n), ("982", N982)):
                summary[f"{metric_name}|{gname}|dist_{scope}"] = round(float(d[:s].mean()), 6)
            for i in range(n):
                per_sample_rows.append({"metric": metric_name, "kind": "distance", "group": gname,
                                        "idx": i, "id": int(ids[i]), "value": float(d[i]), "extra": "lower_is_better"})
            print(f"[{metric_name}] {gname} dist_1000={summary[f'{metric_name}|{gname}|dist_1000']:.4f} "
                  f"dist_982={summary[f'{metric_name}|{gname}|dist_982']:.4f}", flush=True)

with open(f"{args.out}/p1_{args.model}_per_sample.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["metric", "kind", "group", "idx", "id", "value", "extra"])
    w.writeheader()
    w.writerows(per_sample_rows)
with open(f"{args.out}/p1_{args.model}_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("P1_MODEL_DONE", args.model, flush=True)

