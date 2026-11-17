"""CLIP-L 图像塔核验（CPU，≤4 张，只验证离线加载与前向，不做全量）。

必须区分两个模型：
  A. openai/clip-vit-large-patch14（HF 转存的 OpenAI 权重）—— MindEye2 官方图像指标所用
     （notebook cell 27: clip.load("ViT-L/14") + encode_image）
  B. laion/CLIP-ViT-L-14-laion2B-s32B-b82K —— 非官方图像指标模型，不得冒充
官方预处理（cell 27，覆盖 clip 包默认）：Resize(224, BILINEAR) + CLIP mean/std，无 CenterCrop。
"""
import hashlib
import json
import os
import resource
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
HFH = f"{ROOT}/hf_home/hub"
S1 = f"{ROOT}/run/full_stage1"
GTD = f"{ROOT}/assets/ground_truth"
M = "s1_textalign_stage1_FINAL_BEST_32"
OUTD = f"{ROOT}/evaluation/gpu_missing_metrics"
os.makedirs(OUTD, exist_ok=True)
os.environ["HF_HOME"] = f"{ROOT}/hf_home"
os.environ["HF_HUB_CACHE"] = HFH

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

rep = {"official_evidence": {
    "notebook": "source/TextAlign-mindeye2-231850a/src/final_evaluations.ipynb",
    "cell": 27,
    "code": 'clip_model, preprocess = clip.load("ViT-L/14", device=device); '
            "preprocess = transforms.Compose([Resize(224, BILINEAR), Normalize(CLIP mean/std)]); "
            "two_way_identification(all_recons, all_images, clip_model.encode_image, preprocess, None)",
    "official_model": "OpenAI ViT-L/14（openai 原版权重；HF openai/clip-vit-large-patch14 为同一权重的 HF 转存）",
    "feature": "encode_image 最终层（含 vision projection，768 维）",
    "preprocess": "Resize(224, BILINEAR) + Normalize；无 CenterCrop（官方覆盖了 clip 包默认 bicubic+crop）",
    "laion_model_verdict": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K 训练数据(2B)+方案不同，"
                           "与本指标无关；仅可用于 MindEye2 内部的 bigG→L 投影转换等其它用途",
}}

def dir_info(repo, snap):
    d = f"{HFH}/{repo}/snapshots/{snap}"
    out = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                out[f] = os.path.getsize(p)
    return out

snap_openai = "32bd64288804d66eefd0ccbe215aa642df71cc41"
files_openai = dir_info("models--openai--clip-vit-large-patch14", snap_openai)
sha_st = None
st_path = f"{HFH}/models--openai--clip-vit-large-patch14/snapshots/{snap_openai}/model.safetensors"
if os.path.exists(st_path):
    h = hashlib.sha256()
    with open(st_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    sha_st = h.hexdigest()

laion_snaps = os.listdir(f"{HFH}/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/snapshots") \
    if os.path.isdir(f"{HFH}/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/snapshots") else []
files_laion = dir_info("models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K", laion_snaps[0]) if laion_snaps else {}

rep["openai_clip_vit_large_patch14"] = {
    "local": True, "revision": snap_openai, "files": files_openai,
    "model_safetensors_sha256": sha_st,
}
rep["laion_clip_vit_l14"] = {"local": bool(laion_snaps), "revision": laion_snaps[0] if laion_snaps else None,
                             "files": files_laion, "use_for_official_image_metric": False}

# ---- 离线加载 + 前向（≤4 张）----
from transformers import CLIPVisionModelWithProjection

model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",
                                                      local_files_only=True, torch_dtype=torch.float32)
model.eval().requires_grad_(False)
rep["offline_load"] = {"success": True, "env": "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 local_files_only=True",
                       "dtype": "fp32", "note": "文本塔 P2 已用同缓存权重完成，本核验针对视觉塔"}

def load_pt(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return torch.load(p, map_location="cpu", weights_only=False)

gt = load_pt(f"{GTD}/all_images.pt").float()
A = load_pt(f"{S1}/{M}_all_enhancedrecons.pt").float()
ids = torch.load(f"{S1}/{M}_all_ids.pt", map_location="cpu", weights_only=False).long()
sel = [0, 1]
imgs = torch.cat([gt[sel], A[sel]])

@torch.no_grad()
def embed(x):
    ts = []
    for xi in x:
        if xi.shape[-1] != 224:
            xi = F.interpolate(xi[None], size=(224, 224), mode="bilinear", align_corners=False)[0]
        t0 = time.time()
        e = model(pixel_values=((xi[None] - CLIP_MEAN) / CLIP_STD)).image_embeds
        ts.append(time.time() - t0)
        yield e[0], ts[-1]

embs, times = [], []
for e, t in embed(imgs):
    embs.append(e)
    times.append(t)
E = torch.stack(embs)

rep["forward_test"] = {
    "n_images": len(sel) * 2, "device": "cpu",
    "output_dim": int(E.shape[1]),
    "s_per_image": [round(t, 2) for t in times],
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    "emb_norms": [round(float(E[i].norm()), 4) for i in range(len(sel) * 2)],
    "preprocess": "Resize(224, BILINEAR) + CLIP mean/std（与官方 cell 27 一致）",
    "note": "OpenAI clip 包与 HF 视觉塔的数值等价性留待 GPU 轮用 clip.load 原包对 ~10 张图核对（HF_PROXY 验证项）",
}

with open(f"{OUTD}/clipl_image_tower_preflight.json", "w", encoding="utf-8") as fh:
    json.dump(rep, fh, indent=2, ensure_ascii=False)
print(json.dumps({k: rep[k] for k in ("offline_load", "forward_test")}, ensure_ascii=False))
print("CLIPL_PREFLIGHT_DONE")
