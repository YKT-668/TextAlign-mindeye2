"""导出 bigG 图像特征（A/B/GT 各 1000），供 CPU 阶段复用。

严格使用与 MindEye2 官方评测一致的编码器与预处理：
FrozenOpenCLIPImageEmbedder(arch="ViT-bigG-14", version="laion2b_s39b_b160k",
                            output_tokens=True, only_tokens=True)
preprocess = kornia resize(224, bicubic) -> (x+1)/2 -> normalize(CLIP mean/std)
"""
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
S1 = f"{ROOT}/run/full_stage1"
S2 = f"{ROOT}/run/full_stage2"
GTD = f"{ROOT}/assets/ground_truth"
OUT = f"{ROOT}/evaluation/precomputed_features/bigg"
MODEL = "s1_textalign_stage1_FINAL_BEST_32"
WEIGHTS = (f"{ROOT}/hf_home/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/"
           "snapshots/743c27bd53dfe508a0ade0f50698f99b39d03bec/open_clip_pytorch_model.bin")

os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
t_start = time.time()
info = {}

wsize = os.path.getsize(WEIGHTS)
info["weights"] = {"path": WEIGHTS, "bytes": wsize, "expected_bytes": 10158638769,
                   "size_ok": wsize == 10158638769}
print("weights", wsize, "ok=", wsize == 10158638769, flush=True)

raw = torch.load(f"{S1}/{MODEL}_all_enhancedrecons.pt", map_location="cpu", weights_only=False).float()
ref = torch.load(f"{S2}/{MODEL}_stage2_img2img.pt", map_location="cpu", weights_only=False).float()
gt = torch.load(f"{GTD}/all_images.pt", map_location="cpu", weights_only=False).float()
blurry = torch.load(f"{S1}/{MODEL}_all_blurryrecons.pt", map_location="cpu", weights_only=False).float()
ids1 = torch.load(f"{S1}/{MODEL}_all_ids.pt", map_location="cpu", weights_only=False).long()
ids2 = torch.load(f"{S2}/{MODEL}_stage2_ids.pt", map_location="cpu", weights_only=False).long()

assert raw.shape == (1000, 3, 224, 224), raw.shape
assert ref.shape == (1000, 3, 256, 256), ref.shape
assert gt.shape == (1000, 3, 224, 224), gt.shape
assert blurry.shape == (1000, 3, 224, 224), blurry.shape
assert torch.equal(ids1, ids2), "A/B ids mismatch"
assert ids1.numel() == 1000 and len(torch.unique(ids1)) == 1000
info["ids"] = {"n": 1000, "ab_equal": True, "ascending": bool(torch.all(ids1[1:] > ids1[:-1])),
               "head": [int(x) for x in ids1[:5]], "tail": [int(x) for x in ids1[-5:]]}
print("ids ok", info["ids"]["head"], flush=True)

SRC = f"{ROOT}/source/TextAlign-mindeye2-231850a/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.append(f"{SRC}/generative_models/")
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder  # noqa: E402

embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14", version="laion2b_s39b_b160k", output_tokens=True, only_tokens=True,
)
embedder.to(dev).eval()
info["encoder"] = {"class": "FrozenOpenCLIPImageEmbedder", "arch": "ViT-bigG-14",
                   "version": "laion2b_s39b_b160k", "output_tokens": True, "only_tokens": True,
                   "output_shape": "[N,256,1664]",
                   "preprocess": "kornia resize224 bicubic; (x+1)/2; CLIP mean/std",
                   "dtype_saved": "float16"}
print("embedder loaded", flush=True)


@torch.no_grad()
def embed(x, batch=20, tag=""):
    outs = []
    n = x.shape[0]
    for i in range(0, n, batch):
        c = x[i:i + batch].to(dev, non_blocking=True).float()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            t = embedder(c)
        outs.append(t.reshape(t.shape[0], -1).half().cpu())
    return torch.cat(outs, 0)


def to256(x):
    return F.interpolate(x, size=(256, 256), mode="bicubic", align_corners=False) if x.shape[-1] != 256 else x


blurry256 = to256(blurry)
raw_blend = (to256(raw) * 0.75 + blurry256 * 0.25).clamp(0, 1)
ref_blend = (ref * 0.75 + blurry256 * 0.25).clamp(0, 1)
info["blend"] = {"formula": "0.75*recon + 0.25*blurryrecons @256", "status": "DIAGNOSTIC_NOT_OFFICIAL",
                 "reason": "官方 sd_image_var_autoenc.pth 本地缺失，无法逐字节确认与 stabilityai/sd-vae-ft-mse 一致"}
print("blend prepared", flush=True)

groups = {
    "A_raw": (raw, "A 组：Stage1 unCLIP 重建（未融合）"),
    "B_refined": (ref, "B 组：GIT caption 引导 SDXL img2img 精炼（未融合）"),
    "GT": (gt, "GT：真实刺激图像"),
    "A_raw_blend075_DIAGNOSTIC": (raw_blend, "诊断：A 的 .75/.25 融合（非官方复现）"),
    "B_refined_blend075_DIAGNOSTIC": (ref_blend, "诊断：B 的 .75/.25 融合（非官方复现）"),
}
files = {}
for name, (imgs, desc) in groups.items():
    t0 = time.time()
    f = embed(imgs, tag=name)
    dt = time.time() - t0
    path = f"{OUT}/{name}_features.pt"
    torch.save({"features": f, "ids": ids1.clone(),
                "meta": {"group": name, "description": desc, "n": 1000, "shape": list(f.shape),
                         "dtype": "float16", "encoder": info["encoder"]}}, path)
    files[name] = {"path": path, "bytes": os.path.getsize(path), "seconds": round(dt, 1),
                   "shape": list(f.shape), "nan": int(torch.isnan(f).sum()), "inf": int(torch.isinf(f).sum())}
    print(f"[{name}] {tuple(f.shape)} {dt:.1f}s nan={files[name]['nan']} inf={files[name]['inf']}", flush=True)

# 抽查 ID 对应：重算 3 张，与已存特征比余弦
src_map = {"A_raw": raw, "B_refined": ref, "GT": gt}
checks = []
for i in (0, 432, 999):
    for name in ("A_raw", "B_refined", "GT"):
        rec = embed(src_map[name][[i]], batch=1, tag="spot").float()
        saved = torch.load(files[name]["path"], map_location="cpu", weights_only=False)["features"][i].float()
        cos = float(F.cosine_similarity(rec.reshape(1, -1), saved.reshape(1, -1)))
        checks.append({"group": name, "idx": i, "cosine": round(cos, 6), "ok": cos > 0.999})
info["spot_checks"] = checks
info["spot_all_ok"] = all(c["ok"] for c in checks)
print("spot", checks, flush=True)

info["files"] = {k: {kk: vv for kk, vv in v.items()} for k, v in files.items()}
info["peak_memory_allocated_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
info["peak_memory_reserved_GB"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
info["elapsed_s"] = round(time.time() - t_start, 1)
info["gpu"] = torch.cuda.get_device_name(0)

lines = []
for name, d in files.items():
    h = hashlib.sha256()
    with open(d["path"], "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    d["sha256"] = h.hexdigest()
    lines.append(f"{h.hexdigest()}  {os.path.basename(d['path'])}")

with open(f"{OUT}/feature_manifest.json", "w", encoding="utf-8") as fh:
    json.dump(info, fh, indent=2, ensure_ascii=False)
h = hashlib.sha256()
with open(f"{OUT}/feature_manifest.json", "rb") as fh:
    for chunk in iter(lambda: fh.read(8 << 20), b""):
        h.update(chunk)
lines.append(f"{h.hexdigest()}  feature_manifest.json")
with open(f"{OUT}/SHA256SUMS", "w") as fh:
    fh.write("\n".join(lines) + "\n")

print("DONE", json.dumps({"elapsed_s": info["elapsed_s"],
                          "peak_alloc_GB": info["peak_memory_allocated_GB"],
                          "peak_reserved_GB": info["peak_memory_reserved_GB"],
                          "spot_all_ok": info["spot_all_ok"]}, ensure_ascii=False), flush=True)

