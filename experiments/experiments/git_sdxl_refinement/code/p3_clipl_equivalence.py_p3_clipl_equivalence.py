"""P3 CLIP-L 官方实现等价性核验（门禁）。

对比对象：
  A. OpenAI 官方 clip 包（git+CLIP，ViT-L-14.pt，URL 内嵌 SHA256 b8cca3fd...）+ encode_image
  B. HF openai/clip-vit-large-patch14 → CLIPVisionModelWithProjection.image_embeds

核验层次：
  1. 官方权重文件 SHA256 校验（URL 内嵌哈希）
  2. 权重逐张量映射对比（visual.* → HF vision_model.*，覆盖率 + max abs diff）
  3. 16 张内容各异样本（GT 均匀抽样）在完全相同官方预处理下的输出对比：
     逐样本 cosine、max abs error
门禁标准：权重映射完整；cosine ≥ 0.99999；max abs error ≤ 1e-4。
预处理参数依据：cell 8 Resize((256,256))（默认 bilinear antialias）→ cell 27
Resize(224, BILINEAR) + Normalize（官方覆盖 clip 包默认 bicubic+CenterCrop，无 CenterCrop）。
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import GTD, official_preprocess, load_pt, log_msg, env_snapshot  # noqa: E402

OUT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/evaluation/cpu_mindeye2_p3_clip_swav/clip_l"
os.makedirs(OUT, exist_ok=True)
LOG = f"{OUT}/equivalence_check.log"
PT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/tmp/ViT-L-14.pt"
PT_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"   # URL 内嵌官方哈希
os.environ["HF_HOME"] = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/hf_home"
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"] + "/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
torch.set_num_threads(2)

rep = {"pt_sha256_expected": PT_SHA256, "preprocess_evidence": {
    "cell8": "transforms.Resize((256,256)) — 默认 BILINEAR, antialias=True（tensor 输入）",
    "cell27": "transforms.Compose([Resize(224, BILINEAR), Normalize(CLIP mean/std)]) — 覆盖 clip 包默认 bicubic+CenterCrop，无 CenterCrop",
    "feature": "clip_model.encode_image → visual.proj 投影后 768 维（官方 cell 27 传 encode_image，feature_layer=None）",
}}

# ---- 1) 官方权重 SHA256 ----
h = hashlib.sha256()
with open(PT, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 22), b""):
        h.update(chunk)
rep["pt_sha256_actual"] = h.hexdigest()
rep["pt_sha256_ok"] = h.hexdigest() == PT_SHA256
log_msg(LOG, f"ViT-L-14.pt sha256 ok={rep['pt_sha256_ok']}")
assert rep["pt_sha256_ok"], "官方权重 SHA256 不符，禁止继续"

# ---- 2) 16 张内容各异样本 ----
sel = list(np.linspace(0, 999, 16).astype(int))
gt = load_pt(f"{GTD}/all_images.pt").float()
imgs = gt[sel]
log_msg(LOG, f"16 diverse GT images, indices={sel}")

t0 = time.time()
env_snapshot("before openai clip", LOG)

# ---- 3) OpenAI clip 包 ----
import clip  # noqa: E402

clip_model, _ = clip.load("ViT-L/14", device="cpu",
                          download_root=os.path.dirname(PT))   # 复用已下载并校验过的 ViT-L-14.pt
clip_model.eval()
log_msg(LOG, f"openai clip loaded dtype={next(clip_model.parameters()).dtype}")
with torch.inference_mode():
    xb = official_preprocess(imgs, norm="clip", size=224)
    emb_openai = clip_model.encode_image(xb).float()
rep["openai"] = {"n": 16, "dim": int(emb_openai.shape[1]),
                 "time_s": round(time.time() - t0, 1),
                 "sd_dtype_note": "clip.load 在 CPU 自动 model.float()（fp16 存储 → fp32 计算）"}

# ---- 权重逐张量映射对比 ----
from transformers import CLIPVisionModelWithProjection  # noqa: E402

hf = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",
                                                   local_files_only=True, torch_dtype=torch.float32)
hf.eval()
hf_sd = hf.state_dict()
oa_sd = clip_model.state_dict()

mapping = {
    "visual.class_embedding": "vision_model.embeddings.class_embedding",
    "visual.conv1.weight": "vision_model.embeddings.patch_embedding.weight",
    "visual.positional_embedding": "vision_model.embeddings.position_embedding.weight",
    "visual.ln_pre.weight": "vision_model.pre_layrnorm.weight",
    "visual.ln_pre.bias": "vision_model.pre_layrnorm.bias",
    "visual.ln_post.weight": "vision_model.post_layernorm.weight",
    "visual.ln_post.bias": "vision_model.post_layernorm.bias",
    "visual.proj": "visual_projection.weight",
}
NL = 24
for i in range(NL):
    b = f"visual.transformer.resblocks.{i}."
    h_ = f"vision_model.encoder.layers.{i}."
    mapping[b + "ln_1.weight"] = h_ + "layer_norm1.weight"
    mapping[b + "ln_1.bias"] = h_ + "layer_norm1.bias"
    mapping[b + "ln_2.weight"] = h_ + "layer_norm2.weight"
    mapping[b + "ln_2.bias"] = h_ + "layer_norm2.bias"
    mapping[b + "mlp.c_fc.weight"] = h_ + "mlp.fc1.weight"
    mapping[b + "mlp.c_fc.bias"] = h_ + "mlp.fc1.bias"
    mapping[b + "mlp.c_proj.weight"] = h_ + "mlp.fc2.weight"
    mapping[b + "mlp.c_proj.bias"] = h_ + "mlp.fc2.bias"
    mapping[b + "attn.out_proj.weight"] = h_ + "self_attn.out_proj.weight"
    mapping[b + "attn.out_proj.bias"] = h_ + "self_attn.out_proj.bias"

weight_rows = []
for ok, hk in mapping.items():
    if ok not in oa_sd or hk not in hf_sd:
        weight_rows.append({"openai_key": ok, "hf_key": hk, "status": "MISSING"})
        continue
    a, b = oa_sd[ok].float(), hf_sd[hk].float()
    if a.shape != b.shape and ok == "visual.proj" and a.shape == b.shape[::-1]:
        a = a.T                      # openai proj 为 (in,out)，HF Linear 权重为 (out,in)
    if a.shape != b.shape:
        weight_rows.append({"openai_key": ok, "hf_key": hk, "status": "SHAPE_MISMATCH",
                            "openai_shape": list(a.shape), "hf_shape": list(b.shape)})
        continue
    weight_rows.append({"openai_key": ok, "hf_key": hk, "status": "OK",
                        "max_abs_diff": (a - b).abs().max().item(), "shape": list(a.shape)})

# attn in_proj 拆 q/k/v（openai 单矩阵 vs HF 三矩阵）
for i in range(NL):
    b_ = f"visual.transformer.resblocks.{i}.attn.in_proj_weight"
    W = oa_sd[b_].float()
    bw = oa_sd[b_.replace("weight", "bias")].float()
    for name, sl in (("q", slice(0, 1024)), ("k", slice(1024, 2048)), ("v", slice(2048, 3072))):
        hw = f"vision_model.encoder.layers.{i}.self_attn.{name}_proj.weight"
        hb = f"vision_model.encoder.layers.{i}.self_attn.{name}_proj.bias"
        dw = (W[sl] - hf_sd[hw].float()).abs().max().item()
        db = (bw[sl] - hf_sd[hb].float()).abs().max().item()
        weight_rows.append({"openai_key": b_ + f"[{name}].weight", "hf_key": hw, "status": "OK",
                            "max_abs_diff": dw})
        weight_rows.append({"openai_key": b_ + f"[{name}].bias", "hf_key": hb, "status": "OK",
                            "max_abs_diff": db})
del clip_model, oa_sd
import gc  # noqa: E402
gc.collect()

n_ok = [r for r in weight_rows if r["status"] == "OK"]
n_bad = [r for r in weight_rows if r["status"] != "OK" or r["max_abs_diff"] > 1e-6]
covered_hf = {r["hf_key"] for r in n_ok}
hf_vision_keys = {k for k in hf_sd if k.startswith("vision_model.")} | {"visual_projection.weight"}
uncovered = sorted(hf_vision_keys - covered_hf)
rep["weight_mapping"] = {
    "n_compared": len(weight_rows), "n_ok": len(n_ok), "n_problem": len(n_bad),
    "problems": n_bad[:10],
    "max_abs_diff_overall": max(r["max_abs_diff"] for r in n_ok),
    "hf_vision_keys_total": len(hf_vision_keys),
    "hf_keys_uncovered_by_openai": uncovered,
    "note": "fp16→fp32 精确上 cast 后逐张量对比；HF 视觉塔全部键应被 openai visual.* 覆盖",
}
# ---- 4) HF 输出对比 ----
env_snapshot("before HF", LOG)
with torch.inference_mode():
    xb = official_preprocess(imgs, norm="clip", size=224)
    emb_hf = hf(pixel_values=xb).image_embeds.float()

cos = torch.nn.functional.cosine_similarity(emb_openai, emb_hf, dim=1)
abs_err = (emb_openai - emb_hf).abs().max().item()
rep["output_comparison"] = {
    "n": 16, "cosine_min": round(float(cos.min()), 8),
    "cosine_mean": round(float(cos.mean()), 8),
    "max_abs_error": float(abs_err),
    "per_sample_cosine": [round(float(c), 8) for c in cos],
}
rep["gate"] = bool(cos.min() >= 0.99999 and abs_err <= 1e-4
                   and len(n_bad) == 0 and not uncovered)
rep["verdict"] = "PASS_OFFICIAL_EQUIVALENCE" if rep["gate"] else "FAIL_OFFICIAL_EQUIVALENCE"
rep["gate_rule"] = "权重映射完整无问题键且 HF 键全覆盖 + cosine_min>=0.99999 + max_abs_error<=1e-4"

with open(f"{OUT}/equivalence_check.json", "w", encoding="utf-8") as fh:
    json.dump(rep, fh, indent=2, ensure_ascii=False)
log_msg(LOG, f"VERDICT={rep['verdict']} cos_min={rep['output_comparison']['cosine_min']} "
             f"max_abs={abs_err:.3e} weight_problems={len(n_bad)} uncovered={len(uncovered)}")
print("EQUIVALENCE_VERDICT:", rep["verdict"])
