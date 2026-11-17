"""P3 CLIP-L 图像特征全量提取（CPU，3000 张 = A/B/GT 各 1000）。

模型：openai/clip-vit-large-patch14（HF 转存的 OpenAI ViT-L/14），CLIPVisionModelWithProjection，
特征 = image_embeds（encode_image 最终层，768 维）——等价性以 p3_clipl_equivalence.py 门禁为准。
协议：cell 8（Resize256 bilinear antialias）+ cell 27（Resize224 bilinear + CLIP norm，无 CenterCrop）。
资源约束：CPU only、inference_mode、batch=1、线程=2、每 50 张原子检查点、断点续算。
"""
import gc
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import (P3, S1, S2, GTD, M, official_preprocess, load_pt,  # noqa: E402
                       atomic_save_array, env_snapshot, log_msg, verify_alignment)

os.environ["HF_HOME"] = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/hf_home"
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"] + "/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

torch.set_num_threads(2)
OUT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/evaluation/cpu_mindeye2_p3_clip_swav/clip_l"
os.makedirs(OUT, exist_ok=True)
CKPT_EVERY = 50
LOG = f"{OUT}/extract_clipl_full.log"

from transformers import CLIPVisionModelWithProjection  # noqa: E402

st_path = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/hf_home/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/model.safetensors"
h = hashlib.sha256()
with open(st_path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 22), b""):
        h.update(chunk)
log_msg(LOG, f"CLIP-L safetensors SHA256: {h.hexdigest()}")

model = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-large-patch14", local_files_only=True, torch_dtype=torch.float32)
model.eval()
log_msg(LOG, "model loaded (fp32, offline)")

align = verify_alignment()
assert align["manifest_equals_tensor_order"], f"manifest 与张量顺序不一致: {align}"
log_msg(LOG, f"alignment OK: {align}")

groups = {
    "A_raw": f"{S1}/{M}_all_enhancedrecons.pt",
    "B_refined": f"{S2}/{M}_stage2_img2img.pt",
    "GT": f"{GTD}/all_images.pt",
}

t_start = time.time()
env_snapshot("start", LOG)
for gname, gpath in groups.items():
    out_path = f"{OUT}/{gname}_features.npy"
    meta_path = f"{OUT}/{gname}_features.meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    if meta and meta.get("n_done") == 1000 and meta.get("safetensors_sha256") == h.hexdigest():
        log_msg(LOG, f"{gname}: complete, skip")
        continue
    feats = np.load(out_path) if os.path.exists(out_path) else np.zeros((1000, 768), dtype=np.float32)
    start_i = meta.get("n_done", 0) if meta else 0
    x_all = load_pt(gpath).float()
    log_msg(LOG, f"=== {gname} start_idx={start_i} shape={list(x_all.shape)} ===")
    t0, n_done = time.time(), start_i
    with torch.inference_mode():
        for i in range(start_i, 1000):
            xb = official_preprocess(x_all[i:i + 1], norm="clip", size=224)
            e = model(pixel_values=xb).image_embeds.numpy().astype(np.float32)
            assert np.isfinite(e).all(), f"non-finite feature at {gname}:{i}"
            feats[i] = e[0]
            n_done = i + 1
            if n_done % CKPT_EVERY == 0 or n_done == 1000:
                atomic_save_array(feats[:n_done] if n_done < 1000 else feats, out_path)
                json.dump({"n_done": n_done, "dim": 768, "dtype": "float32",
                           "safetensors_sha256": h.hexdigest(), "group": gname,
                           "protocol": "cell8_resize256_bilinear+cell27_resize224_clipnorm_no_center_crop",
                           "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                          open(meta_path + ".tmp", "w"))
                os.replace(meta_path + ".tmp", meta_path)
                el = time.time() - t0
                log_msg(LOG, f"{gname}: {n_done}/1000 ({n_done * 1.0 / max(el, 1e-9):.2f} img/s, "
                             f"RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}MB, "
                             f"MemAvail check in env log)")
    if n_done == 1000:
        assert np.isfinite(feats).all()
        atomic_save_array(feats, out_path)
        log_msg(LOG, f"{gname}: DONE 1000/1000")
    del x_all
    gc.collect()

env_snapshot("end", LOG)
log_msg(LOG, f"CLIPL_FULL_EXTRACT_DONE total={time.time() - t_start:.0f}s")
