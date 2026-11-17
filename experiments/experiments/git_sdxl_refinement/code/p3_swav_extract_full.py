"""P3 SwAV 全量特征提取（CPU，3000 张 = A/B/GT 各 1000）。

协议：MindEye2 cell 8（Resize256 bilinear antialias）+ cell 31（Resize224 bilinear + ImageNet
norm）+ avgpool(2048) 特征；官方距离 = scipy.spatial.distance.correlation（后续脚本计算）。
资源约束：CPU only、inference_mode、batch=1、线程=2、每 100 张原子检查点、断点续算。
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import (P3, ROOT, S1, S2, GTD, M, official_preprocess, load_pt,  # noqa: E402
                      atomic_save_array, env_snapshot, log_msg, verify_alignment)

torch.set_num_threads(2)
OUT = f"{P3}/swav"
os.makedirs(OUT, exist_ok=True)
CKPT = "/root/.cache/torch/hub/checkpoints/swav_800ep_pretrain.pth.tar"
SWAV_SHA256 = "029d0b8c2e70bcee3f8beb70cc104ef51585c090224d33fb9c6f51d146cfd1eb"
CKPT_EVERY = 100
LOG = f"{OUT}/extract_swav_full.log"

import hashlib  # noqa: E402
h = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()
assert h == SWAV_SHA256, f"SwAV 权重 SHA256 不符: {h}"
log_msg(LOG, f"SwAV checkpoint SHA256 verified: {h}")

import resource  # noqa: E402
from torchvision.models import resnet50  # noqa: E402
from torchvision.models.feature_extraction import create_feature_extractor  # noqa: E402

model = resnet50(weights=None)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
mapped = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
info = model.load_state_dict(
    {k: v for k, v in mapped.items() if not k.startswith(("projection_head.", "prototypes."))},
    strict=False)
assert all(k.startswith("fc.") for k in info.missing_keys), f"非 fc 缺失键: {info.missing_keys}"
assert not info.unexpected_keys, f"意外键: {info.unexpected_keys}"
feat_model = create_feature_extractor(model, return_nodes=["avgpool"])
feat_model.eval()
del sd, mapped
log_msg(LOG, "model built: torchvision resnet50 + SwAV backbone (strict accounting, fc excluded)")

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
    if meta and meta.get("n_done") == 1000 and meta.get("sha256") == h:
        log_msg(LOG, f"{gname}: complete, skip")
        continue
    feats = np.load(out_path) if os.path.exists(out_path) else np.zeros((1000, 2048), dtype=np.float32)
    start_i = meta.get("n_done", 0) if meta else 0
    x_all = load_pt(gpath).float()
    log_msg(LOG, f"=== {gname} start_idx={start_i} shape={list(x_all.shape)} ===")
    t0, n_done = time.time(), start_i
    with torch.inference_mode():
        for i in range(start_i, 1000):
            xb = official_preprocess(x_all[i:i + 1], norm="imagenet", size=224)
            f = feat_model(xb)["avgpool"].flatten(1).numpy().astype(np.float32)
            assert np.isfinite(f).all(), f"non-finite feature at {gname}:{i}"
            feats[i] = f[0]
            n_done = i + 1
            if n_done % CKPT_EVERY == 0 or n_done == 1000:
                atomic_save_array(feats[:n_done] if n_done < 1000 else feats, out_path)
                json.dump({"n_done": n_done, "dim": 2048, "dtype": "float32", "sha256": h,
                           "group": gname, "protocol": "cell8_resize256_bilinear+cell31_resize224_imagenet",
                           "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                          open(meta_path + ".tmp", "w"))
                os.replace(meta_path + ".tmp", meta_path)
                el = time.time() - t0
                log_msg(LOG, f"{gname}: {n_done}/1000 ({n_done * 1.0 / max(el, 1e-9):.2f} img/s, "
                             f"RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}MB)")
    # 校验完整性（全量时）
    if n_done == 1000:
        assert np.isfinite(feats).all()
        atomic_save_array(feats, out_path)
        log_msg(LOG, f"{gname}: DONE 1000/1000")
    del x_all
    import gc
    gc.collect()

env_snapshot("end", LOG)
log_msg(LOG, f"SWAV_FULL_EXTRACT_DONE total={time.time() - t_start:.0f}s")
