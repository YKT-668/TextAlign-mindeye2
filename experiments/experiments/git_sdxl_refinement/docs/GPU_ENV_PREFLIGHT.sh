#!/usr/bin/env bash
# A100 GPU 环境预检（offline）。在 A100 上、环境装好后运行。
set -uo pipefail
ROOT=/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114
export HF_HOME=$ROOT/hf_home
export HF_HUB_OFFLINE=1

echo "=== [1] Python / Torch / CUDA ==="
python - <<'PY'
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "vram_GB", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
PY

echo "=== [2] import 检查 ==="
python - <<'PY'
mods = ["transformers","diffusers","open_clip","einops","omegaconf","safetensors","h5py","numpy","webdataset","pytorch_lightning"]
for m in mods:
    try:
        __import__(m); print("OK", m)
    except Exception as e:
        print("MISSING", m, type(e).__name__)
PY

echo "=== [3] 本地资源离线加载（HF_HUB_OFFLINE=1）==="
python - <<'PY'
from huggingface_hub import hf_hub_download

checks = [
    ("microsoft/git-large-coco", "model.safetensors", 1576861900),
    ("microsoft/git-large-coco", "config.json", None),
    ("microsoft/git-large-coco", "preprocessor_config.json", None),
    ("openai/clip-vit-large-patch14", "pytorch_model.bin", None),
    ("openai/clip-vit-large-patch14", "vocab.json", None),
    ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "open_clip_pytorch_model.bin", 10158638769),
    ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "open_clip_config.json", None),
    ("laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "open_clip_pytorch_model.bin", 1710631365),
    ("laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "open_clip_config.json", None),
]
import os
for repo, fn, exp in checks:
    try:
        p = hf_hub_download(repo, fn, local_files_only=True)
        sz = os.path.getsize(p)
        if exp and sz != exp:
            print("SIZE_MISMATCH", repo, fn, sz, "!=", exp)
        else:
            print("OK offline", repo, fn, sz)
    except Exception as e:
        print("WARN offline-miss", repo, fn, type(e).__name__, str(e)[:80])
PY


echo "=== [4] 资产存在性 ==="
for f in assets/models/ours_s1_checkpoint/s1_textalign_stage1_FINAL_BEST_32_last.pth \
         assets/models/mindeye2/unclip6_epoch0_step110000.ckpt \
         assets/models/mindeye2/bigG_to_L_epoch8.pth \
         assets/models/mindeye2/zavychromaxl_v30.safetensors \
         assets/data/betas_all_subj01_fp32_renorm.hdf5 \
         assets/data/wds_subj01_new_test_0.tar \
         assets/ground_truth/all_images.pt assets/ground_truth/all_captions.pt; do
  [ -f "$ROOT/$f" ] && echo "OK $f ($(stat -c%s "$ROOT/$f"))" || echo "MISSING $f"
done
echo "=== 预检结束 ==="
