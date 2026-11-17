#!/usr/bin/env bash
# 在 A100 服务器上安装 GPU 推理环境（CUDA 12.4）。
# 用法：bash install_gpu_env.sh  （需要在 A100 上、已挂载 /mnt/data）
set -euo pipefail
ROOT=/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114
VENV=${VENV:-/mnt/data/gpu_venv}

echo "[1] 创建虚拟环境 $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip

echo "[2] 安装 PyTorch 2.5.0 + CUDA 12.4"
if [ -n "${PIP_MIRROR:-}" ]; then
  echo "[info] 使用镜像: $PIP_MIRROR"
  pip install --retries 10 --timeout 120 torch==2.5.0 torchvision==0.20.0 --index-url "$PIP_MIRROR"
else
  pip install --retries 10 --timeout 120 torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
fi

echo "[3] 安装其余依赖"
if [ -n "${PIP_MIRROR:-}" ]; then
  PIP_ARGS="--retries 10 --timeout 120 --index-url $PIP_MIRROR"
else
  PIP_ARGS="--retries 5 --timeout 60"
fi
pip install $PIP_ARGS \
  transformers==4.46.3 diffusers==0.31.0 open_clip_torch==2.26.1 \
  einops==0.8.0 omegaconf==2.3.0 pytorch_lightning==2.4.0 accelerate==1.1.1 \
  safetensors==0.4.5 h5py==3.12.1 "numpy==1.26.4" webdataset==0.2.100 \
  deepspeed==0.15.4 "dalle2-pytorch==1.16.3" ftfy==6.3.1 regex==2024.11.6 timm==1.0.11 \
  hf_transfer==0.1.9

echo "[4] 可选 xformers（失败可忽略，代码已 XFORMERS_DISABLED=1）"
pip install xformers==0.0.28.post3 || echo "[warn] xformers 安装失败，可忽略"

echo "[5] 校验"
python -c "import torch, torchvision, transformers, diffusers, open_clip, einops, omegaconf, safetensors, h5py; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
echo "[done] 环境安装完成：$VENV"
