#!/usr/bin/env bash
# ConceptAlign — GIT Caption-guided SDXL Img2Img 补充实验下载环境
# 所有路径都落在 150GB 持久化数据盘 /mnt/data 上。
export ROOT=/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114
export VENV=$ROOT/download_venv
export PATH=$VENV/bin:$PATH

export HF_HOME=$ROOT/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_ASSETS_CACHE=$HF_HOME/assets
export TMPDIR=$ROOT/tmp
export PIP_CACHE_DIR=$ROOT/tmp/pip

export HF_HUB_DOWNLOAD_TIMEOUT=180
export HF_HUB_ETAG_TIMEOUT=60
export PYTHONUNBUFFERED=1

# 本机可用 RAM 仅约 3.8 GiB。经实测（hf-xet 1.8.0 >= 1.7.0，27MB 测试文件无 OOM），
# 启用高性能模式可将下载速度从 ~40 KiB/s 提升到 ~90 KiB/s（约 2.2x），故启用。
export HF_XET_HIGH_PERFORMANCE=1
# hf_transfer（0.1.9，已安装）：对 LFS 大文件启用多线程分块下载。
# 实测同一条线下 bigG 单文件 1.5MB/s -> ~10MB/s（约 6x）；对 xet 仓库无影响。
export HF_HUB_ENABLE_HF_TRANSFER=1
