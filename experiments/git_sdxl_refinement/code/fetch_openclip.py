#!/usr/bin/env python3
"""下载 OpenCLIP 权重（仅 open_clip 格式，避免重复下载 HF 分片/safetensors）。

open_clip 2.26.1 的 download_pretrained_from_hf 默认只请求
`open_clip_pytorch_model.bin`，没有 safetensors 回退。
因此 bigG 仓库 40.6GB 中只有 10.2GB 是必需的，其余（HF 分片 20GB +
open_clip_model.safetensors 10GB）全部跳过。
"""
import sys
from huggingface_hub import snapshot_download

PATTERNS = [
    "open_clip_pytorch_model.bin",
    "open_clip_config.json",
    "config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "README.md",
]

REPOS = [
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
]

for r in REPOS:
    print("downloading", r, flush=True)
    snapshot_download(r, token=True, allow_patterns=PATTERNS, max_workers=2)
    print("done", r, flush=True)
print("OpenCLIP ALL DONE", flush=True)

