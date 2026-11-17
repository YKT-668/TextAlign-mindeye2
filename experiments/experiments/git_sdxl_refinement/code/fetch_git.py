#!/usr/bin/env python3
"""下载 GIT 模型到 HF 缓存（排除重复的 pytorch_model.bin，只留 safetensors）。"""
import os
from huggingface_hub import snapshot_download

snapshot_download(
    "microsoft/git-large-coco",
    revision="b644b0b41274649f87de8846fa15f9a22dad7583",
    allow_patterns=["*.json", "*.txt", "*.safetensors"],
    ignore_patterns=["pytorch_model.bin"],
    token=True,
)
print("GIT downloaded OK")
