#!/usr/bin/env python3
"""下载 SDXL conditioner 所需的 openai/clip-vit-large-patch14（FrozenCLIPEmbedder 依赖）。

只取 CLIPTokenizer + CLIPTextModel 需要的文件（~0.65GB），
跳过同样内容的 tf_model.h5 / flax_model.msgpack / rust_model.ot（共 ~5GB）。
"""
from huggingface_hub import snapshot_download

REPO = "openai/clip-vit-large-patch14"
PATTERNS = [
    "config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
]
print("downloading", REPO, flush=True)
snapshot_download(REPO, token=True, allow_patterns=PATTERNS)
print("done", REPO, flush=True)

