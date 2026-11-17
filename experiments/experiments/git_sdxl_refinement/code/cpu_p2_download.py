"""P2 权重下载：MiniLM（句向量）+ CLIP-B（图文）。仅下载，不加载。"""
import os

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf_home")

jobs = [
    ("sentence-transformers/all-MiniLM-L6-v2",
     ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json",
      "vocab.txt", "special_tokens_map.json", "1_Pooling/config.json", "modules.json"]),
    ("openai/clip-vit-base-patch32",
     ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt", "tokenizer.json",
      "tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json"]),
]
for repo, pats in jobs:
    print("DOWNLOAD", repo, flush=True)
    snapshot_download(repo, allow_patterns=pats, local_dir=None, token=True)
    print("DONE", repo, flush=True)
print("ALL_DOWNLOADS_DONE", flush=True)
