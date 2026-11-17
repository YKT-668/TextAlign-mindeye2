"""后台下载 P2 文本模型到大文件友好方式（urllib + Range 续传），小文件用 hf_hub_download。"""
import os
import shutil
import time
import urllib.request

from huggingface_hub import hf_hub_download

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
os.environ.setdefault("HF_HOME", f"{ROOT}/hf_home")
BASE = "/mnt/data/cpu_models"

JOBS = [
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2",
     ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
      "special_tokens_map.json", "1_Pooling/config.json", "modules.json"]),
    ("clipb", "openai/clip-vit-base-patch32",
     ["config.json", "vocab.json", "merges.txt", "tokenizer.json",
      "tokenizer_config.json", "special_tokens_map.json"]),
]

for tag, repo, smalls in JOBS:
    d = f"{BASE}/{tag}"
    os.makedirs(d, exist_ok=True)
    for fn in smalls:
        try:
            p = hf_hub_download(repo, fn, token=True)
            dst = os.path.join(d, fn)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.realpath(p), dst)
        except Exception as e:
            print(f"[warn] small file {fn}: {type(e).__name__}", flush=True)
    dst = f"{d}/pytorch_model.bin"
    url = f"https://huggingface.co/{repo}/resolve/main/pytorch_model.bin"
    for attempt in range(30):
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=30) as r, open(dst, "ab" if have else "wb") as f:
                while True:
                    b = r.read(262144)
                    if not b:
                        break
                    f.write(b)
            print(f"[{tag}] attempt {attempt} done size={os.path.getsize(dst)}", flush=True)
            break
        except Exception as e:
            print(f"[{tag}] attempt {attempt} {type(e).__name__} size={os.path.getsize(dst) if os.path.exists(dst) else 0}", flush=True)
            time.sleep(2)
    print(f"[{tag}] FINAL size={os.path.getsize(dst) / 1e6:.1f}MB", flush=True)
print("DL_ALL_DONE", flush=True)
