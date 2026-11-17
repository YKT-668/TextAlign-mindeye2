"""只下载 P1 所需权重文件（不实例化模型，内存占用极小）。"""
import os

import torch
from torchvision.models import (AlexNet_Weights, EfficientNet_B1_Weights,
                               Inception_V3_Weights)

CKPT = "/root/.cache/torch/hub/checkpoints"
os.makedirs(CKPT, exist_ok=True)

jobs = [
    ("alexnet", AlexNet_Weights.IMAGENET1K_V1.url),
    ("inception", Inception_V3_Weights.DEFAULT.url),
    ("effnet_b1", EfficientNet_B1_Weights.DEFAULT.url),
    ("swav_rn50", "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"),
]
total = 0
for name, url in jobs:
    fn = os.path.join(CKPT, os.path.basename(url))
    if os.path.exists(fn) and os.path.getsize(fn) > 1e6:
        print(f"SKIP {name} exists {os.path.getsize(fn)/1e6:.0f}MB", flush=True)
        total += os.path.getsize(fn)
        continue
    print(f"DL {name} <- {url}", flush=True)
    torch.hub.download_url_to_file(url, fn, progress=False)
    sz = os.path.getsize(fn)
    total += sz
    print(f"OK {name} {sz/1e6:.0f}MB", flush=True)
print(f"TOTAL {total/1e6:.0f}MB", flush=True)
