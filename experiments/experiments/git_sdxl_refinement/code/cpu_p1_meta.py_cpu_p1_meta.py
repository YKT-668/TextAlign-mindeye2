"""生成 P1 交付元数据：evaluation_config.json / model_manifest.json / evaluation_log.txt / SHA256SUMS"""
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
OUT = f"{ROOT}/evaluation/cpu_mindeye2_p1"
CKPT = "/root/.cache/torch/hub/checkpoints"


def sha256(p, chunk=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def rss_of(model):
    p = f"{ROOT}/logs/p1_{model}.time.log"
    if not os.path.exists(p):
        return None
    txt = open(p, errors="ignore").read()
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
    return round(int(m.group(1)) / 1e6, 2) if m else None


cfg = {
    "scope": {"primary": 1000, "secondary": "shared982 = first 982 of the 1000 (ascending ids)",
              "note": "982-way 用 982 张候选重建矩阵，不是从 1000-way 结果里截取"},
    "python": sys.version.split()[0],
    "torch": __import__("torch").__version__,
    "device": "cpu",
    "env": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": 3, "MKL_NUM_THREADS": 3,
            "PYTHONPATH_EXTRA": "/mnt/data/cpu_pkgs (scikit-image 0.26 target install)"},
    "skimage": __import__("skimage").__version__,
    "preprocess": {
        "PixCorr": "Resize(425, bilinear) on [0,1] tensor; Pearson on flattened RGB; mean over images",
        "SSIM": "Resize(425); skimage.color.rgb2gray; ssim(gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)",
        "AlexNet(2)/(5)": "Resize(256,bilinear)+ImageNet norm; features.4 / features.11",
        "InceptionV3": "Resize(342,bilinear)+ImageNet norm; avgpool",
        "EfficientNet-B1": "Resize(255,bilinear)+ImageNet norm; avgpool; scipy correlation distance",
        "identification": "official two_way_identification: r=Pearson(GT,recon); acc=mean_j(#{i: r[i,j]<r[j,j]})/(N-1)",
    },
    "status": {
        "PixCorr": "OFFICIAL_REPRODUCED", "SSIM": "OFFICIAL_REPRODUCED",
        "AlexNet(2)": "OFFICIAL_REPRODUCED", "AlexNet(5)": "OFFICIAL_REPRODUCED",
        "InceptionV3": "OFFICIAL_REPRODUCED", "EfficientNet-B1": "OFFICIAL_REPRODUCED",
        "CLIP-ViT-L14": "BLOCKED_SLOW_ON_CPU",
        "SwAV-ResNet50": "BLOCKED_MISSING_WEIGHT",
        "blend075": "DIAGNOSTIC_NOT_OFFICIAL",
    },
    "stats": {"bootstrap": 10000, "seed": 2027, "ci": "95% percentile", "test": "wilcoxon signed-rank",
              "multiplicity": "Holm across all P1 delta p-values (1000 scope)"},
}
json.dump(cfg, open(f"{OUT}/evaluation_config.json", "w"), indent=2, ensure_ascii=False)

models = [
    ("AlexNet IMAGENET1K_V1", "torchvision official weights URL", f"{CKPT}/alexnet-owt-7be5be79.pth"),
    ("InceptionV3 DEFAULT", "torchvision official weights URL", f"{CKPT}/inception_v3_google-0cc3c7bd.pth"),
    ("EfficientNet-B1 DEFAULT", "torchvision official weights URL", f"{CKPT}/efficientnet_b1-c27df63c.pth"),
]
man = {"models": [], "blocked": [
    {"model": "SwAV-ResNet50", "reason": "dl.fbaipublicfiles.com 不可达（403/超时）；仅下到 23.6MB/~800MB",
     "status": "BLOCKED_MISSING_WEIGHT"},
    {"model": "CLIP ViT-L/14 (HF proxy)", "reason": "实测 14 s/image（4 核）→ 3000 张约 11.7 小时，本轮不可行",
     "status": "BLOCKED_SLOW_ON_CPU"}],
    "reused_local": [
        {"model": "openai/clip-vit-large-patch14 (HF, OpenAI 权重)", "used_for": "上一轮 A/B 图文指标（GPU）"},
        {"model": "laion/CLIP-ViT-bigG-14", "used_for": "bigG 图像特征（上一轮已导出）"}]}
for name, src, p in models:
    if os.path.exists(p):
        man["models"].append({"model": name, "source": src, "bytes": os.path.getsize(p),
                              "sha256": sha256(p), "offline_ok": True})
json.dump(man, open(f"{OUT}/model_manifest.json", "w"), indent=2, ensure_ascii=False)

lines = ["CPU P1 evaluation log summary", ""]
for m in ["lowlevel", "effnet", "alexnet", "inception", "clipl"]:
    lg = f"{ROOT}/logs/p1_{m}.log"
    if os.path.exists(lg):
        tail = [l for l in open(lg, errors="ignore").read().splitlines() if l.strip()][-6:]
        lines += [f"--- {m} (last lines) ---"] + tail + [f"RSS_GB={rss_of(m)}", ""]
lines += ["BLOCKED: SwAV (weight unreachable), CLIP-L (14 s/img on 4-core CPU)"]
open(f"{OUT}/evaluation_log.txt", "w").write("\n".join(lines) + "\n")

checks = []
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    if os.path.isfile(p) and f != "SHA256SUMS":
        checks.append(f"{sha256(p)}  {f}")
open(f"{OUT}/SHA256SUMS", "w").write("\n".join(checks) + "\n")
print(json.dumps({"models": man["models"], "rss": {m: rss_of(m) for m in ["effnet", "alexnet", "inception"]}}, indent=2, ensure_ascii=False))
print("META_DONE")
