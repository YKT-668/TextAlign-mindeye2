"""P2 模型清单：记录来源、版本、SHA256（含 HF 缓存 blob 名对照）。"""
import hashlib
import json
import os
import subprocess

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
OUT = f"{ROOT}/evaluation/cpu_mindeye2_p2_caption"
os.makedirs(OUT, exist_ok=True)


def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


cands = {
    "minilm_weights": "/mnt/data/cpu_models/minilm/pytorch_model.bin",
    "clipb_weights": "/mnt/data/cpu_models/clipb/pytorch_model.bin",
    "swav_weights": "/root/.cache/torch/hub/checkpoints/swav_800ep_pretrain.pth.tar",
}
clip_l_snap = f"{ROOT}/hf_home/hub/models--openai--clip-vit-large-patch14"
for c in os.listdir(clip_l_snap + "/snapshots") if os.path.isdir(clip_l_snap + "/snapshots") else []:
    d = f"{clip_l_snap}/snapshots/{c}"
    for fn in os.listdir(d):
        if fn.endswith((".bin", ".safetensors")):
            cands[f"clip_l_{fn}"] = os.path.realpath(f"{d}/{fn}")

man = {"created": "2026-11-15", "python": subprocess.run(
    ["/mnt/data/gpu_venv/bin/python", "-c", "import sys;print(sys.version.split()[0])"],
    capture_output=True, text=True).stdout.strip(), "models": {}}
for k, p in cands.items():
    if os.path.exists(p):
        man["models"][k] = {"path": p, "bytes": os.path.getsize(p), "sha256": sha(p)}
    else:
        man["models"][k] = {"path": p, "status": "MISSING"}

man["versions"] = subprocess.run(
    ["/mnt/data/gpu_venv/bin/python", "-c",
     "import torch,transformers,nltk,rouge_score,numpy,scipy;"
     "print(torch.__version__,transformers.__version__,nltk.__version__,numpy.__version__,scipy.__version__)"],
    capture_output=True, text=True,
    env={**os.environ, "PYTHONPATH": "/mnt/data/cpu_pkgs"}).stdout.strip().split()

man["sources"] = {
    "minilm": "hf:sentence-transformers/all-MiniLM-L6-v2 (pytorch_model.bin)",
    "clipb": "hf:openai/clip-vit-base-patch32 (pytorch_model.bin)",
    "clip_l": "hf:openai/clip-vit-large-patch14 (本地缓存)",
    "meteor": "nltk.meteor_score + wordnet corpus (nltk_data)",
    "rouge": "google rouge_score 0.1.x (rouge1/2/L/Lsum, use_stemmer=True)",
}
man["official_mapping"] = {
    "meteor": "OFFICIAL_EQUIVALENT — evaluate.load('meteor') 内部即 nltk.meteor_score",
    "rouge": "OFFICIAL_EQUIVALENT — evaluate.load('rouge') 内部即 google rouge_score",
    "sentence": "OFFICIAL_EQUIVALENT — all-MiniLM-L6-v2 mean pooling",
    "clip_b": "OFFICIAL_REPRODUCED — CLIP text tower (等价 get_text_features)",
    "clip_l": "OFFICIAL_REPRODUCED — CLIP-L text tower",
}
json.dump(man, open(f"{OUT}/model_manifest.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(man, indent=2, ensure_ascii=False))
