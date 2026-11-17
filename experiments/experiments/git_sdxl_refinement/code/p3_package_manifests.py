"""P3 打包：model_manifest.json、evaluation_config.json、SHA256SUMS（追加模式）。"""
import hashlib
import json
import os
import time

P3 = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/evaluation/cpu_mindeye2_p3_clip_swav"
HF_ST = ("/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/hf_home/hub/models--openai--clip-vit-large-patch14"
         "/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/model.safetensors")
SWAV = "/root/.cache/torch/hub/checkpoints/swav_800ep_pretrain.pth.tar"
VIPT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/tmp/ViT-L-14.pt"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


model_manifest = {
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "models": {
        "clip_l_image": {
            "hf_repo": "openai/clip-vit-large-patch14",
            "hf_revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
            "file": HF_ST,
            "sha256": sha256(HF_ST),
            "official_equivalence": json.load(open(f"{P3}/clip_l/equivalence_check.json"))["verdict"],
            "official_pt_reference": {"file": VIPT, "sha256": sha256(VIPT),
                                      "note": "OpenAI clip 包 ViT-L-14.pt（等价性核验基准，非评价用权重）"},
            "feature": "image_embeds (visual_projection 后 768 维)",
            "laion_model_used": False,
        },
        "swav": {
            "file": SWAV,
            "sha256": sha256(SWAV),
            "official_url": "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
            "architecture": "torchvision resnet50 backbone (318/318 backbone keys matched); "
                            "projection_head/prototypes excluded (recorded); fc absent (recorded)",
            "feature": "avgpool 2048 维（官方 cell 31；未使用 projection_head/prototypes）",
        },
    },
    "libraries": {"torch": "2.5.0+cu124 (CPU)", "torchvision": "0.20.0",
                  "transformers": "4.46.3", "numpy": "2.5.4", "scipy": "1.17.1",
                  "clip_pkg": "1.0 (openai/CLIP@d05afc4, 仅用于等价性核验)"},
}
with open(f"{P3}/model_manifest.json", "w", encoding="utf-8") as fh:
    json.dump(model_manifest, fh, indent=2, ensure_ascii=False)

config = {
    "protocol": "MindEye2 final_evaluations.ipynb（TextAlign-mindeye2-231850a）",
    "preprocess": {
        "stage1_cell8": "transforms.Resize((256,256)) — 默认 BILINEAR, antialias=True",
        "clip_l_stage2_cell27": "Resize(224, BILINEAR) + Normalize(CLIP mean/std)，无 CenterCrop",
        "swav_stage2_cell31": "Resize(224, BILINEAR) + Normalize(ImageNet mean/std)",
        "erratum": "cell 8 为 bilinear（此前文档误记 bicubic），已在 P3 修正",
    },
    "features": {"dtype": "float32", "clip_l_dim": 768, "swav_dim": 2048,
                 "batch": 1, "threads": 2, "inference_mode": True},
    "alignment": json.load(open(f"{P3}/statistics/p3_metrics_summary.json"))["alignment"],
    "scopes": [1000, 982],
    "scope_note": "982 = 同一固定候选池前 982 列；逐对 Pearson 不受行删除影响，等价于独立 982 矩阵",
    "metrics_official": ["CLIP-L Pairwise 2-way Identification", "SwAV correlation distance"],
    "metrics_supplementary": ["Strict N-way Top-1", "Mean Rank", "CLIP-L GT↔重建 Pearson 相似度",
                              "SwAV 2-way/Top-1（新增，非官方）"],
    "runtime": {"swav_total_s": 344, "clip_l": "见 clip_l/extract_clipl_full.log"},
}
with open(f"{P3}/evaluation_config.json", "w", encoding="utf-8") as fh:
    json.dump(config, fh, indent=2, ensure_ascii=False)

lines = [f"# generated {time.strftime('%Y-%m-%d %H:%M:%S')}"]
for base, _, files in os.walk(P3):
    for f in sorted(files):
        if f == "SHA256SUMS" or f.endswith(".pyc") or "__pycache__" in base:
            continue
        p = os.path.join(base, f)
        lines.append(f"{sha256(p)}  {os.path.relpath(p, P3)}")
with open(f"{P3}/SHA256SUMS", "a", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"manifest entries={len(lines) - 1}")
print("P3_PACKAGE_DONE")
