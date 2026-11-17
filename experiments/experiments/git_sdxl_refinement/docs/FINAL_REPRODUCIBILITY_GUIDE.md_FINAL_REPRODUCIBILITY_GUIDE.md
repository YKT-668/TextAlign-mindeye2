# FINAL：复现指南（Reproducibility Guide）

本指南说明如何在全新环境中从归档恢复本补充实验的全部正式结果并复现评价。
归档分支：`archive/git-sdxl-refinement-final-2026-11-17`（GitHub）；
大文件：HF 私有 Dataset `ykt668/textalign-mindeye2-data`，目录
`supplementary/git_sdxl_refinement/2026-11-17/`。

## 1. 环境准备

```bash
# Python 3.12 + 依赖（版本以 GPU_ENV_LOCK.txt / model_manifest.json 为准）
pip install torch==2.5.0 torchvision==0.20.0 transformers==4.46.3 \
            numpy==2.5.4 scipy==1.17.1 safetensors huggingface_hub
# 评价阶段无需 GPU；CPU 4 核即可，内存建议 ≥8GB
export HF_HUB_OFFLINE=0   # 首次拉取权重时保持联网
```

## 2. 获取代码与文档

```bash
git clone -b archive/git-sdxl-refinement-final-2026-11-17 \
    https://github.com/YKT-668/TextAlign-mindeye2.git
cd TextAlign-mindeye2/experiments/git_sdxl_refinement
# 校验：对照 SHA256SUMS
sha256sum -c SHA256SUMS
```

## 3. 获取正式数据（HF 私有，需只读 token）

```bash
huggingface-cli login   # 或 export HF_TOKEN=...
hf download ykt668/textalign-mindeye2-data --repo-type dataset \
    --include "supplementary/git_sdxl_refinement/2026-11-17/*" --local-dir ./hf_data
# 逐文件 SHA256 校验
python - <<'EOF'
import hashlib, csv
for row in csv.DictReader(open("FINAL_ARCHIVE_MANIFEST.csv")):
    ...
EOF
```

## 4. 重建评价（如需从头复算）

### 4.1 CLIP-L + SwAV（P3，官方口径）

```bash
python scripts/p3_clipl_extract_full.py     # CLIP-L 特征，CPU ≈70 min（需 openai/clip-vit-large-patch14 本地缓存）
python scripts/p3_swav_extract_full.py      # SwAV 特征，CPU ≈6 min（需 swav_800ep_pretrain.pth.tar，
                                            #   SHA256 029d0b8c...cfd1eb 放 /root/.cache/torch/hub/checkpoints/）
python scripts/p3_metrics_and_stats.py      # 官方 2-way / SwAV 距离 / 补充口径 / 统计
# 或直接使用已归档特征（evaluation/cpu_mindeye2_p3_clip_swav/features/*.npy）跳过提取
```

### 4.2 已归档结果直接读取

| 结果 | 文件 |
|---|---|
| CLIP-L 2-way 与全部统计 | `statistics/p3_metrics_summary.json` |
| 逐样本 | `statistics/p3_per_sample.csv`、`p1/p2 per_sample CSV` |
| bigG image-image 评价 | `evaluation/per_sample_metrics.csv`、`eval_summary.json` |
| P1 官方 6 项 | `cpu_mindeye2_p1/metrics_summary.csv` + `stats.json` |
| P2 caption | `cpu_mindeye2_p2_caption/caption_summary.json` + extension |

## 5. 权重来源（不重分发，引用既有位置）

| 权重 | 来源 | 校验 |
|---|---|---|
| S1 checkpoint | ykt668/textalign-mindeye2-model @ masterGit/a127f929 | 见归档报告核实记录 |
| SDXL / unCLIP / bigG_to_L | pscotti/mindeyev2 @ 3f20ccbe（官方公开） | HF blob SHA |
| CLIP-L（评价） | openai/clip-vit-large-patch14 @ 32bd6428 | a2bf730a...（safetensors）|
| OpenAI ViT-L-14.pt（等价性基准） | openaipublic.azureedge.net（URL 内嵌 b8cca3fd...） | b8cca3fd41...a03836 |
| SwAV | dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar | 029d0b8c2e...cfd1eb |

## 6. 关键协议要点（复现时不可省略）

1. 预处理两段式：`Resize((256,256))`（bilinear, antialias）→ `Resize(224, BILINEAR)` + 各自
   mean/std；**无 CenterCrop**；不要用 `F.interpolate`（antialias=False，数值不同）；
2. 指标 = 平均 Pairwise 2-way Identification（官方 cell 17），不是 N-way Top-1；
3. 982 口径 = 同一固定池前 982 列；不要用目录文件名排序重建配对，必须使用
   `REFINEMENT_EVALUATION_MANIFEST.csv` / `all_ids.pt`；
4. 逐样本检验为描述性（共享候选池）；20×300 池为敏感性；
5. 原始 NSD beta / 刺激数据不在归档内，需经 NSD 官方渠道获取。

## 7. 时间线与溯源

- 推理：2026-01-15（A800-40GB，见 FINAL_1000_INFERENCE_REPORT.md）
- P1/P2/P3 评价：2026-11-15/16（CPU 4 核）
- 归档：2026-11-17（GitHub tag `git-sdxl-refinement-final-2026-11-17` + HF
  `supplementary/git_sdxl_refinement/2026-11-17`）
