# CPU P1 MindEye2 官方评测报告（图像质量 + 高级视觉语义）

日期：2026-11-15 ｜ 机器：**4 vCPU / 3.87GB RAM（可用 ~1GB）/ 5.9GB swap / 无 GPU**

## 最终状态

# `P0_MIGRATION_PASS_P1_PARTIAL`

- **P0 迁移校验：全 PASS**
- **P1：8 项指标中 6 项完成**（PixCorr、SSIM、AlexNet-2、AlexNet-5、InceptionV3、EfficientNet-B1）
  **2 项 BLOCKED**：CLIP ViT-L/14（CPU 过慢）、SwAV（权重源不可达）

---

## 1. P0：迁移后完整性检查

| 项 | 结果 |
|---|---|
| 挂载 | `/dev/vdb` → `/mnt/data`，**ext4**，`rw,noatime` |
| UUID | `5a9d9737-e795-4d2a-9c93-3d7b443f9054` ✅ 与预期一致 |
| 独立数据盘 | ✅ 系统盘 `/dev/vda3`（196G，146G 可用）/ 数据盘 `/dev/vdb`（147G，25G 可用） |
| 正式产物 SHA256（19 项） | ✅ **19/19 OK**（FULL_1000 10 项 + EVALUATION 9 项；1 条 "improperly formatted" 是清单末尾的 `DONE` 说明行，非失败项） |
| bigG 特征 SHA256（6 项） | ✅ **6/6 OK** |
| 断链 | hf_home / assets / run / evaluation / reports 全部 **0** |
| run/ 中 4 条绝对软链 | ✅ 目标全部可达（`unclip6`、`bigG_to_L`、`betas`、`new_test 0.tar`） |
| GPU | ❌ 无（`nvidia-smi` 报无法通信）→ 全程强制 CPU |
| Python / PyTorch | 3.12.7 / **2.5.0+cu124（CPU 模式，`torch.cuda.is_available()==False`）** |
| 其它依赖 | torchvision 0.20.0、numpy 2.5.4、scipy 1.17.1、timm 1.0.11、Pillow 13.0.0、open_clip 2.20.0 |
| scikit-image | **0.26.0**（以 `--target /mnt/data/cpu_pkgs` 独立安装，**未改动任何原有环境**） |
| scikit-learn / nltk / sentence-transformers / evaluate | 未安装（P1 不需要；未为凑清单而安装） |

> 结论：**数据与运行环境均通过**。唯一限制是 **RAM 极小**（vscode-server 常驻约 2GB，实测可用仅 0.2–1.9GB，期间多次触发 swap）。

## 2. P1 已完成指标（1000 与 shared982 两种口径）

方向：PixCorr/SSIM/Identification **越高越好**；Distance **越低越好**。

| 指标 | 口径 | A 组 (raw) | B 组 (refined) | Δ (B−A) | 95% CI (bootstrap 10k, seed 2027) | Wilcoxon p | Holm p | 改善/变差 |
|---|---|---|---|---|---|---|---|---|
| **PixCorr** | 1000 | 0.349108 | **0.354903** | **+0.005795** | [0.004716, 0.006900] | 2.29e-25 | **0** | 661 / 339 |
| PixCorr | 982 | 0.347531 | 0.353322 | +0.005791 | — | — | — | 646 / 336 |
| **SSIM** | 1000 | 0.376164 | **0.411509** | **+0.035345** | [0.034029, 0.036660] | 1.82e-163 | **0** | **989 / 11** |
| SSIM | 982 | 0.374918 | 0.410266 | +0.035347 | — | — | — | 971 / 11 |
| **AlexNet(2) N-way acc** | 1000 | **0.977492** | 0.973730 | −0.003762 | — | — | — | 排名法 300 更好 / 385 更差 |
| AlexNet(2) | 982 | 0.977733 | 0.973966 | −0.003767 | — | — | — | — |
| **AlexNet(5) N-way acc** | 1000 | **0.993159** | 0.989296 | −0.003863 | — | — | — | 223 / 325 |
| AlexNet(5) | 982 | 0.993131 | 0.989287 | −0.003844 | — | — | — | — |
| **InceptionV3 N-way acc** | 1000 | **0.962539** | 0.948914 | −0.013625 | — | — | — | 355 / 513 |
| InceptionV3 | 982 | 0.962359 | 0.948658 | −0.013701 | — | — | — | — |
| **EfficientNet-B1 dist ↓** | 1000 | **0.625586** | 0.647990 | **+0.022404**（更差） | [0.016290, 0.028792] | 1.32e-11 | **0** | 393 / 607 |
| EfficientNet-B1 dist ↓ | 982 | 0.625823 | 0.648030 | +0.022207 | — | — | — | — |

补充（逐样本排名口径，识别类指标的 per-sample 值 = 1-based 排名，越低越好）：

| 指标 | 平均排名 Δ | 中位排名 Δ | 95% CI | Wilcoxon p | Holm p |
|---|---|---|---|---|---|
| AlexNet(2) | +3.759 | 0.0 | [1.754, 5.929] | 2.53e-4 | 2.53e-4 |
| AlexNet(5) | +3.859 | 0.0 | [2.260, 5.660] | 2.34e-6 | 5.0e-6 |
| InceptionV3 | +13.611 | +1.0 | [7.721, 19.754] | 4.62e-10 | 0 |

> **Holm 校正**：对 P1 全部 6 个指标的 delta p 值做 Holm（1000 口径），**全部仍显著**（校正后 p ≈ 0 或 ≤2.5e-4）。

### 结论方向

- **低层（像素/结构）：B 显著更好** —— SSIM 在 **989/1000** 张上升，PixCorr 661/1000 上升。img2img 让图像更"自然、平滑、结构完整"。
- **高层语义：B 显著更差** —— AlexNet-2/5 与 InceptionV3 的 N-way 识别准确率全部下降，EfficientNet-B1 相关距离上升。
- 这与上一轮 GPU 端的 bigG 结果（CLIP 图像相似度 0.34516→0.33842，1000-way 检索 Top-1 53.1%→42.9%）**方向完全一致**：精炼提升"像真实图像的观感"，但降低"与真实刺激的语义对应"。

## 3. 75/25 融合诊断组（DIAGNOSTIC_NOT_OFFICIAL）

| 指标 | A_blend075 | B_blend075 |
|---|---|---|
| PixCorr (1000) | 0.387034 | 0.392967 |
| SSIM (1000) | 0.413427 | 0.442110 |

- 因官方 `sd_image_var_autoenc.pth` 本地缺失、替代 VAE 无法逐字节证明一致 → **仅作诊断**，不得写成官方复现。
- 原始 A/B（未融合）仍是正式主比较；诊断组未覆盖任何原始数据。

## 4. BLOCKED 项（明确原因，未用替代模型冒充）

| 指标 | 状态 | 原因（实测证据） |
|---|---|---|
| **CLIP ViT-L/14** | `BLOCKED_SLOW_ON_CPU` | 官方用 OpenAI `clip` 包；本地以 HF `openai/clip-vit-large-patch14`（同源权重）代理。实测 **14 秒/张**（113s 完成 8 张）→ 3000 张约 **11.7 小时**，本轮不可行。已尝试 fp16（更慢）与仅视觉塔（仍慢） |
| **SwAV-ResNet50** | `BLOCKED_MISSING_WEIGHT` | 官方 `torch.hub facebookresearch/swav` 指向 `dl.fbaipublicfiles.com`；该站 **403/超时**，仅下到 **23.6MB / ~800MB** 后停滞。按"不使用来源不明第三方权重"的要求，未替换 |

> 上一轮 GPU 已算出的 bigG 图像相似度/1000-way 检索**不需要重算**，可直接复用（`evaluation/precomputed_features/bigg/`）。

## 5. 资源实测

| 模型 | 耗时 | 峰值 RSS |
|---|---|---|
| PixCorr + SSIM（含诊断融合组，5000 图） | ~8 min | — |
| EfficientNet-B1 | 4.6 min（20:28:51→20:33:27） | 2.10 GB |
| AlexNet-2/5（单次前向双层） | 1.5 min（20:33:27→20:34:58） | 2.61 GB |
| InceptionV3 | 11.0 min（20:34:58→20:46:00） | 2.68 GB |
| CLIP-L（中止） | 8 张 / 113 s | — |

- 期间 **Swap 使用最高约 3.3GB**，`MemAvailable` 最低约 **220MB**；已按约束单模型串行、分批、无并发大任务。
- 未触发"持续换页即停机"的硬门限（任务全部完成，仅 CLIP-L 因速度中止）。

## 6. 交付物

目录：`evaluation/cpu_mindeye2_p1/`

| 文件 | 说明 |
|---|---|
| `metrics_summary.csv` | 全部指标 × (A/B/DELTA) × (1000/982) 的 mean/std/median/方向 |
| `p1_accuracy_summary.csv` | 识别类官方口径准确率（acc_N1000 / acc_N982 / hit1 / median_rank） |
| `per_sample_metrics.csv` | 统一逐样本指标（metric, group, idx, id, value, direction） |
| `lowlevel_per_sample.csv` / `lowlevel_summary.json` | PixCorr / SSIM 明细 |
| `p1_{alexnet,inception,effnet}_per_sample.csv` + `_summary.json` | 各模型明细 |
| `stats.json` | delta、bootstrap CI、Wilcoxon、Holm 校正 |
| `evaluation_config.json` | 口径、预处理、环境、指标状态（OFFICIAL / DIAGNOSTIC / BLOCKED） |
| `model_manifest.json` | 模型来源 + 字节数 + SHA256 + 离线可用性 + 被 BLOCKED 原因 |
| `evaluation_log.txt` | 各模型运行尾部日志与 RSS |
| `features/` | 各模型 fp16 特征缓存（可复算，未删除） |
| `SHA256SUMS` | **15 项**产物校验 |

模型权重 SHA256（本轮新下载，共 **385MB** < 1.5GB 预算）：

```
7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02  alexnet-owt-7be5be79.pth (244,408,911 B)
0cc3c7bd75056d25e46cba549dc184522069b81e9787eff6df84f397bd52a5ef  inception_v3_google-0cc3c7bd.pth (108,949,747 B)
c27df63ce6eb17ef8bcea58922fd3a254cba910c720f41ee89d64d99fb7a4ddf  efficientnet_b1-c27df63c.pth (31,600,329 B)
```

## 7. 回答任务书 10 问

| # | 问题 | 回答 |
|---|---|---|
| 1 | CPU 数据盘迁移是否完全通过？ | ✅ **通过**：UUID 一致、ext4、独立盘、挂载点 `/mnt/data`、0 断链 |
| 2 | 原始 19 项 + 新增 bigG 6 项 SHA 是否全通过？ | ✅ **19/19 + 6/6 全 OK**，无失败项 |
| 3 | CPU 环境是否满足评测需求？ | ⚠️ **部分满足**：小模型（AlexNet/Inception/EffNet/PixCorr/SSIM）可跑；**bigG/CLIP-L 级模型在 4 核 + 1GB 可用内存下不可行** |
| 4 | 实际完成了哪些指标？ | PixCorr、SSIM、AlexNet-2、AlexNet-5、InceptionV3（N-way）、EfficientNet-B1（correlation distance），均含 1000 与 982 |
| 5 | 哪些指标仍缺权重？ | **SwAV-ResNet50**（源站不可达）；**CLIP ViT-L/14**（有代理权重但 CPU 太慢，14 s/张） |
| 6 | 1000 与 982 两套结果是否都完成？ | ✅ 两者都完成；982 用 **982×982 候选矩阵**独立计算，不是截取 |
| 7 | A/B 差异及统计显著性如何？ | **低层显著更好**（SSIM +0.0353，989/1000 改善；PixCorr +0.0058）；**高层显著更差**（AlexNet-2 −0.0038、AlexNet-5 −0.0039、Inception −0.0136、EffNet 距离 +0.0224）；Holm 校正后全部仍显著 |
| 8 | 官方融合诊断结果如何？ | PixCorr 0.3870→0.3930、SSIM 0.4134→0.4421（B 仍更好）；标注 `DIAGNOSTIC_NOT_OFFICIAL` |
| 9 | 实际 CPU 耗时与 RAM 峰值？ | 核心计算约 **25 分钟**（lowlevel ~8min + effnet 4.6 + alexnet 1.5 + inception 11）；峰值 RSS **2.68GB**（Inception），Swap 最高 ~3.3GB |
| 10 | 是否存在不能用于论文的代理结果？ | 有 1 项：**CLIP ViT-L/14 若后续用 HF 代理权重**必须标注 `HF_CLIP_L_PROXY`（本轮未出数）；SwAV 无结果；其余 6 项均为 `OFFICIAL_REPRODUCED` |

## 8. 下一阶段建议（P2/P3）

1. **换 8–16 核 / 32GB 机器**（或至少 16GB + 8 核）：可在 ~1 小时内补齐 CLIP ViT-L/14（14s/张 → 8 核约 5s/张），并重跑 bigG 校验。
2. **SwAV**：换可达网络或由用户在本地放好 `swav_800ep_pretrain.pth.tar` 到 `/root/.cache/torch/hub/checkpoints`，再用同一脚本续算（特征有 `.ok` 标记，可跳过已完成）。
3. **Brain Correlation**：仍需 `gnet_multisubject.pt` + `brain_region_masks.hdf5`（本轮按要求未下载）。
4. **Caption 指标**：A/B 相同，只需算一次；需 `evaluate`/`nltk`/`sentence-transformers`（本轮未装，避免为凑清单装包）。
5. 不要把 RSA/CCD/2AFC 等**脑表征指标**当作 SDXL 精炼的效果；A/B 在这类指标上完全相同。

## 9. 本轮边界

未下载 GNet/brain masks/大型数据集；未重新推理；未训练；未修改任何原始产物与原始 MindEye2 代码；未覆盖既有 evaluation/reports；未租 GPU；新增权重下载 385MB（未超 1.5GB）；未删除任何 checkpoint 或缓存。

