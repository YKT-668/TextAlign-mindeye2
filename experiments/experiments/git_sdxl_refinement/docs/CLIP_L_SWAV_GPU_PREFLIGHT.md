# CLIP-L / SwAV GPU 预检报告（PREFLIGHT）

日期：2026-11-16 ｜ 机器：4 vCPU / 3.87GB RAM / 5.9GB swap / 无 GPU
本轮：仅预检、轻量验证与 GPU 执行包准备；**未租 GPU、未执行全量推理、未下载大型模型**。

## 0. 最终门禁

# `READY_FOR_SHORT_GPU_RENTAL`

理由：协议、统计、权重、预处理、输入对齐、执行脚本全部就绪，无遗留阻塞项。
可选替代：SwAV/CLIP-L 在当前 CPU 上实测速度显著好于上轮悲观估计（见 §5），
SwAV 全量 CPU ≈6 分钟、CLIP-L 全量 CPU ≈70–130 分钟——若接受时长，**不租 GPU 也可完成**。

---

## 1. SwAV 模型加载验证（✅ PASS）

权重：`/root/.cache/torch/hub/checkpoints/swav_800ep_pretrain.pth.tar`
113,703,565 B，SHA256 `029d0b8c...cfd1eb`（与记录一致），官方源 `dl.fbaipublicfiles.com/deepcluster`。
官方架构：notebook cell 31 `torch.hub.load('facebookresearch/swav:main','resnet50')` + `avgpool`。

检查结果（`evaluation/gpu_missing_metrics/swav_preflight_cpu.json`）：

| 项 | 结果 |
|---|---|
| checkpoint 键名 | 328 键，`module.*` 前缀 |
| 键名映射 | 剥离 `module.` 后与 torchvision ResNet50 完全一致，**0 shape mismatch** |
| missing keys（显式记录） | 仅 `fc.weight, fc.bias`（SwAV backbone 无分类头；fc 在 avgpool 之后，不在特征路径） |
| unexpected keys（显式记录） | 仅 SwAV 投影头/原型：`projection_head.0/1/3.*`（10 键）+ `prototypes.weight`（官方 ResNet 自带，torchvision 无；均不在特征路径） |
| backbone 完整性 | conv1/bn1/layer1-4 全部键形状精确匹配 |
| strict 加载 | strict=False（上述差异逐键列出，**未静默忽略**） |
| CPU 前向（48 张，官方 Resize(224,BILINEAR)+ImageNet norm） | **0.11 s/张**，峰值 RSS 1.47GB，输出 **2048 维**（avgpool） |
| 合理性 sanity | 前 48 张 A 组相关距离 **0.3547 ± 0.107**（官方 MindEye2 表量级 ≈0.32；子集值非正式结果） |

## 2. CLIP-L 图像塔核验（✅ PASS，附 1 个 GPU 轮验证项）

**版本区分（关键判定）：**

| 模型 | 用途 | 判定 |
|---|---|---|
| `openai/clip-vit-large-patch14`（OpenAI ViT-L/14 的 HF 转存） | **MindEye2 官方图像指标**：cell 27 `clip.load("ViT-L/14")` + `encode_image` | ✅ 使用 |
| `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | 训练数据/方案不同 | ❌ **不得用于本指标**（本地有缓存，但仅限其它用途） |
| `openai/clip-vit-large-patch14` 文本塔 | P2 caption CLIP-L 指标（已完成） | 与图像塔无关，不混淆 |

官方口径（cell 27，逐字核对）：特征 = `encode_image` 最终层（768 维）；
预处理 = **cell 8 两段式**：先把所有图像 Resize 到 **256（bicubic）**（`all_recons/all_images` 若非 256），
再 **Resize(224, BILINEAR) + CLIP mean/std，无 CenterCrop**（官方覆盖了 clip 包默认
bicubic+CenterCrop，GPU 脚本按官方两段实现；A_raw 224→256→224 的往返也已如实复刻）。

本地权重与离线能力（`clipl_image_tower_preflight.json`）：

| 项 | 值 |
|---|---|
| 缓存 revision | `32bd64288804d66eefd0ccbe215aa642df71cc41` |
| 文件 | config/safetensors/bin/tokenizer 全套齐备（safetensors 1.71GB，SHA256 见 JSON） |
| 离线加载 | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 local_files_only=True` **成功** |
| CPU 前向（4 张） | 输出 768 维，1.4–2.6 s/张，峰值 RSS 2.05GB |

**GPU 轮验证项（不阻塞租赁）**：OpenAI `clip` 包与 HF 视觉塔的数值等价性——
租到 GPU 后装 `pip install git+https://github.com/openai/CLIP.git`，对 ~10 张图比对
`clip.load("ViT-L/14").encode_image` 与 HF `image_embeds`（预期 <1e-4）。在此之前，
结果标注保持 `HF_CLIP_L_PROXY`。

## 3. 缺失 Caption 文件（✅ 已解决，无需重跑 GIT）

`all_git_generated_captions.pt`（GIT 看 GT 图生成的 caption）已从**官方数据集仓库**
`pscotti/mindeyev2` 下载（55,125 B；git blob oid `8dca9586...`；本地 SHA256 `adfd766b...`），
落到 `assets/ground_truth/`。详见 `reports/CAPTION_ASSET_COMPLETENESS_AUDIT.md`。
predcaptions 来自 `clip_convert(prior_out)`（脑→预测嵌入），与 GIT 看图 caption 输入不同，
**确认不可复用**——本轮用官方文件直接补齐，避免了任何 GIT 重推理。

## 4. GPU 执行包（`evaluation/gpu_missing_metrics/`）

| 文件 | 说明 |
|---|---|
| `extract_features_gpu.py` | 统一提取脚本：`--model clipl|swav`，官方预处理，离线权重，SwAV 内置 SHA256 校验 |
| `run_missing_metrics.py` | 指标计算：CLIP-L 官方 2-way（+Top-1/Mean Rank）、SwAV 相关距离；1000/982 双口径；时间戳输出不覆盖 |
| `build_input_manifest.py` | 统一 A/B/GT 输入清单（ids 唯一升序、张量形状、SHA256） |
| `make_sha256_manifest.py` | 产物 SHA256SUMS（追加模式） |
| `gpu_env_requirements.txt` | 依赖清单（版本与既有环境一致） |
| `README_GPU_RUNBOOK.md` | 完整 Runbook：小样本验证 → 全量 → 指标 → 归档，串行互斥、断点续算、日志 |
| `swav_preflight_cpu.json` / `clipl_image_tower_preflight.json` | 本轮预检证据 |

**脚本可用性已实测（小样本，CPU 回退模式）**：`--limit 4 swav`（2048 维 ✓）与
`--limit 2 clipl`（768 维 ✓）均完整跑通 A_raw/B_refined/GT 三组并落盘 `.pt`+.ok，
日志见 `logs/gpu_pkg_test_swav.log`、`logs/gpu_pkg_test_clipl2.log`。
GPU 上只需把 batch 提到 64 并加 `--dtype fp16`。

断点续算：每组特征 `.pt` + `.ok` 标记（含 n/dim/dtype），重跑自动跳过已完成组；
日志逐 batch 记录 img/s。两模型**串行**执行，不同时占用 GPU。

## 5. 资源与时间预算（环境安装与模型计算分开）

### 环境安装（一次性）

| 场景 | 耗时 |
|---|---|
| 预装 PyTorch 镜像（推荐） | **2–5 分钟**（仅补 transformers/scipy） |
| 裸机 pip 全量（torch ~2.5GB） | 机房快网 5–20 分钟；本项目线路 ~1MB/s 则 40–60 分钟 |

### 模型计算（3000 张 = A 1000 + B 1000 + GT 1000）

| GPU | CLIP-L ViT-L/14（fp16） | SwAV ResNet50 | 显存峰值（估计） |
|---|---|---|---|
| 16GB（如 4090/A4000） | ~2–4 分钟 | ~1–2 分钟 | ~4GB ✅ |
| 24GB（3090/4090/A5000） | 同上 | 同上 | ~4GB ✅ |
| A100 40GB | ~1–2 分钟 | <1 分钟 | ~4GB ✅ |

> 注意：CLIP-L/SwAV 的时长为**架构外推估计**，未使用 A100 历史 bigG（10B 参数、
> 33 img/s）耗时直接折算——bigG 比 ViT-L 大一个数量级，不可类比；本轮给出的
> CPU 实测（CLIP-L 1.4–2.6 s/张、SwAV 0.11 s/张）是唯一实测锚点，GPU 按单卡
> ≥50–200× CPU 吞吐估算，并在 GPU 轮首 2 分钟内以小样本实测校准。

### 建议

- **推荐 16GB 或 24GB**：显存需求 ~4GB，A100 40GB 明显过剩；
- 主机 RAM 32GB 建议：张量加载（fp32 600MB×3 + 模型 ~2GB）在 16GB RAM 机器亦可，
  但 32GB 更稳；
- 租赁时长：预装镜像下 **30–60 分钟足够**（含环境 5 分钟 + 小样本校准 + 全量 + 指标 + 归档）。

### CPU 替代路径（零成本，实测）

| 模型 | 实测 s/张 | 3000 张 |
|---|---|---|
| SwAV | 0.11 | **≈6 分钟** |
| CLIP-L | 1.4–2.6 | ≈70–130 分钟（本轮轻载实测；上轮 14 s/张系 swap 压力所致） |

若不赶时间，可先在本机 CPU 跑 SwAV（6 分钟），CLIP-L 是否租 GPU 由时长接受度决定。

## 6. 执行边界确认

未租 GPU；未执行 3000 张全量（本轮最大前向 48 张 SwAV + 4 张 CLIP-L）；未下载大型模型
（仅 55KB 官方 caption 文件）；未重新生成 Caption；未修改任何原始结果；未删除任何文件；
未进入 Empty Prompt 与 GNet 实验。
