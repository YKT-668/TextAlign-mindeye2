# A100 40GB 可行性审计（ConceptAlign GIT img2img）

日期：2026-11-15
结论：**A100 40GB 够用（PASS，有条件）** —— 实测峰值 28.1GB（nvidia-smi，第一阶段），保守上界 **30GB**，40GB 余量 **约 9.5GB**。
最终判定与完整证据链见 **`reports/FINAL_A100_40G_RENTAL_DECISION.md`** → `A100_40G_READY`。

---

## 一、实测数据（唯一权威来源）

来源：`reports/provenance/final14_archive__followup_experiments__recovery_metadata__READMЕ3_A100_FINAL_INFERENCE_REPORT.md` 第 6 节（A100 80GB，`nvidia-smi` 实测）：

| 阶段 | 峰值显存 | 说明 |
|---|---|---|
| 模型加载 + 扩散初始化 | ~10.5 GB | 常驻基线 |
| **重建峰值（SDXL denoising）** | **~28.1 GB** | 全流程最高点 |
| 表示推理（3000 样本） | ~10.5 GB | 不跑 SDXL 时 |

历史结论原文：`RECONSTRUCTION_GUARANTEED_ON_48GB`（max 28.1GB << 48GB）。
→ 对 40GB 而言：28.1GB < 40GB，**余量 ≈ 11.9GB（≈30%）**。

---

## 二、显存构成拆解（代码级核对，均为最小配置）

### 阶段 1：脑信号 → unCLIP 重建（`src/recon_inference_run.py`）
| 组件 | 占用（实测） | 依据 |
|---|---|---|
| unCLIP denoiser（SDXL 架构，fp32） | **10.005 GB**（2.5013B 参数） | ckpt `state_dict/model.*` 元数据实测 |
| unCLIP conditioner（fp16，仅文本塔） | **1.635 GB** | 0.8176B 参数（CLIP-L 0.123B + bigG 文本 0.6946B）×2B；`del model.visual` + `.half()` |
| unCLIP VAE（first_stage，fp32） | **0.335 GB** | ckpt `first_stage_model.*` |
| MindEye2/TextAlign 主体（fp32） | **8.929 GB** | backbone 7.612 + diffusion_prior 1.039 + ridge 0.258 + text_head 0.020 |
| GIT 文本解码器（fp32） | **1.577 GB** | `model.safetensors`（可 CPU offload） |
| 激活 / 采样缓冲 | 4–6 GB | `minibatch_size=1`、`num_samples=1`、38 步、latent 96×96、CFG batch=2 |
| **合计** | **22.48 GB 权重 + 激活 ≈ 28–30 GB** | 与 nvidia-smi 实测 28.1GB 吻合 |

> 只有 `--save_images` 或 `--export_official_pts` 时才加载 unCLIP（L748 `need_unclip`）。我们**需要 S1 重建图**，所以走 28.1GB 路径。
> 模型侧无 EMA 副本：`sd_xl_base.yaml` / `unclip6.yaml` 均未设 `use_ema`，`DiffusionEngine(use_ema=False)` 默认，`ema_scope()` 是空操作。

### 阶段 2：SDXL img2img（`src/enhanced_recon_inference.ipynb`）
| 组件 | 占用 | 依据 |
|---|---|---|
| Zavy SDXL base（fp16） | ~6.5 GB | safetensors 头解析：2,515 个张量，**全部 F16** |
| CLIP-L 文本塔（FrozenCLIPEmbedder） | ~0.5 GB | cell 3 |
| OpenCLIP bigG 文本塔（FrozenOpenCLIPEmbedder2） | ~3 GB | cell 3（fp32 权重） |
| bigG **图像**塔（FrozenOpenCLIPImageEmbedder） | **0（不加载）** | cell 4：仅 `plotting or num_samples>1` 时加载；我们 `num_samples=1`、`plotting=False` |
| 采样激活（768²，CFG 双批） | 数 GB | `num_steps=25`、`cfg scale=5`、`img2img_timepoint=13`，全程 `autocast(fp16)` |

> 阶段 2 无官方单独实测值。按权重级实测推算：Zavy 6.50 + CLIP-L 文本 0.49 + bigG 文本 2.78 = 9.77GB 权重，加激活 4–6GB → **保守上界 18GB**。
> **两个阶段的实测/推算结论与最终判定见 `reports/FINAL_A100_40G_RENTAL_DECISION.md`**（28.1GB 已确认为第一阶段、nvidia-smi 口径）。

---

## 三、40GB 结论与前提条件

**够用**，但必须同时满足：

1. **两个阶段分进程跑**（S1 重建结果落盘 → 再启 img2img 进程）。
   若同进程加载 unCLIP(fp32 14GB) + Zavy(6.5GB) + GIT(1.5GB) + 激活，会逼近 30–34GB，余量骤降到 <6GB。
2. **保持 `num_samples=1`、`plotting=False`**（否则加载 bigG 图像塔，+4–7GB）。
3. **不加 refiner**（官方 img2img 本身无 refiner，审计已确认）。
4. **保持 `minibatch_size=1`**；GIT 若 OOM 可按代码注释移到 CPU（`clip_text_model.to('cpu')`）。
5. 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 降低碎片化。
6. 推理机上不要同时跑第二个吃显存的进程。

### 版本/硬件备注
- 40GB A100 基本都是 **PCIe** 版：HBM 带宽 ~1555 GB/s（80GB SXM 为 ~2039 GB/s）→ 采样约慢 1.3×，显存不受影响。
- 40GB 卡常见实例配置为 8–30 vCPU + **60–120GB 主机内存**。**必须确认 ≥64GB**；推荐 96GB。
  - 理由：unCLIP ckpt 用 `torch.load(map_location="cpu")`（L657）会在 CPU 常驻 17GB 后再搬 GPU；S1 的 26.8GB ckpt 已用 `mmap=True`（L459）对 RAM 友好，但 `ckpt` 对象仍占页缓存。
  - 60GB 也能跑，但不要在同进程同时保留两份大对象。

---

## 四、40 / 48 / 80GB 对比

| 档位 | 峰值 28.1GB 结论 | 余量 | 建议 |
|---|---|---|---|
| 24GB | ❌ 不够（官方明确：只能跑 representation） | -4GB | 不选 |
| **40GB** | ✅ 够（需满足上节 6 条） | 实测余量 ~11.4GB / 保守余量 ~9.5GB | **可选，性价比优先** |
| 48GB | ✅ 稳（历史判定 GUARANTEED） | ~19.4GB | 可选 |
| 80GB | ✅ 富余 | ~51.4GB | 省心/可开 num_samples>1 |

---

## 五、OOM 应急清单（按优先级）

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
2. GIT 移到 CPU（S1 阶段）
3. 确认 `plotting=False, num_samples=1`
4. 确认两阶段分进程、`torch.cuda.empty_cache()` 后退出
5. 极端情况：unCLIP denoiser 改 fp16（会轻微影响画质，需记录为偏差）
6. 最后手段：换 80GB

---

## 六、待 GPU 实测核对项

- [ ] `nvidia-smi` 确认 total ≈ 40960MiB、机器 RAM ≥ 64GB
- [ ] S1 单图跑通后 `torch.cuda.max_memory_allocated()` 记录（预期 ≤ 29GB）
- [ ] img2img 单图跑通后同样记录（预期 ≤ 28GB）
- [ ] 1000 张端到端时间（阶段 1 + 阶段 2）

> 若 S1 + img2img 同进程实测 > 38GB，则改分进程（本报告默认建议）。
