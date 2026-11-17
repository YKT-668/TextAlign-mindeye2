# CPU P3：CLIP-L + SwAV 全量评价报告

日期：2026-11-16 ｜ 机器：4 vCPU / 3.87GB RAM / 无 GPU（全程 CPU，串行，inference_mode，batch=1，线程=2）
产物：`evaluation/cpu_mindeye2_p3_clip_swav/`（clip_l/ swav/ statistics/ + 三个 manifest/清单）

## 0. 最终门禁

# `CPU_P3_COMPLETE`

- CLIP-L：**PASS_OFFICIAL_EQUIVALENCE**（OpenAI clip 包 vs HF，16 图 + 320 张量权重映射）
- SwAV：**PASS_OFFICIAL_PROTOCOL**（权重 SHA256 + 318/318 backbone 键 + 官方 avgpool 口径）
- 两模型 1000 与 982 双口径评价、配对统计、Holm 校正：全部完成
- 指标字典与命名审查：`reports/FINAL_EVALUATION_METRIC_DICTIONARY.csv`、
  `reports/FINAL_METRIC_NAMING_AND_PAPER_IMPACT_AUDIT.md`

---

## 1. CLIP-L 官方等价性门禁（PASS）

| 检查 | 结果 |
|---|---|
| OpenAI 官方权重 SHA256（URL 内嵌） | `b8cca3fd...a03836` ✅（`tmp/ViT-L-14.pt`，932,768,134 B） |
| 权重逐张量映射 | **320/320 OK，max_abs_diff = 0.0**（fp16→fp32 上 cast 后逐位相同；含 q/k/v 拆分与 visual.proj 转置约定） |
| HF 键覆盖 | vision_model.* 全部键被 openai visual.* 覆盖（无未覆盖键） |
| 16 张内容各异样本输出对比 | **cosine_mean = 1.0，cosine_min = 0.99999976，max_abs_error = 0.0** |
| 预处理参数依据 | cell 8 `Resize((256,256))`（默认 BILINEAR, antialias=True）→ cell 27 `Resize(224, BILINEAR)` + CLIP norm，**无 CenterCrop**（官方覆盖 clip 包默认） |
| clip 包版本 | openai/CLIP@d05afc4，`clip.load` 在 CPU 自动 `model.float()` |
| laion 模型 | 未使用（`laion/CLIP-ViT-L-14-laion2B-s32B-b82K` 非本项官方模型） |

> 证据：`clip_l/equivalence_check.json`、`clip_l/equivalence_check.log`、脚本 `scripts/p3_clipl_equivalence.py`。
> 门槛标准（cos ≥0.99999，max abs ≤1e-4）全部满足且大幅优于门槛（max abs = 0）。
> **勘误**：cell 8 为 bilinear（此前预检报告误记 bicubic）；预检用的 `F.interpolate`
> 缺 antialias 与 cell8 段，P3 全量已改用官方 `transforms.Resize` 两段式。

## 2. SwAV 官方协议核验（PASS）

| 检查 | 结果 |
|---|---|
| 权重 SHA256 | `029d0b8c...cfd1eb` ✅（脚本内校验，不符即退出） |
| 架构 | torchvision ResNet50 backbone，**318/318 键形状精确匹配，0 shape mismatch** |
| 键差异（显式记录） | fc（ckpt 无）、projection_head/prototypes（SwAV 特有）——均不在特征路径 |
| 特征位置 | **avgpool（2048 维）**= 官方 cell 31；未使用 projection_head/prototypes 替代 |
| 预处理 | 官方两段式（同上）+ ImageNet mean/std |
| 官方距离 | `scipy.spatial.distance.correlation`（=1−Pearson r）逐图均值，越低越好 |

**Sanity 差异来源分解**（`swav/sanity_gap_decomposition.json`）：预检 48 张 0.3547 vs
全量官方协议 **0.35055 ± 0.098**（前 48 张子集均值 0.3498，子集效应 −0.0008）。
0.355 vs 论文参考值 ≈0.32 的差距**主要来自参考值口径**：0.32 是 MindEye2 自身重建图的
均值，本项目 A/B 是不同生成图像，不存在必须等于 0.32 的约束。协议无问题，未做任何
"向参考值靠拢"的修改。

## 3. 运行规模与耗时

| 模型 | 图像数 | 耗时 | 吞吐 | 峰值 RSS |
|---|---|---|---|---|
| SwAV ResNet50 | 3×1000 | **344 s（5.7 min）** | 8.9 img/s | 1403 MB |
| CLIP-L ViT-L/14 | 3×1000 | **4201 s（70.0 min）** | 0.70–0.73 img/s | 2338 MB |
| 合计（串行） | 6000 次前向 | ≈75.7 min | — | 无 swap 压力 |

特征：fp32，CLIP-L 768 维 / SwAV 2048 维；NaN/Inf 检查通过（逐断言 + 全量断言）；
每 50/100 张原子检查点（tmp+os.replace），支持断点续算；manifest 对齐
（REFINEMENT_EVALUATION_MANIFEST.csv 1000 id 与张量行序逐行相等，升序）。

## 4. 正式结果

### 4.1 官方指标（确证族，1000 主口径）

| 官方指标 | A (raw unCLIP) | B (GIT+SDXL 精炼) | Δ=B−A | 95% Bootstrap CI | Wilcoxon z | log10 p | Holm log10 p | 改善/变差/持平 |
|---|---|---|---|---|---|---|---|---|
| **CLIP 2-way Identification** ↑ | **0.9319** | 0.9165 | **−0.0154** | [−0.0239, −0.0072] | −4.56 | −5.29 | **−5.29 ✅** | 404 / 515 / 81 |
| **SwAV correlation distance** ↓ | **0.3505 ± 0.098**（med 0.3380） | 0.3633 ± 0.109（med 0.3521） | **+0.0127（更差）** | [+0.0089, +0.0166] | +5.91 | −8.48 | **−8.18 ✅** | 428 / 572 / 0 |

两个官方指标在 Holm 校正后均显著：**B 组（精炼）在高层语义上显著更差**——与 P1 的
AlexNet/Inception/EffNet 结论方向完全一致。

参照（非本项目数字）：MindEye2 论文其自身重建的 CLIP 2-way ≈0.9312（本项目 A=0.9319，
量级一致）、SwAV ≈0.3205（其图像；本项目 A=0.3505，差异来自生成图不同，见 §2）。

### 4.2 补充指标（非官方，单独列出，不冒充官方）

| 补充指标 | A | B | Δ | 备注 |
|---|---|---|---|---|
| CLIP-L Strict N-way Top-1 ↑ | 0.130 | 0.113 | −0.017 | log10 p = −0.63，**不显著**（CI 含 0） |
| CLIP-L Mean Rank ↓ | 69.07 | 84.43 | +15.35 | 补充描述 |
| CLIP-L GT↔重建 Pearson 相似度 ↑ | 0.7003 ± 0.0997 | 0.6932 ± 0.1228 | −0.0070 | log10 p = −1.42，描述性 |
| SwAV Strict N-way Top-1 ↑ | 0.338 | 0.250 | −0.088 | log10 p = −6.74（补充指标） |
| SwAV 2-way Identification（新增）↑ | 0.9811 | 0.9734 | −0.0077 | 官方未对 SwAV 做 identification |
| SwAV Mean Rank ↓ | 19.90 | 27.59 | +7.69 | 补充描述 |

### 4.3 982 敏感性口径（与 1000 对照）

| 指标 | 1000 Δ | 982 Δ | 方向一致 |
|---|---|---|---|
| CLIP-L 2-way（官方） | −0.015368 | −0.015046 | ✅ |
| SwAV distance（官方） | +0.012742 | +0.012826 | ✅ |
| SwAV Top-1 | −0.088 | −0.086 | ✅ |
| CLIP-L Top-1 | −0.017 | −0.016 | ✅ |

982 与 1000 全部方向一致、量级几乎相同（982 = 同一固定候选池前 982 列，等价于独立
982 Pearson 矩阵）。

## 5. 统计口径声明

- Bootstrap 10k（seed 2027）：重采样单位 = 逐样本 delta；候选池固定为全部 1000 张 GT；
  CI 仅描述本固定刺激集，不外推新刺激，不解释为独立实验区间；
- Wilcoxon p 以 log10 下界表述（无 p=0 记录）；
- Holm 确证族 = 仅两个官方指标（CLIP-L 2-way、SwAV distance）；补充指标 p 值为描述性；
- McNemar（Top-1）与池层面分析遵循 P1 统计审计的限定（共享候选池 → 描述性；
  20×300 池为敏感性分析，非独立实验）；
- 统计显著 ≠ 效果量大：Δ 的绝对量级（1.3%~1.5%）应与效应量一并报告。

## 6. 结论

B 组（GIT caption + SDXL img2img 精炼）在本轮补齐的两个 MindEye2 官方高层语义指标上
**均显著更差**（CLIP 2-way −1.54pp，SwAV distance +0.0127），且与 P1 全部高层指标、
GPU 轮 bigG 指标方向一致：**精炼提升低层质量（P1: SSIM/PixCorr ↑），但系统性降低
与真实刺激的语义对应**。至此 MindEye2 官方图像指标族 8 项中 7 项已按官方口径完成
（PixCorr/SSIM/AlexNet×2/Inception/CLIP-L/EffNet/SwAV），仅 Brain Correlation 待
GNet 资产补齐（来源已确认，见 `reports/P3_FOLLOWUP_EXPERIMENTS_PREPARATION.md`）。

## 7. 产物与校验

```
evaluation/cpu_mindeye2_p3_clip_swav/
├── clip_l/    A_raw/B_refined/GT_features.npy(+meta)  equivalence_check.json/log
│              extract_clipl_full.log
├── swav/      A_raw/B_refined/GT_features.npy(+meta)  extract_swav_full.log
│              sanity_gap_decomposition.json
├── statistics/  p3_metrics_summary.json  p3_per_sample.csv
├── model_manifest.json  evaluation_config.json  SHA256SUMS (21 项)
```

脚本（新增，未改动任何既有脚本）：`scripts/p3_common.py, p3_swav_extract_full.py,
p3_clipl_extract_full.py, p3_clipl_equivalence.py, p3_metrics_and_stats.py,
p3_swav_sanity_decomp.py, p3_package_manifests.py`
日志：`logs/p3_swav_full.log, p3_clipl_full.log, p3_equivalence.log, p3_metrics.log`
