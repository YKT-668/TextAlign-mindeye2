# FINAL：GIT-Caption-Conditioned SDXL Refinement 补充实验正式记录

归档日期：2026-11-17 ｜ 本文件为项目收尾的正式实验记录，供论文写作与未来恢复使用。
代码基线：MindEye2 官方源码 `archive/final14@231850a`（TextAlign-mindeye2-231850a）。

---

## 1. 实验背景与研究目的

ConceptAlign 在 1000 张 NSD 测试图像上完成两阶段脑解码与生成：第一阶段由脑信号重建图像
（unCLIP 系列），第二阶段在其上叠加 **GIT caption 引导的 SDXL img2img refinement**。
本补充实验回答：**额外的 caption-conditioned refinement 是否值得？**——它在多大成本下
改变了重建图像与真实刺激的语义对应（CLIP-L 2-way Identification）。

## 2. A/B 两组的具体区别（重要措辞澄清）

| 组 | 定义 |
|---|---|
| **A（ConceptAlign）** | 仅第一阶段：脑信号 → 重建图像（`all_enhancedrecons`）。**注意**：A 并非"完全没有 GIT/Caption"——第一阶段本身使用项目既有的 caption 相关组件；A 的准确含义是**不包含额外的第二阶段 caption-conditioned SDXL refinement**。 |
| **B（ConceptAlign + GIT-SDXL）** | 与 A 完全相同的第一阶段输出（同一 1000 张、同一顺序、同一 ID），叠加 GIT 预测 caption 引导的 SDXL img2img refinement（Zavy SDXL base）。 |
| GT | NSD 真实刺激图像（同 ID 同序；像素级顺序验证：配对 pixcorr 为随机配对的 6.3 倍，95.6% 样本为正，见 `reports/FINAL_GIT_REFINEMENT_EVALUATION.md` §2）。 |

因此本实验比较的是"**增加整个 SDXL refinement 流程**"的效果；结果中的下降**不能**完全
归因于 caption 文本本身（SDXL 图像先验、img2img 平滑均包含在内）。

## 3. 推理流程

- **第一阶段**（A）：官方 `recon_inference_run.py`（archive/final14@231850a 原文件），
  1000 张、每图 3 次重复平均、官方随机种子；输出七类：`enhancedrecons`、`blurryrecons`、
  `clipvoxels`、`prior_out`、`backbones`、`predcaptions`、`ids`（另有 `captions.txt`、
  `verify_stage1_1000.json`）。
- **第二阶段**（B）：严格复刻 `enhanced_recon_inference.ipynb` cell 3-5：
  `plotting=False, num_samples=1, img2img_timepoint=13, guider.scale=5, num_steps=25,
  seed=42, 无 refiner, use_ema=False`；negative prompt 与 `vector_suffix`（来自 `txt=""`）
  与官方一致；输出 `stage2_img2img.pt`（1000,3,256,256 fp32）+ `stage2_ids.pt`。
- 两阶段验证：`verify_stage1_1000.json` / `verify_stage2_1000.json` 均 ALL_PASS
  （shape/dtype/ID 一致/数值范围/无 NaN/Inf）。

## 4. 数据集与口径

- **主评测集**：1000 张唯一 ID（NSD new_test，与 MindEye2 官方 1000 评测集同源）。
- **敏感性口径**：982 张 = 同一固定候选池前 982 列（与项目既有 shared982 一致）；
  逐对 Pearson 不受行删除影响，等价于独立 982 相关矩阵。

## 5. 模型、checkpoint、参数与环境

| 项 | 值 | 来源/SHA256 |
|---|---|---|
| 第一阶段模型 | TextAlign S1（ykt668/textalign-mindeye2-model） | masterGit 分支，commit a127f929（历史）；实际核实见归档报告 |
| SDXL | Zavy SDXL base（Zavychromaxl v30） | pscotti/mindeyev2 快照 3f20ccbe |
| unCLIP/VAE | unclip6_epoch0_step110000.ckpt、bigG_to_L_epoch8.pth | 同上 |
| CLIP-L（评价） | openai/clip-vit-large-patch14 @ 32bd6428 | 与 OpenAI ViT-L-14.pt（b8cca3fd...）320 张量逐位一致、16 图输出 max_abs=0 |
| SwAV（评价） | swav_800ep_pretrain.pth.tar | 官方 dl.fbaipublicfiles.com，SHA256 029d0b8c...cfd1eb |
| bigG（图像特征） | laion/CLIP-ViT-bigG-14-laion2B-39B-b160k | 特征已导出并归档（HF） |
| 推理环境 | A800-40GB、torch 2.5.0+cu124 | `GPU_ENV_LOCK.txt` 逐包锁定 |
| 评价环境 | CPU 4C，torch 2.5.0+cu124(CPU)、transformers 4.46.3、numpy 2.5.4、scipy 1.17.1 | `evaluation/cpu_mindeye2_p3_clip_swav/model_manifest.json` |

## 6. 推理时间统计方法

- 第一阶段：进程内计量 `elapsed_s`，**5411.0 s = 90.2 min**（14:56:44 启动 → 16:27:05 落盘）；
- 第二阶段：墙钟 **31.1 min**（16:28:41 → 16:59:48；≈1690 s 推理 + ~180 s 加载）；
- **合计 121.3 min**（90.2 + 31.1）；
- 舍入说明：`reports/FINAL_1000_INFERENCE_REPORT.md` 第 62 行写作"121.5 分钟"，为该报告的
  算术/舍入不一致（90.2+31.1=121.3）；以原始计量值（5411.0 s、31.1 min）为准，
  本文采用 **~121.3 min**，未修改历史报告；
- 单张成本：第一阶段 ≈5.26 s/张；第二阶段 ≈1.72 s/张。

## 7. CLIP-L 2-way Identification 评价协议（官方口径）

MindEye2 `final_evaluations.ipynb` **Cell 17** `two_way_identification`（MindEyeV2 Cell 14 逐字一致）：

```python
r = np.corrcoef(reals, preds)[:N, N:]
c_j = np.sum(r < np.diag(r)[:, None], 0)     # 正确目标胜过的候选数
perf = np.mean(c_j) / (N - 1)                # = 平均 Pairwise 2-way Identification
```

- 编码器：`clip.load("ViT-L/14").encode_image`（官方 cell 27）。本项目使用 HF
  `openai/clip-vit-large-patch14` 视觉塔，**经 320 张量权重映射 + 16 图输出对比证明与
  OpenAI 官方逐位等价**（max_abs_error = 0.0，cosine_min = 0.99999976）。
- 预处理：cell 8 `Resize((256,256))`（bilinear, antialias）→ cell 27 `Resize(224, BILINEAR)`
  + CLIP mean/std，**无 CenterCrop**（官方覆盖 clip 包默认 bicubic+CenterCrop）。
- 特征：encode_image 最终层（768 维，含 visual_projection）。
- **命名纪律**：该指标是平均 Pairwise 2-way Identification，**不是** Strict N-way Top-1
  （补充口径，单独报告），也**不是**1000-way 分类准确率。
- 禁用替代：laion/CLIP-ViT-L-14-laion2B-s32B-b82K 不属于本协议。

## 8. 统计检验与显著性

- 配对设计：同一 1000 张、同一候选池、同一顺序（REFINEMENT_EVALUATION_MANIFEST 逐行核对）。
- 主统计：Wilcoxon signed-rank（log10 下界表述）；Bootstrap 10k 95% CI（重采样单位 =
  逐样本 delta；候选池固定，CI 不外推新刺激）。
- 多重比较：Holm-Bonferroni，校正后仍显著（log10 p = −5.29）。
- 敏感性：982 口径方向一致。
- 诚实性限定：共享候选池 → 逐样本检验为描述性；20×300 随机子池高度重叠（平均重叠 90），
  仅作敏感性分析，不作为独立重复实验证据。

## 9. 完整实验结果

### 9.1 主结果（1000 张，已确认口径）

| Method | Caption-conditioned refinement | Inference time (1000 images) | CLIP-L 2-way |
|---|---|---:|---:|
| ConceptAlign | No | 90.2 min | **93.19%** |
| ConceptAlign + GIT-SDXL | Yes | ~121.3 min | **91.65%** |

- Δ（B−A）= **−0.0154**（−1.54 pp）；95% CI **[−0.0239, −0.0072]**（Bootstrap 10k）；
  Wilcoxon log10 p = −5.29；**Holm 校正后仍显著**；
- 982 敏感性：Δ = −0.0150（0.9311 → 0.9161），方向一致；
- 改善/变差/持平：404 / 515 / 81。

### 9.2 配套官方指标（同协议、同批图像）

| 指标 | A | B | Δ | 显著性 |
|---|---|---|---|---|
| SwAV correlation distance ↓（官方） | 0.3505±0.098 | 0.3633±0.109 | +0.0127（更差） | Holm ✅（log10 p −8.18） |
| Strict N-way Top-1（CLIP-L，补充） | 0.130 | 0.113 | −0.017 | 不显著（log10 p −0.63） |
| Mean Rank（CLIP-L，补充） | 69.07 | 84.43 | +15.35 | 描述性 |
| SwAV 2-way（新增补充） | 0.9811 | 0.9734 | −0.0077 | log10 p −11.76（描述性） |

### 9.3 其它既有关键结果（背景参照）

- 低层质量：SSIM 0.3762→0.4115（↑989/1000）、PixCorr +0.0058（P1，Holm 显著）；
- 高层语义（P1）：AlexNet-2/5、InceptionV3 2-way 全部显著下降；EffNet-B1 距离上升；
- bigG 自定义口径：image-image 相似度 0.3452→0.3384、1000-way 检索 Top-1 53.1%→42.9%；
- Caption 指标（text-text）与 A/B 无关（共享同一批 predcaptions）。


## 10. 结果解释与论文结论

**论文结论（准确表述）**：反事实语义约束使 ConceptAlign 在**不增加额外
Caption-conditioned SDXL refinement** 的情况下，仍能获得较好的语义对应能力
（CLIP-L 2-way 93.19%）；额外精炼增加约 31.1 min（+34.5%）推理成本，但语义对应
**没有改善反而显著下降**（−1.54 pp，CI 全为负）。

**机制**：B 组 prompt adherence 上升（+0.0047）证明文本条件确实生效，但条件文本是
GIT 预测 caption（非真值）；SDXL img2img（timepoint=13, steps=25）同时引入强图像先验与
平滑，把图像推向"caption 语义 + SDXL 先验"，偏离真实刺激。

**归因边界**：本实验比较的是"增加整个 SDXL refinement 流程"的效果；下降不能完全归因于
caption 文本本身（区分 caption 与 SDXL 先验需 Empty Prompt 对照，本轮未执行，
见 §14 与 `reports/P3_FOLLOWUP_EXPERIMENTS_PREPARATION.md`）。

## 11. 实验局限性

1. 单一受试者（NSD subj01）、单一生成模型、单组超参（timepoint=13 等）；
2. A 组 224px / B 组 256px 在评价前按官方协议统一 Resize 256→224（消除尺寸混杂）；
   B 组原生分辨率更高、理论上采样损失更小，对 B 是有利偏差——结论方向不受影响；
3. 共享候选池使逐样本检验为描述性；20×300 池只作敏感性；
4. 未做 Empty Prompt / GT-caption 上界对照，无法完全分离 caption 与 SDXL 先验贡献；
5. CLIP-L 2-way 度量语义对应，不代表低层保真（SSIM/PixCorr 方向相反，须并列报告）。

## 12. 代码、数据、模型来源与版本

- 官方代码：MindEye2/TextAlign（GitHub 公开仓库），本地快照 `source/TextAlign-mindeye2-231850a`
  = archive/final14@231850a；
- 本项目代码：`scripts/`（P1/P2/P3 评价与统计、归档脚本）→ GitHub 归档分支
  `archive/git-sdxl-refinement-final-2026-11-17`；
- 数据：NSD new_test 1000 图（受 NSD 协议约束，不重分发；经官方 pscotti/mindeyev2 获取）；
- 权重：S1 checkpoint（ykt668/textalign-mindeye2-model @ masterGit/a127f929）、
  SDXL/VAE/bigG（pscotti/mindeyev2 @ 3f20ccbe）、CLIP-L（openai @ 32bd6428）、
  SwAV（fbaipublicfiles，029d0b8c...）；
- 环境：`GPU_ENV_LOCK.txt` + `evaluation/cpu_mindeye2_p3_clip_swav/model_manifest.json`。

## 13. 文件位置、SHA256 与恢复方法

- **正式输出（HF Dataset 私有）**：`supplementary/git_sdxl_refinement/2026-11-17/`
  下 `stage1/`、`stage2/`（七个 .pt + captions + ids + verify JSON）；
- **评价结果**：`evaluation/cpu_mindeye2_p1|p2|p3*/`（小型汇总同步 GitHub；逐样本与
  特征在 HF）；
- **恢复方法**：
  1. `git clone -b archive/git-sdxl-refinement-final-2026-11-17 https://github.com/YKT-668/TextAlign-mindeye2`
  2. `hf download ykt668/textalign-mindeye2-data --repo-type dataset --include "supplementary/git_sdxl_refinement/2026-11-17/*"`
     （私有仓库需 HF token）；
  3. 校验：对照 `reports/FINAL_ARCHIVE_MANIFEST.csv` 逐文件 SHA256；
  4. 复现评价：`python scripts/p3_clipl_extract_full.py && python scripts/p3_metrics_and_stats.py`
     （CPU ≈76 min）；
  5. 详细步骤见 `reports/FINAL_REPRODUCIBILITY_GUIDE.md`。

## 14. 本轮未开展的实验及原因

| 实验 | 原因 |
|---|---|
| Empty Prompt 对照 / GT-caption 上界 | 收尾轮禁止新实验；方案已备好（P3_FOLLOWUP），建议下次 24GB GPU 会话执行 |
| GNet Brain Correlation | 资产已确认可获取（1.27GB + masks 均在 pscotti/mindeyev2），计算待下次 GPU 会话 |
| 人工盲测 | 材料就绪但无人评分；执行说明已归档 |
| 982 独立重推理 / 重训练 / 超参搜索 | 不改变结论方向，成本高 |
| 修改历史报告中的"121.5" | 历史文件不修改；本文 §6 已注明以原始计量为准（121.3 min） |

