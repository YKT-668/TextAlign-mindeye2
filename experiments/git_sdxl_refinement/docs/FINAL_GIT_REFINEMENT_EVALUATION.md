# GIT 文本引导精炼 A/B 对照评测（1000 张）

日期：2026-11-15 ｜ 机器：A800-SXM4-40GB ｜ 仅做推理后处理的图像级配对对照
（**不是**训练阶段，也**没有**改变第一阶段的脑表征或 decoded visual tokens）

## 结论

### **`REFINEMENT_MIXED_RESULTS`**

**在"与真实图像的语义一致度"上没有任何提升，反而是显著下降；但在"与自身预测 caption 的一致度"上显著上升。**

| 指标（1000 张配对） | A 组（raw，unCLIP） | B 组（+GIT caption → Zayv img2img） | Δ = B − A | 95% CI | 显著 |
|---|---|---|---|---|---|
| CLIP 图像-图像 vs GT（mean） | **0.34516** | 0.33842 | **−0.00674** | [−0.00841, −0.00509] | ✅ 负 |
| 同上（median） | 0.33955 | 0.32873 | −0.00772 | — | — |
| CLIP 图像-文本 vs **GT caption** | **0.20451** | 0.18434 | **−0.02017** | [−0.02247, −0.01791] | ✅ 负 |
| 图像 vs **预测 caption**（prompt adherence，仅辅助） | — | — | **+0.00469** | [+0.00288, +0.00649] | ✅ 正 |
| 1000-way 检索 Top-1（gen→GT） | **53.1%** | 42.9% | **−10.2 pp** | — | — |
| 1000-way 检索 Top-5（gen→GT） | **78.3%** | 68.7% | −9.6 pp | — | — |
| 1000-way 检索 MRR（gen→GT） | **0.6426** | 0.5442 | −0.0984 | — | — |
| 中位排名（gen→GT） | **1** | 2 | — | — | — |

> 逐样本计数（图像-图像）：**364 改善 / 636 变差 / 0 持平**；Wilcoxon 符号秩 **p = 3.71e-17**
> 逐样本计数（图像-文本）：305 改善 / 695 变差；p ≈ 0

**解释（机制层面，可核对）**：`prompt adherence` 上升说明**文本条件确实生效**；但条件文本是 **GIT 预测 caption（并非真值）**，且 SDXL img2img 在 `timepoint=13 / steps=25` 下会强烈引入模型的图像先验与平滑——结果是图像被推向"captions 的语义 + SDXL 先验"，**离真实刺激更远**。因此这不是"精炼没生效"，而是"精炼生效了，但把误差放大了"。

---

## 1. 数据冻结与完整性检查

| 检查 | 结果 |
|---|---|
| A 组 `enhancedrecons` shape | (1000, 3, 224, 224) fp16 ✅ |
| B 组 `stage2_img2img` shape | (1000, 3, 256, 256) fp32 ✅ |
| GT `all_images` shape | (1000, 3, 224, 224) fp32 ✅ |
| 唯一 ID | 1000 ✅ |
| A/B ID 逐行完全一致 | ✅ `torch.equal == True` |
| ID 升序（np.unique 顺序） | ✅ |
| captions（GT / 预测） | 各 1000 条，0 空 ✅ |
| 数值范围 / 有限性 | 均在 [0,1]、无 NaN/Inf ✅ |
| 未使用 `blurryrecons` 作为 A 组基线 | ✅（A = `enhancedrecons`；blurry 仅用于 GT 顺序校验） |
| 使用完整 1000 行（非 976 中间快照） | ✅（976 仅为进程内的安全快照，正式产物是 1000 行） |
| 图像可正常读取 | ✅ 3000 张 PNG 全部导出可读 |
| 输入只读 | ✅ 未修改/删除任何正式产物 |

## 2. 对齐审计（本轮从 INFERRED 升级为 VERIFIED）

### 2.1 GT 图像 ↔ ID 顺序：**VERIFIED（像素级证据）**

模型从脑信号回归的 `blurryrecons` 与该图像的 GT 应显著相关；若顺序错位则相关性塌陷为随机。

| 量 | 值 |
|---|---|
| 配对 pixcorr（blurry_i vs GT_i）均值 | **0.4432** |
| 配对 pixcorr 中位数 | 0.4687 |
| 打乱配对（同一批 GT 随机置换）均值 | **0.0705** |
| 配对为正的比例 | **95.6%** |

→ 配对相关性是随机配对的 6.3 倍，且 95.6% 样本为正 → **GT 图像顺序与 ID 一一对应（VERIFIED）**。

### 2.2 GT Caption ↔ GT 图像：**VERIFIED（CLIP 证据）**

用 `openai/clip-vit-large-patch14` 计算 GT 图像与 GT caption 的相似度：

| 量 | 值 |
|---|---|
| 配对相似度均值（对角） | **0.2613** |
| 非配对均值（999,000 对） | **0.0841** |
| 配对中位数 | 0.2610 |

→ 对角显著高于非对角（3.1×）→ **caption 与 GT 图像逐行对应（VERIFIED）**。

> 因此第三组指标（image-text）**可以**作为正式指标使用，无需标记 BLOCKED_MAPPING。
> 证据文件：`evaluation/eval_summary.json` 的 `gt_image_order` / `gt_caption_order` 字段；脚本 `scripts/eval_git_refinement.py`。

---

## 3. 指标一：CLIP 图像语义相似度（图像-图像）

**编码器与预处理完全复用项目原评测协议**：`FrozenOpenCLIPImageEmbedder(arch="ViT-bigG-14", version="laion2b_s39b_b160k", output_tokens=True, only_tokens=True)`（见 `final_evaluations.ipynb` cell 375），token 展平后按 `utils.batchwise_cosine_similarity` 方式计算；A / B / GT **同一编码器、同一预处理**、分批（batch=20）提取。

> 协议细节：与项目原用法一致——直接把 [0,1] 的张量送入该 embedder（其内部 `preprocess` 做 resize(224,bicubic)+`(x+1)/2`+CLIP 归一化），**A/B/GT 完全同构**，因此配对比较不受该约定影响。

| 组 | mean | median | std |
|---|---|---|---|
| A raw | **0.34516** | 0.33955 | 0.04753 |
| B refined | 0.33842 | 0.32873 | 0.05353 |

配对差 `delta = refined − raw`：mean **−0.00674**，median **−0.00681**，95% CI **[−0.00841, −0.00509]** → **显著为负**。

分布图：`evaluation/fig_delta_hist_image.png`、`evaluation/fig_scatter_raw_vs_refined.png`

## 4. 指标二：1000-way 图像检索

检索库 = 同一批 1000 张 GT（未改变候选集合）；同一编码器与预处理。

| 方向 | 组 | Top-1 | Top-5 | MRR | Median rank |
|---|---|---|---|---|---|
| 生成 → GT | **A raw** | **0.5310** | **0.7830** | **0.6426** | **1** |
| 生成 → GT | B refined | 0.4290 | 0.6870 | 0.5442 | 2 |
| GT → 生成 | **A raw** | **0.5640** | **0.8150** | **0.6768** | **1** |
| GT → 生成 | B refined | 0.4860 | 0.7200 | 0.5944 | 2 |

随机基线 Top-1 = 0.001（1/1000）。CDF 图：`evaluation/fig_retrieval_cdf.png`。
→ **两个方向、全部四个检索指标一致下降**。

## 5. 指标三：Image-Text 相似度 + Prompt Adherence

文本编码：`openai/clip-vit-large-patch14`（`CLIPModel.get_text_features`，与 `final_evaluations.ipynb` 的图文协议一致）。

| 比较对象 | A raw | B refined | Δ | 95% CI |
|---|---|---|---|---|
| vs **GT caption**（正式语义指标） | **0.20451** | 0.18434 | **−0.02017** | [−0.02247, −0.01791] |
| vs **预测 caption**（prompt adherence，**仅辅助**） | 基线 | — | **+0.00469** | [+0.00288, +0.00649] |

> 严格区分：**GIT 预测 caption 是第二阶段的输入条件，不是标签**。因此上表第二行**不能**当作语义准确度，只用于说明"条件确实被注入并生效"。

## 6. 统计分析

- 全部指标基于**同一组 1000 个配对样本**
- Bootstrap **10,000** 次，**seed = 2027**，95% **percentile** CI
- 补充：Wilcoxon 符号秩检验

| 指标 | 平均提升 | 中位数提升 | 95% CI | 改善/持平/下降 | 是否跨 0 | Wilcoxon p |
|---|---|---|---|---|---|---|
| 图像-图像 | −0.00674 | −0.00681 | [−0.00841, −0.00509] | 364 / 0 / 636 | 否（全负） | 3.7e-17 |
| 图像-文本（GT cap） | −0.02017 | −0.01597 | [−0.02247, −0.01791] | 305 / 0 / 695 | 否（全负） | ≈0 |
| Prompt adherence | +0.00469 | +0.00277 | [+0.00288, +0.00649] | 539 / 0 / 461 | 否（全正） | — |

**没有任何一个 GT 语义指标的置信区间跨 0**，因此结论不是"证据不足"，而是"**有统计显著证据表明精炼降低了 GT 语义一致度**"。

## 7. 视觉质量与人工盲测

- 未引入新依赖做 LPIPS（按要求不新增大量依赖；如需可后续单独启用）
- 已生成 **150 组**盲测材料：`evaluation/human_blind_review/`
  - 每组：`sample_XXXX_gt.png`（GT）+ `sample_XXXX_left.png` / `sample_XXXX_right.png`
  - A/B 左右位置**按 seed=2027 随机**，文件名不含方法名
  - `review_sheet.csv`：含 `id`、左右图文件名，以及待填列（subject / attributes / relations / errors / overall_quality / preference）
  - `ANSWER_KEY_DO_NOT_OPEN_BEFORE_RATING.csv`：独立存放左右对应关系，**评分完成前不要打开**
- **未编造任何人工评价结果**，等待人工填写

## 8. 推理时间审计

| 阶段 | 墙钟时间 | 进程内计量 | 单张耗时 | 构成 |
|---|---|---|---|---|
| 第一阶段（BrainNetwork + Prior + GIT + unCLIP） | 14:56:44 → 16:27:05（**90.2 min**） | `elapsed_s = 5411.0` | **5.26 s/张**（含 ~152s 加载） | 见下 |
| 第二阶段（SDXL img2img） | 16:28:41 → 16:59:48（**31.1 min**） | tqdm 1.72 s/it | **1.72 s/张**（含 ~180s 加载≈1.69 s/张） | 单步去噪 25 步 → 约 33 步/1000 图 |
| **完整流程** | **121.5 min** | — | **7.2 s/张** | — |

**精炼的额外成本**：**+31.1 分钟 / 1000 张（+34.4% 墙钟）**，即 **+1.72 s/张**（相对第一阶段 +32.7%）。

> **无法拆分** BrainNetwork/Prior、GIT caption、unCLIP 的各自耗时：第一阶段脚本**没有分组件计时器**，日志里只有整体进度条。按要求**不以估算代替实测**，此处明确标记 `COMPONENT_BREAKDOWN = NOT_INSTRUMENTED`。

---

## 9. 结果解释与限制

**为什么精炼会降低 GT 语义一致度？**

1. **条件信号的准确性上限**：第一阶段 GIT caption 本身并不完全正确（cap 与 GT caption 的 CLIP 相似度均值仅 0.20 量级）。img2img 忠实执行该条件 → 把图像推向"caption 描述的内容"，而 caption 偏差会被放大。
2. **SDXL 先验注入**：`img2img_timepoint=13`（约保留一半噪声步长）+ `guider.scale=5` + 25 步，使 SDXL 的生成先验替代了 unCLIP 保留下来的低频结构；`256→768→256` 的重采样还额外丢失细节。
3. **证据方向一致**：`prompt adherence ↑` 与 `GT 语义 ↓` 同时出现，正是"条件生效但条件不准"的典型signature，而不是随机波动（p=3.7e-17）。

**限制（如实说明）**

- 这是**图像级后处理消融**，不涉及脑表征；不能据此推断 MindEye2/TextAlign 的表征能力变化。
- 单一超参数设定（`timepoint=13, scale=5, steps=25`）下的结论；未做超参扫描（本轮要求不重新生成）。
- 编码器为项目原用的 bigG token 相似度与 CLIP-L 图文相似度；换用其它指标（LPIPS/SSIM/DINO）可能给出不同幅度，但本次未引入新依赖。
- 人工盲测**尚未评分**，因此本报告的结论仅基于自动指标。

## 10. 交付物清单

| 类别 | 路径 |
|---|---|
| A/B/GT 配对 manifest（1000 行） | `reports/REFINEMENT_EVALUATION_MANIFEST.csv` |
| 逐样本指标 CSV（1000 行） | `evaluation/per_sample_metrics.csv` |
| 统计/对齐汇总 JSON | `evaluation/eval_summary.json` |
| 图像相似度分布图 | `evaluation/fig_delta_hist_image.png` |
| A vs B 散点图 | `evaluation/fig_scatter_raw_vs_refined.png` |
| 检索 CDF 图 | `evaluation/fig_retrieval_cdf.png` |
| 导出图像（各 1000 张） | `evaluation/images/{raw,refined,gt}/` |
| 人工盲测（150 组 452 文件） | `evaluation/human_blind_review/`（含 `review_sheet.csv` 与独立 `ANSWER_KEY_DO_NOT_OPEN_BEFORE_RATING.csv`） |
| 评测脚本 | `scripts/eval_git_refinement.py` |
| 运行日志 | `logs/eval_refinement.log` |
| SHA256 清单 | `reports/EVALUATION_SHA256.sha256` |

**SHA256（关键文件）**

```
5dc5eb4220f65b5c0ec1c06387a3a105770ef30a2e2353adb2c91343ccdfb1a2  reports/REFINEMENT_EVALUATION_MANIFEST.csv
57aa2ba128f23ffde780b56591e70f2a89f07defd5e15a2fd4578a54338ce6db  evaluation/per_sample_metrics.csv
e84f9f7137b5ca412b0726846e70fc7d6749dd794913b46f0a8e1420883920f9  evaluation/eval_summary.json
4deb4136e8bd04aaecea9ac8b132068dfa7f5f739e72db4753aae443d2db2d70  evaluation/fig_delta_hist_image.png
4c4e124f3263cfda8035c1215b712351aa01f55a741b6c3575f97246d484ab06  evaluation/fig_scatter_raw_vs_refined.png
c78c873f20ce4f86227fec500892058db1bbe8f73dfb668d86d993950c1f087b  evaluation/fig_retrieval_cdf.png
f2ff6a6d77eb4df4dbfe40b39077731779846357a8ec39b7791b95587a696a6d  evaluation/human_blind_review/review_sheet.csv
1e2759852c39e902be84bd16626edc417ddaff60e67b398d01d4f50a34f2ff64  evaluation/human_blind_review/ANSWER_KEY_DO_NOT_OPEN_BEFORE_RATING.csv
0abced29aca343abc8b8150a802e3e8794def8daf9f64b3f5fbbbe88f0dc10a1  scripts/eval_git_refinement.py
```

## 11. 最终结论标签

**`REFINEMENT_MIXED_RESULTS`**

- **GT 语义准确度（主指标）：显著下降** —— 图像-图像 −0.0067（p=3.7e-17）、图像-文本 −0.0202、检索 Top-1 −10.2pp
- **Prompt adherence（辅助）：显著上升** —— +0.0047（p<0.01）
- **时间成本：+31.1 分钟 / 1000 张（+34.4%）**

**给论文的直接结论**：在当前 checkpoint 与固定超参下，**GIT caption 引导的 SDXL img2img 精炼不能提升脑解码图像的语义准确度，反而显著降低；其实际作用是提高与预测 caption 的一致性**。若要提升语义指标，应先提升 caption 准确度或降低 img2img 强度（更小的 `steps`/更大的 `timepoint`），并用本套脚本做配对复测。


