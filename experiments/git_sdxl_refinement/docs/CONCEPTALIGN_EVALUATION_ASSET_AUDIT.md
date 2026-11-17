# ConceptAlign 既有评测体系与资产审计

目标：判断"哪些已完成、哪些可直接复用、哪些缺、哪些与 img2img 无关"。
可信度标注：VERIFIED / INFERRED / BLOCKED。

## 1. 已完成的评测结果（VERIFIED，直接来自仓库内产物）

### 1.1 主结果审计（`audit_runs/s1_textalign_stage1_FINAL_BEST_32_20260115_004557/audit_report_final.md`）

协议：**shared982（Strict）**，IS-RSA: SKIP，bootstrap 一致性 PASS。

| 指标 | Point Est | Boot Mean | Diff | 95% CI |
|---|---|---|---|---|
| **982-way FWD@1** | **0.24644** | — | — | — |
| 982-way FWD@5 | 0.58554 | — | — | — |
| 982-way BWD@1 | 0.13340 | — | — | — |
| 982-way BWD@5 | 0.35234 | — | — | — |
| **300-way × 30 FWD@1** | 0.43444 ± 0.02438 | — | — | — |
| 300-way × 30 BWD@1 | 0.25822 ± 0.02095 | — | — | — |
| **2AFC_B2I** | 0.99047 | 0.98948 | 0.00099 | [0.98824, 0.99064] |
| 2AFC_I2B | 0.97845 | 0.97748 | 0.00096 | [0.97464, 0.98010] |
| 2AFC_RandomCheck | 0.50543 | — | — | — |
| **RSA_Pearson** | 0.35994 | 0.37868 | 0.01875 | [0.35701, 0.40165] |
| RSA_Spearman | 0.26074 | 0.26326 | 0.00253 | [0.24135, 0.28661] |
| **CCD_Hard_K1** | 0.56823 | 0.56741 | 0.00082 | [0.53666, 0.60082] |
| CCD_Hard_K2 | 0.40428 | 0.40321 | 0.00107 | [0.37373, 0.43381] |
| CCD_Random_K2 | 0.94908 | 0.94891 | 0.00018 | [0.93483, 0.96232] |

证据资产：`brain_clip.pt`（= `clipvoxels` 同源）、`ids.json`、`evals/all_images_bigG_1664_mean.pt`（GT bigG 均值特征）。
CCD 修复说明：bootstrap 改为"全局固定负样本池 → 逐图 0/1 正确性 → 再 bootstrap"，与点估计一致（Diff<0.002）。

### 1.2 MindEye2 官方格式表（`tables/`）

| 文件 | 行数 | 内容 | 可信度 |
|---|---|---|---|
| `s1_textalign_stage1_FINAL_BEST_32_all_enhancedrecons.csv` | 16 值（无表头名） | 顺序与官方 cell 38 一致：PixCorr 0.3930 / SSIM 0.4447 / AlexNet(2) 0.9769 / AlexNet(5) 0.9925 / Inception 0.9632 / CLIP 0.9312 / EffNet-B 0.6257 / SwAV 0.3205 / FwdRetr 0.9912 / BwdRetr 0.9906 / BrainCorr nsd_general 0.0088 / V1 0.0103 / V2 0.0094 / V3 0.0073 / V4 0.0049 / higher_vis 0.0046 | **INFERRED**（有产物，但未记录样本数/是否融合/是否真加载 GNet） |
| `..._caption_metrics.csv` | 3 值 | 不完整（官方应为 24 项） | **BLOCKED**（不完整） |

> ⚠️ **该 16 值表存在两处需要复核的地方**：① BrainCorr 仅 0.005–0.010，明显低于 MindEye2 论文量级（0.2–0.3），可能是 GNet 未正确加载或单位/掩码问题；② 未记录是 1000 还是 982 样本、是否执行 `.75/.25` 融合。**不得直接引用为正式结果**。

## 2. 可复用的评测工具（VERIFIED：文件存在）

| 脚本 | 作用 |
|---|---|
| `eval/hard_retrieval.py` | 难样本检索 |
| `tools/eval_ccd_embed.py` | CCD（含 **margin** mean/median/std + bootstrap CI） |
| `tools/eval_rsa_embed.py` | RSA（Pearson/Spearman） |
| `tools/rerun_all_rsa_shared982.py` / `rerun_all_twoafc_shared982.py` / `rerun_all_ccd_shared982.py` | shared982 协议重跑 |
| `tools/run_ccd_ablation.py`、`tools/rerun_ccd_with_hardneg_mask.py` | CCD 消融/掩码 |
| `tools/run_efficiency_pipeline.py` + `make_efficiency_figures.py` | 数据效率实验（figures 已存在） |
| `tools/build_main_results_table.py` | 汇总主表 |

## 3. 覆盖率判定

| 类别 | 状态 | 说明 |
|---|---|---|
| **L1：982-way FWD/BWD** | ✅ 已有 | shared982；与 A/B 无关（用 clipvoxels） |
| **L1：2AFC** | ✅ 已有 | B2I 0.9905 / I2B 0.9784 |
| **L2：CCD-H K=1/2** | ✅ 已有 | K1 0.568 / K2 0.404，含 random K2 |
| **L2：CCD K=4/8** | ❌ 未见 | 仓库内无 K>2 的实现/结果 |
| **L2：Object/Attribute/Relation 分解** | ❌ 未见 | 无对应脚本/结果 |
| **L2：Δ-margin** | 🟡 代码有 | `eval_ccd_embed.py` 计算 margin，但报告中未单独列 |
| **L2：Hard Retrieval M=50/100** | 🟡 部分 | `hard_retrieval.py` 存在；结果未在报告中系统列出 |
| **L3：RSA** | ✅ 已有 | Pearson 0.360 / Spearman 0.261 |
| **数据效率** | 🟡 图存在 | `Fig_efficiency_*` 已生成，原始数值需回溯 |
| **Cross-LLM** | 🟡 图存在 | `Fig_isrsa_heatmap_textalign_llm_v2` 已生成，评测数值需回溯 |
| **Human evaluation** | ❌ 未见 | 本轮已生成 img2img 盲测材料（另一件事） |

## 4. 与 A/B 精炼实验的关系（关键判断）

| 指标族 | 依赖 | A/B 是否可能变化 | 结论 |
|---|---|---|---|
| 2AFC / RSA / CCD / Fwd-Bwd Retrieval | **clipvoxels / brain_clip**（脑表征） | **不会变** | **不得**用于证明精炼效果；A/B 共用同一份结果 |
| Caption 指标（METEOR/ROUGE/Sentence/CLIP-B/L） | predcaptions（文本） | **不会变** | 同上 |
| PixCorr / SSIM / AlexNet / Inception / CLIP / EffNet / SwAV / BrainCorr | **生成图像** | **会变** | 这些才是 A/B 的合法指标 |

> 这正是任务书里强调的：**不要用 RSA/CCD 等脑表征指标重复计算 A/B 效果**。

## 5. 样本口径（必须区分）

| 口径 | 数量 | 用途 |
|---|---|---|
| **1000**（全量 new_test 唯一 ID） | 1000 | 本轮 A/B 精炼评测 |
| **shared982** | 982 | ConceptAlign 既有 L1/L2/L3 结果（与论文表对齐） |

两者存在 18 张的差集（前 982 为 shared982，后 18 为额外），**任何跨口径比较必须显式声明**。

---

## 6. 阶段 F：人工盲测材料检查（VERIFIED）

目录：`evaluation/human_blind_review/`

| 检查项 | 结果 |
|---|---|
| 组数 | **150 组**（left/right/gt 各 150 张，共 450 张） |
| 文件总数 | 452（450 图 + `review_sheet.csv` + `ANSWER_KEY_...csv`） |
| `review_sheet.csv` 行数 | 150 ✅ |
| `ANSWER_KEY` 行数 | 150 ✅ |
| 抽样编号一致 | ✅ sheet 与 key 的 sample/ID 完全对应 |
| ID 唯一性 | ✅ 150 个唯一 ID |
| 左右随机化 | ✅ `left_is` 分布 raw=73 / refined=77（seed=2027） |
| 方法名泄露（文件名） | ✅ **0**（无 raw/refined/unclip/img2img/stage2 字样） |
| 方法名泄露（表头） | ✅ 无 raw/refined 列 |
| 评分列是否被预填 | ✅ `preference` 全空（未编造任何评分） |
| 图像三元组完整性 | ✅ 缺失 0 |
| 答案与评价表分离 | ✅ ANSWER_KEY 独立文件，文件名自带"DO_NOT_OPEN_BEFORE_RATING" |

**判定：材料具备正式发放条件**（可直接发给评分者，评价完成后再打开答案键）。
本轮**未打开**答案键进行任何推断，也未编造评分。

