# FINAL：指标命名统一审查与论文影响审计（只读审查，未修改任何历史文件）

日期：2026-11-16 ｜ 依据：`final_evaluations.ipynb` Cell 17/27 逐字核对 + 
`evaluation/p1_protocol_final_audit/protocol_audit.json` 数值复算 + `FINAL_P1_PROTOCOL_AND_STATISTICAL_AUDIT.md`

## 0. 核心结论

官方 `two_way_identification`（Cell 17）= **平均 Pairwise 2-way Identification**。
此前 P1 报告中的 "N-way acc（96%–99%）" 命名**不准确**：同数据下 Strict N-way Top-1 仅
0.17–0.52，二者是不同指标。96%–99% 数值本身**正确**，只需改名。

## 1. 统一指标字典

已生成：**`reports/FINAL_EVALUATION_METRIC_DICTIONARY.csv`**（24 项指标：英文名/中文名/
计算方式/方向/是否官方/是否新增/结果文件/论文推荐名）。命名三条铁律：
1. `2-way Identification`（官方）≠ `Strict N-way Top-1`（补充）≠ `Mean Rank`（补充）；
2. SwAV 的官方指标是 **correlation distance**；SwAV 的 2-way/Top-1 是我们新增的补充口径；
3. CLIP-L 图像指标（P3）与 CLIP-L 文本指标（P2 caption）同权重不同用途，不可互称。

## 2. 历史报告问题清单（只读审查结果）

| # | 文件 | 位置/字段 | 原文/问题 | 建议替换 |
|---|---|---|---|---|
| 1 | `CPU_P1_MINDEYE2_EVALUATION_REPORT.md` | §2 表头 "AlexNet(2) N-way acc"、"InceptionV3 N-way acc"、正文 "N-way 识别准确率" | 把官方 2-way 口径称为 N-way | "AlexNet(2) 2-way Identification（mean pairwise，官方口径）" |
| 2 | `evaluation/cpu_mindeye2_p1/p1_accuracy_summary.csv` | 列名 `acc_N1000` / `acc_N982` | "acc_N" 暗示 N-way Top-1 | `acc_2way_N1000` / `acc_2way_N982`（新列名重导出，原文件保留） |
| 3 | `evaluation/cpu_mindeye2_p1/stats.json` 等 | 识别类 delta 的表述 | 与 #1 同源 | 同 #1 |
| 4 | `REFINEMENT_EVALUATION_MATRIX.csv` | 行 4–6 备注 "N-way via np.corrcoef" | 同 #1 | "2-way Identification via np.corrcoef (official cell 17)" |
| 5 | `MINDEYE2_OFFICIAL_EVALUATION_AUDIT.md` §0-C | ""2-way Identification" 实际是 N-way（N=1000）" | 表述误导：它确为 2-way 口径（对全体候选取平均的 pairwise 期望），只是在 N×N 矩阵上计算 | ""2-way Identification 在 N×N Pearson 矩阵上计算，等价于对 N-1 个候选取平均的两两 2AFC 期望；不是 Strict Top-1"" |
| 6 | `CPU_P1_STATISTICAL_AUDIT.md` §4/§8 | "以池层面（20×300）重采样结果为准"、"20/20 显著" | 20 个池高度重叠（均重叠 90），非独立重复实验 | "池层面结果为候选池敏感性分析（20/20 方向一致），不作为独立显著性证据" |
| 7 | `evaluation/cpu_mindeye2_p1_stat_audit/pool_level_audit.json` | 字段 `p_wilcoxon_pool_level` | 字段名暗示推断级 p 值 | 读取时按 sensitivity 解释（附录 A 已注明），不改历史文件 |
| 8 | `CPU_P1_MINDEYE2_EVALUATION_REPORT.md` §3 | "75/25 融合诊断组" | 已正确标注 DIAGNOSTIC_NOT_OFFICIAL | 无需修改（保持现状即可） |
| 9 | `FINAL_GIT_REFINEMENT_EVALUATION.md` | "CLIP 图像-图像相似度 / 1000-way 检索" | 属自定义 bigG-token 指标，与官方 CLIP-L 2-way 不同编码器不同公式 | 引用时加 "(custom bigG-token metric; not MindEye2 official)" |
| 10 | `CONCEPTALIGN_EVALUATION_ASSET_AUDIT.md` §1.2 | 16 值官方格式表 | 数值为 2-way 口径但表头未注明；BrainCorr 量级异常已有脚注 | 表头补 "(2-way Identification row)"；BrainCorr 行维持"不得直接引用"警示 |
| 11 | P2 caption 报告 | CLIP-L 文本相似度 | 与 P3 CLIP-L 图像指标同名不同义 | 论文中分别写 "CLIP-L text similarity (caption)" 与 "CLIP 2-way Identification (image)" |

## 3. 论文影响（final20 已冻结，只列需改位置，不直接修改论文文件）

以 final20-1（Supplement 基线）为准，需修改的位置类型：

| 类型 | 位置 | 修改建议 |
|---|---|---|
| A. 主/附表指标行名 | 凡出现 "N-way Accuracy / Acc_N" 的行 | 改为 "2-way Identification"；数值不变（0.97xx/0.94xx 等） |
| B. 方法描述 | 评价协议段落 | 采用 §1 字典中的计算方式描述；补充 "Strict N-way Top-1 (supplementary)" 仅在附表出现 |
| C. 显著性表述 | 涉及 20×300 池的句子 | 按 #6 替换为敏感性表述 |
| D. 自定义指标 | bigG 相似度/1000-way 检索 | 加 custom 标注，不与官方 2-way 并列为"官方复现" |
| E. 新增结果 | CLIP-L 2-way（P3）、SwAV distance（P3） | 进入官方口径表：CLIP-L 2-way A=…/B=…（见 P3 报告）；SwAV distance A=…/B=… |
| F. Brain Correlation | 若 final20 含该行 | 当前数值不得引用（历史 16 值表 BrainCorr 量级异常）；待资产补齐后重算 |

## 4. 边界

本审计为只读；`reports/` 与 `evaluation/` 下所有历史文件未做任何修改或删除；
所有修正以新表格/新报告呈现（本文件 + 指标字典 CSV + P1 协议终审修正表）。
