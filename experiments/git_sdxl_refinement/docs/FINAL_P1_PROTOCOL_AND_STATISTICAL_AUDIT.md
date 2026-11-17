# FINAL：P1 指标协议终审 + 统计推断终审

日期：2026-11-16 ｜ 执行：Cline ｜ 性质：只审计与轻量复算，未重跑模型、未修改既有结果
产物目录：`evaluation/p1_protocol_final_audit/`、`evaluation/cpu_mindeye2_p1_stat_audit/`（仅新增文件）

---

## 一、N-way / 2-way 协议终审（最高优先级）

### 1.1 官方源代码位置与原始公式（逐字核对，VERIFIED）

| 仓库 | 文件 | Cell | 函数 |
|---|---|---|---|
| TextAlign-mindeye2-231850a | `source/TextAlign-mindeye2-231850a/src/final_evaluations.ipynb` | **CELL 17**（27 个 code cell 之一） | `two_way_identification()` |
| MindEyeV2 | `source/MindEyeV2/src/final_evaluations.ipynb` | **CELL 14**（56 cells） | 同一函数，**逐字一致** |

原始代码（两仓库逐字节相同）：

```python
@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
    preds = model(...)   # 重建特征
    reals = model(...)   # GT 特征
    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)
    success = r < congruents            # success[i,j] = r[i,j] < r[j,j]
    success_cnt = np.sum(success, 0)    # c_j = #{i: R[i,j] < R[j,j]}
    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)   # ← 被审公式
        return perf
```

官方打印标签（cell 23/25/27）：**"2-way Percent Correct"**。调用处（cell 27，CLIP 指标）：

```python
clip_model, preprocess = clip.load("ViT-L/14", device=device)
preprocess = transforms.Compose([Resize(224, BILINEAR), Normalize(CLIP mean/std)])  # 覆盖 clip 默认
all_per_correct = two_way_identification(all_recons, all_images, clip_model.encode_image, preprocess, None)
```

### 1.2 归属判定（不凭名称推断）

**判定：该公式属于 ①平均 Pairwise 2-way Identification。**

数学论证：`acc = mean_j c_j/(N-1)`，其中 `c_j = #{i: R[i,j] < R[j,j]}` 是正确目标 j 胜过的候选数。
对均匀随机抽取的对手 i，`P(目标胜出) = c_j/(N-1)`。因此 acc 恰为
"正确目标 vs 随机对手"的 2-way 强迫选择期望正确率（对所有 N-1 个对手取平均）。

**它不是 ②Strict N-way Top-1**：Top-1 要求正确目标严格大于全部 N-1 个候选（`c_j = N-1`）。
本轮实测同数据下两者差距巨大（Top-1 仅 0.17–0.52，官方口径 0.95–0.99），见表 1.3。

**③Mean Rank：官方 notebook 未使用**（final_evaluations.ipynb 无 mean rank 输出）。
仅作为补充口径计算（恒等式 `acc = 1 − (mean_rank − 1)/(N−1)` 已数值验证，误差 < 1e-12）。

### 1.3 三口径对比（A/B × 1000/982 × 3 模型，全部基于已保存 R 矩阵复算）

**表 1：Pairwise 2-way（官方口径）= 此前报告中的 96%–99% 数值**

| 模型 | 组 | 1000 | 982 | 1000 Δ(B−A) |
|---|---|---|---|---|
| AlexNet(2) | A_raw | 0.977492 | 0.977733 | |
| AlexNet(2) | B_refined | 0.973730 | 0.973966 | **−0.003762** |
| AlexNet(5) | A_raw | 0.993159 | 0.993131 | |
| AlexNet(5) | B_refined | 0.989296 | 0.989287 | **−0.003863** |
| InceptionV3 | A_raw | 0.962539 | 0.962359 | |
| InceptionV3 | B_refined | 0.948914 | 0.948658 | **−0.013625** |

**表 2：Strict N-way Top-1（同一数据，非官方口径）与 Mean Rank（补充）**

| 模型 | 组 | Top-1 @1000 | Top-1 @982 | Mean rank @1000 | Mean rank @982 |
|---|---|---|---|---|---|
| AlexNet(2) | A_raw | 0.317000 | 0.318737 | 23.485 | 22.844 |
| AlexNet(2) | B_refined | 0.297000 | 0.298371 | 27.244 | 26.540 |
| AlexNet(5) | A_raw | 0.517000 | 0.521385 | 7.834 | 7.738 |
| AlexNet(5) | B_refined | 0.459000 | 0.461303 | 11.693 | 11.509 |
| InceptionV3 | A_raw | 0.213000 | 0.217923 | 38.424 | 37.926 |
| InceptionV3 | B_refined | 0.166000 | 0.170061 | 52.035 | 51.367 |

三种实现的交叉校验全部通过：12/12 行与既有报告值一致（<1e-6）；Top-1 的
`c_j == N-1` 实现与 `argmax_i R[i,j] == j` 实现逐样本完全一致；rank 恒等式误差 0。

### 1.4 命名修正（原报告名称不准确，生成修正版表格，不覆盖原文件）

此前 `CPU_P1_MINDEYE2_EVALUATION_REPORT.md` / `p1_accuracy_summary.csv` 中的
**"AlexNet(2) N-way acc"、"acc_N1000" 等命名不准确**。

> **96%–99% 的数值应命名为：2-way Identification（mean pairwise，MindEye2 官方
> `two_way_identification` 口径）。它们不是 N-way Top-1 准确率。**

修正版表格：`evaluation/p1_protocol_final_audit/CORRECTED_METRIC_NAMING_TABLE.csv`
（旧文件全部保留原样；论文引用时使用修正命名，并注明方向：B 组在高层语义 2-way 上显著更差）。

---

## 二、统计推断终审

### 2.1 20×300 随机候选池的两两重叠（本轮新增，可复现）

原池脚本 `scripts/cpu_p1_pool_audit.py` 使用 `SEED=2027, np.random.default_rng`，
20 个池**可精确复现**（复算 A/B 池均值与 20/20 计数与 `pool_level_audit.json` 完全一致）。

两两 ID 重叠（20×20 矩阵见 `pool_overlap_audit.json`）：

| 统计量 | 值 |
|---|---|
| 190 个池对的重叠 min / mean / max | **70 / 90.17 / 112** |
| 独立池理论期望（超几何 1000,300,300） | 90 ± 14.0 |
| P(两独立池零重叠) | 2.3e-58 |
| 1000 个 ID 的覆盖 | 全部 1000 个被覆盖；单 ID 最多入选 20 池 |

### 2.2 独立性判定

**20 个池不独立。** 任意两池期望重叠 ≈90（实测 90.17），它们是对同一 1000 候选总体
的高度交叠抽样。因此：

1. **"20/20 方向一致的符号检验 p≈1.9e-6"不能作为 20 次独立重复实验的显著性证据**——
   该 p 值只反映"20 个交叠子池上方向完全一致"这一事实，池间相关使有效样本量远小于 20；
2. **保留其作为候选池敏感性/稳健性结果**：结论表述为
   「在 20 个随机 300 子池上 B 均更差（20/20），Δ 稳定在 −0.0036～−0.0134，
   池内 sd 远小于效应量」；
3. **正式统计推断与探索性结果分离**（见 2.5）。

### 2.3 10k Bootstrap 复核

- **重采样单位**：逐样本 delta `d_j = (cB_j − cA_j)/(N−1)`，n=1000，10k 次有放回，
  percentile 95% CI（seed 2027）；
- **候选池是否固定**：固定。1000 口径候选池 = 全部 1000 张 GT；982 口径 = 同一固定池前 982 列
  （逐对 Pearson 不受行删除影响，数学上等价于独立 982 相关矩阵）；
- **适用范围**：CI 仅描述"这 1000 张固定刺激上的平均 delta"的重抽样不确定性。
  由于样本共享候选池（非独立），CI 偏窄，**不可外推到新刺激总体**；
  与池层面敏感性分析互为补充，不是推断意义上的"独立实验置信区间"。

### 2.4 McNemar 检验复核

- 输入：`hit1_j = (c_j == N−1)`，即 **Strict Top-1** 逐样本 0/1 正确性（非 2-way 连续量）；
- discordant counts：b = A 对 B 错、c = A 错 B 对（记录于 `nway_audit.json`，如
  AlexNet(5)@1000 b=128, c=70）；实现 `chi2=(|b−c|−1)²/(b+c)`（连续校正），p 值以 log10 记录；
- 限定：A/B 共享候选池同样使 0/1 项非独立，McNemar p 亦为**描述性**证据；
  且它检验的是 Top-1 差异，与 2-way 主口径不同，报告时必须分开表述。

### 2.5 探索性 vs 正式统计推断（最终口径）

| 层级 | 内容 | 状态 |
|---|---|---|
| **正式主结果** | A/B 全量 1000 上的点估计（表 1.3）；方向与效应量 | ✅ 确定性计算 |
| **描述性配对检验** | Wilcoxon/符号翻转（1000 主口径）、bootstrap CI、McNemar | ✅ 保留，但须声明"共享候选池 → 描述性" |
| **敏感性分析** | 20×300 池层面 20/20 方向一致 | ✅ 保留为敏感性，**不再作为独立性显著性** |
| ** Holm 校正** | 6 项指标 log10 空间校正后全部显著 | ✅ 维持（属描述性检验族） |
| **p=0 表述** | 已改为下界（p < 1e-162 等） | ✅ 维持 |

推荐论文表述：「由于 1000 个重建项共享同一候选池，逐项配对检验为描述性比较；
显著性以固定池上的效应量与 20×300 池层面方向一致性（敏感性分析）共同支撑，
不作独立重复实验的显著性声明。」

本轮未重训模型、未重跑推理；所有修正写入新文件，旧文件未动。

**产物**：`evaluation/p1_protocol_final_audit/{protocol_audit.json, protocol_comparison.csv, CORRECTED_METRIC_NAMING_TABLE.csv}`；
`evaluation/cpu_mindeye2_p1_stat_audit/pool_overlap_audit.json`（新文件）；
脚本 `scripts/p1_protocol_final_audit.py`、`scripts/p1_pool_overlap_audit.py`。
