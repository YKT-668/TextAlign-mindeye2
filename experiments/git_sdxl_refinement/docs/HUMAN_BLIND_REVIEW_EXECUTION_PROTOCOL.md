# 人工盲测正式执行说明（Human Blind Review Protocol）

对象：`evaluation/human_blind_review/`（150 组三元组 + 评分表 + 独立答案键）
材料生成日期：2026-11-15（当时审计：150 组完整、0 方法名泄露、左右随机 raw=73/refined=77、
答案键独立、preference 列全部空白未预填）

## 1. 评价者看到的材料

每位评价者获得：
1. `review_sheet.csv`（150 行，每行一个样本）；
2. 450 张 PNG：每行 `left_png` / `right_png` / `gt_png` 三张图；
3. **不得**获得 `ANSWER_KEY_DO_NOT_OPEN_BEFORE_RATING.csv`（评分全部提交后才打开）。

三张图的关系：left 与 right 是同一脑样本的两种重建结果（A=unCLIP 原始重建，
B=GIT caption + SDXL img2img 精炼，随机左右），gt 是该样本的真实刺激图像（NSD）。

## 2. 每个样本的评分流程（约 1.5–2 分钟/组）

1. **先看 GT**（顶部真实图像）建立参照；
2. 并排比较 left 与 right，针对以下 5 个维度分别打分（1–5 整数分）：

| 列名 | 维度 | 1 分含义 | 5 分含义 |
|---|---|---|---|
| `subject` | 主体一致性 | 主体与 GT 完全不符 | 主体与 GT 完全一致 |
| `attributes` | 属性保留 | 颜色/纹理/数量等错乱 | 属性高度吻合 |
| `relations` | 空间关系 | 空间布局错误 | 空间布局正确 |
| `errors` | 明显伪影缺陷 | 大量伪影/结构崩坏 | 几乎无伪影 |
| `overall_quality` | 综合自然度与 GT 相似 | 非常差 | 非常好 |

3. 二选一：`preference` 填 `left` 或 `right`（整体更接近 GT 的一张；若实在无法判断，
   允许填 `tie`，统计时单独处理）；
4. 评分期间不得使用任何脚本/指标辅助，不得讨论。

## 3. 建议增加的三个正式维度（本轮新增建议，需在发放前冻结口径）

现有 5 列已可拆出任务书要求的三个维度，对应关系：
- **自然度（naturalness）** → `errors` + `overall_quality` 中与伪影相关的部分；
  如需独立维度，在表中新增 `naturalness` 列（1–5）；
- **GT 相似度（GT similarity）** → `overall_quality` 主口径；可新增 `gt_similarity` 列；
- **语义保留（semantic preservation）** → `subject` + `attributes` + `relations` 的综合；
  可新增 `semantic` 列。
> 两个方案二选一并在发放前冻结：A）沿用现有 5 列，论文中按上述映射聚合；
> B）在 CSV 追加 3 列（left/right 共用同一行，打分针对"更接近 GT 的那张"）。
> 推荐 A：避免重复打分负担；本说明默认方案 A，不修改既有 CSV 结构（新增列属方案 B）。

## 4. 评分汇总与统计（评分回收后由脚本执行，不含任何人工干预）

1. 打开 ANSWER_KEY，将每组 left/right 映射回 A/B；
2. 主指标：preference rate（B 被选比例），二项检验 vs 50%（含 tie 剔除与 tie=0.5 两种口径）；
3. 分维度：对 5 个维度分别做 A/B 的配对 Wilcoxon 符号秩（同一评价者内配对）；
4. 多评价者：报告每维度的评价者间一致性（Krippendorff's α 或 Kendall's W）；
5. 分层检查：左右位置效应（left 被选率应接近 50%，否则存在位置偏差）；
6. 效果量：preference rate 的 95% bootstrap CI（重采样单位=评价者×样本）。

## 5. 发放前检查清单（发放人执行，10 分钟）

- [ ] 452 个文件齐全（450 图 + 2 CSV）；
- [ ] review_sheet 150 行、id 唯一、preference 列全空；
- [ ] ANSWER_KEY 不随材料发放；
- [ ] 抽查 5 组：三张图内容对应同一 id；
- [ ] 文件名不含 raw/refined/unclip/img2img/stage 字样。

## 6. 边界

本说明仅描述流程；**未生成、未预测、未虚构任何评分**；答案键在本轮仍未打开。
