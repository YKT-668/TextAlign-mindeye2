# GIT-Caption-Conditioned SDXL Refinement 补充实验（最终归档）

归档日期：2026-11-17 ｜ 分支：`archive/git-sdxl-refinement-final-2026-11-17`

## 实验目的

在 ConceptAlign 两阶段脑解码流程（1000 张 NSD 测试图）上，检验额外的
GIT caption 引导 SDXL img2img refinement 是否值得：B 组 = A 组（第一阶段 unCLIP 重建）
之上叠加 GIT 预测 caption 引导的 SDXL img2img（timepoint=13, scale=5, steps=25, seed=42）。

## 主结果（1000 张，CLIP ViT-L/14 官方 2-way Identification）

| Method | Caption-conditioned refinement | Inference time | CLIP-L 2-way |
|---|---|---:|---:|
| ConceptAlign (A) | No | 90.2 min | **93.19%** |
| ConceptAlign + GIT-SDXL (B) | Yes | ~121.3 min | **91.65%** |

Δ(B−A) = **−1.54 pp**，95% CI [−0.0239, −0.0072]（Bootstrap 10k），Holm 校正后显著。
982 敏感性口径方向一致（−0.0150）。SwAV distance（官方）同样显著变差（0.3505→0.3633）。

**注意**：A 组并非"无 GIT/Caption"——A 不含的是额外的第二阶段 caption-conditioned
refinement。本实验比较的是"增加整个 SDXL refinement 流程"的效果，下降不能完全归因于
caption 文本本身。

## 目录结构

- `docs/` — 全部正式报告（实验记录、协议审计、统计审计、指标字典、命名审查、复现指南）
- `code/` — 评测/统计/归档脚本与环境锁定文件
- `results_summary/` — 小型结果汇总（JSON/CSV；逐样本与大文件在 HF 私有 Dataset）

## 大文件位置（HF 私有 Dataset）

`ykt668/textalign-mindeye2-data` → `supplementary/git_sdxl_refinement/2026-11-17/`
（stage1 七类输出、stage2 输出、bigG 特征、P3 特征、逐样本评价、日志、SHA256SUMS）。
NSD 原始数据受协议约束不在归档内，需经官方渠道获取。

## 复现

见 `docs/FINAL_REPRODUCIBILITY_GUIDE.md` 与 `docs/FINAL_GIT_SDXL_REFINEMENT_EXPERIMENT.md`。
