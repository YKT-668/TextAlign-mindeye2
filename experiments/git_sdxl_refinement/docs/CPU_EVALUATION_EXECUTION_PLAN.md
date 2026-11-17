# CPU 评测执行计划与迁移建议

## 0. 最终状态判定

# `CPU_MIGRATION_READY`

理由：正式产物 **19/19 SHA256 通过**、**0 断链**、数据盘 ext4/UUID 正常、无后台写入进程、GPU 已空闲。
同时，评测资产层面为 **`EVALUATION_PARTIALLY_READY`**：图像级官方指标可在补齐少量权重/依赖后完成；Brain Correlation 与 caption 相对项缺资产。

---

## 1. 逐条回答任务书 13 问

| # | 问题 | 回答 |
|---|---|---|
| 1 | 当前是否可以安全结束 GPU 租赁？ | **可以**。唯一建议在释放前做的可选动作：导出 A/B/GT 的 bigG 图像嵌入（约 2.5GB，3–5 分钟），否则 CPU 重算需 4–7 小时 |
| 2 | 150GB 数据盘是否完整安全？ | **是**。ext4 / UUID `5a9d9737-…` / 19 项 SHA 全 OK / 0 断链 / 32G 可用 |
| 3 | CPU 实例推荐配置？ | **8–16 vCPU、32GB RAM（若不做 bigG 则 16GB 亦可）、系统盘 60GB、数据盘挂 `/mnt/data`**；需装 `scikit-image / scikit-learn / evaluate / nltk / sentence-transformers`（都是小包） |
| 4 | 哪些官方指标可直接用现有数据？ | **PixCorr、SSIM**（仅缺 scikit-image）；**CLIP 2-way**（用 HF CLIP-L 代理）；**caption 绝对项**（CLIP-L 版）；**300-way retrieval**（用已缓存的 bigG 嵌入或现算） |
| 5 | 哪些需新增权重？ | AlexNet(233MB)、InceptionV3(104MB)、EfficientNet-B1(30MB)、SwAV(~100MB)、CLIP-B(600MB)、sentence-transformer(90MB)；CLIP ViT-L/14 可用现有 HF CLIP-L 替代 |
| 6 | 哪些需新增数据？ | `brain_region_masks.hdf5`、`gnet_multisubject.pt`（BrainCorr）；`all_git_generated_captions.pt`（caption 相对项；或用 GIT 在 CPU 重新生成） |
| 7 | 哪些指标必须用 GPU？ | 严格意义上**没有**；但 **bigG 对 3000 张图的嵌入**在 8 vCPU 上要 4–7 小时 → 实务上"建议 GPU 或提前缓存" |
| 8 | 哪些可完全在 CPU 完成？ | PixCorr、SSIM、AlexNet-2/5、Inception、EffNet-B1、SwAV、CLIP-L 图像/文本、METEOR/ROUGE/Sentence、人工盲测 |
| 9 | 哪些 ConceptAlign 结果可直接复用？ | **982-way FWD/BWD、300-way×30、2AFC(B2I/I2B)、CCD-H K1/K2、RSA(Pearson/Spearman)** —— 均为表征层，A/B 共用 |
| 10 | 哪些实验需要补做？ | ① A/B 的官方图像指标（PixCorr/SSIM/AlexNet/Inception/CLIP/EffNet/SwAV）② 官方 N-way（1000）Identification ③ CCD K=4/8、Object/Attribute/Relation 分解（如论文需要）④ hard retrieval M=50/100 的系统结果 ⑤ efficiency / cross-LLM 的数值回溯 |
| 11 | Empty Prompt 实验是否值得做？ | **值得**（低成本、能分离"文本条件"与"SDXL 先验"）。但必须知道：`""` **不是**"无文本条件"（见 `EMPTY_PROMPT_CONTROL_PREFLIGHT.md`） |
| 12 | 预计磁盘 / RAM / 时间？ | **磁盘**：模型新增 ≈1.2GB（不含 GNet/masks）+ 输出 <1GB；**RAM**：峰值 8–12GB（不含 bigG）；**时间**（8 vCPU，3000 张 A/B/GT）：PixCorr+SSIM 5–10 min、AlexNet/Inception/EffNet/SwAV 合计 10–20 min、CLIP-L 图像 8–15 min、caption 5 min、N-way 统计 2–3 min → **总计约 40–60 min**（不含 bigG） |
| 13 | 是否还需再租 GPU / 最小时长？ | 若采纳"释放前导出 bigG 嵌入"→ **不需要**；若未导出 → 需 **1 张 A100-40G，最短 1 小时**（仅跑 3000 张 bigG 前向 + 导出） |

---

## 2. 执行优先级清单

### P0（迁移后第一件事，失败即停止）
1. `findmnt /mnt/data` + `blkid` 确认 UUID/文件系统；
2. `sha256sum -c reports/FULL_1000_SHA256.sha256`（10 项）与 `reports/EVALUATION_SHA256.sha256`（9 项）；
3. 断链扫描：`find hf_home assets run -xtype l | wc -l` 必须为 0；
4. 4 条绝对软链存在性（`run/cache/*`、`run/data/*`）。

### P1（纯 CPU、零新增权重，先出结果）
5. 安装 `scikit-image`（SSIM 必需）；
6. 计算 A/B **PixCorr + SSIM**（含/不含官方 `.75/.25` 融合两种版本，明确标注）；
7. 复用既有 `evaluation/per_sample_metrics.csv` 出对比表。

### P2（需下载小权重，仍纯 CPU）
8. AlexNet-2/5、InceptionV3、EfficientNet-B1、SwAV → 官方 N-way Identification 与距离指标（A/B/GT）；
9. 与 `tables/…_all_enhancedrecons.csv` 的 16 值做交叉复核（重点查 BrainCorr 异常）。

### P3（需补数据）
10. 下载/生成 `all_git_generated_captions.pt` → caption 相对项；
11. 下载 `gnet_multisubject.pt` + `brain_region_masks.hdf5` → Brain Correlation（A/B 的"图像→脑"指标）。

### P4（补实验）
12. Empty Prompt（C 组）与 GT-Caption 上界：见 `EMPTY_PROMPT_CONTROL_PREFLIGHT.md`；
13. CCD K=4/8、Object/Attribute/Relation、hard retrieval M=50/100 的整理与补算（表征层，与 A/B 无关）；
14. 人工盲测评分（150 组，`evaluation/human_blind_review/`）。

---

## 3. 需要用户决策的 3 件事

1. **是否在释放 GPU 前导出 bigG 嵌入（强烈建议）** —— 3–5 分钟 vs CPU 4–7 小时；
2. 是否允许下载 **GNet + brain masks**（BrainCorr 的唯一门槛，体积未知）；
3. 是否允许回收 **8.7G** 中间 checkpoint（`run/full_stage*_ckpt/`）以缓解 32G 余量。

## 4. 本轮未做（严格遵守边界）

未安装依赖、未下载模型、未运行评测、未修改原始脚本、未删除/迁移任何文件、未启动 GPU 任务、未退租。
