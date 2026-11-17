# P3 后续实验准备（Empty Prompt / GNet / 盲测汇总）

日期：2026-11-16 ｜ 本轮只整理，不执行任何生成或新评价

## A. 人工盲测（就绪，可正式发放）

材料审计与执行规范见 **`reports/HUMAN_BLIND_REVIEW_EXECUTION_PROTOCOL.md`**。
要点：150 组材料完整、左右随机（raw=73/refined=77）、0 方法名泄露、答案键独立未开启；
评分 5 维度（subject/attributes/relations/errors/overall_quality）+ preference 二选一；
推荐直接沿用现有 5 列（映射到自然度/GT 相似度/语义保留），发放前按检查清单过一遍即可。

**判定：可以正式发放。**

## B. Empty Prompt 对照（整理自 EMPTY_PROMPT_CONTROL_PREFLIGHT.md，结论不变）

### B.1 定位与命名

- 唯一变量：`prompt = ""`；其余 8 项（输入图像、Zavy SDXL base、`img2img_timepoint=13`、
  `guider.scale=5`、`num_steps=25`、`seed=42`、`num_samples=1`、negative prompt 与
  `vector_suffix`）必须与 B 组逐项一致；
- **措辞红线**：只能称 **"empty-string prompt 对照"**。实测（CLIP-L 文本塔）空串嵌入
  与单字 "a" 余弦 0.8228，非零向量——**不得**称为"无文本条件 / text-free"。

### B.2 需要输出的结果

1. C 组 1000 张 `run/empty_prompt/..._stage2_emptyprompt.pt`（shape (1000,3,256,256)，
   ID 与 A 完全一致、无 NaN/空白，`verify_stage2` 同款校验 ALL_PASS）；
2. 回归自检：先跑 `--prompt_mode pred` 若干张与 `run/full_stage2/` **逐位一致**才允许继续；
3. A/B/C 三方配对：PixCorr/SSIM/AlexNet/Inception/CLIP-L 2-way/EffNet/SwAV（复用 P1/P3
   同一套脚本）；核心问题：B vs C 的差 = caption 的贡献；
4. （可选诊断）D 组 GT-caption 上界，标注 `DIAGNOSTIC_UPPER_BOUND`。

### B.3 资源与成本

| 项 | 估计 |
|---|---|
| 显存 | 峰值 ~20.4GB（与 B 组相同）→ 需 24GB 级 GPU |
| 耗时 | ≈31 min/1000 张 + 模型加载 ≈5 min；C（+可选 D）合计 ≈1–1.5 h |
| 磁盘 | +786MB/组（.pt），PNG 可选另计 |
| 建议租赁 | 24GB 单卡 2 小时档位即可（含回归自检与校验） |

**判定：值得进入下一阶段**（信息量高：可把 MIXED_RESULTS 归因为 caption 不准 vs SDXL 先验）。

## C. GNet / Brain Correlation（资产问题已解决，建议继续）

### C.1 官方来源（本轮实测核实，HF API 直查）

| 资产 | 官方来源 | 状态 |
|---|---|---|
| `gnet_multisubject.pt` | `https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/gnet_multisubject.pt`（**1,271,493,801 B**，本轮 API 实测存在） | ❌→**可获取**（此前"来源不明"的判断已过时） |
| `brain_region_masks.hdf5` | 同仓库根目录 `brain_region_masks.hdf5`（小文件，含 V1–V4/higher_vis/nsd_general 掩码） | ❌→**可获取** |
| 真值 beta | `betas_all_subj01_fp32_renorm.hdf5` | ✅ 本地已有（pscotti 快照） |
| GNet8_Encoder 代码 | `source/MindEyeV2/src/models.py`（本地官方源码） | ✅ 已有 |

### C.2 协议要点（官方 cell 33–37）

- 输入 = 融合后重建（`.75*recons + .25*blurryrecons`，Resize 256）；
- GNet 由**生成图像**预测 beta，逐体素 Pearson 对 6 个脑区取均值；
- 真值 = 每图 3 次 trial 平均。

### C.3 成本与建议

| 项 | 估计 |
|---|---|
| 下载 | 1.27GB + 小文件，5MB/s 线路 ≈5–10 分钟 |
| 计算 | GNet 前向 3000 张：GPU ≈2–5 分钟（显存 ~3–4GB）；CPU 慢（估计 0.5–2 s/张，25–100 分钟）|
| 附加数据 | blurryrecons 已在本地（`run/full_stage1/..._all_blurryrecons.pt`）|

**判定：值得继续**。两个此前 BLOCKED 的资产均可从官方仓库直接获取，成本约等于一次
短 GPU 会话中的几分钟计算；该指标是唯一合法的"图像→脑"A/B 指标，补齐后 MindEye2
官方 16 项主表仅剩 caption 相对项之外的全部可复现。**建议下一阶段与 Empty Prompt
共用同一次 GPU 租赁完成。**

> 本轮未下载 gnet 权重、未运行任何 GNet 计算。
