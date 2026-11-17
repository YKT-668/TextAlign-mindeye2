# Caption 资产完整性审计（P2 收尾）

日期：2026-11-16 ｜ 性质：只审计 + 官方小文件下载 + CPU 轻量补算；**未重跑 GIT、未重新生成 Caption**

## 0. 结论速览

| 项 | 结果 |
|---|---|
| `all_git_generated_captions.pt` | **已补齐**（官方来源，55,125 B；此前本地缺失） |
| `predcaptions.pt` | ✅ 1000 条（`s1_textalign_stage1_FINAL_BEST_32_all_predcaptions.pt`） |
| `all_ids.pt` | ✅ 1000 个唯一升序 ID |
| GT captions | ✅ 1000 条（`assets/ground_truth/all_captions.pt`） |
| 被阻塞的两个指标族 | **已全部补算完成（OFFICIAL_REPRODUCED）** |
| predcaptions 能否复用为 git_generated | ❌ **不能**（输入不同，见 §3） |

## 1. 缺失文件对应哪两个指标（官方 cell 40–53 证据）

MindEye2 caption 三项式需要三套文本：`all_captions`（GT 人工标注）、
`all_predcaptions`（脑→GIT 预测）、`all_git_generated_captions`（**GIT 直接看 GT 图像**生成）。
缺失文件阻塞的是涉及第三套的两个组合（5 个指标 × 2 + 相对项）：

| 指标族 | 官方定义（cell 42/44/46/50/52） | 性质 |
|---|---|---|
| **`*_brain_img`** | `all_predcaptions` ↔ `all_git_generated_captions` | text-text，无图像输入 |
| **`*_img_ref`**（官方称 GROUND） | `all_git_generated_captions` ↔ `all_captions` | text-text，作归一化分母 |
| `*_relative` | `brain_img / img_ref` | 相对归一化 |

（第三项 `*_brain_ref` 即 P2 已完成的绝对项，不受影响。）

## 2. 官方文件获取（无损，KB 级）

| 属性 | 值 |
|---|---|
| 来源 | `https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/evals/all_git_generated_captions.pt` |
| 大小 | 55,125 B（与 HF API 报告一致） |
| git blob oid | `8dca9586fcbfef4c6b8cbc119c857e99555c6e5d`（HF API 报告值） |
| 本地 SHA256 | `adfd766b6b17bfdab3b5170e86f2b5a2b0e039521c32cedb42489583d5d2e64a` |
| 本地路径 | `assets/ground_truth/all_git_generated_captions.pt`（新增文件，未覆盖任何东西） |
| 内容核验 | **1000 条字符串**；样例：`"a large number of people standing around a table with carrots"`（典型 GIT/COCO 风格） |

下载 55KB 文件不属于"下载大型模型"，也未触发任何 GIT 推理。

## 3. predcaptions 复用判定：❌ 不可复用

来源证据（`source/TextAlign-mindeye2-231850a/src/recon_inference.ipynb` cell 6/8）：

```python
pred_caption_emb = clip_convert(prior_out)          # 脑→diffusion prior 预测的 CLIP 图像嵌入
generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
```

- **predcaptions**：GIT 以"脑活动经 prior 预测出的 CLIP 嵌入"为条件生成——它描述的是
  *"脑认为图像应该是什么"*；
- **all_git_generated_captions**：GIT 以 **GT 图像像素** 经视觉编码器为条件生成——描述
  *"真实图像里有什么"*。

两者的 GIT 输入条件在表示空间上完全不同（预测嵌入 ≠ 图像编码输出），二者数值上不可能相同，
也不应相同——官方正是利用这一差异构造 `relative = brain/img` 归一化。**结论：无任何
无损复用方案；本轮已用官方文件补齐，替代了"重新运行 GIT"路线。**

## 4. 补算结果（`evaluation/cpu_mindeye2_p2_caption_extension/`）

实现与 `cpu_p2_caption.py` 完全一致（nltk METEOR / google rouge_score /
MiniLM / CLIP-B / CLIP-L 文本塔）；brain_ref 三项与 P2 已发布值**逐位一致**
（0.361447 / 0.333716 / 0.243895），实现一致性得到交叉验证。

### 4.1 ROUGE / METEOR（均值）

| 指标 | brain_ref（已有） | **brain_img（新）** | **img_ref（新）** | **relative（新）** |
|---|---|---|---|---|
| ROUGE-1 | 0.361447 | 0.455332 | 0.521683 | 0.872814 |
| ROUGE-L | 0.333716 | 0.423833 | 0.473953 | 0.894251 |
| METEOR | 0.243895 | 0.333791 | 0.419676 | 0.795354 |

### 4.2 句嵌入 / CLIP 文本相似度（对角余弦均值）

| 指标 | **brain_img（新）** | **img_ref（新）** | **relative（新）** |
|---|---|---|---|
| Sentence（MiniLM） | 0.512774 | 0.719007 | 0.713170 |
| CLIP-B（ViT-B/32 文本塔） | 0.747817 | 0.834027 | 0.896634 |
| CLIP-L（ViT-L/14 文本塔） | 0.663987 | 0.776591 | 0.855002 |

### 4.3 解读

- **img_ref（GROUND）** 一致高于 brain 侧：GIT 看真实图生成的 caption 与 GT 人工标注的
  重合度，天然高于"从脑预测"的 caption——与 MindEye2 论文模式一致；
- **relative（brain/img）0.71–0.90**：脑侧 caption 达到了"看图 caption"质量的
  71%–90%，与官方论文中 ~0.8–0.95 的量级相符（具体数值随模型而异）；
- 这些指标与 SDXL 精炼 A/B **无关**（文本侧 A/B 共享同一批 predcaptions），
  仅用于模型整体 caption 质量定位。

## 5. 状态汇总（P2 24 项官方表格口径）

| 族 | 状态 |
|---|---|
| `*_brain_ref`（5 指标） | ✅ P2 已完成 |
| `*_brain_img`（5 指标） | ✅ **本轮补齐** |
| `*_img_ref`（5 指标） | ✅ **本轮补齐** |
| `*_relative`（6 项，含 ROUGE-1/L、METEOR、Sentence、CLIP-B/L） | ✅ **本轮补齐** |
| ROUGE-2 / ROUGE-Lsum（brain_ref） | ✅ P2 已完成（官方表格只取 24 项主口径） |

产物：`caption_extension_summary.json`（含 provenance）、脚本
`scripts/cpu_p2_caption_extension.py`、日志 `logs/caption_extension.log`。
输出写入新目录，未修改 `cpu_mindeye2_p2_caption/` 任何既有文件。

## 6. 边界确认

未重新运行 GIT；未重新生成 Caption；未下载大型模型（55KB）；未修改/删除任何既有结果；
A/B 精炼评价不受影响（caption 指标本来就不区分 A/B）。
