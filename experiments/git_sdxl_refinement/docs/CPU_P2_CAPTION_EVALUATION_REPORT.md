# P2 Caption 评价报告（METEOR / ROUGE / Sentence / CLIP-B / CLIP-L）

对象：`s1_textalign_stage1_FINAL_BEST_32` 的 1000 条 GIT 预测 caption 与对应的 GT caption
产物目录：`evaluation/cpu_mindeye2_p2_caption/`
机器：4 vCPU / 3.87GB RAM / 无 GPU（CPU-only，串行单模型）

## 0. 结论速览

| 指标 | 结果 | 状态 |
|---|---|---|
| METEOR | **0.2439** | ✅ 完成（OFFICIAL_EQUIVALENT） |
| ROUGE-1 / 2 / L / Lsum | **0.3614 / 0.1112 / 0.3337 / 0.3337** | ✅ 完成（OFFICIAL_EQUIVALENT） |
| Sentence Similarity（all-MiniLM-L6-v2） | **0.4665**（中位 0.4789） | ✅ 完成（OFFICIAL_EQUIVALENT） |
| CLIP-B（ViT-B/32 文本塔） | **0.7294**（中位 0.7376） | ✅ 完成（OFFICIAL_REPRODUCED） |
| CLIP-L（ViT-L/14 文本塔） | **0.6291**（中位 0.6392） | ✅ 完成（OFFICIAL_REPRODUCED） |

**五项文本指标全部完成（5/5）**；A/B 共享同一批预测 caption，故只算一套，**不能**用它解释 SDXL 精炼效果。

- **A/B 共享同一批预测 caption**：文本指标只算一套，**不能**用来解释 SDXL 精炼效果；
- 上述均为 **brain→text（预测 caption ↔ GT caption）** 绝对项；
- MindEye2 官方三项式（brain↔ref / brain↔image / image↔ref）中的 **image↔* 两项 BLOCKED**：
  缺 `all_git_generated_captions.pt`（GIT 直接看图生成的 caption），见 §6。

## 1. 配对正确性验证（VERIFIED）

| 检查 | 结果 |
|---|---|
| 预测 caption 数 | 1000（`s1_textalign_stage1_FINAL_BEST_32_all_predcaptions.pt`） |
| GT caption 数 | 1000（`assets/ground_truth/all_captions.pt`） |
| id 数 | 1000（`..._all_ids.pt`），**唯一** |
| 配对方式 | 同序逐行配对（第 i 条 ↔ 第 i 条，id 升序一致） |
| 空 caption | 0 条 |
| 与 per-sample CSV 一致性 | 1000 行、1000 个唯一 id、均值与 summary 完全一致 |

## 2. 实现与"官方/代理"标注

MindEye2 官方 notebook 用 `evaluate.load('meteor')`、`evaluate.load('rouge')`、
`sentence-transformers/all-MiniLM-L6-v2`、`openai/clip-vit-base-patch32`、`openai/clip-vit-large-patch14`。

| 指标 | 本实现 | 标注 | 说明 |
|---|---|---|---|
| METEOR | `nltk.translate.meteor_score`（word_tokenize 分词） | **OFFICIAL_EQUIVALENT** | `evaluate` 的 meteor 内部就是该函数，超参默认 α=0.9,β=3,γ=0.5 |
| ROUGE | `rouge_score.RougeScorer(['rouge1','rouge2','rougeL','rougeLsum'], use_stemmer=True)`，取 mean F | **OFFICIAL_EQUIVALENT** | `evaluate` 的 rouge 内部就是 google rouge_score |
| Sentence | all-MiniLM-L6-v2，**mean pooling + L2** 后取对角余弦 | **OFFICIAL_EQUIVALENT** | 与 sentence-transformers 默认 pipeline 等价（未装 ST 包，用 transformers 复现） |
| CLIP-B | `CLIPTextModelWithProjection`（clip-vit-base-patch32）文本塔，归一化后对角余弦 | **OFFICIAL_REPRODUCED** | 与 `CLIPModel.get_text_features` 数值等价 |
| CLIP-L | 同上（clip-vit-large-patch14，本地缓存权重） | **OFFICIAL_REPRODUCED** | 文本塔可跑，无需图像前向 |

> 说明：`evaluate` 包本身未安装（其 metric 脚本需联网从 HF 拉取），因此**直接调用其底层库**；
> 忠实度由"同一函数、同一超参"保证，已在 `caption_summary.json.status` 中逐项标注。

## 3. 环境与权重溯源

| 项 | 值 |
|---|---|
| torch | 2.5.0+cu124（CPU 运行） |
| transformers | 4.46.3 |
| nltk | 3.10.4（wordnet + punkt + punkt_tab 语料） |
| numpy / scipy | 2.5.4 / 1.17.1 |
| MiniLM 权重 | 90,888,945 bytes，SHA256 `c3a85f238711653950f6a79ece63eb0ea93d76f6a6284be04019c53733baf256`（= HF 官方 blob 名，可验证） |
| CLIP-B 权重 | 605,247,071 bytes，SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f` |
| CLIP-L 权重 | 本地 HF 缓存，safetensors `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`(1.71GB)；bin `f1a17cdbe0f36fec524f5cafb1c261ea3bbbc13e346e0f74fc9eb0460dedd0d3` |
| SwAV（下一轮用） | 113,703,565 bytes，SHA256 `029d0b8c2e70bcee3f8beb70cc104ef51585c090224d33fb9c6f51d146cfd1eb` |

新增下载量（本轮）：MiniLM 90.9MB + CLIP-B 605.2MB + SwAV 113.7MB ≈ **810MB < 1GB 预算**。

## 4. 结果明细

| 指标 | 均值 | 中位数 | 备注 |
|---|---|---|---|
| METEOR | 0.243895 | — | nltk 默认超参 |
| ROUGE-1 | 0.361447 | — | F-measure 均值 |
| ROUGE-2 | 0.111172 | — | |
| ROUGE-L | 0.333716 | — | |
| ROUGE-Lsum | 0.333716 | — | 单句 caption 时与 ROUGE-L 相同 |
| Sentence Sim | 0.466508 | 0.478917 | MiniLM |
| CLIP-B | 0.729425 | 0.737579 | 文本塔 |
| CLIP-L | 0.629149 | 0.639187 | 文本塔 |

**如何解读**
- ROUGE-L 0.334 / METEOR 0.244 属于"语义大致相关、用词差异较大"的区间：
  GT 是 NSD 人工标注（如 "People shopping in an open market for vegetables."），
  预测是 GIT 风格短句（如 "a group of people standing around a table."）；
- **CLIP-B 0.729 > CLIP-L 0.629**：两者计算方式完全一致（同 tokenizer 超参、同归一化、同对角余弦），
  差异来自模型本身——CLIP-B/32 的文本空间更"词面/短语化"，CLIP-L/14 更"整体语义"，
  在"预测短句 vs 人工长句"这一场景下 B 反而更高；
- 三者量纲不同（ROUGE/METEOR 是 n-gram/对齐分数，MiniLM/CLIP 是余弦），**不可横向比较**；
- 这些数值只描述"预测 caption 与 GT 的接近程度"，**与 SDXL 精炼无关**（A/B 同一条 caption）。

## 5. CLIP-B 获取记录

- 权重 `pytorch_model.bin` 605,247,071 bytes，SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`；
- 下载策略：单连接仅 0.15MB/s 且常断流 → 改用 **aria2c `-x8 -s8` 多段并发（≈1.06MB/s）** 完成；
- tokenizer 用小文件 `config.json / vocab.json / merges.txt / tokenizer_config.json / special_tokens_map.json`
  （无 `tokenizer.json` → 使用 slow `CLIPTokenizer`，与官方 CAPTION 预处理等价）；
- 脚本以"**文件大小精确等于 605,247,071 且无 `.aria2` 残留**"作为就绪判据，避免误用残缺权重。

## 6. 未完成项与原因

| 项 | 状态 | 原因 |
|---|---|---|
| brain↔image（预测 caption ↔ GIT 看图 caption） | **BLOCKED_MISSING_REFERENCE** | 缺 `all_git_generated_captions.pt` |
| image↔ref | **BLOCKED_MISSING_REFERENCE** | 同上 |
| 三项式汇总 | BLOCKED | 缺上述两项 |

> 补齐路径（下一轮，可选）：用本地 `microsoft/git-large-coco` 对 1000 张 GT 图重新生成 caption
> （CPU 约 10–20 分钟）→ 写入 `all_git_generated_captions.pt` → 重跑本脚本即得完整三项式。
> 注意：这是**新增推理**，需用户确认后执行。

## 7. 产物

| 文件 | 说明 |
|---|---|
| `caption_summary.json` | 5 项指标 + 官方/代理标注 + 状态 |
| `caption_per_sample.csv` | 1000 行：id、meteor、rouge1、rougeL、sentence、clip_b、clip_l、pred、gt |
| `model_manifest.json` | 权重路径/大小/SHA256、库版本、来源、官方映射 |
| `SHA256SUMS` | 上述文件归档校验 |
