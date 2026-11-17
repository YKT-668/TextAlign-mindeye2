# Empty Prompt 控制实验预审（仅审计，不执行生成）

## 1. 现有 B 组的文本条件链路（VERIFIED，代码级）

`scripts/run_stage2_img2img.py` 完全复刻 `enhanced_recon_inference.ipynb` cell 3-5，文本条件构造如下：

```python
# (1) 初始化时，用空字符串计算一次 vector_suffix
batch = {"txt": "", ...}
out = base_engine.conditioner(batch)
crossattn      = out["crossattn"]
vector_suffix  = out["vector"][:, -1536:]          # ← 来自 txt=""

# (2) 每张图：cond 用"预测 caption"
openai_clip_text            = base_text_embedder1(prompt)        # CLIP-L text
clip_text_tokenized, emb    = base_text_embedder2(prompt)        # OpenCLIP bigG text
clip_text_emb               = torch.hstack((emb, vector_suffix))
c  = {"crossattn": torch.cat((openai_clip_text, clip_text_tokenized), -1),
      "vector":    clip_text_emb}
uc = {"crossattn": crossattn_uc, "vector": vector_uc}            # ← 固定 negative prompt
```

关键事实：
| # | 事实 | 含义 |
|---|---|---|
| 1 | `vector_suffix` 来自 **`txt=""`**，且**三组共用** | 空文本信息**已经**恒定注入，B/C 之间不变 |
| 2 | `uc`（无条件分支）使用**官方默认 negative prompt**（`painting, extra fingers, ... anime`，24 个短语） | CFG 的"负条件"仍是文本 |
| 3 | 精炼强度由 `guider.scale=5` + `img2img_timepoint=13` + `num_steps=25` 控制 | C 组必须**完全保持一致**，否则不是单因素对照 |

## 2. "空 Prompt"到底是不是"无文本条件"？（实测证据，CPU 仅文本塔）

用 `openai/clip-vit-large-patch14` 的文本塔实测（max_length=77，取 EOS 池化，L2 归一）：

| 输入 | 非 padding token 数 | embedding 范数 | 与 empty 的余弦 |
|---|---|---|---|
| `""`（空串） | **2**（BOS+EOS） | 29.235 | 1.0000 |
| `"a"` | 3 | 29.252 | **+0.8228** |
| negative prompt（24 短语） | 72 | 29.179 | +0.2486 |
| `"a man is standing in front of a table."` | 12 | 28.310 | +0.3729 |

**结论（VERIFIED）**：`""` 并非零向量或退化输入，它给出一个**真实的"空语义"文本嵌入**，且与单字 `"a"` 高度相似（cos 0.82）。

> 因此 C 组的正确表述是：**"空字符串 Prompt"**，而**不能**表述为"无文本条件 / text-free"。
> 真正的"无文本条件"需要把 CFG 的 cond 也替换为 `uc`（改变采样语义，属算法改动）——**本轮不做、也不建议做**。

## 3. C 组（Empty Prompt）执行方案（待批准后执行）

**目标**：在**只改文本条件**的前提下，测量"A/B 的差异有多少来自 caption、多少来自 SDXL 先验"。

**必须固定的 8 项**（与 A/B 完全一致）：
1. 输入图像：`run/full_stage1/..._all_enhancedrecons.pt`（同一 1000 张、同一顺序）
2. 模型：Zavy SDXL base（同一 safetensors）
3. `img2img_timepoint = 13`
4. `guider.scale = 5`
5. `num_steps = 25`
6. `seed = 42`（循环外 `torch.manual_seed(42)` 一次，逐图顺序一致 → 噪声序列与 A/B 相同）
7. `num_samples = 1`；尺寸链 224 → Resize 768 → 输出 Resize 256
8. **negative prompt 与 `vector_suffix` 不变**（保持 `batch_uc` 原样）

**唯一变量**：`prompt = ""`。

**实现方式（不改现有脚本、不覆盖现有产物）**：
- 复制 `scripts/run_stage2_img2img.py` 为 `scripts/run_stage2_prompt_variant.py`，增加 `--prompt_mode {pred,empty,gt}`；
- 输出目录：`run/empty_prompt/`（C）与 `run/gt_caption_upper/`（D，上界诊断）；
- `pred` 模式用于**回归自检**：其结果应与 `run/full_stage2/` **逐位一致**（若不一致，说明实现引入了额外变量，必须停止）。

**验收**：C 组 1000 张 shape=(1000,3,256,256)、ID 与 A 完全一致、无 NaN/空白；然后复用 `evaluation/per_sample_metrics.csv` 的同一套指标脚本做 A/B/C 三方配对比较。

## 4. GT-Caption 上界实验（诊断用，不可作为方法成绩）

- 实现：同一脚本 `--prompt_mode gt`，prompt 取 `assets/ground_truth/all_captions.pt`。
- 用途：给出"若 caption 完全正确，img2img 能把 CLIP 图像相似度推到多少"的**上界**。
- **报告规范**：必须标注 `DIAGNOSTIC_UPPER_BOUND`，**不得**与 A/B 的方法结果并列展示，也不得声称"我们的方法"。

## 5. 成本与风险

| 项 | 估计 |
|---|---|
| C 组 1000 张（GPU） | ≈ 31 min（与 B 相同）+ 加载；输出 786MB |
| D 组（GT 上界）1000 张 | 同上，另 786MB |
| 显存 | 与 B 相同（峰值 ~20.4GB） |
| 磁盘 | +1.6GB（两组 .pt）+ PNG（可选） |
| 风险 | 若 C 组同时改变 negative prompt 或 `vector_suffix`，则不再是单因素对照 → 必须严格固定 |

## 6. 结论

- **值得做**（成本 ≈1 小时 GPU / 组，信息量高：能把"MIXED_RESULTS"归因为"caption 不准"还是"SDXL 先验"）。
- **必须在报告中精确措辞**为"empty-string prompt"，不得写成"无文本条件"。
- 本轮**未执行**任何生成；仅完成代码与参数审计。
