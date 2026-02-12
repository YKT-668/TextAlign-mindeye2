# subj01 no-hard 训练排查报告（TextAlign-mindeye2）

日期：2025-12-30

目标：在“最小干预、可复现、证据链完整”的前提下，判断 subj01 的 no-hard（v1/v3）是否 **真的在训练 & 有有效梯度**，还是训练逻辑/参数配置有问题。

本报告只基于仓库现有日志与 checkpoint，并通过新增脚本在本机复现读取与梯度 sanity。

---

## 结论（必须一句话明确）

结论 B：no-hard 分支**在训练且梯度非 0**，但监督信号很弱/易饱和，导致最终表现仍接近随机。

（说明：本次证据显示 `loss_text` 非 0、`text_head` 梯度非 None 且非 0；同时 v3 的 `loss_text` 在短程训练中快速下降/饱和，且 v3 的训练配置与 v2 存在显著差异（尤其 batch 与训练步数/epoch 安排），更符合“信号弱导致学不动/学偏”的现象，而不是“完全没训练”。）

---

## Step 2.1 定位 v3 的训练入口与日志（实际生效入口）

### 训练入口（从日志中确认）

- v3、v2 的实际入口脚本均为：
  - [src/Train_textalign.py](src/Train_textalign.py)

证据：v3/v2 日志中多处出现 `.../src/Train_textalign.py:<line>`。

### v3 日志文件（subj01）

在 [train_logs/](train_logs/) 中定位到：

- v3（短程，10 epochs）：
  - [train_logs/s1_textalign_coco_train_long_v3.log](train_logs/s1_textalign_coco_train_long_v3.log)
  - 关键片段：`s1_textalign_coco_train_long_v3 starting with epoch 0 / 10`

- v3（长程，150 epochs）：
  - [train_logs/s1_textalign_coco_train_long_v3.train_textalign_py.log](train_logs/s1_textalign_coco_train_long_v3.train_textalign_py.log)
  - 关键片段：`s1_textalign_coco_train_long_v3 starting with epoch 0 / 150`

> 注意：长程 v3 日志文件中未出现 tqdm postfix 风格的 `train/loss_text=...` 标量行（本次解析工具按该格式匹配），因此标量统计使用 v3 short log（10 epochs）作为可解析证据源。

### 启动命令原文

当前 workspace 的 v3 short log 未直接打印完整启动命令；但仓库文档给出了同类训练命令模板（包含关键 env flag）：

- [src/readme2.md](src/readme2.md)
  - 示例包含：`MINDEYE_TEXTALIGN=1`、`MINDEYE_TEXTALIGN_SCALE=0.05`、`python src/Train_textalign_v3.py ...`

同时，日志中确认的“实际生效 hard-negative 状态”为：

- ` [TextAlign] hard-negative disabled (no hardneg will be loaded/used).`

该状态由 [src/Train_textalign.py](src/Train_textalign.py) 的以下逻辑控制：

- `--use_hardneg/--no-use_hardneg` 参数
- 环境变量 `MINDEYE_DISABLE_HARDNEG=1` 可强制关闭
- 若 hardneg 文件不存在也会退化成无 cf 的 InfoNCE

---

## Step 2.2 Checkpoint keys 对比（v2 vs v3）

### checkpoint 路径

- v2 last：
  - `/mnt/work/repos/textalign-mindeye2-model/models/subj01/s1_textalign_coco_train_long_v2_last.pth`
- v3 last：
  - `/mnt/work/repos/textalign-mindeye2-model/models/subj01/s1_textalign_coco_train_long_v3/last.pth`

### 运行脚本与输出

- 脚本：
  - [tools/ckpt_keydiff.py](tools/ckpt_keydiff.py)
- 输出：
  - [debug_artifacts/ckpt_keydiff.txt](debug_artifacts/ckpt_keydiff.txt)
  - [debug_artifacts/ckpt_keydiff.json](debug_artifacts/ckpt_keydiff.json)

### 关键结论（state_dict keys）

- v2 vs v3：
  - `num_keys` 都是 70
  - `intersection=70`、`only_in_A=0`、`only_in_B=0`
  - 关键层（关键词匹配）在两边都存在：
    - `text_head.mlp.*`
    - `backbone.backbone_linear.*`
    - `backbone.clip_proj.*`
    - `ridge.linears.0.*`

这意味着：**v3 并不是“少了 TextAlign 模块/关键层没保存”这种结构性错误**。

---

## Step 2.3 单步梯度 sanity（最关键，1 batch forward+backward，无 optimizer.step）

### 运行脚本与输出

- 脚本：
  - [tools/one_batch_gradcheck.py](tools/one_batch_gradcheck.py)
- 输出：
  - [debug_artifacts/one_batch_gradcheck.txt](debug_artifacts/one_batch_gradcheck.txt)

### 关键证据（直接摘录自输出）

- `loss_text` 非 0：
  - `loss_text=4.076996326446533`
- `text_head` 关键参数梯度非 None 且非 0：
  - `text_head.mlp.1.weight ... grad_abs_mean=1.467e-04`
  - `text_head.mlp.3.bias ... grad_abs_mean=9.201e-05`
- `backbone.backbone_linear.weight` 在训练脚本里被冻结（`requires_grad=False`），因此 `grad=None` 是预期行为：
  - `backbone.backbone_linear.weight: requires_grad=False grad=None`

### “如果 grad 为 0，打印跳过条件”的核对

脚本也会打印 `shared_hits` 与 `MINDEYE_TEXTALIGN(_SCALE)` 等关键 flag；本次单 batch 中：

- `MINDEYE_TEXTALIGN=1`、`MINDEYE_TEXTALIGN_SCALE=0.05`
- `shared_hits=32/32`（即 batch 中样本 id 都可在 teacher id2row 中命中）

因此：**no-hard 的 TextAlign loss 分支没有被 if/continue 跳过**。

---

## Step 2.4 参数是否真的更新（start/end diff）

### 运行脚本与输出

- 脚本：
  - [tools/weight_update_probe.py](tools/weight_update_probe.py)
- 输出：
  - [debug_artifacts/weight_update_probe.txt](debug_artifacts/weight_update_probe.txt)

### 关键证据（参数差分）

本次按你的模板用 v2 last 作为 baseline（start/ref），对比 v3 last：

- TextAlign head 权重：
  - `delta_l2=31.3746`、`cosine=0.6492`
- backbone_linear 权重：
  - `delta_l2=541.3143`、`cosine=0.5929`
- 任意 bias（text_head.mlp.3.bias）：
  - `delta_l2=0.3258`、`cosine=0.5902`

这些数值清晰表明：**v3 与 v2 的参数状态显著不同**（不是“几乎没动”的情况）。

> 重要解释：本步骤按你给的命令模板使用了 “v2 last vs v3 last” 的差分，它证明“不是同一个权重/确实发生了更新或分叉”。
> 但严格意义上的 “v3 start vs v3 end” 需要 v3 的 start ckpt（或日志中最早的 ckpt）才能完成闭环。若你能提供 v3 start ckpt 路径，我可以把 probe 直接改为 v3 start/end 对比。

---

## Step 2.5 训练日志关键标量统计（loss_text / loss_total / lr）

### 运行脚本与输出

- 脚本：
  - [tools/parse_train_log.py](tools/parse_train_log.py)
- 输出：
  - [debug_artifacts/trainlog_stats.txt](debug_artifacts/trainlog_stats.txt)（v2 vs v3 short）
  - [debug_artifacts/trainlog_stats_v3short.txt](debug_artifacts/trainlog_stats_v3short.txt)

### 关键现象

1) v2（150 epoch 训练日志片段）

- `train/loss_text`：均值约 1.576（在采样点上逐步下降 1.73 → 1.46）
- `train/loss`：约 1.97～1.98
- `train/lr`：warmup 后到 3e-4

2) v3（10 epoch short log，可解析）

- `train/loss_text`：从 3.1 快速下降到 0.649（随 step 递减，后期进入饱和区）
- `train/loss`：基本稳定在 3.52～3.56
- `train/lr`：OneCycle 从 3e-4 退火到接近 0（末尾到 1.2e-08）

### 推断

v3 的 `loss_text` 明显存在并下降，说明 no-hard 并非 loss=0 或完全断梯度；但下降很快、且整体 `train/loss` 基本不动，符合“TextAlign-only 的弱监督在当前训练设置里很快饱和/被别的项主导或无法转化为有效表征”的模式。

---

## 训练配置一致性核对（v2 vs v3）

以下从日志中抽取（v2: long log；v3: short log；两者均来自 Train_textalign.py 的打印）：

| 配置项 | v2（s1_textalign_coco_train_long_v2） | v3（s1_textalign_coco_train_long_v3 short） |
|---|---:|---:|
| subj | 1 | 1 |
| num_sessions | 40 | 40 |
| num_epochs | 150（log 显示 0/150） | 10（log 显示 0/10） |
| batch_size | 8 | 32 |
| num_iterations_per_epoch | 3750 | 937 |
| total_steps | 562500 | 9370 |
| use_prior | False（loss_prior=0） | False（loss_prior=0） |
| blurry_recon | False（按日志配置推断） | False（按日志配置推断） |
| MINDEYE_TEXTALIGN_SCALE | 未从日志直接打印（默认为 0.05，可被 env 覆盖） | 未从日志直接打印（默认为 0.05，可被 env 覆盖） |
| hard negatives | disabled（log 明确打印 disabled） | disabled（log 明确打印 disabled） |
| negatives 来源 | in-batch（InfoNCE） | in-batch（InfoNCE） |

关键差异：v3 short 的总训练步数只有 9370（而 v2 long 的 total_steps=562500），并且 batch 变大为 32。这会直接改变 in-batch negatives 规模、梯度噪声与收敛形态。

---

## 最终判断：是否存在训练 bug？

可以排除“no-hard 没在训练”的典型 bug：

- `loss_text` 明确非 0（见 one-batch gradcheck）
- `text_head` 梯度明确非 None 且非 0（见 one-batch gradcheck）
- ckpt keys 完整，v2/v3 key 集合一致（见 ckpt_keydiff）

更符合的解释是：no-hard 在当前设定下监督信号较弱/易饱和，导致最后仍随机。

---

## “是否认结果”建议（≤5 行）

建议认结果（非 bug）：no-hard 确实在训练且有梯度，但 in-batch InfoNCE 在 shared1000 的监督信号不足/易饱和，无法学到可泛化的脑→语义对齐。

若要进一步确认并可能修复：优先检查 [src/Train_textalign.py](src/Train_textalign.py) 中 `valid_mask.any()==False` 时的 `continue` 是否在实际训练中频繁触发（shared_hits 很低会导致有效更新步稀疏），以及 `MINDEYE_TEXTALIGN_SCALE` 与 batch size/温度 `tau` 的组合是否导致梯度过小。
