# Fig01_ccd_acc1_per_subj_bar.png：数据来源与含义说明

本报告回答两个问题：
1) `cache/model_eval_results/shared982_ccd/figures/Fig01_ccd_acc1_per_subj_bar.png` 用的“数据文件”是什么？
2) 这张图当时是做什么用的？代表了什么、有什么意义？

> 目标：你可以把同一份数据导入在线绘图软件自行复刻/重排版。

---

## 1. 这张图的直接数据文件是什么？

**直接输入数据文件（用于绘图）**：

- `/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_ccd/ccd_summary.csv`

依据：仓库内用于生成该 PNG 的脚本是 `tools/make_ccd_figures.py`，其中 `fig01_per_subj_bars()` 明确读取 `ccd_summary.csv`，并筛选 `eval_repr == "pooled_mean"` 后绘制并保存到：

- `/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_ccd/figures/Fig01_ccd_acc1_per_subj_bar.png`

另外，`cache/model_eval_results/shared982_ccd/figures/figures_manifest.json` 的 Figure 1 条目也记录了这张图的用途说明（what/comment/usage）。

---

## 2. 这张图的“更底层”数据来源是什么？（可选了解）

`ccd_summary.csv` 是一个“汇总表”，每一行对应一个模型（通常用 `group` + `tag` 表示）、一个被试 `subj`、一种表示汇总方式 `eval_repr` 下的评测结果。

在 `ccd_summary.csv` 里，每行都带有一个 `metrics.json` 路径（列名就叫 `metrics.json`），例如：

- `cache/model_eval_results/shared982_ccd/<group>/<tag>/<eval_repr>/metrics.json`

这些 `metrics.json` 才是每个 job 的原始评测输出；`ccd_summary.csv` 把它们汇总成便于排序、比较、出图的表。

如果你只关心复刻 Fig01 这张图：**直接用 `ccd_summary.csv` 就够了**。

---

## 3. 图里每个元素具体代表什么？

这张图是“按被试分列的横向条形图”，用于展示：

- **每个被试（subj）上，各个模型的 `CCD@1` 分数**

脚本的具体做法（高层概括）：

- 先筛选 `eval_repr == "pooled_mean"`（这是为了避免同一模型出现 pooled/tokens 两行导致重复）
- 对每个 `subj` 单独作一列子图
- y 轴是模型标识（`label = group/tag`）
- x 轴是 `ccd_acc1`（图中用“CCD@1 (higher is better)”）
- 每个条形长度 = 该模型在该被试上的 CCD@1

你在 `ccd_summary.csv` 中会看到关键列：

- `group`：模型来源/类别（例如 official、ours 等）
- `tag`：具体模型/训练版本名
- `subj`：被试编号（例如 01/02/05/07）
- `eval_repr`：特征汇总方式（`pooled_mean` 或 `tokens_mean`）
- `ccd_acc1`：CCD@1 主指标
- `ccd_acc1_ci95_lo / ccd_acc1_ci95_hi`：bootstrap 95% CI（这张 bar 图本身没画 CI，但数据里有）
- `n_eval`：用于评测的样本数（此数据集里常见是 909）

---

## 4. CCD@1 是什么？它衡量了什么能力？

从仓库的 `ccd_summary.md`（评测说明）可以提炼出这套任务的核心：

- **任务**：caption discrimination (K+1-way)
- **每张图像**：将“正确 caption（GT）”与 K 个“负 caption”一起参与排序
- **CCD@1 的含义**：正确 caption 是否能排在第一名（也就是胜过所有负例）
- 因此 CCD@1 是一个 $[0,1]$ 的准确率；**越高表示模型越稳定地区分真 caption 与负 caption**

在该 shared982 CCD 配置里，你还会看到：

- `neg_mode`：负例模式（例如 `hardneg`）
- `k_neg`：K 的取值（例如 1）
- `seed`：随机种子（用于随机负例时的可复现）
- `bootstrap`：bootstrap 次数（用于 CI）
- `clip_pretrained`：文本编码器的预训练配置标识（OpenCLIP 的 pretrained 名称）

**简化理解**：这张图衡量的是“模型在跨模态空间里，能不能把正确 caption 排到负例前面”。

---

## 5. 这张 Fig01 当时是做什么用的？

它属于 `shared982_ccd` 评测包里自动生成的一组编号图（Fig01…Fig09）。

**Fig01 的用途（面向研发/选型的快速视图）**：

- 在“每个被试内部”，快速比较不同模型版本的 caption 判别能力
- 直观看出：
  - 哪些模型在某个被试上特别强（条更长）
  - 哪些模型异常弱（条特别短），通常提示特征空间/投影/预处理不匹配

`figures_manifest.json` 对 Fig01 的描述也很明确：

- 内容：每个被试一列横向条形图，展示各模型在 shared982 上的 CCD@1（pooled_mean）。
- 评价：用于快速看出同一被试下不同模型的 caption 判别能力差异；条形越长越好。
- 建议：优先在每个被试内挑选 CCD@1 靠前的模型作为 caption-alignment 的候选。

---

## 6. 这张图代表了什么、有什么意义？

从“论文/汇报叙事”的角度，它表达的是：

- CCD@1 是一个非常直接的“caption 对齐/判别”指标
- **按被试拆开看**，可以看到不同被试的难度差异、以及方法是否在多个被试上稳定提升
- 由于 `neg_mode` 常用 hard negative（更难），CCD@1 提升通常意味着：
  - 模型的跨模态对齐更可靠
  - 对“语义相近但错误”的 counterfactual caption 更有区分能力

从“工程诊断”的角度，它的意义是：

- 当你有很多 group/tag 的候选模型时，这是最省时间的一张“筛选图”
- 你可以先用它把每个被试的 top-k 候选挑出来，再去做更细的分析（例如误差条、差值、逐样本诊断等）

---

## 7. 如何用在线绘图软件复刻（建议流程）

1) 导入 CSV：`cache/model_eval_results/shared982_ccd/ccd_summary.csv`
2) 过滤：`eval_repr == pooled_mean`
3) 分面（facet）或拆分：按 `subj` 分 4 个 panel（01/02/05/07）
4) y 轴：`group + "/" + tag`（脚本里叫 label）
5) x 轴：`ccd_acc1`
6) 排序：在每个 subj panel 里按 `ccd_acc1` 从小到大或从大到小排序（原脚本是升序）

如果你需要把 CI 和 n_eval 也带上（更像论文图），可以额外使用：

- `ccd_acc1_ci95_lo / ccd_acc1_ci95_hi`
- `n_eval`

---

## 8. 备注：pooled_mean / tokens_mean 为什么会重复？

`ccd_summary.csv` 同一 (group, tag, subj) 通常会有两行：

- `eval_repr = pooled_mean`
- `eval_repr = tokens_mean`

这就是“重复行问题”的来源。

Fig01 这张 bar 图为了避免重复，明确只取 `pooled_mean`。
