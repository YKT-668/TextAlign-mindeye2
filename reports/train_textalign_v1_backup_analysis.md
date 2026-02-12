# Train_textalign_v1_backup.py 训练参与模块与张量维度报告

> 面向命令：
>
> ```bash
> nohup env \
>   CUDA_VISIBLE_DEVICES=0 \
>   MINDEYE_TEXTALIGN=1 \
>   MINDEYE_TEXTALIGN_SCALE=0.05 \
>   LOG_STEP_INTERVAL=50 \
>   python src/Train_textalign_v1_backup.py \
>     --model_name s1_textalign_coco_train_long_v2 \
>     --data_path /home/vipuser/MindEyeV2_Project/src \
>     --cache_dir "$HF_HOME" \
>     --subj 1 \
>     --num_sessions 40 \
>     --num_epochs 10 \
>     --batch_size 8 \
>     --hidden_dim 1024 \
>     --n_blocks 4 \
>     --no-use_prior \
>     --no-blurry_recon \
>     --no-use_image_aug \
>   > train_logs/s1_textalign_coco_train_long_v2.log 2>&1 &
> ```
>
> 本报告基于仓库当前代码与本地数据文件的“实测 shape”。
> - 训练入口：`src/Train_textalign_v1_backup.py`
> - TextAlign 模块定义：`src/models_textalign.py`

---

## 1. 结论速览：本次训练到底训练了哪些“模型/模块”？

本次命令会构建并参与前向/反传的模块（按数据流顺序）：

1) **RidgeRegression（可训练）**
- 位置：`src/Train_textalign_v1_backup.py` 内部类 `RidgeRegression`
- 作用：把 fMRI 体素向量投影到 `hidden_dim=1024`

2) **BrainNetwork（可训练）**
- 位置：`src/models_textalign.py` 类 `BrainNetwork`
- 作用：把 `hidden_dim=1024`（seq_len=1）映射成 **OpenCLIP bigG token embedding 形状** `[B, 256, 1664]`
- 同时输出用于 CLIP 对比学习的投影 `clip_voxels`（也是 token 级别）

3) **TextAlignHead（参与前向/反传，但“默认不会被 optimizer 更新”）**
- 位置：`src/models_textalign.py` 类 `TextAlignHead`
- 作用：把 `[B,256,1664]` 的 token 表示平均池化后 MLP 映射到文本向量 `[B,768]`，与 teacher 文本特征对齐
- 重要：当前脚本的 optimizer 参数组 **没有** 包含 `model.text_head` 的参数，因此它的权重即使产生梯度也不会被更新（详见第 4 节）。
  - 这意味着 TextAlign loss 仍然会给 `BrainNetwork` 产生梯度（通过 `text_head` 反传到 backbone tokens），但 `text_head` 自身不会学习。

4) **FrozenOpenCLIPImageEmbedder（冻结，仅做 teacher）**
- 位置：`src/Train_textalign_v1_backup.py` 中 `FrozenOpenCLIPImageEmbedder`
- 作用：把 COCO 图像编码成 bigG tokens，作为 `clip_target`（训练目标）
- 状态：`eval()` 且 `requires_grad_(False)`，不参与训练更新

本次命令**明确关闭**的模块：
- `--no-use_prior`：**Diffusion Prior / unCLIP** 不会创建、不会训练
- `--no-blurry_recon`：**SD VAE + blurry reconstruction head** 不会创建、不会训练
- `--no-use_image_aug`：图像增强不会启用

但需要注意：
- 你没有显式传 `--clip_scale 0`，所以 `clip_scale` 仍然是默认值 `1.0`，因此 **CLIP 对比损失仍会计算**，也就仍然需要图像与 `clip_img_embedder`。

---

## 2. 数据与文件：训练数据来自哪里？基础维度是什么？

### 2.1 fMRI betas（体素向量）
- 文件：`betas_all_subj01_fp32_renorm.hdf5`
- Dataset key：`betas`
- 实测 shape：
  - `betas.shape == (30000, 15724)`
  - 含义：每个样本（trial / image）对应一个 15724 维体素向量

因此本次命令（subj01）的体素维度为：
- `num_voxels = 15724`

### 2.2 COCO 图像（用于 CLIP teacher）
- 文件：`coco_images_224_float16.hdf5`
- Dataset key：`images`
- 实测 shape：
  - `images.shape == (73000, 3, 224, 224)`，dtype `float16`

训练时图像按需懒加载（不会一次性读入 73k）。

### 2.3 TextAlign teacher 文本特征（CLIP-L text）
- 文件：`data/nsd_text/train_coco_text_clip.pt`
- 实测内容：
  - `image_ids.shape == (9000,)`
  - `text_feats.shape == (9000, 768)`

这意味着：只有当训练 batch 中的 `global COCO image id` 落在 `image_ids` 这 9000 个集合内时，才会参与 TextAlign loss（否则该样本的 TextAlign loss 为 0）。

### 2.4 hard-negative 文本特征（本仓库当前缺失）
- 脚本期望路径：`data/nsd_text/train_coco_captions_hard_negs_clip.pt`
- 本工作区现状：该文件 **不存在**，因此 `USE_HARD_NEG=False`，hard-neg 分支不会启用。

---

## 3. 模型结构（清晰拆解）

本次训练总体计算图可以概括为：

```
WebDataset behav -> voxel_idx + image_id
        |                   |
        |                   +-> LazyH5Images -> image [B,3,224,224]
        |                                  |
        |                                  +-> FrozenOpenCLIPImageEmbedder -> clip_target [B,256,1664]
        v
betas_hdf5[voxel_idx] -> voxel [B,1,15724]
        |
        +-> RidgeRegression (Linear 15724->1024) -> voxel_ridge [B,1,1024]
                    |
                    v
             BrainNetwork (mixer + big linear) ->
                backbone tokens [B,256,1664]
                clip_voxels     [B,256,1664]

Loss 1 (CLIP contrastive): clip_voxels vs clip_target
Loss 2 (TextAlign InfoNCE): TextAlignHead(backbone) -> t_pred [B,768] vs t_pos [B,768]
```

### 3.1 RidgeRegression（脑信号到 hidden_dim）
- 输入：`voxel0`，shape `[B_subj, 1, num_voxels]`
- 实际 sub1：`num_voxels=15724`
- 模块：`Linear(15724 -> hidden_dim=1024)`
- 输出：`[B_subj, 1, 1024]`

### 3.2 BrainNetwork（hidden_dim 到 bigG tokens）
构造参数（本次命令）：
- `h=hidden_dim=1024`
- `seq_len=1`
- `n_blocks=4`
- `clip_size=1664`
- `out_dim=clip_emb_dim*clip_seq_dim = 1664*256 = 425984`
- `blurry_recon=False`
- `clip_scale=1.0`（默认）

内部主要部件：
- `mixer_blocks1`: 4 个 token-mixing block（对最后一维 1024 做 MLP + 残差）
- `mixer_blocks2`: 4 个 channel-mixing block（对 `seq_len` 做 MLP + 残差；但 `seq_len=1` 时几乎退化）
- `backbone_linear`: `Linear(h*seq_len -> out_dim)`，即 `Linear(1024 -> 425984)`
- reshape 成 tokens：`[B, 425984] -> [B, 256, 1664]`
- `clip_proj`: projector MLP，把 `[B,256,1664] -> [B,256,1664]`（作为对比学习分支输出）

输出：
- `backbone`: `[B, 256, 1664]`
- `clip_voxels`: `[B, 256, 1664]`（当 `clip_scale>0` 时）
- `b`：当 `blurry_recon=False` 时是占位 tensor，不会在训练中使用

### 3.3 TextAlignHead（tokens 到 text embedding）
构造参数：
- `token_dim=1664`
- `hidden_dim=2048`
- `text_dim=768`

前向：
- 输入 `x`：`[B_valid, 256, 1664]`
- mean-pool：`[B_valid, 1664]`
- MLP：`LayerNorm(1664) -> Linear(1664->2048) -> GELU -> Linear(2048->768)`
- 输出：`t_pred` shape `[B_valid, 768]`

### 3.4 FrozenOpenCLIPImageEmbedder（teacher）
- 构造：`FrozenOpenCLIPImageEmbedder(arch="ViT-bigG-14", version="laion2b_s39b_b160k", output_tokens=True)`
- 输入：`image` shape `[B, 3, 224, 224]`
- 输出：`clip_target`（脚本把它当作 token 表示使用）
- 训练中处理：no_grad + eval + requires_grad(False)

---

## 4. “哪些参数真的在训练更新？”（非常关键）

脚本里 optimizer 的参数组（`opt_grouped_parameters`）只包含：
- `model.ridge`
- `model.backbone`
- （若 `use_prior=True` 才会加）`model.diffusion_prior`

**脚本没有把 `model.text_head` 加入 optimizer。**

因此在当前实现下：
- `TextAlignHead` 虽然参与前向并产生梯度，但 optimizer 不会更新它的权重。
- TextAlign loss 依然会通过 `text_head` 把梯度传到 `backbone`（即 BrainNetwork 输出 tokens），从而影响 `model.backbone` 的更新。

如果你的真实意图是“训练 TextAlignHead”，那你需要把 `model.text_head.parameters()` 加入 optimizer 参数组（这属于代码改动；本报告只做现状分析）。

---

## 5. 训练过程中张量维度如何变化？（按一次 train step 展开）

下述维度以 **单卡/单进程**、`subj_list=[1]`、`args.batch_size=8` 为例。

> 注意：脚本里会执行 `batch_size = max(1, batch_size // len(subj_list))`。
> - 单被试时 `len(subj_list)=1`，所以 `batch_size` 仍为 8。
> - 若多被试训练，会先均分到每个被试，再在后面 concat。

### 5.1 从 WebDataset batch 取索引
WebDataset 输出四个张量：`behav0, past_behav0, future_behav0, old_behav0`。

脚本只用到了 `behav0` 的两个字段：
- `image_idx = behav0[:, 0, 0]`：全局 COCO image id（用于取图像与 TextAlign 对齐）
- `voxel_idx = behav0[:, 0, 5]`：用于索引 betas HDF5 的行号

因此可以推断：
- `behav0` shape 形如 `[B_subj, 1, K]`，且 `K >= 6`

同时脚本会强制一个约束：
- 如果 batch 内 `image_idx` 有重复，会丢弃该 batch 并继续取下一个（直到无重复）。

### 5.2 懒加载图像与 CLIP teacher
- `image0`：去重后的 image ids，长度 `B_subj`
- `img_tensor = lazy_coco.get(image0)`
  - `img_tensor` shape `[B_subj, 3, 224, 224]`
- 拼接后（单被试不变）：
  - `image` shape `[B, 3, 224, 224]`（这里 `B=8`）
  - `global_ids` shape `[B]`

CLIP teacher：
- `clip_target = clip_img_embedder(image)`（no_grad）
- 期望 shape：`[B, 256, 1664]`

### 5.3 取 betas 并做 ridge
- `voxel0_np = betas[voxel_sorted_idx]`：shape `[B_subj, 15724]`
- `voxel0 = torch.tensor(voxel0_np).unsqueeze(1)`：shape `[B_subj, 1, 15724]`

Ridge：
- `voxel_ridge_list[0] = model.ridge(voxel_list[0], subj_idx=0)`
- 输出 `voxel_ridge` shape `[B, 1, 1024]`

### 5.4 BrainNetwork 输出 tokens
- `backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)`

其中：
- `backbone` shape `[B, 256, 1664]`
- `clip_voxels` shape `[B, 256, 1664]`（因为 `clip_scale>0`）
- `blurry_image_enc_`：本次 `blurry_recon=False`，不使用

### 5.5 TextAlign loss（只在 teacher 覆盖到的样本子集上计算）

先由 `global_ids` 映射到 teacher 行号：
- `rows = [id2row.get(gid, -1) ...]` -> tensor shape `[B]`
- `valid_mask = rows >= 0` -> shape `[B]`

只对有效样本：
- `B_valid = valid_mask.sum()`
- `backbone_tokens_valid = backbone[valid_mask]` -> `[B_valid, 256, 1664]`
- `t_pos = text_feats_teacher[rows_valid]` -> `[B_valid, 768]`
- `t_pred = model.text_head(backbone_tokens_valid)` -> `[B_valid, 768]`

InfoNCE：
- `logits = t_pred @ t_pos.T / tau` -> `[B_valid, B_valid]`
- `loss_text = CrossEntropy(logits, labels)` -> scalar

最终加权：
- `loss += alpha_text * L_text`，其中本命令 `alpha_text=0.05`

### 5.6 CLIP 对比损失（MixCo / SoftCLIP）
脚本将 tokens flatten 到向量：
- `clip_voxels.flatten(1)` -> `[B, 256*1664] = [B, 425984]`
- `clip_target.flatten(1)` -> `[B, 425984]`
- normalize 后进入：
  - 前期（epoch < mixup_pct*num_epochs）：`utils.mixco_nce(...)`
  - 后期：`utils.soft_clip_loss(...)`

最终加权：
- `loss += clip_scale * loss_clip`，本次 `clip_scale=1.0`

### 5.7 反传与更新
- `accelerator.backward(loss)`（在后续代码段中执行）
- `optimizer.step()`
- scheduler step

更新到的参数（按当前 optimizer 配置）：
- ✅ `model.ridge.*`
- ✅ `model.backbone.*`
- ❌ `model.text_head.*`（不在 optimizer 中）

---

## 6. 与你的命令逐项对齐（启用/禁用清单）

- `--subj 1`：只训练 subj01（因此 `num_voxels=15724`）
- `--batch_size 8`：单被试时每步总 batch `B=8`（每个 step 的图像/voxel/clip_target 都是 8 条）
- `--hidden_dim 1024`：Ridge 输出与 BrainNetwork 隐层维度
- `--n_blocks 4`：BrainNetwork mixer block 数
- `--no-use_prior`：Diffusion prior 不存在、不训练
- `--no-blurry_recon`：VAE/Convnext blurry recon 不存在、不训练
- `--no-use_image_aug`：无图像增强
- `MINDEYE_TEXTALIGN=1` + `MINDEYE_TEXTALIGN_SCALE=0.05`：TextAlign loss 启用，但仅对 `teacher image_ids` 命中的样本生效

---

## 7. 你可以用哪些日志快速验证本报告的 shape？

脚本本身已经在 TextAlign 部分提供了早期 step 的调试打印（epoch 0 前 3 step）：
- batch_size
- shared_hits（teacher 覆盖的样本数）
- use_hard_neg

如果你想额外核对关键张量 shape，最省事的位置通常是：
- `clip_target` 产生后
- `voxel0_np/voxel0` 产生后
- `voxel_ridge`、`backbone`、`clip_voxels` 产生后

---

## 8. 关键定位（代码入口点）

- 数据索引字段：`behav0[:,0,0]`（image id），`behav0[:,0,5]`（voxel_idx）
- betas 加载与 shape：`betas_all_subj0{s}_fp32_renorm.hdf5` 的 `betas`
- 图像加载：`LazyH5Images` + `coco_images_224_float16.hdf5`
- CLIP teacher：`FrozenOpenCLIPImageEmbedder`
- 主干模型：`RidgeRegression` + `BrainNetwork`
- TextAlign：`train_coco_text_clip.pt` + `TextAlignHead` + `text_align_loss`

---

## 9. 附：本报告中出现的核心 shape 汇总表

| 名称 | 含义 | shape（本次命令典型值） |
|---|---|---|
| `betas` | subj01 fMRI 体素库 | `[30000, 15724]` |
| `image` | COCO 图像 batch | `[B, 3, 224, 224]`（B=8） |
| `clip_target` | bigG 图像 tokens（teacher） | `[B, 256, 1664]` |
| `voxel0` | 体素 batch（取自 betas） | `[B, 1, 15724]` |
| `voxel_ridge` | ridge 后 hidden | `[B, 1, 1024]` |
| `backbone` | 预测 bigG tokens（backbone 输出） | `[B, 256, 1664]` |
| `clip_voxels` | 对比学习分支 tokens | `[B, 256, 1664]` |
| `t_pos` | teacher 文本向量 | `[B_valid, 768]` |
| `t_pred` | 预测文本向量 | `[B_valid, 768]` |
| `logits` | TextAlign InfoNCE logits | `[B_valid, B_valid]` |

