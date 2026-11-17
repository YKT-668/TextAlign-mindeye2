# Ours S1 第一阶段重新推理预检报告

> 日期：2026-11-14。只读审计，未运行 GPU/训练/推理，未下载新增大型文件。结论标注 VERIFIED / INFERRED / BLOCKED。

## 执行摘要

**可以只重跑第一阶段推理、无需重新训练**：Ours S1 checkpoint、正式推理代码、必要数据（beta + new_test）、unCLIP/GIT 依赖均可用。不确定项：评估用 `coco_images_224_float16.hdf5`（21GB）是否必须，以及 GPU 尚未租。

## 1. 是否可以不训练、只执行第一阶段重新推理

**可以**。三要素齐备：checkpoint（VERIFIED）、推理代码（VERIFIED，GitHub 公开 `YKT-668/TextAlign-mindeye2`）、输入数据（VERIFIED）。

## 2. S1 checkpoint 是否可用

**可用（VERIFIED）**。
- `ykt668/textalign-mindeye2-model` @ `a127f9295fd5656ac63ae436f07e61b80bf4efce`（master），路径 `checkpoints/s1_textalign_stage1_FINAL_BEST_32/last.pth`。
- 大小 26,788,144,501 ✓；SHA256 `eb59c4262c5f6316285d5ad78b945af27c0fb565444278da6b83551584b8e91f` ✓（与预期完全一致）。
- 本机无本地副本，需下载 26.8GB。
- 未在 4GB CPU 上加载 checkpoint。结构由代码默认 + provenance 推断：`models_textalign.BrainNetwork` `h=4096, n_blocks=4, clip_size=768, seq_len=2` + `TextAlignHead(token_dim=1664, hidden_dim=2048, text_dim=768)`（INFERRED，最终以 checkpoint metadata 为准）。

## 3. 必需 beta 和 new_test 数据是否可用

**可用（VERIFIED）**，均在 `pscotti/mindeyev2` @ `3f20ccbe614876c83894e148cc36530c4b04caa7`：

| 文件 | 大小 | SHA256 | 结论 |
|---|---|---|---|
| `betas_all_subj01_fp32_renorm.hdf5` | 1,886,882,048 | 71992eb8… ✓ | 必需 |
| `wds/subj01/new_test/0.tar` | 38,410,240 | 16f810c0… ✓ | 必需（new_test） |
| `unclip6_epoch0_step110000.ckpt` | 17,980,447,939 | 6bd7033d… ✓ | 必需（unCLIP） |
| `bigG_to_L_epoch8.pth` | 7,086,032 | 2f540faf… | 必需（GIT 转换） |
| `brain_region_masks.hdf5` | 841,773 | — | 评估辅助，可不下 |
| `COCO_73k_subj_indices.hdf5` | 629,114 | — | ID 辅助，可不下 |
| `shared1000.npy` | 73,128 | d909f565… | 评估映射，可不下 |
| `subj01_annots.npy` | 24,360,128 | ba9c7b9a… | 评估标注，可不下 |
| `test_73k_images.npy` | 8,128 | 59d772c1… | ID→行号（对齐用，建议下） |
| `vox_include_subj01.npy` | — | — | **MISSING** |

- `vox_include_subj01.npy`（5.59MB）在该 revision **不存在**。判断为训练期数据准备产物，推理不需要（INFERRED：官方 `recon_inference.ipynb` 与我们的脚本均未加载它；README3 资源清单未列它；betas 已 `_renorm` 归一化）。
- `sd_image_var_autoenc.pth`（blurry 解码）不在本清单，blurry_recon=False 可跳过（INFERRED）。
- 遵守 NSD 数据协议，不绕过访问限制。

## 4. unCLIP 和 GIT 权重依赖

| 依赖 | 类型 | 说明 |
|---|---|---|
| `unclip6_epoch0_step110000.ckpt`（16.75GB） | A 单独下载 | 必需 |
| `bigG_to_L_epoch8.pth`（7.09MB） | A 单独下载 | 必需 |
| `microsoft/git-large-coco` | B 运行时自动 | 必需（~1.5GB） |
| OpenCLIP ViT-bigG-14 + CLIP L | B 运行时自动 | 必需（~5GB） |
| `sd_image_var_autoenc.pth`（335MB） | A 单独下载 | 可选（blurry） |
| `zavychromaxl_v30.safetensors`（已下载） | — | 第二阶段用，第一阶段不用 |

## 5. 正确的代码版本

- 两候选版本几乎相同：`archive/final14@231850a…`（40,631B）与 `master@80bb079c…`（40,695B）的 `recon_inference_run.py` 头部/导出段逐字一致，仅差 64 字节（差异未定位，INFERRED）。
- `models_textalign.BrainNetwork`：`h=4096, in_dim=15724, out_dim=768, seq_len=2, n_blocks=4, clip_size=768` + TextAlign head。与 checkpoint 结构匹配（INFERRED）。
- **与 MedARC 官方差异（VERIFIED）**：官方 `models.py` 用 `clip_size=1664`（bigG）无 TextAlign head；我们的 `models_textalign.py` 用 `clip_size=768`（CLIP L）+ TextAlign head。**必须用 `models_textalign.BrainNetwork`**。
- 建议采用 `archive/final14@231850a`（与 readme5-1.md 归档一致，含 `--export_official_pts`）。

## 6. 数据和 ID 对齐

- `new_test/0.tar` 含 3000 条行为记录，去重后 **1000 个唯一图像 ID**，每图 3 次 fMRI 重复（官方 Cell[8] 取 3 次平均，VERIFIED）。
- GT `all_images.pt`（[1000,3,224,224]）覆盖同一 1000 张；**S1 new_test 的 1000 与四被试 shared982 是不同口径，勿混用**（任务明确要求）。
- 对齐链：`test_73k_images.npy`（NSD ID→0-999 行号）映射到 `all_images.pt`；重跑后以 `all_ids.pt` 为准对齐。
- **BLOCKED**：因原三文件缺失，无法预先确认 GT 行序与 IDs 精确对应；拿到重跑结果后用 `test_73k_images.npy` + `all_ids.pt` 对齐。

---

## 7. 新增下载清单及大小

必需新增（A 类，约 45.6GB）：
| 文件 | 大小 |
|---|---|
| `last.pth`（checkpoint） | 26.8GB |
| `unclip6_epoch0_step110000.ckpt` | 16.75GB |
| `betas_all_subj01_fp32_renorm.hdf5` | 1.76GB |
| `wds/subj01/new_test/0.tar` | 38.4MB |
| `bigG_to_L_epoch8.pth` | 7.09MB |
| `sd_image_var_autoenc.pth`（blurry 可选） | 335MB |

可选/评估：`coco_images_224_float16.hdf5`（21GB，仅评估 cossim）；`test_73k_images.npy`（8KB，建议下）。运行时自动：GIT ~1.5GB + OpenCLIP ~5GB。

## 8. 当前数据盘空间是否充足

**充足（VERIFIED）**。137G 可用；新增 ~45.6GB + 运行时 ~6.5GB + zavy 剩余 ~6.7GB ≈ **~59GB** < 137G。硬链接不产生双倍占用。

## 9. GPU 配置和推理耗时估算

- **GPU**：A100 80GB 足够；48GB 也可（README3 实测精炼峰值 28.1GB，表示推理 ~10.5GB）。
- **耗时**（README3 实测口径，非本机）：3000 样本表示推理 ~3 分钟；1000 张 unCLIP ~95 分钟（1000×5.7s）。合计 **约 1.5–2 小时**。
- **存储**：enhancedrecons [1000,3,224,224] float16 ≈ 287MiB + captions/ids（KB级）+ 中间文件（backbones/clipvoxels/prior_out 各 ~0.85GB）≈ 总计 **~2.5–3GB**（可最小输出只留三件必需）。

## 10. 当前阻塞问题

1. **GPU 未租**（需 24GB+，推荐 48–80GB）。
2. `vox_include_subj01.npy` 缺失（判断非必需，INFERRED）。
3. `coco_images_224_float16.hdf5`（21GB）是否必须取决于是否做 CLIP cossim 评估。
4. 7-10 运行的确切 commit 无直接证据（两候选版本几乎相同，任一可用，INFERRED）。

## 11. 推荐的最小执行方案

1. GPU 服务器 clone `YKT-668/TextAlign-mindeye2` @ `archive/final14`（`231850a`）。
2. 下载：checkpoint、unclip ckpt、beta、new_test/0.tar、bigG_to_L、test_73k_images.npy；`--no-blurry_recon` 跳过 autoenc。
3. 正式命令（独立 rerun 目录 + 新运行编号）：
   ```bash
   python src/recon_inference_run.py --data_path=... --cache_dir=... \
     --output_dir=.../rerun_v2/s1_textalign_stage1_FINAL_BEST_32 \
     --model_name=s1_textalign_stage1_FINAL_BEST_32 \
     --subj=1 --new_test --seed 42 --export_official_pts --max_save 1000
   ```
4. 输出 enhancedrecons（原始 unCLIP 重建）+ predcaptions + ids，不覆盖旧 checksum/provenance。
5. 之后用 `test_73k_images.npy` + `all_ids.pt` 对齐 GT，再做方案 A img2img 精炼。

*注：本报告只审计、不执行。不得自行开始新增大型文件下载或 GPU 实验。*
