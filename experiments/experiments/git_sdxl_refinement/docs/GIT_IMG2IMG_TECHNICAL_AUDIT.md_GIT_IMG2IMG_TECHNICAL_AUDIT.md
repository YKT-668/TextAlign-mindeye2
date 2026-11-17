# GIT IMG2IMG 技术审计报告

> 状态标注说明：**VERIFIED**=有代码/文件证据确认；**INFERRED**=根据已有证据推断；**BLOCKED**=缺少必要材料暂无法确认。
> 审计日期：2026-11-14。本报告只读，未运行任何 GPU/推理/训练，未修改项目文件。

## 执行摘要

1. 官方 MindEye2「GIT 文本引导 SDXL img2img 精炼」流程已**完全还原**（VERIFIED，来自 `enhanced_recon_inference.ipynb` 的 6 个 cell）。
2. 官方精炼阶段**不使用 SDXL Refiner**，只用一个 SDXL base 变体权重 `zavychromaxl_v30.safetensors` + unclip6 采样器配置（VERIFIED）。SDXL Refiner 在本补充实验中**不需要**。
3. 官方精炼的 img2img 输入是**第一阶段未精炼的 unCLIP 重建图**（`*_all_recons.pt`，256×256），caption 是第一阶段 GIT 生成并保存的 `*_all_predcaptions.pt`（numpy 数组）。
4. 我们自己的推理代码**不在本服务器、也不在任何 HF 仓库中**（BLOCKED）。仅能根据 provenance/manifest 推断：本地仓库在 `/mnt/work/repos/TextAlign-mindeye2`，推理入口为 `src/recon_inference_run.py`（带 `--export_official_pts`）。
5. `step6d_full_ours_llmhard_all_enhancedrecons.pt` 是**第二阶段（img2img 精炼后）的输出**（INFERRED：命名与官方 `*_all_enhancedrecons.pt` 一致 + provenance 描述「SDXL-enhanced reconstruction」）。**没有发现单独保存的未精炼 unCLIP 重建文件**，这是方案 A 的一个关键疑点（详见第 9/10 节）。
6. Ours S1 三个文件在 HF 上确实缺失（VERIFIED，文件名+SHA256 双重检索）；其唯一已知位置是原 A100 服务器 `/mnt/aaai_ablation_disk`（`/dev/vdc`）上的 `aaai_readme3_full_reconstruction_maxthroughput_20260710_150646/ours_llmhard/`（VERIFIED 来自 checksum 文件路径）。
7. 已下载资产中，`zavychromaxl_v30.safetensors` 与 GT 是必需的；SDXL Refiner 非必需（保留但不使用）。

---

## 一、官方 MindEye2 GIT 精炼流程（VERIFIED）

证据来源：`$ROOT/source/MindEyeV2/src/enhanced_recon_inference.ipynb`（commit `76a2609243ba1bacb450e47e3dc78de1f78a9c00`），共 6 个 cell。

### 1.1 第二阶段（精炼）需要的输入

Cell [2] 加载了 5 个文件：

| 文件 | 是否真正需要 | 用途 |
|---|---|---|
| `evals/all_images.pt`（GT） | **否**（仅评估/可视化） | 仅 `plotting=True` 或 `num_samples>1` 时用于 cossim 选图/画图；`num_samples=1` 且 `plotting=False` 时**完全不参与生成** |
| `*_all_recons.pt`（未精炼重建） | **是**（img2img 条件图像） | 精炼的输入图像 |
| `*_all_predcaptions.pt`（GIT caption） | **是**（文本条件） | `prompt = all_predcaptions[[img_idx]][0]` |
| `*_all_clipvoxels.pt` | **否** | 被加载但循环中**从未使用**（遗留） |
| `*_all_blurryrecons.pt` | **否** | 仅用于 `plotting` 时显示 |

结论（VERIFIED）：生成真正必需 = **未精炼重建图 + caption**；GT / blurry / clipvoxels 只用于评估或可视化。

### 1.2 精炼实现（按代码执行顺序，Cell [3]-[5]）

1. **加载未精炼重建**：`all_recons = torch.load(..._all_recons.pt)`，原始尺寸 **256×256**（第一阶段保存时 `imsize=256`）。
2. **放大到 768×768**：`transforms.Resize((768,768))(all_recons)`。原因：SDXL 的 VAE 8× 下采样，768/8=96 → latent 形状 `[1,4,96,96]`；代码 `assert image.shape[-1]==768` 确认。
3. **图像编码为 latent**：`z = base_engine.encode_first_stage(image*2-1)`（图像归一到 [-1,1] 后 VAE 编码）。
4. **两个文本编码器**：`FrozenCLIPEmbedder(layer=hidden, layer_idx=11)`（OpenAI CLIP L/14）+ `FrozenOpenCLIPEmbedder2(ViT-bigG-14)`（OpenCLIP）。两者输出拼接：`crossattn = cat(openai_clip_text, clip_text_tokenized)`，`vector = hstack(clip_text_emb, vector_suffix)`（vector_suffix 为尺寸/crop 条件）。
5. **加噪（img2img）**：`noise=randn_like(z)`；`sigmas=discretization(num_steps=25)`；`init_z = (z + noise*append_dims(sigmas[-13])) / sqrt(1+sigmas[0]^2)`；再 `sigmas = sigmas[-13:]`。
6. **去噪**：`sampler.num_steps = sigmas.shape[-1]-1 = 12`；循环 `range(12)` 调 `sampler_step(..., gamma=0)`。
7. **解码保存**：`samples_x = decode_first_stage(noised_z)`；`samples = clamp((samples_x+1)/2, 0, 1)`；逐张 `vstack`；最后 `Resize((256,256))` 后 `torch.save` 为 `*_all_enhancedrecons.pt`。

### 1.3 实际参数（VERIFIED，与任务引用一致）

```python
num_samples = 1                      # Cell[4] L184
img2img_timepoint = 13               # Cell[4] L185
base_engine.sampler.guider.scale = 5 # Cell[4] L186  (CFG)
base_engine.sampler.num_steps = 25   # Cell[5] L204
```
- **采样器**：来自 `unclip6.yaml` 的 `EulerEDMSampler` + `VanillaCFG`（`guider_config.scale=5.0`），但 `num_steps` 由 38（Cell[3] 设的）在循环内覆盖为 **25**。
- **实际去噪步数**：`sigmas[-13:]` 取 13 个 sigma → `num_steps = 13-1 = 12` → **去噪 12 步**（VERIFIED 来自代码逻辑）。
- **输入/输出分辨率**：输入重建 256→768×768（latent 96×96），输出 768×768 后 resize 回 **256×256** 保存。
- **随机种子**：`seed_everything(42)`（Cell[2]），每张图**不单独设种子**，共享全局确定性序列（VERIFIED）。

### 1.4 官方需要的权重（VERIFIED，按 notebook 实际加载代码判断）

| 权重 | 是否加载 | 说明 |
|---|---|---|
| `zavychromaxl_v30.safetensors` | **是**（精炼阶段） | `base_ckpt_path = '.../zavychromaxl_v30.safetensors'`（Cell[3] L130），作为 SDXL base 引擎权重 |
| SDXL Base `sd_xl_base_1.0.safetensors` | **否**（精炼阶段） | 未在 enhanced_recon 中加载；只在 `unclip6.yaml` 的训练配置 `ckpt_config` 里被引用 |
| SDXL Refiner | **否** | enhanced_recon 完全不引用 refiner（`sd_xl_refiner.yaml` 存在于配置目录但未被 notebook 使用）|
| unCLIP checkpoint `unclip6_epoch0_step110000.ckpt` | 是（**第一阶段**） | 在 `recon_inference.ipynb` Cell[7] 加载，用于生成未精炼重建 |
| 文本编码器 CLIP L + OpenCLIP bigG | 是（自动下载） | 通过 transformers/open_clip 自动加载 |

**结论（VERIFIED）：官方 GIT 精炼阶段不需要单独的 SDXL Refiner 权重。**

### 1.5 是否存在图像融合

**否（VERIFIED）**。精炼输出 = img2img 去噪后 `decode_first_stage` 直接得到，无任何与 blurry/原图的加权融合、无后处理。blurry recons 仅用于 `plotting` 可视化（Cell[5] L209-211、L222-223）。不要把第一阶段内部「blurry 低层重建」当作精炼融合——两者是独立产物。

### 1.6 GIT 在整体流程中的位置（VERIFIED）

- **第一阶段**（`recon_inference.ipynb`）：fMRI → ridge → backbone → `clip_voxels`；`diffusion_prior.p_sample_loop(timesteps=20)` → `prior_out`；`clip_convert`（`bigG_to_L_epoch8.pth`）→ GIT（`microsoft/git-large-coco`）`generate(max_length=20)` → **caption**，保存为 numpy 数组 `*_all_predcaptions.pt`。
- **unCLIP 生成**（第一阶段）：`utils.unclip_recon(prior_out, ...)`，`z = torch.randn(num_samples,4,96,96)` **从随机 latent 开始**，条件 = `prior_out`（crossattn）+ `vector_suffix`（vector），**不使用 caption**（caption 是并行产物，供第二阶段用）。
- **第二阶段**：加载保存好的 caption 作为文本条件做 img2img。

---

## 二、我们自己的推理代码审计

### 2.1 代码位置与版本（部分 BLOCKED）

| 项 | 结论 |
|---|---|
| 本地仓库路径 | **BLOCKED**（本服务器不存在）。已知历史路径 `/mnt/work/repos/TextAlign-mindeye2`（VERIFIED，来自 `RECOVERY_MANIFEST.json` 的 `root` 字段） |
| 当前分支 | **BLOCKED**（git 信息在 `ykt668/textalign-mindeye2-model` 的 `ENV_SNAPSHOT/git_remote_v.txt`、`git_status_short.txt` 中，本次未抓取） |
| 当前 commit | **BLOCKED**（`ENV_SNAPSHOT/git_rev_parse_head.txt`，41 字节） |
| 是否有未提交修改 | **BLOCKED**（`ENV_SNAPSHOT/git_status_short.txt`、`git_diff_stat.txt`） |
| 生成 Ours S1 的代码版本 | **INFERRED**：推理入口 `src/recon_inference_run.py`（`--export_official_pts --max_save 1000`），见 readme3.md / READMЕ3 报告 |

本服务器、`/mnt/work`、`/mnt/data/TextAlign-AAAI` 均无该仓库；HF 数据/模型仓库也**不含** `recon_inference_run.py`/`models_textalign.py`（模型仓库仅备份 3 个 .py 工具脚本，其余代码在 GitHub）。

### 2.2 我们的推理链（INFERRED）

推理命令（VERIFIED 来自 readme3.md）：
```bash
python src/recon_inference_run.py --data_path=... --cache_dir=... \
  --output_dir=.../evals/s1_textalign_stage1_FINAL_BEST_32 \
  --model_name=s1_textalign_stage1_FINAL_BEST_32 \
  --subj=1 --new_test --seed 42 --export_official_pts --max_save 1000
```

输出 7 个 `.pt`（VERIFIED 来自 FINAL_OUTPUT_CHECKSUMS.sha256 / READMЕ3 报告）：`all_backbones`、`all_clipvoxels`、`all_prior_out`、`all_blurryrecons`、`all_enhancedrecons`、`all_predcaptions`、`all_ids`。

推断链路（与官方 `recon_inference.ipynb` 同构 + TextAlign head / hard-neg 改动）：
fMRI → ridge → backbone（`models_textalign.py`，含 TextAlign head）→ clip_voxels → diffusion prior → prior_out →（GIT 生成 caption，保存）→ unCLIP（随机 latent 开始）→ 未精炼重建 →（第二阶段 img2img）→ enhancedrecons。

- GIT 是否真实执行：**VERIFIED**（存在 `all_predcaptions.pt`，第一阶段链含 GIT `generate`）。GIT 输入 = `prior_out` 经 `clip_convert`；caption 被保存；unCLIP 从随机 latent 开始（官方 `unclip_recon` 中 `z=randn`，VERIFIED）；caption 不进入第一阶段生成，只进入第二阶段 img2img。

### 2.3 `step6d_full_ours_llmhard_all_enhancedrecons.pt` 是什么

**INFERRED（倾向 B：第二阶段精炼后图像）**：
- 命名 `*_all_enhancedrecons.pt` 与官方第二阶段输出完全一致（VERIFIED）。
- fig4 `MODEL_PROVENANCE_REPORT.md` 明确写「SDXL-enhanced reconstruction」（VERIFIED）。
- **注意**：7 个导出文件里**没有** `all_recons.pt`（未精炼 unCLIP）。若方案 A 需要「未精炼重建」作 img2img 输入，该文件**缺失**（BLOCKED，见第 10 节）。
- 因源码不可得，无法用「实际代码」严格证明；以上为命名约定 + provenance 的强推断。

### 2.4 输出数据格式

| 文件 | 格式（来源） |
|---|---|
| `*_all_enhancedrecons.pt` | `[1000,3,224,224]` float16（VERIFIED，fig4 provenance）；官方是 256×256，我们是 224×224（INFERRED：对齐 GT 224） |
| `*_all_predcaptions.pt` | **BLOCKED**（官方为 numpy 字符串数组 shape(N,)；任务提示我们可能是 `list[str]`） |
| `*_all_ids.pt` | **BLOCKED**（应为一维 int ID；s2/s5/s7 的 ids.pt 仅 9414 字节≈1000 int，可参考） |
| GT `all_images.pt` | `[1000,3,224,224]` float32（VERIFIED） |
| GT `all_captions.pt` | BLOCKED（待文件落盘后确认） |

### 2.5 与官方 notebook 的兼容性

官方：`prompt = all_predcaptions[[img_idx]][0]` → 要求 `all_predcaptions` 支持 `[[idx]]` 索引（numpy 数组）。
- 若我们是 `list[str]`：`list[[idx]]` 会 `TypeError`，**不兼容**。
- 最小转换（在第二阶段脚本里做，不改源文件）：`all_predcaptions = np.array(all_predcaptions)`；或改索引为 `all_predcaptions[img_idx]`。

命名 `*_all_recons.pt`（官方第一阶段未精炼）vs `*_all_enhancedrecons.pt`（精炼后）：**不只是命名差异**，语义不同（未精炼 vs 精炼），tensor 不能直接互换，除非确认我们的 enhancedrecons 实为未精炼输出（BLOCKED）。

### 2.6 最小改动方案（仅提议，不实施）

- **直接复用**：官方 `enhanced_recon_inference.ipynb` 第二阶段逻辑（Cell[2]-[5]）；官方 `sgm`（Stability generative-models）模块。
- **需补充**：一个独立第二阶段脚本（读重建图 + caption，做 img2img 精炼，输出新 enhancedrecons），不修改 `recon_inference_run.py`，不覆盖现有输出。
- **需转换**：caption 格式（如 2.5）；重建图 224→768（官方 VAE 要求 768）。
- **需额外下载**：文本编码器（CLIP L + OpenCLIP bigG，运行时自动拉）；`unclip6_epoch0_step110000.ckpt` 仅重跑第一阶段时需要。
- **需依赖**：`sgm`、`omegaconf`、`einops`、`transformers`、`open_clip_torch`、`diffusers`、`torchvision`、`accelerate`。
- **是否改现有代码**：否，独立脚本即可。

---

## 三、已下载资产与 Ours S1 缺失

### 3.1 已下载资产清单（未完成文件的 SHA 引用远端元数据，不本地计算）

| 项目 | 文件名 | 来源 | 大小 | SHA256 | 是否必需 |
|---|---|---|---|---|---|
| GT 图像 | `evals/all_images.pt` | pscotti/mindeyev2 @3f20ccbe | 602,113,195 | 93698254…566bcd1 | 是（评估） |
| GT Caption | `evals/all_captions.pt` | pscotti/mindeyev2 @3f20ccbe | 813,044 | 5ac53118…a5a1cd（已落盘） | 是（评估） |
| MindEye2 Zavy | `zavychromaxl_v30.safetensors` | pscotti/mindeyev2 @3f20ccbe | 6,938,040,682 | f1270c1f…6411b75 | **是（精炼必需）** |
| SDXL Base | `sd_xl_base_1.0.safetensors` | stabilityai/…base-1.0 @4621659 | 6,938,078,334 | 31e35c80…893f7e5b | 参考（官方精炼用 zavy，不用此） |
| SDXL Refiner | `sd_xl_refiner_1.0.safetensors` | stabilityai/…refiner-1.0 @5d4cfe8 | 6,075,981,930 | 7440042b…7d9c5f | **否（精炼不需要）** |

provenance（已完整下载、SHA 已算）：`MODEL_PROVENANCE_REPORT.md`（2442B，09ed7872…）、`FINAL_OUTPUT_CHECKSUMS.sha256`（5218B，b3ccc399…）、`READMЕ3_A100_FINAL_INFERENCE_REPORT.md`、`readme3.md`、`CANONICAL_SHARED982_SOURCE.md`、`N1000_ALL_IDS_ALIGNMENT.tsv`（在 `reports/provenance/`）。

### 3.2 Ours S1 三个缺失文件调查

SHA（VERIFIED 来自 FINAL_OUTPUT_CHECKSUMS.sha256）：
- enhancedrecons = `df704fca629453c79f80b83a859694e468f1aad994257d31314e86dc7f8a85b2`
- predcaptions = `a6b49190a20f79ce990bc647d7cca032e9503383877c3728bec647ef8c069758`
- ids = `0a837980a67886e2dfc5e0f3b08facdeab7a0d0b7d9ae6309f1f604ca7812d3c`

调查结论：
1. **其他本地副本**：无（本机无 `/mnt/work`、`/mnt/data/TextAlign-AAAI`）。
2. **历史备份**：2026-04-12 HF 快照（114GB）只含 s2/s5/s7，**不含 s1 step6d**（VERIFIED，HF_UPLOAD_MANIFEST.json 148 文件清单）。s1 只有 2026-01-15 的 `brain_clip.pt`+`ids.json`（shared982 评估），非 step6d。
3. **大小/SHA 记录**：有 SHA（checksum 文件）；大小只有 fig4 provenance 的「约 287MiB，[1000,3,224,224] float16」。
4. **原服务器路径**：有（VERIFIED）：`/mnt/aaai_ablation_disk/aaai_readme3_full_reconstruction_maxthroughput_20260710_150646/ours_llmhard/`，磁盘 `/dev/vdc`（295GB），2026-07-10 生成。
5. **输出目录/推理命令**：有（readme3.md / READMЕ3 中的 `recon_inference_run.py` 命令）。
6. **可恢复归档**：无 HF 归档；唯一恢复点 = 原 A100 服务器的 `/dev/vdc`（若被保留/可重挂）。

**恢复路径（只建议，不执行）**：重挂原 A100 实例 `/dev/vdc`，从上述 `ours_llmhard/` 目录拷三个文件并核对 SHA。**不要**擅自连接旧服务器/传输/改远端数据（需人工授权）。

### 3.3 数据对齐（BLOCKED）

因 Ours S1 三文件缺失，无法完成 1000 张一一对应验证（**BLOCKED**）。
- 线索：GT [1000,3,224,224]；ids.pt（缺失）应为一维 int；`CANONICAL_SHARED982_SOURCE.md` 提到「四个 N=1000 all_ids 完全一致」「shared982 ∩ N=1000 = 982/982」，即存在 shared982（982）与 N=1000 两套口径。
- 风险：必须用 ids.pt 对齐精炼前后，**不能按文件名顺序假设**（官方也是 `np.unique(test_images_idx)` 去重后生成）。

---

## 四、评测方案与计算资源

### 4.1 评测指标

我们的评测代码 `src/run_debug.py`（BLOCKED，本机无），但 s1 审计报告（20260115）已给出可用指标集：2AFC（B2I/I2B）、CCD（Hard K1/K2、Random K2）、RSA（Pearson/Spearman）、retrieval（FWD/BWD@1/@5）、CLIP cossim、PixCorr、SSIM、AlexNet（官方 `final_evaluations.ipynb` 里有 AlexNet/Inception/PixCorr/SSIM 等）。

- **直接衡量语义准确度**：CLIP 余弦相似度（image-image 或 image-text）、2AFC、CCD、RSA。适合精炼前后对比。
- **衡量视觉相似度/质量**：PixCorr、SSIM、AlexNet（2-way 相似度）、Inception。
- **不适合本次对比**：纯检索类（FWD/BWD@k 需要大量负样本池，与单图精炼对比关联弱，可作为次要）。

**推荐最小评测组合（工作量最小）**：
1. `CLIP cossim`（精炼前后 vs GT，官方 `utils`/`final_evaluations` 里已有 `FrozenOpenCLIPImageEmbedder` 计算代码）；
2. `PixCorr`（`utils.pixcorr`，官方已实现）；
3. （可选）`SSIM` / `AlexNet 2-way`（官方 `final_evaluations.ipynb` 有实现）。

### 4.2 推理时间设计（建议，不执行）

- 计时边界：模型加载（SDXL engine + 文本编码器，一次性）与「单张 img2img 精炼」（encode→加噪→去噪→decode）分开计时。
- 单张耗时参考（来自 READMЕ3 实测，非本轮实测）：SDXL 去噪 ~280ms/step，精炼单张 ~5.7s（含模型加载/扩散初始化摊薄后首张偏慢）。
- 1000 张估算：~95 分钟（README3 报告口径 1000×5.7s），**这是官方实测估计，非本轮实测**。
- 公平性：精炼前/后都用同一 GT、同一种子、同一 batch 顺序；只比较单张 img2img 的 wall-clock，不含模型加载。

### 4.3 GPU 与存储（估算 vs 实测区分）

- **官方实测（READMЕ3）**：模型加载+扩散初始化 ~10.5GB；精炼峰值 ~28.1GB；48GB 卡足够，24GB 卡仅能跑 representation 不能跑 SDXL 精炼（VERIFIED 来自 READMЕ3 第 6 节）。
- **推荐 GPU**：48GB（A6000/A40 级别）。
- **最低可尝试**：40GB（A100 40G），风险：28GB 峰值 + 文本编码器开销可能吃紧。
- **存储**：最终精炼输出（256×256 或 224×224 float16）= 1000×3×256×256×2 ≈ 393MB（或 224 口径 ≈ 301MB）；无需保留中间 latent（img2img 直接从重建图重新编码即可）。若保留 768×768 float16 中间图 ≈ 3.5GB。
- **明确区分**：以上 GPU/耗时是「官方报告实测 + 代码推算」，不是本机实测。

---

## 五、官方 vs 我们的区别（小结）

| 维度 | 官方 | 我们（推断） |
|---|---|---|
| 第一阶段保存 | `all_recons`（未精炼）、`all_blurryrecons`、`all_predcaptions`、`all_clipvoxels` | 7 文件（无单独 `all_recons`，有 `all_backbones`/`all_prior_out`/`all_ids`） |
| 重建分辨率 | 256×256 | 224×224（对齐 GT） |
| caption 格式 | numpy 字符串数组 | 疑似 `list[str]`（BLOCKED） |
| 精炼权重 | zavy（SDXL base 变体）+ unclip sampler | 推断一致（step6d 输出已含 SDXL enhanced） |
| 第二阶段 | 独立 notebook `enhanced_recon_inference.ipynb` | 集成在 `recon_inference_run.py`（`--export_official_pts` 一次导出） |

## 六、推荐的最小实施方案（只提议）

1. 取回 Ours S1 三文件（否则 BLOCKED）——优先从原 A100 `/dev/vdc` 恢复。
2. 写独立第二阶段脚本（不改现有入口、不覆盖输出）：
   - 输入：`step6d_full_ours_llmhard_all_enhancedrecons.pt`（若作为「精炼前」基准）+ `all_predcaptions.pt`（格式转换如 2.5）。
   - 若方案 A 的「原始重建」指的是未精炼 unCLIP，则需先补出未精炼重建（当前缺失，见阻塞问题）。
   - 复用官方 Cell[3]-[5] 逻辑与参数（zavy + unclip sampler + num_samples=1 + img2img_timepoint=13 + scale=5 + num_steps=25 + seed 42）。
3. 精炼后输出 `*_all_enhancedrecons_v2.pt`，用 ids.pt 对齐做精炼前后对比。
4. 评测：CLIP cossim + PixCorr（+ 可选 SSIM/AlexNet）。

## 七、当前阻塞问题

1. **Ours S1 三文件缺失**（enhancedrecons/predcaptions/ids），HF 与本地均无，唯一线索是原 A100 的 `/dev/vdc` 磁盘路径。
2. **我们的源码不可得**（`recon_inference_run.py`/`models_textalign.py` 不在本机与 HF），无法用「实际代码」严格证明 step6d enhancedrecons 的生成方式与 caption/ids 的精确格式。
3. **「原始重建图像」语义待确认**：方案 A 的 img2img 输入是「S1 的 enhancedrecons（已精炼）」还是「未精炼 unCLIP（缺失）」——这决定是否需要补出未精炼重建。
4. **caption 格式兼容性**（list[str] vs numpy）待文件确认。

## 八、关键结论证据索引

| 结论 | 状态 | 证据 |
|---|---|---|
| 精炼参数 num_samples=1/img2img_timepoint=13/scale=5/num_steps=25 | VERIFIED | enhanced_recon Cell[4][5] |
| 精炼去噪 ~12 步 | VERIFIED | Cell[5] `sigmas[-13:]`, `num_steps=shape[-1]-1` |
| 精炼不用 Refiner | VERIFIED | enhanced_recon 无 refiner 引用；refiner 仅存在配置目录 |
| 精炼用 zavy 作 SDXL base 权重 | VERIFIED | Cell[3] `base_ckpt_path=...zavychromaxl_v30.safetensors` |
| img2img 输入 256→768，latent 96×96 | VERIFIED | Cell[2] Resize(768)；Cell[5] `assert 768`, `encode_first_stage` |
| 无图像融合，输出=img2img 直出 | VERIFIED | Cell[5] decode→clamp→save；blurry 仅 plotting |
| GIT caption 生成在第一阶段、精炼只加载 | VERIFIED | recon Cell[6][8]；enhanced Cell[2][5] |
| unCLIP 从随机 latent 开始、条件=prior_out 非 caption | VERIFIED | utils.unclip_recon `z=randn` |
| 我们仓库路径 /mnt/work/repos/TextAlign-mindeye2 | VERIFIED | RECOVERY_MANIFEST.json root |
| 我们仓库 commit/分支 | BLOCKED | 在模型仓库 ENV_SNAPSHOT（未抓取） |
| step6d enhancedrecons 是精炼后图像 | INFERRED | 命名 + fig4 provenance |
| Ours S1 三文件 SHA | VERIFIED | FINAL_OUTPUT_CHECKSUMS.sha256 |
| Ours S1 原磁盘路径 /mnt/aaai_ablation_disk(/dev/vdc) | VERIFIED | checksum 路径 + READMЕ3 第 11 节 |
| 2026-04-12 快照不含 s1 step6d | VERIFIED | HF_UPLOAD_MANIFEST.json（148 文件，仅 s2/s5/s7） |

---

*报告结束。本报告为只读审计，未运行 GPU/推理/训练，未修改项目文件（仅新增本报告与 `reports/_notebook_extract/`、`reports/_model_repo/`、`scripts/` 下若干只读辅助脚本）。*
