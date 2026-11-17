# A100 40GB 迁移与首轮样本测试报告

日期：2026-11-15 ｜ 机器：A800-SXM4-40GB / 16 vCPU / 47GB RAM / Ubuntu 22.04.3

## 最终状态

| 门禁 | 结果 |
|---|---|
| `STAGE1_SAMPLE_PASS` | ✅ |
| `STAGE1_STABILITY_PASS` | ✅ |
| `STAGE2_SAMPLE_PASS` | ✅ |
| **`READY_FOR_FULL_1000_INFERENCE`** | ✅（见第 15 节估算与建议） |

---

## 1. GPU 型号与显存

| 项 | 值 |
|---|---|
| GPU | **NVIDIA A800-SXM4-40GB**（A100 同架构 SM80，NVLink 受限版） |
| 显存 | 40960 MiB（torch 报 42.5 GB total_memory） |
| Driver / CUDA | 550.127.05 / 12.4 |
| 空闲占用 | 10 MiB |

## 2. CPU 与 RAM

| 项 | 值 |
|---|---|
| vCPU | 16 |
| RAM | 47 GiB 总 / 实测可用 43–44 GiB |
| Swap | 原 5.9G + **本轮新增 16G**（`/mnt/data/swapfile`，防 OOM 兜底） |
| 实测峰值 RSS | Stage A 41.32 GB，Stage B 42.21 GB（**接近上限，务必保持 47GB 及以上**） |

## 3. 环境安装结果

- venv：`/mnt/data/gpu_venv`（6.0 GB）
- `torch 2.5.0+cu124`、`python 3.12.7`、CUDA 12.4 可用、A800 识别正确
- 19 个依赖全部导入 OK（transformers 4.46.3 / diffusers 0.31.0 / h5py / webdataset / deepspeed / dalle2_pytorch / timm / matplotlib …）
- `GPU_ENV_PREFLIGHT.sh`：**[1]~[4] 全 PASS**（4 个 HF 仓库离线加载 + 8 个资产存在性）
- GIT `from_pretrained` 离线实测：`GitProcessor` + `GitForCausalLMClipEmb`（394.196 M 参数）OK

### 环境偏差（必须记录，均为依赖/兼容修复，**未改动任何权重与算法**）

| # | 问题 | 处理 |
|---|---|---|
| 1 | `open_clip_torch==2.26.1` 的 attention 已改 batch-first，与 sgm 的 LND 调用不兼容（`attn_mask` [77,77] vs (1,1)） | **降级 open_clip_torch==2.20.0**（timm 版 LND），权重不变 |
| 2 | 锁定文件里的 `dalle2-pytorch==1.16.3` **上游不存在** | 改用 **1.15.6**（项目 setup.sh 未锁版本） |
| 3 | 缺 `matplotlib`（脚本第 77 行强制 import） | 安装 3.11.2 |
| 4 | 缺 `sd_image_var_autoenc.pth`（ckpt 强制 `blurry_recon=True`，`--no-blurry_recon` 会被 ckpt args 覆盖） | 下载 `stabilityai/sd-vae-ft-mse`（334.6MB），**仅做键名重映射**（`query/key/value/proj_attn` → `to_q/to_k/to_v/to_out.0`），`strict=True` 加载 **0 missing / 0 unexpected** |
| 5 | `src/evals` 是指向 `/mnt/work/mindeye_data_real/evals` 的**断链**，`makedirs(exist_ok=True)` 报 FileExistsError | 创建目标目录（不改仓库、不删数据） |
| 6 | 代码打开 `coco_images_224_float16.hdf5`（21GB）但**全程未使用该变量** | 用官方 `all_images.pt`（1000×3×224×224）生成同结构 h5（301MB），非伪造 GT |
| 7 | cuDNN 9.1 SDPA 在 CLIP 文本上报 "No execution plans support the graph" | 运行时禁用 cudnn/flash/mem_efficient SDPA，**仅用 math 后端**（数值等价） |

> 下载用镜像：PyPI tuna（3.4MB/s）+ hf-mirror.com（7.4MB/s）；**未重新下载任何已有大型权重**。

## 4. 数据盘迁移结果

| 检查 | 结果 |
|---|---|
| 挂载 | `/dev/vdb` → `/mnt/data`，UUID `5a9d9737-e795-4d2a-9c93-3d7b443f9054`（与 fstab 一致） |
| 自动挂载 | fstab（nofail,noatime,device-timeout=30s）+ `mnt-data-ensure.service` 重试兜底 |
| 容量 | 147G 总，**45G 可用**（95G 已用；含 venv 6G + swap 16G） |
| HF 缓存软/硬链接 | `find hf_home -xtype l` = **0 断链**；绝对链接 = 0 |

## 5. 六个核心文件 SHA256 复核（迁移后重新计算）

| 文件 | 实测 | 参考 | 结果 |
|---|---|---|---|
| `s1_textalign_stage1_FINAL_BEST_32_last.pth` (26.8GB) | `eb59c426…e91f` | `eb59c426…e91f` | ✅ |
| `unclip6_epoch0_step110000.ckpt` (18GB) | `6bd7033d…5652` | `6bd7033d…5652` | ✅ |
| `zavychromaxl_v30.safetensors` (6.9GB) | `f1270c1f…1b75` | `f1270c1f…1b75` | ✅ |
| `bigG_to_L_epoch8.pth` (7MB) | `2f540faf…36cf` | `2f540faf…36cf` | ✅ |
| `betas_all_subj01_fp32_renorm.hdf5` (1.9GB) | `71992eb8…ecdd` | `71992eb8…ecdd` | ✅ |
| `wds_subj01_new_test_0.tar` (38MB) | `16f810c0…5c5` | `16f810c0…5c5` | ✅ |

参考值来自 `manifests/overnight_manifest.tsv`（CPU 阶段下载时记录）。

## 6. GIT / OpenCLIP 离线加载

| 仓库 | 文件 | 字节 | 结果 |
|---|---|---|---|
| microsoft/git-large-coco | model.safetensors | 1,576,861,900 | ✅ |
| openai/clip-vit-large-patch14 | pytorch_model.bin | 1,710,671,599 | ✅ |
| laion/CLIP-ViT-bigG-14 | open_clip_pytorch_model.bin | 10,158,638,769 | ✅ |
| laion/CLIP-ViT-L-14 | open_clip_pytorch_model.bin | 1,710,631,365 | ✅ |

`HF_HUB_OFFLINE=1` 下全部命中；GIT 文本解码器实测 394.196 M 参数。

---

## 7. Stage A：2 张图像完整推理（第一阶段）

**命令**（独立进程，官方脚本未修改）：
```bash
python recon_inference_run.py --data_path=run/data --cache_dir=run/cache \
  --output_dir=run/out_stage1_a --model_name=s1_textalign_stage1_FINAL_BEST_32 \
  --ckpt_path=assets/models/ours_s1_checkpoint/s1_textalign_stage1_FINAL_BEST_32_last.pth \
  --subj=1 --new_test --seed 42 --export_official_pts --max_save 2 --save_images
```

**流程符合性**：每个唯一 ID → 取 3 条 behav 记录（`behav[:,0,5]` 行索引）→ 3 次 BrainNetwork 前向 → `accum/3` 平均 → Prior → GIT caption → unCLIP 重建（`num_samples_per_image=1`）。

**结果**：2/2 完成，ID = **[2950, 2990]**（升序，与 `np.unique` 一致）

| 输出 | shape | dtype | 有限值 | 字节 |
|---|---|---|---|---|
| enhancedrecons | (2,3,224,224) | float16 | ✅ | 603,645 |
| blurryrecons | (2,3,224,224) | float16 | ✅ | 603,635 |
| clipvoxels | (2,256,1664) | float16 | ✅ | 1,705,449 |
| prior_out | (2,256,1664) | float16 | ✅ | 1,705,444 |
| backbones | (2,256,1664) | float16 | ✅ | 1,705,444 |
| predcaptions | 2 条非空 | str | — | 1,228 |
| ids | (2,) | int64 | — | 1,414 |

**caption**：
- 2950 → `a group of people standing around a table.`
- 2990 → `a man is standing in front of a table.`

**图像质量**：可正常打开、非黑屏/纯噪声/空白
- id2950：mean 0.481 / std 0.290 / hf_energy 0.0717 / black 0.01%
- id2990：mean 0.192 / std 0.251 / hf_energy 0.0723 / black 27%（夜景厨房，人物与灶台清晰）
- 人工目视：**可辨识场景**（宴席长桌人群 / 厨房厨师），caption 与画面对应

**资源指标**

| 指标 | 值 |
|---|---|
| `max_memory_allocated` | **28.817 GB** |
| `max_memory_reserved` | **28.865 GB** |
| `nvidia-smi` 峰值 | **28,056 MiB** |
| CPU RAM 峰值 (`ru_maxrss`) | **41.32 GB** |
| 总耗时 | 163.0 s（含加载 ~152s；2 张推理 11s，5.5 s/张） |
| GPU 利用率 | 首图 66%（采样期） |
| 数据盘新增 | 6.82 MB |

> **验证结论**：40GB 判定得到实测支持——峰值 28.87GB，余量 **≈10.6GB**，与历史 28.1GB 完全吻合。

---

## 8. Stage B：20 张稳定性测试

同命令，`--max_save 20`，输出 `run/out_stage1_b`。

| 检查 | 结果 |
|---|---|
| 输出数量 | 20 ✅ |
| IDs | `[2950,2990,3049,3077,3146,3157,3164,3171,3181,3386,3434,3449,3489,3626,3682,3687,3729,3809,3842,3847]` |
| 唯一 / 升序 / 无重复 | ✅ 20 唯一、严格升序 |
| **每个 ID 恰好 3 条 beta 记录** | ✅ 20/20（从 tar 独立复核） |
| 七类输出完整 | ✅ 5 个张量 + captions(20) + ids(20) |
| NaN/Inf | 0 ✅ |
| OOM / 中断 | 无 ✅ |
| 图像质量 | 20 张 hf_energy 0.019–0.141，black_frac ≤4.4%，无空白/纯噪声 |

**资源**：`max_allocated 28.817 GB` / `reserved 28.865 GB` / nvidia-smi **28,056 MiB** / RAM 峰值 **42.21 GB** / 总耗时 **243.1 s**
**稳定速度**：**(243.1 − 152) / 20 ≈ 4.56 s/张**

---

## 9. Stage C：2 张 img2img（第二阶段，独立进程）

第一阶段进程完全退出、GPU 归零（10 MiB）后启动。严格复刻 `enhanced_recon_inference.ipynb` cell 3-5：
Zavy SDXL base / `img2img_timepoint=13` / `guider.scale=5` / `num_steps=25` / `seed=42` / `num_samples=1` / Resize→768 → 输出 Resize→256 / **无 refiner** / `use_ema=False`（`ema_scope()` 空操作）。

| 核对项 | 结果 |
|---|---|
| 输入图像 | `run/out_stage1_a/*_all_enhancedrecons.pt`（224 → Resize 768，与官方 notebook 同一操作） |
| GIT caption 与 ID 对应 | ✅ idx0→2950 `a group of people standing around a table.`；idx1→2990 `a man is standing in front of a table.` |
| 文本条件进入 img2img | ✅ prompt 经 CLIP-L(768) + bigG 文本塔(1280) 拼接后进入 crossattn/vector；输出与输入 **mean\|diff\|=0.077**（确实被改写） |
| 输出 | (2,3,256,256) float32，有限值，IDS 与 Stage A **完全一致** |
| 图像 | 可打开、非空白（mean 0.478/0.204，hf 0.043/0.039） |
| 显存 | `max_allocated` **19.933 GB** / `reserved` **20.07 GB** / nvidia-smi **19,648 MiB** |
| 速度 | 1.91 s/张 |

**环境修复**（本轮定位并解决）：open_clip 2.26 batch-first 导致的 mask 形状错误 → 降级 2.20.0；另加 cuDNN SDPA 后端禁用。

> 目视对比（`run/out_stage2/preview/side_*.png`，左=一阶段输入、右=img2img 输出）：场景结构保留、按 caption 精修，无破坏性伪影。

---

## 10–13. 显存 / RAM / 耗时 / 输出体积汇总

| 阶段 | max_allocated | max_reserved | nvidia-smi 峰值 | RAM 峰值 | 耗时 | 输出体积 |
|---|---|---|---|---|---|---|
| Stage A (2) | 28.817 GB | 28.865 GB | 28,056 MiB | 41.32 GB | 163.0 s | 6.82 MB |
| Stage B (20) | 28.817 GB | 28.865 GB | 28,056 MiB | 42.21 GB | 243.1 s | 67.76 MB |
| Stage C (2) | 19.933 GB | 20.07 GB | 19,648 MiB | ~20 GB | ~120 s | 2.13 MB |

- 第一阶段峰值 28.87 GB → 40GB 卡余量 **≈10.6 GB**（>5GB 要求）
- 第二阶段峰值 20.07 GB → 余量 **≈19 GB**
- 40GB 判定**实测确认**，无需 80GB

## 14. 图像质量与 ID 对齐

| 检查 | Stage A | Stage B | Stage C |
|---|---|---|---|
| ID 与输出顺序一一对应 | ✅ [2950,2990] | ✅ 20 个升序 | ✅ 与 Stage A 完全一致 |
| 图像可打开 | ✅ | ✅ 20/20 | ✅ 2/2 |
| 非黑屏/纯噪声/空白 | ✅ | ✅（hf 0.019–0.141） | ✅ |
| caption 非空且语义匹配 | ✅ | ✅ 20/20 | ✅ 注入 img2img |
| 文件可重新加载 | ✅ 5 个 .pt 全部 torch.load 成功 | ✅ | ✅ |

## 15. 1000 张正式实验耗时估计

| 阶段 | 单位成本 | 1000 张 | 含加载 |
|---|---|---|---|
| 第一阶段 | 4.56 s/张 | 4,560 s | 4,560 + 152 ≈ **78.5 分钟** |
| 第二阶段 | 1.91 s/张 | 1,910 s | 1,910 + 120 ≈ **34 分钟** |
| **合计** | — | — | **≈ 1.9 小时** |

输出体积估计：Stage1 ~3.4 GB + Stage2 ~0.1 GB（按 20 张 67.8MB 线性外推，.pt + PNG）；磁盘余量 45G 充足。

---

## 建议的执行方式（正式 1000 张）

1. 第一阶段：`--max_save 1000`（去掉或无影响），单独进程，预计 ~80 分钟
2. 进程退出、`nvidia-smi` 确认归零
3. 第二阶段：读取第一阶段的 1000 条 enhancedrecons + captions，预计 ~35 分钟
4. 全程保持：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、HF 离线、SDPA math 后端、open_clip 2.20

## 风险与注意事项

1. **RAM 是当前最紧的资源**：峰值 42.2 / 47 GB。1000 张时累计的 CPU 侧预分配张量会更大（enhancedrecons 1000×3×224×224 fp16 ≈ 301MB，clipvoxels 1000×256×1664 fp16 ≈ 852MB），仍在余量内，但**不要同时运行其他大内存进程**；
2. 已加 16GB swap 兜底，若观察到 swapping 明显则说明内存不足；
3. `nvidia-smi` 峰值 28,056 MiB 与 torch reserved 28.87GB 一致，**40GB 安全**；
4. 所有产物路径：`run/out_stage1_a`、`run/out_stage1_b`、`run/out_stage2`（含 preview PNG 与 verify_*.json）。


