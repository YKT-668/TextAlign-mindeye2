# 正式 1000 张推理最终报告（ConceptAlign GIT img2img）

日期：2026-11-15 ｜ 机器：A800-SXM4-40GB / 16 vCPU / 47GB RAM

## 最终状态

| 门禁 | 结果 |
|---|---|
| `STAGE1_FULL_1000_PASS` | ✅ |
| `STAGE2_FULL_1000_PASS` | ✅ |
| **`READY_FOR_EVALUATION`** | ✅ |

两次运行均为**独立进程**，串行执行，未并行占用 GPU。

---

## 1. 实际 GPU 型号

| 项 | 值 |
|---|---|
| GPU | **NVIDIA A800-SXM4-40GB**（A100 同架构 SM80） |
| 显存 | 40960 MiB |
| Driver / CUDA | 550.127.05 / 12.4 |
| 进程前空闲占用 | 10 MiB |

## 2. 环境最终版本（实测通过）

| 组件 | 版本 |
|---|---|
| Python | 3.12.7 |
| PyTorch | 2.5.0+cu124 |
| torchvision | 0.20.0 |
| **open_clip_torch** | **2.20.0**（2.24+ batch-first 与 sgm LND 不兼容，必须 2.20.x） |
| transformers | 4.46.3 |
| diffusers | 0.31.0 |
| dalle2-pytorch | **1.15.6**（上游无 1.16.3） |
| h5py / webdataset / deepspeed | 3.12.1 / 0.2.100 / 0.15.4 |
| matplotlib / hf_transfer | 3.11.2 / 0.1.9 |
| 运行环境 | `/mnt/data/gpu_venv`（6.0 GB） |

> `GPU_ENV_LOCK.txt` 已同步为上述实测版本。

## 3. 第一阶段开始 / 结束时间

| 事件 | 时间 |
|---|---|
| 进程启动（含模型加载） | **14:56:44** |
| 推理循环开始 | ≈14:59:20（加载 ~152 s） |
| 推理完成、产物落盘 | **16:27:05** |
| 进程内计量 `elapsed_s` | **5411.0 s**（90.2 分钟） |

## 4. 第二阶段开始 / 结束时间

| 事件 | 时间 |
|---|---|
| 进程启动（GPU 已确认归零 10 MiB） | **16:28:41** |
| 推理完成、产物落盘 | **16:59:48** |
| 墙钟耗时 | **31.1 分钟** |

## 5. 两阶段实际总耗时

**121.5 分钟 ≈ 2 小时 02 分**（90.2 + 31.1）

| 阶段 | 单张成本 | 1000 张实测 |
|---|---|---|
| 第一阶段 | ≈5.26 s/张 | 5411 s（90.2 min） |
| 第二阶段 | ≈1.72 s/张 | 约 1690 s 推理 + ~180 s 加载 |

## 6. 各阶段 GPU 显存峰值

| 阶段 | `max_memory_allocated` | `max_memory_reserved` | `nvidia-smi` 峰值 |
|---|---|---|---|
| 第一阶段 | 28.817 GB | **28.874 GB** | **28,064 MiB** |
| 第二阶段 | 20.122 GB | **20.374 GB** | **19,958 MiB** |

- 40GB 卡余量：第一阶段 ≈ **10.6 GB**、第二阶段 ≈ **19 GB**
- 与样本测试（28.865 / 20.07 GB）一致，**无随样本数增长的显存泄漏**

## 7. RAM 峰值 / MemAvailable 最低 / Swap

| 指标 | 第一阶段 | 第二阶段 |
|---|---|---|
| 进程峰值 RSS (`ru_maxrss`) | **37.09 GB** | ≈ 20 GB |
| MemAvailable 最低 | **8,904 MB** | **11,860 MB** |
| Swap 使用峰值 | **≈11.4 GB**（22.4G 中剩余 10.99G） | ≈4.2 GB |
| **换页情况** | **推理循环期间 `pswpout` 恒定 5,937,141，零新增** | 仅加载期有换页 |
| 是否触发保护停机 | 否（MemAvailable 始终 >1.5GB，无卡顿/OOM） | 否 |

> 说明：第一阶段的 10.1GB swap-out 全部发生在**模型加载阶段**（14:56–15:01），进入 1000 张循环后 `pswpout` 完全不变——即**无持续换页**。MemAvailable 最低 8.9GB，安全。

---

## 8. 每阶段完成数量

| 阶段 | 目标 | 实际 | 备注 |
|---|---|---|---|
| 第一阶段 | 1000 | **1000** | 1000 唯一 ID，全部 3 次重复平均 |
| 第二阶段 | 1000 | **1000** | 与第一阶段 ID 一一对应 |

## 9. 七类输出文件清单（`run/full_stage1/`）

| 输出 | 文件 | 字节 |
|---|---|---|
| enhancedrecons | `..._all_enhancedrecons.pt` | 301,057,533 |
| blurryrecons | `..._all_blurryrecons.pt` | 301,057,523 |
| CLIP tokens (clipvoxels) | `..._all_clipvoxels.pt` | 851,969,513 |
| prior output | `..._all_prior_out.pt` | 851,969,508 |
| backbone tokens | `..._all_backbones.pt` | 851,969,508 |
| predicted captions | `..._all_predcaptions.pt` | 42,636 |
| IDs | `..._all_ids.pt` | 9,414 |

附加：`..._captions.txt`（37,179 字节）、`images/`（1000 PNG）、`blurry_images/`（1000 PNG）、`preview/`（抽样）、`verify_stage1_1000.json`

第二阶段（`run/full_stage2/`）：
| 输出 | 文件 | 字节 |
|---|---|---|
| img2img 1000 张 (256×256) | `..._stage2_img2img.pt` | 786,433,513 |
| IDs | `..._stage2_ids.pt` | 9,429 |

附加：`preview/`（10 张采样 + 拼图）、`verify_stage2_1000.json`

## 10. 1000 个 ID 对齐验证

| 检查 | 结果 |
|---|---|
| 数量 | 1000 ✅ |
| 唯一性 | 1000 唯一（无重复）✅ |
| 升序（与 `np.unique` 顺序一致） | ✅ |
| 每个 ID 在 tar 中恰好 3 条 beta 记录 | **1000/1000 ✅**（独立遍历 tar 复核） |
| 第二阶段 ID 与第一阶段完全相等 | ✅ `torch.equal == True` |
| 与原始映射一致 | ✅（ID 头 `[2950, 2990, 3049, 3077, 3146]`，尾 `[..., 72948 域内的末尾 5 个]`） |

## 11. Caption 与图像对应检查

- 1000 条 caption，**空字符串 0 条**，全部为 COCO 风格自然语言
- 抽样（idx / ID / caption）：
  - 0 / 2950 / `a group of people standing around a table.`
  - 250 / — / 见 `verify_stage1_1000.json`
  - 500 / — / 同上
  - 750 / — / 同上
  - 999 / — / 同上
- 第二阶段逐张使用**同一 idx 的 caption**（脚本打印 `[img2img] idx=.. id=.. prompt='...'`，日志可逐条核对），ID 映射未错位

## 12. 图像质量抽样检查

第一阶段（1000 张 enhancedrecons）：
| 指标 | 范围 |
|---|---|
| 像素均值 | 0.1096 – 0.8882 |
| 对比度 std | 0.0497 – 0.4004 |
| 高频能量 hf | 0.00952 – 0.16713 |
| 纯黑/空白图像 | **0** |
| NaN/Inf | **0** |

第二阶段（1000 张 img2img）：
| 指标 | 范围 |
|---|---|
| 像素均值 | 0.1283 – 0.8791 |
| 对比度 std | 0.0481 – 0.3930 |
| 高频能量 hf | 0.00416 – 0.09967 |
| 纯黑/空白图像 | **0** |

目视抽样（`run/full_stage2/preview/contact_sheet_10.png`）：宴席长桌、室内人物、远景人群、床上的猫、红色火车、象群、床头柜与台灯、合影、餐点托盘、客厅——**场景合理、清晰、无伪影**。

## 13. 中间保存与断点恢复记录

| 项 | 第一 阶段 | 第二阶段 |
|---|---|---|
| 方式 | 信号触发的**原子快照**（`tmp` + `os.replace`），只读已完成行 | 同 |
| 周期 | 每 **240 s**（≈45 张） | 每 **300 s**（≈175 张） |
| 内容 | `enhancedrecons/blurryrecons/clipvoxels/prior_out/backbones/predcaptions/ids` 的 `[:n]` | `results[:n]` 堆叠 |
| 一致性保证 | 只在 `official["i"]` 自增**之前**的行被保存 → 保存的永远是完整行 | 同 |
| 落盘位置 | `run/full_stage1_ckpt/stage1_ckpt.pt` | `run/full_stage2_ckpt/stage2_ckpt.pt` |
| 实测最后快照 | **rows=976** | 见日志 `[ckpt] stage2 snapshot n=...` |
| 是否改变计算 | **否**（不触碰模型、RNG、顺序、三次平均逻辑） | 否 |

**恢复策略（如实说明）**：发生中断时，快照可用于**保全已完成结果与分析**，但**不做"跳过前 n 张"的续跑**——因为跳过会改变 `torch.randn` 的 RNG 序列，导致与官方单次运行的逐位结果不一致。因此中断后的恢复方式是"整段重跑"，以保证可复现性。本次两阶段**均一次跑通，未触发恢复**。

---

## 14. SHA256 校验清单

见 `reports/FULL_1000_SHA256.sha256`（10 项）。核心摘要：

| 文件 | SHA256 |
|---|---|
| `full_stage1/..._all_enhancedrecons.pt` | `de8522cbdda15eda4dbb4ebe4a82b071adae2fb5c49a0b1917b0bc1634062999` |
| `full_stage1/..._all_blurryrecons.pt` | `015dbe92a27c3bc868c210adb14195590c15baa7a56d9fc20724bf06be1349a7` |
| `full_stage1/..._all_clipvoxels.pt` | `d307a23de7b0b2d2cf9148e9ccbe5cb2857d6dfa3c2cc18548e2d01d580d828a` |
| `full_stage1/..._all_prior_out.pt` | `4536f206bc0678c7689dcb033f7a3aebb5e6b976192aff178c3d2f595ae4b5d3` |
| `full_stage1/..._all_backbones.pt` | `ffef974bb1aa5abad0edb018b95caa69594a778dd4af5890ec96915529276518` |
| `full_stage1/..._all_predcaptions.pt` | `8ef814ec6c231e64832c2a1fd99e3b274114fd3bd93a70f48095512f1feeadd5` |
| `full_stage1/..._all_ids.pt` | `58e5500d841b4dbb6ee29a2b0a71f762dc4a0e85808e2b84dbe015f9a0c757cf` |
| `full_stage1/..._captions.txt` | `093d8eadd63c5aa48b062679eedbf0261f6e52d976cb778e96ad0ecf7b1bf97b` |
| `full_stage2/..._stage2_img2img.pt` | `7d889806eab1a4c8c65b9aab72882f7de452ca65622cc3a1ab832117052e9591` |
| `full_stage2/..._stage2_ids.pt` | `30e89491209a13a07c62675d3986f85e91ce2077c840eb2e8ec56260ab61fb74` |

产物总量 ≈ **3.9 GB**（stage1 .pt 3.16GB + PNG ~180MB + stage2 786MB）。

## 15. VAE 替换与其他环境偏差

| # | 偏差 | 影响面 | 说明 |
|---|---|---|---|
| 1 | `sd_image_var_autoenc.pth` 用 `stabilityai/sd-vae-ft-mse` 补齐（原文件本机缺失） | **仅 blurryrecons** | 代码中 `autoenc` 只出现在 `autoenc.decode(blurry_image_enc / 0.18215)`（L873/L893）；enhancedrecons 走 `utils.unclip_recon`，与 autoenc 无关 → **不影响第一阶段主输出，也不影响第二阶段输入**。加载 `strict=True` 0 missing / 0 unexpected（仅做 `query/key/value/proj_attn → to_q/to_k/to_v/to_out.0` 键名重映射，数值未改） |
| 2 | `open_clip_torch` 2.26.1 → **2.20.0** | 第二阶段文本塔 | 2.24+ 改 batch-first，与 sgm 的 LND 调用冲突（`attn_mask` [77,77] vs (1,1)）。权重文件未变 |
| 3 | `dalle2-pytorch` 1.16.3 → **1.15.6** | PriorNetwork | 上游不存在 1.16.3（项目 setup.sh 未锁版本） |
| 4 | 新增 `matplotlib 3.11.2`、`hf_transfer 0.1.9` | 无算法影响 | 脚本强制 import / 下载加速 |
| 5 | SDPA 后端：禁用 cudnn/flash/mem_efficient，仅 math | 数值等价 | 规避 cuDNN 9.1 frontend "No execution plans" |
| 6 | `src/evals` 断链 → 创建 `/mnt/work/mindeye_data_real/evals` | 无 | 原训练机残留的软链，不影响结果 |
| 7 | `coco_images_224_float16.hdf5` 用官方 `all_images.pt` 生成 | **无** | 代码 L317-318 打开并赋给 `images`，**全文件从未使用**该变量 |
| 8 | 新增 16GB swapfile | 无 | 内存兜底；推理循环期间 `pswpout` 零增长 |

**模型、权重、推理参数、三次重复平均逻辑、随机种子均未修改**；第一阶段使用官方 `recon_inference_run.py`（archive/final14@231850a）原文件，第二阶段严格复刻 `enhanced_recon_inference.ipynb` cell 3-5（`plotting=False`、`num_samples=1`、`timepoint=13`、`scale=5`、`steps=25`、`seed=42`、无 refiner、`use_ema=False`）。

---

## 产物路径速查

```
run/full_stage1/            七类 .pt + captions.txt + images/ + blurry_images/ + preview/
run/full_stage1/verify_stage1_1000.json
run/full_stage1_ckpt/stage1_ckpt.pt
run/full_stage2/            stage2_img2img.pt + stage2_ids.pt + preview/ + verify_stage2_1000.json
run/full_stage2_ckpt/stage2_ckpt.pt
logs/full_stage1.log | full_stage1_monitor.log | full_stage1_progress.tsv
logs/full_stage2.log | full_stage2_monitor.log | full_stage2_progress.tsv
reports/FULL_1000_SHA256.sha256
```

> 样本测试结果（`run/out_stage1_a`、`run/out_stage1_b`、`run/out_stage2`）**完整保留未覆盖**。


