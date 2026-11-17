# 最终终审：A100 40GB 租赁决策

日期：2026-11-15
范围：只做定点核查（不下载、不做全量 SHA 复审计、不改模型、不跑推理）

## 结论

**`A100_40G_READY`**

40GB 足够：预计最大实际显存 **28–30GB**，相对 40GB（nvidia-smi 40960MiB，可用约 39.5GB）保留 **≈9.5–11.5GB**（≥5GB 要求）。
前提是必须遵守第 6 节运行约束；若打算改 `num_samples>1`、开 refiner、或把两阶段塞进同一进程，则必须改租 80GB。

---

## 1. 28.1GB 峰值的真实来源：第一阶段（MindEye2/TextAlign + unCLIP）

**不是第二阶段 SDXL img2img。** 证据链：

| # | 证据 | 内容 |
|---|---|---|
| 1 | A100 报告只登记了一条 GPU 命令 | `python src/recon_inference_run.py --new_test --export_official_pts --max_save 1000` —— **只有第一阶段**脚本 |
| 2 | 报告中的重构产物 | `*_all_enhancedrecons.pt` 来自第一阶段 `official["enhancedrecons"]`，由 `utils.unclip_recon()` 生成 |
| 3 | 报告写 `max_save=2` | 与第一阶段 `--max_save 2` 一致（第二阶段无 `max_save` 参数） |
| 4 | "Reconstruction (2 samples) ~280ms sampling loop" | 对应 unCLIP sampler（38 步、96×96 latent、CFG） |
| 5 | 第二阶段从未在 A100 上跑过 | 报告中无任何 img2img 命令/产物 |

### 测量口径（必须说清楚）

- 历史数字是 **`nvidia-smi` 的 "Max Used"**（原文："Measured using `nvidia-smi` during inference"），**不是** `torch.cuda.max_memory_allocated()`。
- 全仓检索 `nvidia-smi|max_memory|max_reserved`：**不存在**任何 torch allocator 统计日志。
- 关系：`nvidia-smi used ≥ max_memory_reserved ≥ max_memory_allocated`（nvidia-smi 还含 CUDA context、cuDNN workspace、allocator 预留与碎片）→ **28.1GB 本身就是相对 allocated 的保守上界**。

---

## 2. 第一阶段显存拆解（权重级实测 + 代码核对）

权重字节数由**只读 pickle 元数据**的解析器实测（不加载张量）：

| GPU 常驻对象 | 实测大小 | 依据 |
|---|---|---|
| unCLIP denoiser（SDXL UNet，**fp32**） | **10.005 GB**（2.5013B 参数） | ckpt `state_dict/model.*` |
| unCLIP conditioner（**fp16**，仅文本塔） | **1.635 GB** | 0.8176B 参数（CLIP-L 0.123B + bigG 文本 0.6946B）×2B；`del model.visual`（modules.py L521）+ `conditioner.half()`（L665） |
| unCLIP VAE（first_stage，fp32） | **0.335 GB**（0.0837B） | ckpt `first_stage_model.*` |
| **S1 脑模型（fp32）** | **8.929 GB** | `model_state_dict`：backbone 7.612 + diffusion_prior 1.039 + ridge 0.258 + text_head 0.020 |
| GIT-large 文本解码（fp32） | **1.577 GB** | `model.safetensors` |
| CLIPConverter（bigG→L 线性） | ~0.0001 GB | `bigG_to_L_epoch8.pth` 6.8MB |
| **权重小计** | **22.48 GB** | — |
| 瞬态峰值（`.half()` 之前） | **24.12 GB** | `diffusion_engine.to(device)` 时 conditioner 仍 fp32（+1.635GB） |
| 采样激活 | 4–6 GB | unCLIP sampler：38 步、latent 4×96×96、CFG batch=2、fp16 autocast |
| VAE 解码激活 | （含上项） | `decode_first_stage` 768×768 |
| CUDA context / allocator | 0.4–0.8 GB | — |

**推算峰值 = 24.12 + 4~6 + 0.6 ≈ 28.7–30.7GB，与实测 28.1GB 一致。**

> **第一阶段保守上界：30GB**（实测 28.1 + 1.9 安全余量）→ 40GB 余量 `39.5−30 = 9.5GB` → **通过**
> 若 GIT 改 CPU offload：−1.58GB → 上界约 28.5GB

---

## 3. 第二阶段显存（历史未实测，按权重推算）

| 对象 | 大小 | 依据 |
|---|---|---|
| Zavy SDXL base（**fp16**） | 6.50 GB | safetensors 头：2,515 张量全 F16 |
| CLIP-L 文本塔（FrozenCLIPEmbedder，fp32） | 0.49 GB | 0.123B 参数 |
| bigG 文本塔（FrozenOpenCLIPEmbedder2，fp32） | **2.78 GB** | 实测 `.bin`：transformer 2.519 + token_embedding 0.253 + proj/pos 0.007；`visual`（7.38GB）被 `del` |
| bigG **图像**塔 | 0（不加载） | 仅 `plotting or num_samples>1` 时加载；我们 `num_samples=1, plotting=False` |
| GIT | 0（不加载） | 只读第一阶段导出的 `predcaptions.pt` |
| EMA 副本 | 0 | 两份 yaml 无 `use_ema`，`DiffusionEngine(use_ema=False)`，`ema_scope()` 空操作 |
| **权重小计** | **9.77 GB** | — |
| UNet CFG(batch=2) @96×96 + VAE decode 768² | 4–6 GB | 25 步、`img2img_timepoint=13`、`scale=5`、fp16 autocast |
| **保守上界** | **18 GB** | 纯推算，无历史实测 |

> 瓶颈是第一阶段 30GB，不是第二阶段。

### 能否证明

- **能证明**：第一阶段 28.1GB（nvidia-smi，A100 80GB 实测，且是唯一 GPU 命令）。
- **不能证明**：`max_memory_allocated/reserved` 历史值；第二阶段任何实测值。故用**权重级实测 + 代码路径**给出可复算上界（30GB / 18GB），并标注推算。

---

## 4. Beta / new_test 索引语义（术语修正，无真实索引错误）

| 问题 | 结论 | 证据 |
|---|---|---|
| tar 每条记录字段名 | `behav.npy` / `past_behav.npy` / `future_behav.npy` / `olds_behav.npy`（3000×4 = 12000 成员） | `tarfile.getnames()` |
| 哪个字段索引 beta | **`behav[:, 0, 5]`** | `recon_inference_run.py` L324 `voxels['subj01'][behav[:,0,5]]`；`tools/extract_all_features.py` L263 注释 "trial idx -> voxel 行" |
| 行索引还是列索引 | **beta 第一维的行索引（trial 行）**，不是体素列索引 | 实测 row ∈ [0, 29965]，而 beta 第二维只有 15724 |
| 正式代码如何读 | `voxels['subj01'][behav[:,0,5].long()]` | L324-326 |
| beta 两维含义 | **(30000, 15724)**：第一维 = 30000 条 trial（subj01 的 10000 图 × 3 重复）；第二维 = 15724 个体素 | h5py 元数据实测 |
| 是否有独立体素选择索引 | **没有**。S1 用全部 15724 维；`ridge` 权重 0.258GB = 15724×4096×4B 精确吻合 | ckpt 解析 + 代码无 `voxel_idx` |
| 索引是否越界 | 图像 ID ∈ 2950–72948（73k 索引域）；beta 行 ∈ 0–29965 < 30000 ✓；**无越界** | 实测 |

> 先前报告把 `behav[:,0,5]` 写成"voxel 索引 0–29965"，属**术语错误**（易误解为越界），语义本身正确。已修正为"beta 行索引（trial row）"。

### 重复与 1000 张对应关系

| 项 | 结论 |
|---|---|
| 记录数 | 3000（实测） |
| 唯一图像 | 1000（实测） |
| 每图重复 | 恰好 3（实测 `Counter({3: 1000})`） |
| 三次重复如何处理 | **分别前向、输出取平均**，再对平均值做**一次** unCLIP 重建（`accum/3`，L789-809）；`num_samples_per_image=1` |
| 1 或 2 次重复的容错 | 自动复制/补齐到 3（L775-779）；当前数据不触发 |
| 1000 张与 all_ids | `uniq_test_images = np.unique(test_images_idx)` → **升序**；循环按此顺序，`official["ids"][i]=uniq_img`；`*_all_ids.pt` 与 `*_all_enhancedrecons.pt` 逐行一一对应 |

---

## 5. 数据盘迁移脚本安全结论

文件：`scripts/setup_data_disk_automount.sh`

| 检查项 | 结论 | 依据 |
|---|---|---|
| 识别目标盘方式 | ✅ 安全 | 优先用已挂 `/mnt/data` 且含 marker；否则遍历 `lsblk` 的 **ext4** 分区，**只读挂载**到临时目录验证 marker 后 `umount` |
| 误识别系统盘 | ✅ 不会 | 系统盘虽也是 ext4，但不含 marker 目录 → 跳过；只读挂载即使尝试也无写入 |
| 格式化风险 | ✅ 无 | grep `mkfs\|wipefs\|dd \|fdisk\|parted\|> /dev` **零命中**（仅注释含 `/dev/vdX`） |
| 清空目录风险 | ✅ 无 | 无 `rm`；仅 `mkdir -p /mnt/data` |
| fstab 备份/更新 | ✅ 有 | `cp -a /etc/fstab /etc/fstab.bak.<时间戳>`；`sed` 仅删第 2 字段为 `/mnt/data` 的行（不影响 `/mnt/data2`、不碰注释）；追加 UUID 行 |
| UUID 识别 | ✅ 正确 | `blkid -s UUID -o value`；实测 `5a9d9737-e795-4d2a-9c93-3d7b443f9054` 与 `blkid /dev/vdb` 一致 |
| 迁移后重挂原 150GB 盘 | ✅ 可以 | fstab 用 `UUID=` + `nofail` + `device-timeout=30s`；另有 `mnt-data-ensure.service` 40×15s 重试兜底 |
| 缓存软/硬链接有效性 | ✅ 保持 | `find hf_home -xtype l` → **0 断链**；`find hf_home -type l -lname '/*'` → **0 绝对链接**（全为 `../../blobs/...`）；硬链接由同 fs 快照天然保留 |

**建议流程**：数据盘打快照 → 新实例挂载 → 跑 `setup_data_disk_automount.sh` → `findmnt /mnt/data` 校验。

---

## 6. 运行约束（必须全部满足，否则 40GB 不成立）

1. **两阶段独立进程**：先 `recon_inference_run.py`（写 .pt）退出，再跑 `enhanced_recon_inference.ipynb`（读 .pt）。
2. `num_samples = 1`（第一阶段另有断言 `num_samples_per_image == 1`）。
3. `plotting = False`（否则加载 bigG 图像塔 +7.38GB）。
4. `minibatch_size = 1`（代码固定）。
5. **不使用 refiner**（官方 img2img 无 refiner）。
6. `use_ema = False`（配置默认，`ema_scope()` 无额外副本）。
7. 允许 **GIT CPU offload**（`clip_text_model.to('cpu')`），仅实测 OOM 时启用，省 1.58GB。
8. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
9. 推理机不并行跑第二个占显存进程。

> 功能性前提（已核验）：S1 ckpt 元数据 `stage=1, epoch=119, use_prior=True, blurry_recon=True, hidden_dim=4096, n_blocks=4, clip_scale=1.0`。若 `use_prior=False`，`recon_inference_run.py` L749 会直接抛错，无法出重构图。

---

## 7. 推荐配置

| 项 | 推荐 |
|---|---|
| GPU | **A100 40GB（PCIe 可）**；若预算允许或要开 `num_samples>1`/refiner → 80GB |
| 主机 RAM | **≥64GB，推荐 96GB**（unCLIP ckpt 18GB 先常驻 CPU 再搬 GPU；S1 ckpt 26.8GB 已用 `mmap=True`） |
| 数据盘 | 现有 150GB ext4（已用 73G，余 67G） |
| 驱动/CUDA | Driver ≥550、CUDA 12.4（PyTorch 2.5.0 cu124） |
| 带宽注意 | 40GB 多为 PCIe：1555 vs 2039 GB/s → 采样约慢 1.3×（不影响可行性） |

## 8. 上机后必须补测（把推算变实测）

- 第一阶段首图后：`torch.cuda.max_memory_allocated()` / `max_memory_reserved()` + `nvidia-smi`（预期 allocated ≤28GB，nvidia-smi ≤30GB）
- 第二阶段首图后：同上（预期 allocated ≤17GB）
- 若第一阶段 nvidia-smi > 34GB：先 GIT offload；仍超则改租 80GB

