# A100 GPU 租赁前最终门禁审计

> 日期：2026-11-15。本轮只做 CPU 端审计、小文件下载、静态检查。未租 GPU、未推理、未训练、未删除资产。

## 最终结论：GPU_RENTAL_READY（附 5 项 WARN，均不阻塞第一阶段推理）

## 一、六个大文件独立 SHA256 复核 —— PASS

| 文件 | 实际大小 | SHA256（独立重算） | 结果 |
|---|---|---|---|
| bigG_to_L_epoch8.pth | 7,086,032 | 2f540faf…eef36cf | ✅ |
| wds/subj01/new_test/0.tar | 38,410,240 | 16f810c0…64fb5c5 | ✅ |
| zavychromaxl_v30.safetensors | 6,938,040,682 | f1270c1f…6411b75 | ✅ |
| betas_all_subj01_fp32_renorm.hdf5 | 1,886,882,048 | 71992eb8…991ecdd | ✅ |
| unclip6_epoch0_step110000.ckpt | 17,980,447,939 | 6bd7033d…df155652 | ✅ |
| s1_textalign_stage1_FINAL_BEST_32_last.pth | 26,788,144,501 | eb59c426…84b8e91f | ✅ |

- checkpoint 来自 `ykt668/textalign-mindeye2-model` @ `a127f9295fd5656ac63ae436f07e61b80bf4efce`（master），非 main 分支 8.6GB 同名文件。
- 无同名重复、无缓存与正式文件双倍占用（硬链接共享 inode）。

## 二、代码版本 —— PASS

- 已从 `YKT-668/TextAlign-mindeye2` @ `archive/final14`（`231850a314ab85fc99e0238b1eee7ccd56c01155`）下载完整源码到 `$ROOT/source/TextAlign-mindeye2-231850a/`。
- 本地核验 `models_textalign.py`：`BrainNetwork(h=4096, in_dim=15724, out_dim=768, seq_len=2, n_blocks=4, clip_size=768)` + `TextAlignHead(token_dim=1664, hidden_dim=2048, text_dim=768)`；`subject_sizes[1]=15724`。
- 确认用 `models_textalign.BrainNetwork`（含 TextAlign head），不误用官方 `models.BrainNetwork`（clip_size=1664 无 head）。

## 三、数据完整性 —— PASS（beta/new_test）+ WARN（GT caption）

- Beta/new_test：SHA 通过；官方代码逻辑已核（3000→1000 唯一→3 次重复）。内容级检查脚本已写但因终端故障未执行 → WARN（文件已 SHA 验证，风险低）。
- 1000↔shared982：PASS（`S1_NEW_TEST_TO_SHARED982_MAPPING.csv`，shared982==前 982，18 额外在末尾，无错位）。
- GT images [1000,3,224,224] PASS；GT caption 文本级配对 INFERRED → WARN（仅影响评测，不影响生成）。
- `vox_include_subj01.npy` 代码未加载，无需下载。

## 四、GIT 与运行时依赖 —— WARN（下载中）

- GIT `microsoft/git-large-coco` @ `b644b0b…`（11 文件，model.safetensors 1.5GB）已确认，下载已启动（排除重复 pytorch_model.bin）→ WARN（完成后 local_files_only 验证）。
- `modeling_git.py`（GitForCausalLMClipEmb）在本地源码中。
- OpenCLIP：`FrozenOpenCLIPImageEmbedder`/`Embedder2`（ViT-bigG-14）确实需 bigG 权重 ~5GB（一阶段无条件加载 image embedder + 二阶段 text embedder）+ CLIP L ~1.7GB。下载已启动 → WARN。

## 五、第二阶段 img2img —— PASS

Zavy SHA 通过；不重复下 Base/Refiner；参数（num_samples=1/img2img_timepoint=13/scale=5/num_steps=25/seed=42/256→768→256）与官方 notebook 一致；VAE 含于 SDXL 权重内无需独立。

## 六、Python 环境 —— WARN（GPU 另装完整栈）

当前 `download_venv` 仅下载用途（torch 2.15.0+cpu + numpy/h5py + safetensors），缺 transformers/open_clip_torch/diffusers/einops/omegaconf/pytorch_lightning/webdataset/deepspeed/dalle2_pytorch。GPU 推理环境需在 A100 另装（历史参考 PyTorch 2.5.0/CUDA 12.4/Driver 550）。PyTorch 2.6+ 的 `weights_only=True` 默认会破坏 `torch.load`，建议固定 2.5.0。

## 七、checkpoint 加载与 RAM

- 加载：`torch.load(map_location='cpu')` → `checkpoint['model_state_dict']` → `load_state_dict`。
- 26.8GB 文件 ≠ 26.8GB 峰值：load 整份字典 ~26.8GB（含优化器状态）+ 模型参数，**预计峰值 ~30–36GB**。
- unCLIP ckpt 加载峰值 ~17–20GB，与 checkpoint 可能先后驻留。
- **建议主机 RAM：64GB 最低，96GB 更稳妥**。

## 八、显存 / 时间 / 输出空间

- A100 80GB 足够（README3 实测 unCLIP 重建峰值 28.1GB，表示 10.5GB）；48GB 可用，40GB 偏紧。
- 小样本测试：先 max_save=2 跑通全链 → 10–20 张测耗时 → 再估算 1000 张（禁止凭历史 1.5–2h 直接跑 1000）。
- 输出：七类一阶段 ≈3.2GB + 二阶段 ≈0.3–1GB + GIT 1.5GB + OpenCLIP 6.7GB ≈ 12GB 新增；磁盘 86GB 可用，留 20GB 余量后充足。

## 九、数据盘迁移安全

- `/dev/vdb` 独立持久化盘；fstab 用 UUID `5a9d9737-e795-4d2a-9c93-3d7b443f9054`（不依赖设备名），新服务器可凭 UUID 挂载。
- 迁移建议：快照→停止进程→sync→卸载→控制台解绑→确认存在→释放 CPU→A100 挂载→只读验 SHA。**不得 mkfs**。

## 门禁汇总

| 项 | 状态 |
|---|---|
| 六文件 SHA | PASS |
| 代码完整/版本固定 | PASS |
| checkpoint 兼容 | PASS |
| beta/new_test | PASS(SHA)+WARN(内容检查未执行) |
| 1000/shared982 | PASS |
| GT caption 文本 | WARN（评测项，非推理项） |
| GIT 离线 | WARN（下载中） |
| OpenCLIP | WARN（下载中） |
| 第二阶段 Zavy | PASS |
| Python 依赖 | WARN（GPU 另装） |
| RAM 明确 | PASS（64GB 起） |
| 显存合理 | PASS |
| 输出空间 | PASS |
| 数据盘迁移 | PASS |

**最终结论：GPU_RENTAL_READY**（WARN 均不阻塞第一阶段推理；GIT/OpenCLIP 可补齐或 GPU 阶段自动下载；GT caption 为评测项）。

---

## 附录：CPU 收尾轮更新（2026-11-15）

### 1. GIT / OpenCLIP 下载结果
- **GIT** `microsoft/git-large-coco` @ `b644b0b41274649f87de8846fa15f9a22dad7583`：**8/8 文件下载完成**（含 `model.safetensors` 1,576,861,900 B），已排除重复的 `pytorch_model.bin`。显式 revision 离线加载 **OK**；已创建 `refs/main` 指针（支持 `from_pretrained` 无 revision 离线加载）。
- **OpenCLIP** `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`：下载中（17 文件，~5GB 权重）；`laion/CLIP-ViT-L-14-laion2B-s32B-b82K` 紧随其后。完成后验证。
- 与 Zavy 重叠：GIT/OpenCLIP 是文本/图像编码器（CLIP 双塔 + GIT），**不含 SDXL VAE/UNet**，与 `zavychromaxl_v30.safetensors` 无权重重叠（无重复占用）。

### 2. beta / new_test 内容级检查 —— PASS（已在 CPU 完成）
- **beta**：`betas` shape=(30000, 15724) float32，NaN=0/Inf=0（前200），voxel_dim=15724（= subject1）。
- **new_test**：3000 条 behav（各 [1,17]），拼接 (3000,17)；**1000 唯一 ID，每图重复 3 次**；图像 ID 范围 2950–72948；`behav[:,0,5]` 为 **beta 行索引（trial row）** 0–29965（beta 第一维 30000 行内，无越界）；体素维为 beta 第二维 15724（无独立体素选择索引）。
- **GT**：all_images (1000,3,224,224) float32；all_captions 为 numpy (1000,) 真实文本，**0 空**。
- **交叉**：new_test 唯一 ID == test_73k_images（集合完全一致，无错位/缺失）。
- 1000/shared982：前 982 为 shared982，后 18 额外（VERIFIED）。
- GT caption 文本 → 内容级已验证（1000 条真实文本），逐图配对为构造同序（保留 WARN 性质，不影响生成）。

### 3. GPU 环境锁定文件
- `reports/GPU_ENV_LOCK.txt`（Python 3.12.7 / PyTorch 2.5.0 / torchvision 0.20.0 / CUDA 12.4 + transformers/diffusers/open_clip_torch 等固定版本；明确固定 PyTorch 2.5.0 规避 2.6+ 的 weights_only 破坏）
- `reports/install_gpu_env.sh`（A100 安装脚本）
- `reports/GPU_ENV_PREFLIGHT.sh`（离线预检：torch/CUDA/import/local_files_only/资产存在性）

### 4. 28.1GB 显存峰值的历史来源与适用阶段
- 来源：`READMЕ3_A100_FINAL_INFERENCE_REPORT.md` 第 6 节 `GPU Memory Profile`（A100 实测）。
- 适用：第一阶段 unCLIP 重建（"SDXL denoising" 峰值 ~28.1GB，因 unclip6 基于 SDXL 架构）+ 第二阶段 SDXL img2img（同为 SDXL UNet 去噪，峰值同量级）。表示推理（3000 样本）~10.5GB。
- 结论：A100 80GB 充足；48GB 可用；40GB 偏紧。

### 5. 96GB RAM 依据
- checkpoint 加载峰值 ~30–36GB（26.8GB 字典含优化器状态 + 模型参数 + Python 开销）；
- unCLIP ckpt 加载 ~17–20GB（与 checkpoint 可能先后/部分驻留）；
- GIT ~1.5GB + OpenCLIP bigG ~5GB 常驻；
- 峰值叠加 + 操作系统 + 页缓存 → 64GB 偏紧，**96GB 稳妥**。

### 6. 数据盘迁移安全结论
- `/dev/vdb` 独立持久化盘（与系统盘 /dev/vda3 分离），`/mnt/data` ext4。
- `/etc/fstab`：`UUID=5a9d9737-e795-4d2a-9c93-3d7b443f9054 /mnt/data ext4 defaults,nofail 0 2`（与 `blkid` 一致，不依赖设备名）。
- 结论：**可安全迁移**（需云平台确认：数据盘独立于实例生命周期 + 支持快照；若平台不能保证，则 BLOCKED，不释放 CPU）。本轮未执行任何释放/格式化操作。

---

## 最终迁移门禁：READY_FOR_A100_MIGRATION

- 六个大文件 SHA256 全部 PASS；
- 代码固定（archive/final14@231850a，本地完整）；
- checkpoint 与代码兼容；
- beta/new_test 内容级 PASS，1000/shared982 映射 PASS；
- GT 图像/caption 内容级 PASS；
- GIT 离线加载 PASS（含 refs/main）；
- OpenCLIP 下载中（完成后即就绪，不阻塞迁移准备）；
- GPU 环境锁定文件 + 安装/预检脚本就绪；
- 主机 RAM 建议 96GB，A100 80GB 足够；
- 数据盘迁移安全。

**唯一待办：等 OpenCLIP（bigG 5GB + CLIP-L 1.7GB）下载完成并验证离线加载。** 在此之前可先创建数据盘快照、准备 A100 实例；不建议在 OpenCLIP 完成前释放 CPU（以免下载中断）。

---

## 附录二：自动挂载 + A100 40G 审计 + 依赖补齐（2026-11-15 第二轮）

### A. 数据盘自动挂载（已配置并验证）

| 项 | 状态 |
|---|---|
| `/etc/fstab` | `UUID=5a9d9737-e795-4d2a-9c93-3d7b443f9054 /mnt/data ext4 defaults,nofail,noatime,x-systemd.device-timeout=30s 0 2` |
| fstab 校验 | `findmnt --verify` → **0 parse errors, 0 errors**；`mount -a` → rc=0 |
| 自动生成的 unit | `mnt-data.mount`，`What=/dev/disk/by-uuid/5a9d9737-...`（不依赖设备名） |
| 兜底重试服务 | `mnt-data-ensure.service`（oneshot，40×15s 重试；已 enable，`systemd-analyze verify` 通过） |
| 可移植脚本 | `scripts/setup_data_disk_automount.sh`（自包含/幂等/自动探测 UUID） |

> **关键点**：`/etc/fstab` 属于实例系统盘，**不会随数据盘迁移**。新 A100 实例上跑一次：
> ```bash
> bash /mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/scripts/setup_data_disk_automount.sh
> ```
> 若数据盘还没挂上，可把该脚本全文复制到本地先执行（脚本自带设备扫描）。
> 若云平台控制台有"数据盘自动挂载"开关，建议一并开启（双保险）。

### B. A100 40GB 可行性（详见 `A100_40G_VRAM_FEASIBILITY_AUDIT.md`）

- 实测峰值 **28.1GB** → **40GB 够用**，余量 ~11.4GB（≈28%）。
- 必须同时满足：两阶段分进程；`num_samples=1` + `plotting=False`（不加载 bigG 图像塔）；无 refiner；`minibatch_size=1`；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 40GB 多为 PCIe 版（带宽 1555 vs 2039 GB/s）→ 采样约慢 1.3×，显存不受影响。
- 主机内存必须核实 **≥64GB（推荐 96GB）**。
- 反例：24GB 卡不够（官方明确只能跑 representation）。

### C. 依赖补齐（本轮新发现）

| 仓库 | 用途 | 状态 |
|---|---|---|
| `openai/clip-vit-large-patch14` | SDXL conditioner 的 `FrozenCLIPEmbedder`（**此前缺失**） | 下载中（13 文件） |
| `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` | SDXL conditioner 的 `FrozenOpenCLIPEmbedder2` | 8.0GB / 17 文件，进行中 |
| `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | 评测用 CLIP 图像塔 | 排队 |
| `microsoft/git-large-coco` | GIT 文本解码 | ✅ 完成（1.5GB，含 refs/main） |

> 说明：bigG 仓库同时提供 HF 分片格式（`pytorch_model-0000X-of-00002.bin/.safetensors`）和 open_clip 格式（`open_clip_pytorch_model.bin`），两种都会落盘，**离线迁移后才不需要联网**。

### D. 迁移注意：HF 缓存是"共享 blob + 软链接"结构

```
hf_home/hub/blobs/<shard>/<hash>        <- 实际权重（当前 53GB）
hf_home/hub/models--*/snapshots/<rev>/  <- 只是指向 ../../blobs/... 的软链接
```
- 迁移必须**整目录复制并保留链接**：
  ```bash
  rsync -aH --info=progress2 /mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/hf_home/ <目标>/hf_home/
  ```
- **只复制 `models--*` 目录会断链**（GIT 的 1.5GB 权重就在 `hub/blobs/86/861b...`）。
- 当前 `hf_home` = hub 58GB + xet 24MB。
- 用同一块盘做快照/迁移则无此问题（快照保留全部 inode 与链接）。

---

## 附录三：下载全部完成 + 离线校验（2026-11-15 12:05）

### E. 结果：13/13 离线校验 PASS

| 仓库 | 关键文件 | 字节数 | 校验 |
|---|---|---|---|
| `microsoft/git-large-coco` | `model.safetensors` | 1,576,861,900 | ✅ OK |
| `microsoft/git-large-coco` | `config / preprocessor_config / tokenizer` | — | ✅ OK |
| `openai/clip-vit-large-patch14` | `pytorch_model.bin` | 1,710,671,599 | ✅ OK |
| `openai/clip-vit-large-patch14` | `config / vocab / merges / tokenizer_config` | — | ✅ OK |
| `laion/CLIP-ViT-bigG-14-...` | `open_clip_pytorch_model.bin` | **10,158,638,769** | ✅ OK |
| `laion/CLIP-ViT-bigG-14-...` | `open_clip_config.json` | 671 | ✅ OK |
| `laion/CLIP-ViT-L-14-...` | `open_clip_pytorch_model.bin` | **1,710,631,365** | ✅ OK |
| `laion/CLIP-ViT-L-14-...` | `open_clip_config.json` | 564 | ✅ OK |

`HF_HUB_OFFLINE=1` 下 `SUMMARY ok=13 bad=0`；两个大权重与 HF 记录**字节数完全一致**。

### F. 体积优化（共省约 40GB 流量）

| 仓库 | 仓库总大小 | 实际下载 | 省掉的内容 |
|---|---|---|---|
| bigG | 40.64GB | **10.16GB** | HF 分片 20.3GB + `open_clip_model.safetensors` 10.16GB（open_clip 2.26.1 只读 `open_clip_pytorch_model.bin`，无 safetensors 回退） |
| openai CLIP-L | 6.85GB | **0.65GB** | `tf_model.h5` / `flax_model.msgpack` / `rust_model.ot` |
| laion CLIP-L | 6.85GB | **1.71GB** | HF 分片与 safetensors 重复格式 |

加速手段：`hf_transfer 0.1.9`（已写入 `env.sh` 的 `HF_HUB_ENABLE_HF_TRANSFER=1`），单文件从 1.5MB/s 提升到突发 10MB/s（平均约 2MB/s）。

### G. 两个必须知道的坑

1. **`refs/main` 不能有尾随换行**：`huggingface_hub 2.4.0` 的 `try_to_load_from_cache` 会把文件内容**原样**当 revision 名，带 `\n` 就永远找不到快照。已修正为精确 40 字节。
   - 修正后 `hf_hub_download("microsoft/git-large-coco", ...)`（不传 revision）离线 **OK**，即 `from_pretrained("microsoft/git-large-coco")` 可用。
2. **`snapshot_download(local_files_only=True)` 对 GIT 会抛 `IncompleteSnapshotError`**：因为我们**故意**没有下载 `pytorch_model.bin`（与 safetensors 重复）。这**无害**——推理代码走 per-file `from_pretrained`；`GPU_ENV_PREFLIGHT.sh` 已改为逐文件校验。

### H. 磁盘与清理

- `hf_home/hub` = **70GB**（其中大文件与 `assets/` 是**硬链接**，`df` 不会重复计）
- `hf_home/xet` = 31MB
- 清理：bigG 废弃 `.incomplete` **12.54GB** + 数据集残留 **0.94GB**
- `/mnt/data`：73G 已用 / **67G 可用**

### I. 迁移补充提醒

`assets/models/*` 与 `hf_home/hub/blobs/*` 通过**硬链接**共享 inode（如 S1 26.8GB、unclip 18GB、Zavy 6.9GB 均只占一份）。因此：
- 用**整盘快照**迁移最稳；
- 若用 rsync，必须对整个项目目录 `rsync -aH`（`-H` 保留硬链接），否则会把 70GB 复制成 100GB+。

### J. 最终状态

**所有资产下载完毕，离线校验 13/13 PASS → 状态从 `READY_FOR_A100_MIGRATION` 更新为 `ASSETS_COMPLETE_READY_TO_MIGRATE`。**

- 6 个大文件 SHA256：PASS
- 4 个 HF 仓库（GIT / openai CLIP-L / bigG / laion CLIP-L）：离线加载 PASS
- beta/new_test/GT 内容级：PASS
- GPU 环境锁定 + 安装/预检脚本：就绪（预检已改为逐文件校验）
- 数据盘自动挂载：fstab + 重试服务 + 可移植脚本
- 显存：A100 40GB 够用（实测峰值 28.1GB），主机 RAM ≥64GB（推荐 96GB）

**现在可以：创建数据盘快照 → 开通 A100 → 挂盘 → 跑 `setup_data_disk_automount.sh` → 跑 `install_gpu_env.sh` → 跑 `GPU_ENV_PREFLIGHT.sh`。**




