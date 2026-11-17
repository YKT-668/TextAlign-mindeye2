# 最终下载优化与实验资产复核报告

> 日期：2026-11-14。所有速度均为本机实测，未使用理论带宽。

## 1. 当前下载速度与真实瓶颈

- 实测速度：**87–780 KiB/s 大幅波动**（均值 ~350 KiB/s），网络是主要瓶颈。
- 证据：同一时刻对多个独立来源测速都在同量级——HF xet（~300–780）、Civitai B2 CDN（~375）、hf-mirror（更慢/404）。若为本地软件瓶颈，速度会更平稳；大幅波动说明是海外公网线路拥塞。
- 结论：**当前服务器到 HF 的海外线路是硬瓶颈**，本地 4 核 4GB 未饱和（CPU 30%、内存 1.5G 可用、磁盘正常）。

## 2. Xet 实际并发与固定并发

- 当前进程已继承 `HF_XET_HIGH_PERFORMANCE=1`（env.sh 设置，导入 huggingface_hub 前生效；hf-xet 1.8.0 ≥ 1.7.0）。
- xet 日志（配置项，均为 default）：
  - `enable_adaptive_concurrency = true`
  - `ac_initial_download_concurrency = 16`、`ac_min_download_concurrency = 4`、`ac_max_download_concurrency = 124`
  - 实测 `Current concurrency = 4` **始终停在最小值**（自适应控制器因高 RTT/慢响应已降到底）。
- **固定并发 `HF_XET_FIXED_DOWNLOAD_CONCURRENCY`**：在已安装 hf_xet 1.8.0 的 `.so` 二进制中用 `strings` **未找到该变量名**，判断 1.8.0 大概率不支持（该变量可能属于更新版本）。因此未做 16/32 固定并发测试（不支持 + 网络已是瓶颈，符合"不支持/失败即停"原则）。
- 结论：并发不是当前瓶颈；自适应控制器已经试过更高并发并降回 4。

## 3. 传统下载路径（HF_HUB_DISABLE_XET=1）

**跳过/不适用**，理由：
- 传统 HTTP 路径是单连接（无 xet 的分块并发），在单连接已是瓶颈的网络下不会更快；
- HF LFS 文件走签名 CDN URL，直接 Range 请求会遇到与 aria2 相同的 `Authorization failed`（此前已实测）；
- 需要完整下载 32–64MB 才能测速，在"网络是瓶颈"已明确的前提下属于无谓消耗（符合任务"若无法保证严格字节上限就跳过"）。

## 4. 备用模型来源调查（Zavy 精确副本）

| 来源 | 是否同一文件 | 结论 |
|---|---|---|
| `pscotti/mindeyev2`（当前） | 是（SHA=f1270c1f…） | 正在下载 |
| `Eata/SDXL_Model` | 是（SHA、大小完全一致） | xet 内容寻址存储，同 hash 走同一 CAS，**无提速** |
| Civitai（`ZavyChromaXL v3.0`） | 是（SHA、大小完全一致） | 独立来源（Backblaze B2 CDN），实测 ~375 KiB/s，**与当前同量级，不更快**，可作备用 |
| hf-mirror / aria2 | — | 此前已失败（404 / 签名 URL），不重复 |

结论：**没有找到更快的精确文件来源**。Civitai 是唯一独立备用源（速度不占优）。

## 5. 网络测速结果

| 目标 | 首响应 | 采样速度 |
|---|---|---|
| huggingface.co | 3.27s | 33 KiB/s |
| cas-server.xethub.hf.co (AWS us-east) | 2.80s | — |
| cdn-lfs-us-1.hf.co (CloudFront) | 5.03s | — |
| hf-mirror.com（美国 IP） | 10.80s | 1 KiB/s |
| Civitai B2 (b2-large.civitai.com) | 1.98s | 375 KiB/s（1MB 测试） |

- 所有 HF/CAS/CDN 域名都在海外（AWS us-east/CloudFront），无代理。国内/国际测速节点未单独测（避免额外流量），但由 Civitai B2 同样 ~375KiB/s 可知：**瓶颈是智星云公网出口到海外，而非 HF 专用线路**。

## 6. 最终采用的下载配置

**保持不变：HF 官方 + hf_xet 1.8.0 + `HF_XET_HIGH_PERFORMANCE=1`（自适应并发）**。
- 未切换 aria2 / 镜像 / Civitai（均不更快或不可用）。
- 未启用固定并发（1.8.0 不支持 + 网络瓶颈）。

## 7. 是否还值得继续优化

**否**。停止标准已满足：
- 无明显的本地软件瓶颈（并发已自适应降到最低，CPU/内存/磁盘未饱和）；
- 没有找到更快的精确文件来源（Eata 同 CAS、Civitai 同量级）；
- 进一步调整无法稳定提升（速度波动是网络驱动的，非配置驱动）。

## 8. 更新后的必需资产清单

必需（5 项，其中 4 项已完成/在途）：
| 文件 | 大小 | SHA256 | 状态 |
|---|---|---|---|
| GT `evals/all_images.pt` | 602,113,195 | 93698254…566bcd1 | ✅ 已下载 |
| GT `evals/all_captions.pt` | 813,044 | 5ac53118…a5a1cd | ✅ 已下载 |
| Zavy `zavychromaxl_v30.safetensors` | 6,938,040,682 | f1270c1f…6411b75 | ⏳ 下载中 |
| Ours S1 重建/ID（3 个 .pt） | 见 checksum | df704fca… / 0a837980… | ❌ 缺失（仅原 A100 可恢复） |
| 官方配置/代码 | — | MindEyeV2 @76a2609 | ✅ 已下载 |

暂不需要（已从队列移除，未删除、未下载）：
- 独立 SDXL Base 1.0（6.9GB）、SDXL Refiner（6.1GB）、unCLIP 第一阶段 ckpt、fMRI beta 数据。原因：本轮复用历史第一阶段结果；若未来要重跑第一阶段再评估。

## 9. Ours S1 enhancedrecons 的真实含义（二次核验）

**关键结论：`step6d_full_ours_llmhard_all_enhancedrecons.pt` 是第一阶段 unCLIP 原始重建图，不是 GIT img2img 精炼结果。**

代码证据（GitHub 公开仓库 `YKT-668/TextAlign-mindeye2`，commit `231850a314ab85fc99e0238b1eee7ccd56c01155`，父提交 `d4aad858…`，仅改 readme5-1.md）：
- `src/utils.py` `unclip_recon`：`z = torch.randn(num_samples,4,96,96)` **从随机噪声开始**（VERIFIED，实际抓取到代码）。
- `src/recon_inference_run.py` 导出段：`torch.save(official["enhancedrecons"][:filled], f"{model_name}_all_enhancedrecons.pt")`；同一段还保存 blurryrecons/clipvoxels/prior_out/backbones/predcaptions/ids（VERIFIED）。
- 该脚本**只有第一阶段**（GIT caption 生成 + unCLIP），**没有 img2img 精炼阶段**（VERIFIED：脚本结构与官方 `recon_inference.ipynb` 同构，精炼在官方是独立 notebook）。

结论级别：
- 「代码中 enhancedrecons = unCLIP 输出」：**VERIFIED**（代码证据）。
- 「7-10 运行用的正是该代码」：**INFERRED**（强证据：命令含 `--export_official_pts`，与代码导出段一致；7 文件结构一致；但无 A100 git log 直接证明）。
- 因此「step6d enhancedrecons 可作 img2img 输入」：**INFERRED（强）**，可作为方案 A 的候选输入。

**为什么文件名与官方不同**：官方第一阶段保存 `*_all_recons.pt`（未精炼），第二阶段才叫 `*_all_enhancedrecons.pt`；我们的脚本把 unCLIP 输出写进了名为 `enhancedrecons` 的 key 并按官方「enhancedrecons」命名保存——即**命名沿用了官方"增强"叫法，但内容实为第一阶段未精炼输出**。fig4 provenance 里的「SDXL-enhanced」指的是「由 SDXL 架构的 unCLIP 生成」（unclip6.yaml 就是 SDXL base 的微调），而非「img2img 精炼」。

**派生结论**：
- 不需要单独的 `all_recons.pt`（enhancedrecons 就是它）。
- `all_enhancedrecons.pt` 可直接复用为方案 A 的 img2img 输入。
- predcaptions 在代码里按 `official["predcaptions"][:filled]` 保存（用 `len()` 打印），**很可能是 list[str]**（BLOCKED：文件缺失无法最终确认类型）；接入官方第二阶段需 `np.array()` 或改索引。
- GIT caption / IDs / 图像三者的样本顺序是否一一对应：**BLOCKED**（文件缺失，无法验证），需拿到文件后用 ids.pt 对齐。

## 10. 尚未解决的问题

1. **Ours S1 三个 .pt 文件仍缺失**，HF（main/master 分支）、本地均无；唯一恢复点 = 原 A100 `/mnt/aaai_ablation_disk`（`/dev/vdc`）的 `aaai_readme3_full_reconstruction_maxthroughput_20260710_150646/ours_llmhard/`。
2. predcaptions 的精确类型（list[str] vs numpy）与三文件样本对齐，需文件到位后才能确认。
3. 7-10 运行的精确代码 commit 无直接证据（ENV_SNAPSHOT 的 git 信息在 GitHub 上，本机拿不到 git log）。

## 11. 下一步推荐方案

1. **继续让 zavy 下载完成**（约 6.9GB，ETA 数小时，取决于网络波动）。
2. **优先恢复 Ours S1 三文件**（重挂原 A100 `/dev/vdc` 拷贝 + 对 SHA）。
3. 拿到文件后：核对 predcaptions 类型、ids 对齐、enhancedrecons shape（应为 [1000,3,224,224] float16）。
4. 写独立第二阶段脚本（复用官方 Cell[3]-[5]，参数 num_samples=1/img2img_timepoint=13/scale=5/num_steps=25/seed=42，输入 enhancedrecons + predcaptions，输出新的精炼结果）。
5. 评测：CLIP cossim + PixCorr（+可选 SSIM/AlexNet）。

---

## 附：只读核验快照（2026-11-14 13:11 UTC）

### 1. 201MB 重建量 vs 629MB 网络传输量

| 指标 | 数值 | 含义 |
|---|---|---|
| `downloaded_bytes`（201MB） | 201,389,021 B | **文件重建量**：实际写入 `.incomplete` 文件的字节（zavy 已拼装 2.9%） |
| `transfer_bytes`（629→633MB） | 633,174,769 B | **网络传输量**：hf_xet 从网络接收的字节（分块下载到 CAS 缓存 + 预取缓冲） |

- 两者差额（~430MB）= "在途"数据：xet 先下载分块到缓存/预取，攒够才拼装进文件，所以网络字节领先于文件字节。这是正常流水线，不是重复下载。
- **无重复下载、无重试**：`retry_count=0`；`download.jsonl` 中 `file_retry` 事件 0 条、`file_failed` 事件 0 条。xet 内容寻址缓存去重，重启前的 32MB 分块未被重复下载。
- 注：存在一个旧 `.incomplete` 文件（`f1270c1f…9dda8571.incomplete`，262KB，20:17）= 优化重启前遗留的孤儿文件，已不被写入（当前写入的是 `…993c7454.incomplete`）。

### 2. ETA 速度口径与三种计算

- ETA 公式（download_manager 内）：`ETA = 剩余文件字节 / speed`，其中 `speed = 最近速度(cur_bps，120s 滚动窗口) 若>0，否则均值`。
- 之前显示的"16 小时"用的是当时 **最近速度 116KB/s**（13:03:40 快照，剩余 6,736,651,661B ÷ 116,352B/s = 16.1h）。
- 按最新快照（13:11:01，剩余 6,736,651,661 B）计算三种口径：

| 口径 | 速度 | 预计剩余 |
|---|---|---|
| 最近速度（cur_bps，120s 窗口） | 93,495 B/s (91KB/s) | 72,052s ≈ **20.0 小时** |
| 当前速度 | 同上（本系统"当前速度"即 120s 窗口，无独立瞬时值） | ≈ **20.0 小时** |
| 观察均值（status.json avg_bps） | 314,091 B/s (307KB/s) | 21,448s ≈ **6.0 小时** |
| （补充）真实网络均值 = transfer/elapsed | 247,237 B/s (241KB/s) | 27,248s ≈ **7.6 小时** |

说明：`avg_bps`（307KB/s）把"重启时已完成的 603MB"瞬间计入、再除以较短时间，**偏高**；真实网络均值 ~241KB/s 更可信，故**现实 ETA 约 7.6 小时**（速度仍会波动）。

### 3. 4/5 完成口径与两类完成率

- **下载队列**（manifest 5 项：provenance×2 + gt_captions + gt_images + zavy）：
  - 文件数完成率：4/5 = **80%**；
  - 字节完成率：804,322,920 / 7,540,974,581 = **10.7%**。
- **全部实验必需资产**（5 组）：
  1. GT images ✅  2. GT captions ✅  3. Zavy ⏳(2.9%)  4. Ours S1 三文件 ❌(缺失)  5. 官方代码/配置 ✅
  - 资产组完成率：3/5 = **60%**（3 完成、1 进行中、1 缺失）。

### 4. 进程健康与文件增长

- tmux 会话 ALIVE；进程 pid 52740（python，CPU 3.5%、内存 2.4%）正常。
- transfer_bytes 在 35 秒内从 629,913,753 → 633,174,769（+3.26MB ≈ 93KB/s），**网络持续在收数据**。
- 重建量 201MB 短暂不变（等待下一块拼装，xet 正常行为）。
- 无 failed/retry/stall 事件。

### 5. Ours S1 缺失文件的新发现

- **无新的文件位置**：三个输出 `.pt` 仍只在原 A100 `/mnt/aaai_ablation_disk`（`/dev/vdc`）。
- 本轮唯一新增（已记录于正文第 9 节）：GitHub 公开仓库 `YKT-668/TextAlign-mindeye2` 有代码；其 `readme5-1.md` 确认 Ours S1 **checkpoint**（`last.pth`）在 HF Model 仓库 `master` 分支（revision `a127f9295fd5656a`），但该分支（17 文件）**不含** step6d 输出 `.pt`。即：checkpoint 可恢复（可重跑推理），但三个已生成输出文件仍无法从 HF 找回。

