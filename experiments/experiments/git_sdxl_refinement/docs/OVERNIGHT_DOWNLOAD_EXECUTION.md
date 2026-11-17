# 夜间下载执行报告（并发测试 + 资产队列 + 后台运行）

> 日期：2026-11-14 晚。本报告记录并发实测、最终策略、断点续传与后台稳定性。

## A. 并发测试（实测）

| 测试 | 配置 | 实测吞吐 | 结论 |
|---|---|---|---|
| A 单文件 | Zavy 单独（基线） | ~42 KiB/s | 基线 |
| B 双文件 | Zavy + beta（1.76GB）并行 | Zavy 降至 ~30 KiB/s，beta 几乎 0（5min 仅 +1MB） | **总吞吐 ~30 KiB/s < 基线，并行有害** |
| C 三文件 | 未执行 | — | 双文件已明显变差，按决策表跳过 |

- 证据：beta 的 xet 日志持续报 `connection struggling`（连接挣扎），自适应并发停在最小值 4；5 分钟 beta 的 xet 缓存仅 +1MB，Zavy 从 42→30 KiB/s。
- CPU/内存：并行期间 CPU ~1-3%、内存 ~1.3-3.7%，均未饱和（瓶颈是海外网络，非本地资源）。
- 错误/重试：0。
- **最终 Worker 数量：1（顺序下载）**。原因：网络是硬瓶颈，并行拆分带宽导致总吞吐下降、次要文件被饿死。

## B. 下载任务

**当前队列**（`manifests/overnight_manifest.tsv`，共 6 文件 53.6GB）：

| 文件 | revision | 大小 | SHA256 | 状态 |
|---|---|---|---|---|
| `bigG_to_L_epoch8.pth` | pscotti/mindeyev2 @3f20ccbe | 7.09MB | 2f540faf… | ✅ 完成（47s，SHA 通过） |
| `wds/subj01/new_test/0.tar` | pscotti/mindeyev2 @3f20ccbe | 38.4MB | 16f810c0… | ✅ 完成（115s，SHA 通过） |
| `zavychromaxl_v30.safetensors` | pscotti/mindeyev2 @3f20ccbe | 6.94GB | f1270c1f… | ⏳ 下载中 |
| `betas_all_subj01_fp32_renorm.hdf5` | pscotti/mindeyev2 @3f20ccbe | 1.76GB | 71992eb8… | 等待 |
| `unclip6_epoch0_step110000.ckpt` | pscotti/mindeyev2 @3f20ccbe | 16.75GB | 6bd7033d… | 等待 |
| `checkpoints/…/last.pth`（checkpoint） | ykt668/textalign-mindeye2-model @a127f929 | 26.8GB | eb59c426… | 等待 |

- 实际剩余下载量：约 52.4GB（含 zavy 重新下载的部分）。
- 已排除（不下载）：SDXL Base、SDXL Refiner、21GB COCO 评估文件、vox_include、sd_image_var_autoenc。
- GIT（microsoft/git-large-coco）与 OpenCLIP（laion/CLIP-ViT-bigG-14-laion2B-39B-b160k）为运行时自动下载（GPU 阶段），本轮未单独入队。

## C. 断点续传

- **机制**：huggingface_hub + hf_xet 原生缓存 + manifest 状态 + 本地文件大小/SHA 校验。
- **缓存位置**：`$ROOT/hf_home`（HF_HOME），`$ROOT/hf_home/hub`（HF_HUB_CACHE），`$ROOT/hf_home/xet`（Xet 缓存）。
- **恢复逻辑**：启动时 `is_present()` 检查本地文件大小，已完成且大小匹配的跳过；未完成文件重新 `hf_hub_download`（xet 内部重试/续传）；完成后 SHA256 校验，不匹配标记 FAILED。
- **锁机制**：单进程内线程锁保证「同一文件只有一个 worker」；hf_hub_download 自带 per-file 缓存锁。同一目标文件不会有两个进程写入。
- **已验证**：2 个小文件完成 → 硬链接 → SHA 校验通过；重启后已完成文件自动跳过（本次队列重启时 bigG/new_test 之前未下载，zavy 重新下载）。
- **已知限制**：hf_xet 无持久化分块缓存，跨重启的「进行中」大文件会重新下载（本次 zavy 重启丢失了之前约 1GB 的传输进度）。已完成文件不受影响。

## D. 后台状态

- tmux 会话：`conceptalign_img2img_download`。
- Python 进程 PID：89942（queue_download.py）。
- 连续运行：23:01:56 启动，5 次检查（23:02/23:04/23:06/23:09/23:10+）全部存活、无错误、文件持续进展（2 个小文件先后完成）。
- 最近平均速度：~32-42 KiB/s（网络波动）。
- 最新传输：zavy 重新下载中（慢，网络瓶颈）。
- 日志：`$ROOT/logs/download_queue.log`、`status_queue.json`、`worker_status.json`（原 download.log/jsonl/status.json 为上一阶段 Zavy 记录）。
- 数据盘剩余：136G（安全余量充足）。

## E. 最终结论

1. **并行下载是否有效？** 否。单文件 ~42KiB/s，双文件总 ~30KiB/s（beta 被饿死），网络是硬瓶颈。
2. **今晚采用几个 Worker？** 1 个（顺序下载）。
3. **所有任务是否已安全进入后台？** 是。统一队列（6 文件）在 tmux 会话中运行，2 个已完成。
4. **断点续传是否可靠？** 对已完成文件可靠（自动跳过 + SHA 校验）；进行中文件跨重启会重新下载（hf_xet 无持久分块缓存，属已知限制）。
5. **当前是否可以放心退出服务器？** 是。tmux 会话独立于 SSH/对话，退出后继续运行；队列带重试（每文件 3 次）+ 断点跳过。
6. **预计剩余下载时间？** 约 52GB / ~40KiB/s ≈ **15 天**（网络瓶颈所致，波动大；若网络恢复到白天峰值 780KiB/s 则 ~19 小时）。
