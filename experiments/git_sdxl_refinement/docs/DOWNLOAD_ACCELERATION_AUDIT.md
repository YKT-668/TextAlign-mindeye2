# 下载链路诊断与加速优化报告

> 日期：2026-11-14。所有速度均来自本机实际测试，未使用理论带宽。

## 执行摘要

- 原下载工具 = **huggingface_hub 2.4.0 + hf_xet 1.8.0**（非 aria2/wget），单文件顺序下载，xet 默认自适应并发（约 4 个 range 请求，但只建立 ~2 条 TCP 连接）。
- 慢的根本原因 = **海外网络线路拥堵**（HF 及 CAS/CDN 均在 AWS us-east/CloudFront，实测首响应 3~11s、采样仅 1~33KB/s）。
- 优化措施：① 跳过非必需的 SDXL Base + Refiner（总量 20.5GB→7.5GB，省 13GB）；② 启用 `HF_XET_HIGH_PERFORMANCE=1`（实测 ~2.2x 起，重启后实际 ~780KB/s，约 20x，含网络恢复因素）。
- 已安全重启下载（xet 断点续传，未删任何已完成/部分文件）。

## 1. 原先使用的下载工具

**B：huggingface_hub + hf_xet**（VERIFIED）。
- 证据：下载进程 `download_manager.py` 调用 `hf_hub_download`；HF 缓存出现 `.incomplete` 文件；xet 日志（`hf_home/xet/logs/`）持续写入；无 aria2/wget 下载进程。
- 单文件顺序下载（manifest 顺序逐个下载），无多文件并发。
- xet 默认自适应并发（日志显示 concurrency 从 4 起，随带宽调整）。

## 2. 原先是否使用 aria2

**否**。aria2 原本未安装（本次诊断才 `apt install aria2 1.36.0`）。

## 3. 原先是否启用 hf_xet

**是（默认自适应并发）**，但**未启用高性能模式**。
- `HF_HUB_DISABLE_XET` 未设置 → xet 开启。
- `HF_XET_HIGH_PERFORMANCE` 未设置（env.sh 中原本注释为"内存不足不启用"）。
- `HF_HUB_ENABLE_HF_TRANSFER` 未设置（已弃用，正确）。

## 4. 是否存在单连接下载问题

**是（部分）**。实测下载进程（pid 33413）只有 **2 条 TCP 连接**（AWS 54.254.163.78 + Meta 31.13.95.33），xet 的多个 range 请求复用在这极少的连接上，海外单连接成为瓶颈。脚本本身单文件顺序、无多文件并发。

## 5. 网络线路是否存在异常

**是**。5 个主机 DNS 均正常，但 HTTPS 都慢（无代理，`http_proxy/https_proxy/all_proxy` 均 unset）：

| 主机 | DNS | HTTPS 实测 |
|---|---|---|
| huggingface.co | OK | 200，首响应 3.27s，采样 33KB/s |
| cas-server.xethub.hf.co | OK（AWS us-east） | 404（预期），2.80s |
| transfer.xethub.hf.co | OK（AWS us-east） | 403（预期），2.06s |
| cdn-lfs-us-1.hf.co | OK（CloudFront） | 403（预期），5.03s |
| hf-mirror.com | OK（**美国 IP 104.250.143.18**） | 200，首响应 **10.80s**，采样 1KB/s |

结论：海外线路拥堵是主要瓶颈；hf-mirror 也解析到美国 IP、响应更慢，对本服务器并非"国内加速"。

## 6. 官方 HF 与 HF-Mirror 谁更快

**官方 HF 更快**（在本服务器上）。
- 官方 HF：可正常下载（xet），慢但可用。
- HF-Mirror：对本次数据集文件（`/pscotti/mindeyev2/resolve/<commit-sha>/zavychromaxl_v30.safetensors`）返回 **404**（固定 revision 不可用）；首页响应 10.8s。结论：HF-Mirror 对本任务**不可用/不更快**。

## 7. 高性能 hf_xet 是否有帮助

**有帮助**。实测对比：

| 配置 | 实测速度 |
|---|---|
| xet 默认自适应并发 | ~38–41 KiB/s |
| xet + `HF_XET_HIGH_PERFORMANCE=1`（27MB 测试，当时网络拥堵） | ~90 KiB/s（2.2x） |
| 重启后实际（zavy，含网络恢复） | ~780 KiB/s（约 20x） |

说明：部分提速来自高性能模式提高并发，部分来自网络波动恢复；但启用后确实显著提升且稳定。

## 8. 是否成功实现多连接 / 多文件并发

- **多连接（xet 高性能模式）**：✅ 成功（range 并发提高）。
- **aria2 多连接**：❌ 失败。官方 HF 的 LFS 签名 URL 报 `errorCode=24 Authorization failed`（0B/s）；hf-mirror 报 `errorCode=3 Resource not found`。aria2 无法用于 HF LFS 签名 URL。
- **多文件并发**：未采用（跳过后只剩 1 个必需文件 zavy，无并发需求）。

## 9. 实际速度提高多少倍

- 速度：约 38KB/s → 约 780KB/s，**约 20 倍**（含网络恢复因素）。
- 总量：20.5GB → 7.5GB，**减少 63%**（跳过非必需权重）。

## 10. 最终采用的下载方案

**HF 官方 + hf_xet + `HF_XET_HIGH_PERFORMANCE=1` + 跳过非必需权重**。
- 未修改 `download_manager.py` 代码；仅改 `env.sh`（启用高性能模式）和 `manifests/download_manifest.tsv`（移除 base/refiner 并加注释）。
- 未采用 aria2 / HF-Mirror（不适用）。

## 11. 是否安全保留原部分下载文件

**是**。重启前：
- 已完成文件 `all_captions.pt`（813KB）、`all_images.pt`（602MB）已硬链接到 `assets/`，重启后按大小检测跳过，未重下。
- zavy 已传输的 ~32MB 由 xet CAS 缓存断点续传，未删除任何 `.incomplete`/缓存/临时文件。
- 未出现两个进程同时写同一文件。

## 12. 跳过的非必需文件

- `sd_xl_base_1.0.safetensors`（6.94GB）— 跳过（官方精炼用 zavy，不用标准 base）。
- `sd_xl_refiner_1.0.safetensors`（6.08GB）— 跳过（官方精炼不用 Refiner）。
- 依据：GIT 精炼技术审计（`GIT_IMG2IMG_TECHNICAL_AUDIT.md`）VERIFIED 结论。Zavy、GT images、GT captions 保留（必需）。

## 13. 当前下载完成数与剩余容量

- 完成 4/5 文件（provenance×2 + all_captions + all_images），剩余 zavy（6.9GB）。
- 数据盘 `/mnt/data`：137G 可用。

## 14. 当前速度与预计完成时间

- ~780 KB/s（实测），ETA ~2.5 小时（剩余 ~6.9GB）。

## 15. 后台进程是否正常

- 正常。tmux 会话 `conceptalign_img2img_download` 存活，`status.json` 每 5s 更新，无失败/重试。

## 附：环境与工具版本

- huggingface_hub 2.4.0；hf-xet 1.8.0；hf_transfer 未装（已弃用，正确）；aria2 1.36.0（本次新装，未采用）。
- 环境变量：无代理；`HF_HUB_DISABLE_XET` 未设（xet 开）；`HF_XET_HIGH_PERFORMANCE=1`（本次启用）。
