# CLIP-L / SwAV 资源可行性决策

本轮仅做调查与实测评估，**未下载大权重、未跑 GPU**。
当前机器：4 vCPU / 3.87GB RAM（可用 ~1GB）/ 5.9GB swap / 无 GPU。

## 1. 结论

| 问题 | 结论 |
|---|---|
| SwAV 是否有可验证的官方替代下载源 | ✅ **无需替代源——官方源本轮已可直连成功**（详见 §2，实际仅 113.7MB，不是 800MB） |
| CLIP-L 是否值得在 CPU 上跑完 3000 张 | ❌ **不值得**（实测 14.1 s/张 → 约 11.7 小时） |
| 更优方案 | ✅ **短租 1 小时 40GB GPU** 一次性导出 CLIP-L 特征（SwAV 也可顺带完成），成本远低于 12 小时 CPU |
| 是否建议升级 CPU | 若后续还有大量"中小模型 + 文本"任务才考虑；**仅为本项指标升级不划算** |

> ⚠️ 修正：上一轮"SwAV ~800MB、官方源不可达"的判断**部分有误**。本轮实测官方文件
> `swav_800ep_pretrain.pth.tar` **只有 113.7MB**（"800ep" 指 800 epoch，不是体积），
> 且用 8 连接分段下载 100 秒左右即可完成。

## 2. SwAV 官方来源调查（本轮实测，结论已修正）

官方入口（MindEye2 notebook 用）：
```python
swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
```
其 `hubconf.py` 内部走 `torch.hub.load_state_dict_from_url(...swav_800ep_pretrain.pth.tar)`，域名 **`dl.fbaipublicfiles.com`**。

本轮实测（本机，2026-11-15）：

| 测试 | 结果 |
|---|---|
| `HEAD https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar` | **HTTP 200，Content-Length = 113,703,565** |
| aria2c `-x8 -s8` 分段下载 | **完整下载成功**，113,703,565 字节，≈1.1 MiB/s，约 100 秒 |
| 单连接 urllib 下载（上一轮） | 只到 23.6MB 就超时（≈0.1MB/s）→ 上一轮失败是**单连接限速**，不是资源不可得 |
| 文件内容校验 | `torch.load` 成功：dict，**328 个 key**，`module.conv1.weight` 前缀，**28.4M 参数** → ResNet50 backbone |

**结论修正**：
- 之前的"~800MB"是**误判**（把 800ep 当成体积）；
- 官方源**可达**，多连接即可稳定获取；
- 因此**不再需要任何第三方镜像**，"第三方权重无法验证身份"的问题自动消解。

**已归档（本轮）**
```
路径：/root/.cache/torch/hub/checkpoints/swav_800ep_pretrain.pth.tar
大小：113,703,565 bytes
SHA256：029d0b8c2e70bcee3f8beb70cc104ef51585c090224d33fb9c6f51d146cfd1eb
来源：https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar
```
（记录文件：`reports/SWAV_WEIGHT_PROVENANCE.txt`）

**关于"官方 checksum"**：SwAV 仓库与 hubconf **不发布 SHA256**；可用的完整性证据是
① 来源域名 = 官方 `dl.fbaipublicfiles.com`；② Content-Length 与下载字节数一致；
③ state_dict 结构/参数量符合 ResNet50-SwAV。本报告按此口径标注为 **OFFICIAL_SOURCE_NO_PUBLISHED_CHECKSUM**。

**下一步（下一轮，约 10 分钟 CPU）**：直接 `torch.hub.load` 或本地加载该权重，
对 1000+982 张重建图提取特征，补齐 P1 的第 7 项（SwAV 距离）。
本轮按任务要求**未执行** SwAV 前向。

## 3. CLIP-L 在 CPU 上的实测（VERIFIED，本轮沿用上轮实测）

| 项 | 值 |
|---|---|
| 模型 | `openai/clip-vit-large-patch14`（HF，OpenAI 权重） |
| 实现 | `CLIPVisionModelWithProjection`，fp32，batch 8 |
| 实测 | **8 张 / 113 s = 14.1 s/张**（4 核，含 swap 压力） |
| fp16 尝试 | **更慢**（x86 无 fp16 SIMD），已放弃 |
| 仅视觉塔 | 已采用（省 ~0.5GB），仍 14 s/张 |

**外推（线性近似，仅供参考，非实测）**
| 配置 | 估计 s/张 | 3000 张 |
|---|---|---|
| 4 核（当前） | 14.1（实测） | ≈11.7 h |
| 8 核 | ≈7 | ≈5.8 h |
| 16 核 | ≈3.5 | ≈2.9 h |
| A100-40G（短租） | <0.05（同机 bigG 实测 33 张/秒，CLIP-L 更小） | **<1 min** |

> 注意：上表除"4 核"外均为**外推**，未实测；本轮按要求**不再把"升级后 1 小时"当成事实**。

## 4. 建议的预热基准测试方案（50–100 张，下一轮执行）

```text
目标：以实测替代外推，确定 CPU 配置是否需要升级
样本：从 1000 张中固定取前 100 张（idx 0..99），跳过 3 次预热（idx 0..2）
流程：模型加载 → 3 次 warmup → 计时 50 张（batch 8）→ 记录
指标：median / mean s/张、p95、峰值 RSS、swap 增长、MemAvailable 最低
配置对比：当前 4 核 4GB；目标机 8 核 16GB / 8 核 32GB（同一脚本、同一 batch）
判据：若 8 核中位 s/张 ≤ 4 s（→3000 张 ≤3.3h）则升级有意义；否则直接短租 GPU
```

## 5. 短租 GPU 的成本/收益

| 方案 | 时间 | 说明 |
|---|---|---|
| CPU 4 核跑 CLIP-L 3000 张 | ≈11.7 h | 当前机器，几乎不可行 |
| CPU 16 核 | ≈2.9 h（外推） | 仍需大内存；且 SwAV 仍缺权重 |
| **A100-40G 短租 1 小时** | **<5 min 计算** | 可同时导出 CLIP-L 3000 张（并重验 bigG/SwAV 若权重就位），剩余时间用于校验 |

**推荐**：下一次开 GPU 时把"CLIP-L 图像特征 + SwAV（若拿到权重）"一次性导出到
`evaluation/precomputed_features/`，之后全部在 CPU 上复用（与 bigG 同样做法）。

## 6. 与 P1/P2 的关系

- P1 缺的 **CLIP-L 图像 N-way**、**SwAV 距离** 都属于"必须图像特征"的指标 → 走 GPU 导出最划算；
- P2 的 **CLIP-L 文本**指标**不需要**图像前向，本轮已可完成（见 P2 报告）。
