# 最终 GPU 收尾：bigG 特征导出与释放检查

日期：2026-11-15 ｜ GPU：A800-SXM4-40GB ｜ 本阶段为**最后一次 GPU 任务**

## 结论

| 项 | 结果 |
|---|---|
| bigG 特征是否导出成功 | ✅ **成功** |
| 3000 张（A/B/GT）是否全部完成 | ✅ **3000/3000**（另加 2000 张诊断融合组） |
| SHA256 是否通过 | ✅ **6/6 OK** |
| 是否存在异常 | ✅ 无（NaN/Inf=0，ID 对齐，抽查通过） |
| **是否可以安全停止 GPU 实例** | ✅ **可以** |

---

## 1. 前置校验

| 检查 | 结果 |
|---|---|
| bigG 权重文件 | `open_clip_pytorch_model.bin` = **10,158,638,769 B**（与前次审计一致） |
| 权重引用完整性 | 断链 **0** |
| 离线加载 | ✅ `HF_HUB_OFFLINE=1` 下成功 |
| 模型/版本 | `FrozenOpenCLIPImageEmbedder(arch="ViT-bigG-14", version="laion2b_s39b_b160k", output_tokens=True, only_tokens=True)` —— 与 `final_evaluations.ipynb` cell 375 **完全一致**，未替换模型 |
| 预处理 | 该 embedder 自带：`kornia resize(224, bicubic) → (x+1)/2 → normalize(CLIP mean/std)`，A/B/GT 完全同构 |
| A/B ID | `torch.equal(ids1, ids2) == True`，1000 唯一、升序 |
| 原始产物 | 只读，未修改 |

## 2. 关于官方 .75/.25 融合（VAE 一致性判定）

- 官方 `sd_image_var_autoenc.pth` **本地缺失**（我们使用的是 `stabilityai/sd-vae-ft-mse` 转换版，键名重映射后 `strict=True` 0 missing/0 unexpected）。
- **无法逐字节确认**两者与原训练机权重完全一致 → 结论：**不能声称融合结果是"官方复现"**。
- 因此融合组按**诊断组**处理，文件名显式带 `_DIAGNOSTIC`，`feature_manifest.json` 中写明：
  `"status": "DIAGNOSTIC_NOT_OFFICIAL"`，并记录原因。
- 原始 A/B/GT 特征**未受影响**，融合组**未覆盖**任何文件。

## 3. 导出结果

输出目录：`evaluation/precomputed_features/bigg/`

| 文件 | 组 | 形状 | dtype | 大小 |
|---|---|---|---|---|
| `A_raw_features.pt` | A（Stage1 unCLIP 重建，未融合） | (1000, 425984) | float16 | 852 MB |
| `B_refined_features.pt` | B（GIT caption + SDXL img2img，未融合） | (1000, 425984) | float16 | 852 MB |
| `GT_features.pt` | GT（真实刺激） | (1000, 425984) | float16 | 852 MB |
| `A_raw_blend075_DIAGNOSTIC_features.pt` | 诊断：A 的 .75/.25 | (1000, 425984) | float16 | 852 MB |
| `B_refined_blend075_DIAGNOSTIC_features.pt` | 诊断：B 的 .75/.25 | (1000, 425984) | float16 | 852 MB |
| `feature_manifest.json` | 元数据（模型/预处理/ID/耗时/显存/SHA） | — | — | 3.6 KB |
| `SHA256SUMS` | 6 项校验清单 | — | — | 553 B |

**总计 4.26 GB**（5 组）。

> **重要**：425984 = **256 × 1664**，即 embedder 输出的 token 序列按官方方式 `reshape(N, -1)` 展平。
> **保存的是未归一化特征**（与官方一致：官方在计算相似度前才做 `F.normalize(emb.reshape(N,-1), dim=-1)`）。下游使用前**必须 L2 归一化**。
> 每个 `.pt` 是 `dict(features=[1000,425984] fp16, ids=[1000] int64, meta={...})`，自包含 ID 对应关系。

## 4. 校验结果（全部通过）

| 校验项 | 结果 |
|---|---|
| 每组恰好 1000 条 | ✅ 5/5 组 |
| 三组（及诊断组）ID 完全一致、顺序一致 | ✅ `ids_equal_ref == True` |
| 特征维度与模型输出一致 | ✅ 425984 = 256×1664 |
| NaN / Inf | ✅ **0 / 0**（每组） |
| 可重新加载 | ✅ 5 个文件全部 `torch.load` 成功 |
| ID 对应正确（抽查） | ✅ 重算 idx∈{0,432,999} 与已存特征余弦 **0.99951–0.99997** |
| **与既有评测数值交叉验证** | ✅ **完全复现**：导出特征算得 A 0.34516 / B 0.33842 / Δ −0.00674；此前评测为 0.34516 / 0.33842 / −0.00674（max\|diff\| = **3e-6**）；1000-way 检索 A top1 0.5310 / B 0.4290 亦完全一致 |

**SHA256（6/6 `sha256sum -c` 通过）**
```
44b31f5f5ffe5e787c615829a7b4319519f99771f55eb2264340a10c649419ec  A_raw_features.pt
27f58754184ad5204f77c48cd2dc4d97a4360fe59cb29b8cbb22a7a5fa02393a  B_refined_features.pt
d5c743b20a00294b6897f9b9938bd8986733d36e5a828db7568d237d93d5c341  GT_features.pt
270be4ed70805b22b6995d3af590539b634c1267a9b7a9a048b10772e5e3c5b3  A_raw_blend075_DIAGNOSTIC_features.pt
63de90c4bdaede796723470fba4f12d3053d4aaca58880813208fa5c51ff4946  B_refined_blend075_DIAGNOSTIC_features.pt
effc8a1c9ea0a2e88e61b9046690ba82bd567807b14faef6b8329e8959565f9a  feature_manifest.json
```

## 5. 性能记录

| 项 | 值 |
|---|---|
| 总耗时 | **90.6 s**（含加载 ~25s） |
| 单组 1000 张 | **7.8–8.1 s** |
| GPU 峰值 `max_memory_allocated` | **7.947 GB** |
| GPU 峰值 `max_memory_reserved` | **8.311 GB** |
| 显存/时间 | 远低于估算（原估 3–5 分钟） |
| 磁盘变化 | 108G → **112G 已用 / 28G 可用**（+4.26GB） |

## 6. GPU 释放检查

| 检查 | 结果 |
|---|---|
| GPU 推理进程 | ✅ 无 |
| `nvidia-smi` 占用 | ✅ **10 MiB / 40960 MiB，0%** |
| tmux 残留 | ✅ `no server running` |
| 后台写入 | ✅ 无 |
| 新文件 SHA256 | ✅ 已生成并校验 |
| HF 缓存 | ✅ 未清理 |
| checkpoint | ✅ 未删除 |
| 中间文件 | ✅ 未删除 |
| 云平台退租/解绑 | ✅ **未执行**（由用户操作） |

**可以安全停止 GPU 实例。**

## 7. CPU 迁移时需要执行的步骤

1. **迁移数据盘**（整盘快照/挂载，勿格式化），确认挂载点仍为 `/mnt/data`；
2. `blkid /dev/vdX` 核对 UUID = `5a9d9737-e795-4d2a-9c93-3d7b443f9054`；必要时运行 `scripts/setup_data_disk_automount.sh`；
3. 校验正式产物：`sha256sum -c reports/FULL_1000_SHA256.sha256` 与 `reports/EVALUATION_SHA256.sha256`（19 项）；
4. **校验新特征**：`cd evaluation/precomputed_features/bigg && sha256sum -c SHA256SUMS`（6 项）；
5. 断链扫描：`find hf_home assets run -xtype l | wc -l` 应为 0；
6. 重建 CPU 环境（Python 3.12；CPU 版 torch 即可），按 `reports/CPU_EVALUATION_EXECUTION_PLAN.md` 的 P0→P4 推进；
7. **CPU 侧不再需要 bigG 前向**——直接用 `evaluation/precomputed_features/bigg/*.pt`，使用前做 L2 归一化。

> 迁移后 `run/cache`、`run/data` 下 4 条绝对软链需保持 `/mnt/data` 路径一致，否则重建即可（不影响正式产物）。
