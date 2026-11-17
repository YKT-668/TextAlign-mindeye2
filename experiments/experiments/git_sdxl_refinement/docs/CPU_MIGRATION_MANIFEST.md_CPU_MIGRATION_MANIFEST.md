# CPU 迁移清单（迁移前审计）

生成时间：2026-11-15 ｜ 机器：A800-SXM4-40GB（GPU 已空闲）

## 1. 结论摘要

| 项 | 状态 |
|---|---|
| GPU 进程 | ✅ 无（仅 10 MiB 空闲占用） |
| tmux 残留会话 | ✅ 无（`no server running`） |
| 写入正式产物的后台进程 | ✅ 无 |
| 数据盘挂载 / UUID / 文件系统 | ✅ `/mnt/data` ← `/dev/vdb`，ext4，`5a9d9737-e795-4d2a-9c93-3d7b443f9054` |
| 数据盘独立于系统盘 | ✅ 系统盘 `/dev/vda3`，数据盘 `/dev/vdb` |
| 正式产物完整性（19 项 SHA256） | ✅ **全部 OK** |
| HF 缓存软/硬链接 | ✅ 断链 **0**，绝对链接 **0** |
| 容量 | 147G 总 / **108G 已用 / 32G 可用** |
| **是否可安全结束 GPU 租赁** | ✅ **可以**（云平台操作由用户执行） |

## 2. 磁盘与挂载

| 项 | 值 |
|---|---|
| 挂载点 | `/mnt/data` |
| 设备 | `/dev/vdb` |
| UUID | `5a9d9737-e795-4d2a-9c93-3d7b443f9054` |
| 文件系统 | ext4 |
| 挂载选项 | `rw,noatime` |
| 容量 / 可用 | 147G / 32G |
| fstab 条目 | `UUID=5a9d9737-... /mnt/data ext4 defaults,nofail,noatime,x-systemd.device-timeout=30s 0 2` |
| 兜底服务 | `mnt-data-ensure.service`（40×15s 重试） |

## 3. 重要目录大小与文件数

| 目录 | 大小 | 文件数 |
|---|---|---|
| `run/full_stage1/`（A 组七类输出 + 2000 PNG） | 3.1 G | 2020 |
| `run/full_stage2/`（B 组输出 + 预览） | 752 M | 14 |
| `run/out_stage1_a/`（样本测试） | 6.6 M | 15 |
| `run/out_stage1_b/`（20 张稳定性） | 65 M | 69 |
| `run/out_stage2/`（2 张 img2img） | 2.1 M | 7 |
| `evaluation/`（manifest 图像 / 盲测 / 图表 / CSV） | 290 M | 3460 |
| `reports/` | 1.6 M | 54 |
| `scripts/` | 308 K | 38 |
| `logs/` | 1.4 M | 38 |
| `assets/`（模型与数据，与 HF 缓存**硬链接共享**） | 51 G | 8 |
| `hf_home/`（HF 缓存 + xet） | 70 G | 253 |
| `run/full_stage1_ckpt/`（最后快照） | 2.9 G | 1 |
| `run/full_stage2_ckpt/`（最后快照） | 5.8 G | 1 |
| **项目总占用（去重后）** | **86 G** | — |

> 注意：`assets/ 51G + hf_home/ 70G` 之和大于项目总计，因为大权重通过**硬链接**共享 inode（S1 26.8G、unclip 18G、Zavy 6.9G 各只占一份）。

## 4. 正式结果 SHA256 校验（实测）

`reports/FULL_1000_SHA256.sha256`（10 项，全部 **OK**）：

```
OK  run/full_stage1/..._all_backbones.pt
OK  run/full_stage1/..._all_blurryrecons.pt
OK  run/full_stage1/..._all_clipvoxels.pt
OK  run/full_stage1/..._all_enhancedrecons.pt
OK  run/full_stage1/..._all_ids.pt
OK  run/full_stage1/..._all_predcaptions.pt
OK  run/full_stage1/..._all_prior_out.pt
OK  run/full_stage2/..._stage2_ids.pt
OK  run/full_stage2/..._stage2_img2img.pt
OK  run/full_stage1/..._captions.txt
```

`reports/EVALUATION_SHA256.sha256`（9 项，全部 **OK**）：

```
OK  reports/REFINEMENT_EVALUATION_MANIFEST.csv
OK  evaluation/per_sample_metrics.csv
OK  evaluation/eval_summary.json
OK  evaluation/fig_delta_hist_image.png
OK  evaluation/fig_scatter_raw_vs_refined.png
OK  evaluation/fig_retrieval_cdf.png
OK  evaluation/human_blind_review/review_sheet.csv
OK  evaluation/human_blind_review/ANSWER_KEY_DO_NOT_OPEN_BEFORE_RATING.csv
OK  scripts/eval_git_refinement.py
```

**共 19/19 通过，0 失败。**

## 5. HF 缓存与链接完整性

| 检查 | 结果 |
|---|---|
| `hf_home/hub` 断链 | **0** |
| `hf_home/hub` 绝对软链 | **0** |
| `assets/` 断链 | **0** |
| `run/` 断链 | **0** |
| `run/` 绝对软链 | **4**（见下，均在**同一数据盘内**，挂载点不变则安全） |

4 个绝对软链（`run/` 内的运行期视图）：
```
run/cache/unclip6_epoch0_step110000.ckpt        -> $ROOT/assets/models/mindeye2/unclip6_epoch0_step110000.ckpt
run/cache/bigG_to_L_epoch8.pth                  -> $ROOT/assets/models/mindeye2/bigG_to_L_epoch8.pth
run/data/betas_all_subj01_fp32_renorm.hdf5      -> $ROOT/assets/data/betas_all_subj01_fp32_renorm.hdf5
run/data/wds/subj01/new_test/0.tar              -> $ROOT/assets/data/wds_subj01_new_test_0.tar
```
> 因为这 4 条是**绝对路径**，新实例必须把数据盘挂在 **同一路径 `/mnt/data`**，否则会断链（可直接重建，代价很小）。

## 6. 后台进程状态

| 检查 | 结果 |
|---|---|
| `nvidia-smi` | `10 MiB, 0%`（无 CUDA 上下文） |
| tmux | `no server running` |
| python 实验进程 | 无（仅系统 `networkd-dispatcher`、`unattended-upgrade` 与 vscode-server） |
| 写入 `run/`、`evaluation/`、`reports/` 的进程 | 无 |

## 7. 迁移风险与应对

| # | 风险 | 等级 | 应对 |
|---|---|---|---|
| 1 | 数据盘被误格式化/初始化 | **高** | 只挂载、不 mkfs；用 UUID 挂载；`setup_data_disk_automount.sh` 仅做只读探测 |
| 2 | 挂载点变化导致 4 条绝对软链失效 | 中 | 固定挂到 `/mnt/data`；失效则重建软链（`prep_run.py` 可重建） |
| 3 | 用 `rsync`/`cp` 复制而非整盘迁移 → 硬链接被展开 | 中 | **优先整盘迁移/快照**；若必须复制，用 `rsync -aH` |
| 4 | 迁移中 `hf_home` 与 `assets` 只复制其一 → 断链 | 中 | 整盘迁移；或两者一起复制 |
| 5 | 32G 剩余空间在 CPU 阶段不足 | 低-中 | 可回收 `run/full_stage1_ckpt`(2.9G)+`run/full_stage2_ckpt`(5.8G)=**8.7G**（**需用户批准**，本轮未删） |
| 6 | 迁移后未校验就继续评测 | 中 | 先跑 `sha256sum -c` 19 项 + 断链检查（见执行计划 P0） |

## 8. 迁移前建议（可选，需用户批准）

1. **（强烈建议）在释放 GPU 前缓存 bigG 图像嵌入**：A/B/GT 各 1000 张的 bigG token 嵌入（`1000×256×1664` fp16 ≈ 0.85GB/组，共约 2.5GB）目前**只存在于内存中、未落盘**。在 GPU 上约 3–5 分钟即可导出；若不留，CPU 上重算 bigG 需 4–7 小时（8 vCPU）。
2. 可回收 8.7G checkpoint（非必须）。
3. 若计划做官方 N-way（1000-way）Identification 与 .75/.25 融合版本，同样建议在 GPU 上一并导出特征。

## 9. 允许/禁止操作（本轮）

- 本轮**未**执行：退租、解绑、格式化、删除、迁移、安装依赖、下载模型、运行评测。
- 云平台实例退租与数据盘解绑由**用户**在控制台执行。

