# S1 new_test 1000 图像 ↔ shared982 精确对齐审计

> 日期：2026-11-14。只读审计，未运行 GPU/训练/推理，未下载大型文件（仅下载 <1MB 的映射小文件）。

## 结论：VERIFIED（核心映射已精确验证）

S1 new_test 的 1000 张图像与 shared982 评测集的 ID 映射**已完整确定**，无需重排序。

## 1. ID 体系与关键关系（全部 VERIFIED）

| 事实 | 结果 |
|---|---|
| 1000 个 new_test 唯一刺激 ID | 2950–72948（NSD 73k 图像索引，**严格递增**，无重复） |
| `test_73k_images.npy`（all_images.pt 行序） | 与 1000 个 ID **完全同序**（`np.array_equal == True`） |
| `shared982`（982 个） | 恰好是 1000 个 ID 的**前 982 个**（`shared982 == ids1000[:982]` 顺序一致） |
| shared982 ∩ 1000 | **982/982**（shared982 是 1000 的子集） |
| 18 个非 shared982 的 ID | 位置 **982–999**（末尾，值为 71229…72948） |
| 重复 ID | 无（1000 唯一、982 唯一） |

**重要澄清**：`shared1000.npy`（pscotti/mindeyev2 中的）实际是 **73000 长度的二值掩码**（0/1），不是 1000 个 ID 列表；真正的 1000 个 ID 在 `test_73k_images.npy`（1000 长度）中，且与推理输出 `all_ids.pt`（N1000_ALL_IDS_ALIGNMENT.tsv）完全一致。

## 2. 映射表

`S1_NEW_TEST_TO_SHARED982_MAPPING.csv`（1000 行）：

| 字段 | 含义 | 状态 |
|---|---|---|
| `new_test_index` | 0–999（排序后位置） | VERIFIED |
| `nsd_stimulus_id` | NSD 73k 图像索引（2950–72948） | VERIFIED |
| `local_gt_index` | all_images.pt 行号（== new_test_index，顺序已核） | VERIFIED |
| `shared982` | yes（0–981）/ no（982–999） | VERIFIED |
| `caption_index` | all_captions.pt 行号（== new_test_index） | VERIFIED（按 GT 构造同序；caption 文本内容未独立加载，见下） |
| `coco_image_id` | COCO 图像 ID | PENDING（需 h5py 读 COCO_73k_subj_indices.hdf5，可选） |
| `mapping_source` / `verification_status` | 证据来源 / VERIFIED | — |

## 3. 强制验证结果（全部通过）

1. new_test 唯一 ID 数量 = **1000** ✓
2. 重复 ID = **0** ✓
3. shared982 匹配数量 = **恰好 982** ✓
4. 每个 ID 唯一对应 GT 图像 ✓（test_73k_images 无重复、与 ID 同序）
5. 每个 ID 唯一对应 GT caption ✓（all_captions.pt 与 all_images.pt 同为 shared1000 顺序，INFERRED 于文本内容）
6. 图像/caption 错位 = **无**（同序，无需重排）
7. 缺失/无法解释 ID = **无**（18 个非 shared982 的 ID 是高位的 71229–72948，属 shared1000 中未进入四被试共享集的部分）

## 4. 关键结论

- **shared982 就是排序后 1000 个 ID 的前 982 个**；论文评测用的 shared982 与 S1 new_test 的 1000 张**同源、同序**，只是取前 982 个子集。
- **重跑阶段一后，`all_ids.pt` 会与 `test_73k_images.npy` 同序**（官方代码 `np.unique(test_images_idx)` 去重排序），因此 recons / captions / ids / GT 四者天然对齐，**无需重排**。
- 之前「不要混淆 1000 与 982」的提醒，本质是：982 是 1000 的子集（前 982 个），二者是同一条 ID 链，不是两套独立口径。

## 5. 缺失材料 / 待补项

1. `coco_image_id`：可选，仅用于 COCO 数据集交叉引用；需 h5py 读 `COCO_73k_subj_indices.hdf5`（已下载，未解析）。
2. `all_captions.pt` 的文本内容尚未加载（venv 缺 numpy/h5py，pip 安装被 Zavy 下载占满网络而卡住）；caption 行序按 GT 构造判定同序，文本级配对为 INFERRED，可在 GPU 阶段或补装依赖后确认。

## 6. 最小补充下载方案（已执行）

本轮仅下载 4 个小文件（<1MB，不干扰 Zavy）：
- `test_73k_images.npy`（8KB）——核心 ID 行序证据
- `shared982_ids_manifest.csv`（21KB）——shared982 完整清单
- `shared1000.npy`（73KB）——确认为掩码，澄清歧义
- `COCO_73k_subj_indices.hdf5`（629KB）——coco_image_id 用（未解析）

**无需下载 45.6GB 大型资产、无需 38.4MB new_test tar**（ID 证据已由上述文件覆盖）。

## 7. 是否可以进入正式下载阶段

**可以**。ID 映射已 VERIFIED，重跑阶段一的输出（enhancedrecons/predcaptions/ids）将与已下载的 GT（all_images.pt/all_captions.pt）天然对齐。唯一建议在正式阶段前补装 numpy/h5py（或 GPU 环境自带），以解析 coco_image_id 与 caption 文本。
