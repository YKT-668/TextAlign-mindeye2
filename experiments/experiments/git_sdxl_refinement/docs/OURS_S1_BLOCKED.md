# Ours S1 资产下载阻塞记录

**日期**: 2026-11-14
**状态**: BLOCKED（等待用户提供正确位置）

## 目标文件（来自 provenance）
正式 Ours S1（subject 1，`ours_llmhard` 变体）三个输出文件：

| 文件 | SHA256 |
|---|---|
| `step6d_full_ours_llmhard_all_enhancedrecons.pt` | `df704fca629453c79f80b83a859694e468f1aad994257d31314e86dc7f8a85b2` |
| `step6d_full_ours_llmhard_all_predcaptions.pt` | `a6b49190a20f79ce990bc647d7cca032e9503383877c3728bec647ef8c069758` |
| `step6d_full_ours_llmhard_all_ids.pt` | `0a837980a67886e2dfc5e0f3b08facdeab7a0d0b7d9ae6309f1f604ca7812d3c` |

来源引用：
- `final14_archive/followup_experiments/fig4/PHASE_C/MODEL_PROVENANCE_REPORT.md`
- `final14_archive/followup_experiments/recovery_metadata/FINAL_OUTPUT_CHECKSUMS.sha256`
  （checksum 中路径指向原 A100 服务器本地磁盘
  `/mnt/aaai_ablation_disk/aaai_readme3_full_reconstruction_maxthroughput_20260710_150646/ours_llmhard/`）

## 检索结果（文件名 + SHA256 双重确认）
- `ykt668/textalign-mindeye2-data`（固定 revision `2c612c5c…`，即 main，无其它分支/tag）：**不存在**
- `ykt668/textalign-mindeye2-model`（公开）：仅 s2/s5/s7 三个被试的 `all_*` 文件，**无 subject-1/ours_llmhard**，SHA256 全部对不上
- `ykt668/google-cv-internship-private-archive`：仅 2 个无关文件

结论：这三个文件从未被上传到 HF（只存在于原 A100 服务器本地磁盘）。

## 处理方式
用户选择：先跳过 Ours S1，立即下载公开资产（GT、Zavy、SDXL base+refiner），Ours S1 稍后补。
