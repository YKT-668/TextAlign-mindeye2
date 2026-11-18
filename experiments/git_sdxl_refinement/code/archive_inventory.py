"""归档资产盘点：生成最终 manifest 的候选清单（路径/大小/SHA256/类别/去向/理由）。

去向规则：
  github  : 代码、文档、环境锁定、小型结果摘要（<2MB，无大 tensor）
  hf      : 大型正式输出(.pt)、预计算特征、逐样本与日志（私有 Dataset）
  exclude : 缓存/中间产物/重复 PNG/未完成材料/环境本身（记录理由）
只读扫描，不修改任何文件。
"""
import csv
import hashlib
import json
import os

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
OUT_JSON = f"{ROOT}/reports/_archive_inventory_working.json"
M = "s1_textalign_stage1_FINAL_BEST_32"
GIT_MAX = 2 * 1024 * 1024  # github 单文件上限 2MB


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


SELECTION = []
for f in ["all_backbones.pt", "all_blurryrecons.pt", "all_clipvoxels.pt",
          "all_enhancedrecons.pt", "all_ids.pt", "all_predcaptions.pt",
          "all_prior_out.pt", "captions.txt", "verify_stage1_1000.json"]:
    SELECTION.append((f"run/full_stage1/{M}_{f}", "inference_output_stage1", "hf",
                      "第一阶段正式输出(七类+captions+verify)"))
for f in ["stage2_ids.pt", "stage2_img2img.pt", "verify_stage2_1000.json"]:
    SELECTION.append((f"run/full_stage2/{M}_{f}", "inference_output_stage2", "hf", "第二阶段正式输出"))
SELECTION += [
    ("evaluation/per_sample_metrics.csv", "per_sample", "hf+github", "bigG 逐样本评价"),
    ("evaluation/eval_summary.json", "eval_summary", "github+hf", "bigG 评价汇总"),
]
for f in ["metrics_summary.csv", "p1_accuracy_summary.csv", "per_sample_metrics.csv",
          "stats.json", "evaluation_config.json", "model_manifest.json",
          "evaluation_log.txt", "SHA256SUMS", "lowlevel_summary.json",
          "p1_alexnet_summary.json", "p1_effnet_summary.json", "p1_inception_summary.json"]:
    SELECTION.append((f"evaluation/cpu_mindeye2_p1/{f}", "eval_p1", "github+hf", "P1 官方评价结果(小型)"))
SELECTION.append(("evaluation/cpu_mindeye2_p1/lowlevel_per_sample.csv", "per_sample", "hf", "PixCorr/SSIM 逐样本"))
for m in ("alexnet", "inception", "effnet"):
    SELECTION.append((f"evaluation/cpu_mindeye2_p1/p1_{m}_per_sample.csv", "per_sample", "hf", f"P1 {m} 逐样本"))
SELECTION += [(f"evaluation/cpu_mindeye2_p1_stat_audit/{f}", "eval_stats", "github+hf", "P1 统计审计")
              for f in ["nway_audit.json", "pool_level_audit.json", "pool_overlap_audit.json",
                        "p1_corrected_pvalues.csv", "SHA256SUMS"]]
SELECTION += [(f"evaluation/cpu_mindeye2_p2_caption/{f}", "eval_p2", "github+hf", "P2 caption 评价")
              for f in ["caption_summary.json", "caption_per_sample.csv", "model_manifest.json", "SHA256SUMS"]]
SELECTION += [("evaluation/cpu_mindeye2_p2_caption_extension/caption_extension_summary.json",
               "eval_p2", "github+hf", "P2 caption 三项式补全")]
for m in ("clip_l", "swav"):
    for f in ["equivalence_check.json", "equivalence_check.log", "extract_clipl_full.log",
              "extract_swav_full.log", "sanity_gap_decomposition.json"]:
        p = f"evaluation/cpu_mindeye2_p3_clip_swav/{m}/{f}"
        if os.path.exists(f"{ROOT}/{p}"):
            SELECTION.append((p, "eval_p3", "github+hf", "P3 证据/日志"))
    for g in ("A_raw", "B_refined", "GT"):
        SELECTION.append((f"evaluation/cpu_mindeye2_p3_clip_swav/{m}/{g}_features.npy", "features_p3", "hf",
                          "P3 正式特征 fp32"))
SELECTION += [(f"evaluation/cpu_mindeye2_p3_clip_swav/{f}", "eval_p3", "github+hf", "P3 配置/清单")
              for f in ["evaluation_config.json", "model_manifest.json", "SHA256SUMS"]]
SELECTION += [(f"evaluation/cpu_mindeye2_p3_clip_swav/statistics/{f}", "eval_stats", "github+hf", "P3 统计")
              for f in ["p3_metrics_summary.json", "p3_per_sample.csv"]]
for f in os.listdir(f"{ROOT}/evaluation/precomputed_features/bigg"):
    p = f"evaluation/precomputed_features/bigg/{f}"
    if f.endswith((".pt",)):
        SELECTION.append((p, "precomputed_features", "hf", "bigG 预计算特征(GPU 导出)"))
    else:
        SELECTION.append((p, "eval_bigg_meta", "github+hf", "bigG 特征清单/校验"))
for f in sorted(os.listdir(f"{ROOT}/reports")):
    if f.startswith("_") or f == "FINAL_ARCHIVE_MANIFEST.csv":
        continue  # 清单自身不纳入（避免循环哈希）
    p = f"reports/{f}"
    sz = os.path.getsize(f"{ROOT}/{p}")
    if f in ("REFINEMENT_EVALUATION_MANIFEST.csv", "S1_NEW_TEST_TO_SHARED982_MAPPING.csv",
             "FULL_1000_SHA256.sha256", "EVALUATION_SHA256.sha256", "GPU_ENV_LOCK.txt",
             "SWAV_WEIGHT_PROVENANCE.txt", "MODEL_WEIGHT_AVAILABILITY.csv",
             "REFINEMENT_EVALUATION_MATRIX.csv"):
        SELECTION.append((p, "report_meta", "github+hf", "评测映射/清单/环境锁定"))
    elif sz <= GIT_MAX:
        SELECTION.append((p, "report", "github+hf", "报告/审计"))
    else:
        SELECTION.append((p, "report", "hf", f"报告大文件 {sz}B"))
for f in sorted(os.listdir(f"{ROOT}/scripts")):
    if f.endswith((".py", ".sh")):
        SELECTION.append((f"scripts/{f}", "code", "github+hf", "评测/审计/归档脚本"))
for f in ["env.sh", "GPU_ENV_LOCK.txt", "install_gpu_env.sh", "GPU_ENV_PREFLIGHT.sh"]:
    if os.path.exists(f"{ROOT}/{f}"):
        SELECTION.append((f, "env", "github+hf", "环境复现文件"))

EXCLUSIONS = [
    ("run/full_stage1/images/ blurry_images/ preview/", "重建 PNG 可由已归档 tensor+导出脚本恢复，无唯一信息"),
    ("run/full_stage2 无 PNG 目录", "仅 pt/json，已全选"),
    ("evaluation/human_blind_review/", "未完成的盲测材料（无人评分），PNG 可由 tensor 恢复；任务明确排除"),
    ("evaluation/cpu_mindeye2_p1/features/", "P1 特征 memmap 缓存(~2.7GB)，可由官方脚本重建；正式结果已全归档"),
    ("evaluation/images/", "bigG 评测 PNG 导出，可恢复"),
    ("hf_home/", "HF 缓存本身不上传"),
    ("download_venv/ cpu_pkgs/ cpu_p1_venv/ gpu_venv/", "Python 环境，由 requirements/README 重建"),
    ("tmp/", "临时目录"),
    ("source/", "官方 MindEye2 源码，GitHub 公开仓库已有(archive/final14@231850a)"),
    ("assets/", "输入数据(all_images/captions/git captions)已存在于 HF Dataset/官方 revision，引用不重复上传"),
    ("run/cache run/data run/out_stage* run/full_stage*_ckpt", "推理缓存/中间 ckpt，非正式产物"),
    ("betas/wds 等受限数据", "NSD 原始 beta/刺激数据不上传（规则6）"),
]
rows = []
for rel, cat, dest, note in SELECTION:
    p = f"{ROOT}/{rel}"
    if not os.path.isfile(p):
        continue
    sz = os.path.getsize(p)
    d = sha256(p)
    final_dest = dest
    if sz > GIT_MAX:
        final_dest = "hf"
    rows.append({"relative_path": rel, "file_size_bytes": sz, "sha256": d,
                 "category": cat, "archive_destination": final_dest, "notes": note})

with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump({"files": rows, "exclusions": EXCLUSIONS}, fh, indent=1, ensure_ascii=False)
with open(f"{ROOT}/reports/_archive_inventory_working.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["relative_path", "file_size_bytes", "sha256",
                                       "category", "archive_destination", "notes"])
    w.writeheader()
    w.writerows(rows)

by = {}
for r in rows:
    by.setdefault(r["archive_destination"], [0, 0])
    by[r["archive_destination"]][0] += 1
    by[r["archive_destination"]][1] += r["file_size_bytes"]
for k, (n, b) in sorted(by.items()):
    print(f"{k}: {n} files, {b/1e6:.1f} MB")
print("INVENTORY_DONE", len(rows))
