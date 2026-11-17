"""GitHub 归档暂存：把精选文件复制进 repo 的 experiments/git_sdxl_refinement/ 并生成 README 与 SHA256SUMS。

只复制明确审核过的文件（来自 inventory 的 github 目标），禁止 git add .（由调用方显式 add）。
"""
import hashlib
import json
import os
import shutil

ROOT = "/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114"
REPO = "/root/archive_work/TextAlign-mindeye2"
STAGE = f"{REPO}/experiments/git_sdxl_refinement"
INV = json.load(open(f"{ROOT}/reports/_archive_inventory_working.json"))

os.makedirs(f"{STAGE}/docs", exist_ok=True)
os.makedirs(f"{STAGE}/code", exist_ok=True)
os.makedirs(f"{STAGE}/results_summary", exist_ok=True)

# 子目录归类
DOCS_HINTS = ("reports/",)
CODE_HINTS = ("scripts/", "env.sh", "GPU_ENV_LOCK.txt", "install_gpu_env.sh", "GPU_ENV_PREFLIGHT.sh")
SUMMARY_HINTS = ("summary", "audit", "config", "manifest", "verify", "equivalence",
                 "SHA256SUMS", "corrected", "matrix", "dictionary", "pvalues",
                 "accuracy_summary", "stats.json", "eval_summary", "provenance")

copied = []
for f in INV["files"]:
    if "github" not in f["archive_destination"]:
        continue
    rel = f["relative_path"].replace(os.sep, "/")
    src = os.path.join(ROOT, rel)
    if rel.startswith("reports/"):
        dst_dir = f"{STAGE}/docs"
        dst = f"{dst_dir}/{os.path.basename(rel)}"
    elif any(rel.startswith(h) for h in CODE_HINTS) or rel == "env.sh":
        dst_dir = f"{STAGE}/code"
        dst = f"{dst_dir}/{os.path.basename(rel)}"
    else:
        dst_dir = f"{STAGE}/results_summary"
        dst = f"{dst_dir}/{os.path.basename(rel)}"
    os.makedirs(dst_dir, exist_ok=True)
    # 同名冲突处理：保留前缀目录信息
    if os.path.exists(dst):
        stem, ext = os.path.splitext(os.path.basename(rel))
        pfx = rel.split("/")[1] if "/" in rel else "root"
        dst = f"{dst_dir}/{pfx}_{stem}{ext}"
    shutil.copy2(src, dst)
    copied.append({"repo_path": os.path.relpath(dst, REPO), "sha256": hashlib.sha256(open(dst, 'rb').read()).hexdigest()})

# README
readme = """# GIT-Caption-Conditioned SDXL Refinement 补充实验（最终归档）

归档日期：2026-11-17 ｜ 分支：`archive/git-sdxl-refinement-final-2026-11-17`

## 实验目的

在 ConceptAlign 两阶段脑解码流程（1000 张 NSD 测试图）上，检验额外的
GIT caption 引导 SDXL img2img refinement 是否值得：B 组 = A 组（第一阶段 unCLIP 重建）
之上叠加 GIT 预测 caption 引导的 SDXL img2img（timepoint=13, scale=5, steps=25, seed=42）。

## 主结果（1000 张，CLIP ViT-L/14 官方 2-way Identification）

| Method | Caption-conditioned refinement | Inference time | CLIP-L 2-way |
|---|---|---:|---:|
| ConceptAlign (A) | No | 90.2 min | **93.19%** |
| ConceptAlign + GIT-SDXL (B) | Yes | ~121.3 min | **91.65%** |

Δ(B−A) = **−1.54 pp**，95% CI [−0.0239, −0.0072]（Bootstrap 10k），Holm 校正后显著。
982 敏感性口径方向一致（−0.0150）。SwAV distance（官方）同样显著变差（0.3505→0.3633）。

**注意**：A 组并非"无 GIT/Caption"——A 不含的是额外的第二阶段 caption-conditioned
refinement。本实验比较的是"增加整个 SDXL refinement 流程"的效果，下降不能完全归因于
caption 文本本身。

## 目录结构

- `docs/` — 全部正式报告（实验记录、协议审计、统计审计、指标字典、命名审查、复现指南）
- `code/` — 评测/统计/归档脚本与环境锁定文件
- `results_summary/` — 小型结果汇总（JSON/CSV；逐样本与大文件在 HF 私有 Dataset）

## 大文件位置（HF 私有 Dataset）

`ykt668/textalign-mindeye2-data` → `supplementary/git_sdxl_refinement/2026-11-17/`
（stage1 七类输出、stage2 输出、bigG 特征、P3 特征、逐样本评价、日志、SHA256SUMS）。
NSD 原始数据受协议约束不在归档内，需经官方渠道获取。

## 复现

见 `docs/FINAL_REPRODUCIBILITY_GUIDE.md` 与 `docs/FINAL_GIT_SDXL_REFINEMENT_EXPERIMENT.md`。
"""
with open(f"{STAGE}/README.md", "w", encoding="utf-8") as fh:
    fh.write(readme)

# SHA256SUMS
with open(f"{STAGE}/SHA256SUMS", "w") as fh:
    for c in sorted(copied, key=lambda x: x["repo_path"]):
        fh.write(f"{c['sha256']}  {os.path.relpath(c['repo_path'], STAGE)}\n")
    # README 与 SHA256SUMS 自身不计入（README 内容含本文件无哈希需求）
print(f"staged {len(copied)} files")
print("STAGE_DONE")
