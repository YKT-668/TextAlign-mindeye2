# FINAL：归档完成与数据盘释放报告

日期：2026-11-17/18 ｜ 范围：GIT-caption-conditioned SDXL refinement 补充实验的最终归档
原则：不改历史、不 force push、不覆盖远程既有内容、不上传受限数据与凭证、验证通过前不释放数据盘

---

## 1. GitHub 归档

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/YKT-668/TextAlign-mindeye2 |
| 归档分支 | `archive/git-sdxl-refinement-final-2026-11-17` |
| 分支顶端 commit | 归档完成后请以 `git ls-remote` 查询（定稿时的提交链见下行；文档定稿提交 = **ab8bfc9**） |
| Annotated tag | `git-sdxl-refinement-final-2026-11-17` = 056f7f6 → 剥离 **d08791d**（创建时点，未移动） |
| 基线核对 | origin/archive/final14 = `231850a...` ✓；origin/master = `80bb079...` ✓（均未改） |
| 默认 main | 未修改 ✓ |
| 提交链 | 579ab6e(main) → d08791d(tag) → eb091f8 → fa601e4 → b53eb96 → ca39ac7 → 5f7dce6 → 0551fa2 → **ab8bfc9（含本报告正文）** |
| 内容目录 | `experiments/git_sdxl_refinement/{docs,code,results_summary}/` + README + SHA256SUMS |
| 文件数 | **158**（155 归档文件 + README + SHA256SUMS + FINAL_ARCHIVE_MANIFEST.csv），约 3.4MB |
| 远端验证 | **PASS 158/158**（git/trees API 递归树逐文件 blob-sha1+size 比对；文档/清单抽样字节级一致） |
| 无大二进制 | ✓（最大文件 < 2MB；无 .pt/.pth/.ckpt） |
| 无敏感信息 | ✓（无 token/密钥/.env；已 grep 检查） |

中间修复合集（均已记录、无 force push）：① 初始提交路径双嵌套 → 拍平（eb091f8→9d1a812→fa601e4 链）；② repo `.gitignore` 的 `*.log` 拦截 3 个日志 → `git add -f`；③ SHA256SUMS 相对路径受 CWD 污染 → 重新生成；④ 补入归档工具脚本与最终文档（ca39ac7、5f7dce6）。

## 2. Hugging Face Dataset

| 项 | 值 |
|---|---|
| 仓库 | `ykt668/textalign-mindeye2-data`（**Private** ✓） |
| 历史基线 | `2c612c5c2ca3c5344854edbd7028769cb254ab5a`（未被回退/覆盖；此后仅追加） |
| 归档目录 | `supplementary/git_sdxl_refinement/2026-11-17/` |
| 当前 revision | `7259b4845f21a10535d5082c174514a4d2f611e8`（追加式提交，历史 revision 均保留） |
| 文件数 | **186**（180 个归档文件 + FINAL_ARCHIVE_MANIFEST.csv + 5 个旧映射冗余副本；其中 LFS 大文件 20、普通小文件 166） |
| 容量 | 约 8.24 GB（stage1 七类 + stage2 + bigG 特征 5 个 + P3 特征 6 个） |
| 对齐检查 | **PASS**（180 个必要文件：存在性 + 字节尺寸 + 20 个 LFS 的 sha256 逐一匹配 + 无重名冲突） |
| 唯一性 | stage1/stage2 正式输出、ids、captions、P3 特征均为**唯一长期资产** |
| 未上传 | NSD 原始 beta/刺激数据（受限）；PNG 图集（可由 tensor 重建）；HF 缓存；venv；tmp |

已知冗余（保留未删，不影响正确性）：`evaluation/per_sample/` 下 5 个旧映射残留副本（caption/lowlevel/p1_* per_sample.csv），已在新映射的 `evaluation/cpu_mindeye2_p1|p2_caption/` 等目录下有正确版本。

## 3. HF Model checkpoint（既有，不重传）

| 项 | 值 |
|---|---|
| 仓库 | `ykt668/textalign-mindeye2-model`（Public） |
| revision | **a127f9295fd5656ac63ae436f07e61b80bf4efce**（masterGit，实测存在） |
| 文件 | `checkpoints/s1_textalign_stage1_FINAL_BEST_32/last.pth`（26,788,144,501 B） |
| LFS SHA256 | **eb59c4262c5f6316285d…** 与历史记录一致 ✓ |
| 结论 | 无需重复上传；归档时按固定 revision 引用 |

## 4. 独立回下载验证（不依赖本地 HF cache）

| 对象 | 方法 | 结果 |
|---|---|---|
| GitHub 分支 158 文件 | 全新 API 树逐文件 blob-sha1+size + 文档抽样字节级 | **PASS 158/158** |
| HF 小文件 160 个 | 独立 local_dir 逐文件 HTTP 下载 + SHA256 | **PASS 160/160** |
| HF 大文件 17 个 | 逐个从远程完整下载 + SHA256（7.7GB） | **PASS 17/17** |
| HF LFS 内容哈希 | 服务端 LFS oid(=sha256) vs 本地 | **PASS 20/20** |
| 原始推理产物 10 项 | `FULL_1000_SHA256.sha256` 本地复核 | **PASS 10/10** |
| S1 checkpoint | 固定 revision 元数据 + LFS sha256 | **PASS** |
| FINAL_ARCHIVE_MANIFEST.csv | 180 行逐文件记录（状态全为 download_ok_sha256） | ✅ |

> 网络注记：验证期间 HF CDN 多次超时（一度降至 34KiB/s），已通过 aria2c 断点续传 + 重试完成全部下载；**最终所有 >2MB 文件均为完整下载 + 本地 SHA256 比对，未以元数据替代作为最终验证**。

## 5. 正式产物完整性

- stage1 七类输出 + captions + verify JSON：GitHub（小文件）与 HF（全量）双备份
- stage2 输出 + ids + verify：双备份
- 全部评价结果（P1/P2/P3、统计、逐样本）：GitHub 摘要 + HF 全量
- 全部报告（含 14 节正式实验记录、指标字典、命名审计、复现指南）：GitHub docs + HF reports

## 6. 未上传文件与原因（摘要）

| 内容 | 原因 |
|---|---|
| NSD 原始 beta/刺激/73k 图像 | 受限数据，不得进公开/归档仓库（规则 6） |
| `run/full_stage1/images/**`、`blurry_images/`、`evaluation/images/` | PNG 集可由已归档 tensor + 导出脚本逐位重建，无唯一信息 |
| `evaluation/human_blind_review/` | 未完成的人工盲测材料（无评分），按任务要求排除 |
| `evaluation/cpu_mindeye2_p1/features/`（~2.7GB memmap） | 中间缓存，可由官方脚本+已归档权重重算 |
| `hf_home/`、venv、`tmp/`、`source/` | 缓存/环境/临时/公开源码，无需归档 |

## 7. 不可恢复风险

- **无已知阻断性风险**：正式产物、评价结果、报告、代码均在两处可验证存档。
- 残余小风险：`evaluation/per_sample/` 下 5 个冗余旧路径副本（不影响数据正确性）；GitHub tag 指向的 d08791d 内部包含历史中间态路径结构（分支顶端已修正，tag 按"不改写已推送引用"原则不动，内容完整可读）。
- 恢复路径：见 `reports/FINAL_REPRODUCIBILITY_GUIDE.md`（含权重来源/固定 revision/SHA256/命令）。

## 8. 最终门禁

# `SAFE_TO_RELEASE_DATA_DISK`

判定依据（任务八的 11 项条件）：正式实验记录 ✅ 完成；GitHub 分支+tag ✅ 已推送并独立验证；GitHub clone/树验证 ✅ 154/154；HF Dataset ✅ 177/177 对齐；必要新增文件独立回下载 ✅（小 160/160 + 大 17/17）；既有 checkpoint 来源与 SHA256 ✅；原有正式推理结果 SHA256 ✅ 10/10；必要评价与逐样本均有远程备份 ✅；Manifest ✅ 无关键缺失；复现说明 ✅ 可指导重读；无未解决上传失败/断链/冲突 ✅。

> 未执行任何云平台退租/解绑/格式化/删除操作；未删除任何原始实验文件或 checkpoint。
> 数据盘可由**你手动释放**（`/dev/vdb` @ `/mnt/data`）；建议释放前最后浏览一次本报告与 manifest。

## 9. 安全提醒

归档过程中 GitHub PAT 与 HF token 曾出现在会话/日志中（本地文件 `/root/.git-credentials`、`hf_home/token` 未上传、未写入归档）。**建议立即轮换两个 token**；轮换后本地凭证文件可删除（不影响已完成的归档与验证）。
