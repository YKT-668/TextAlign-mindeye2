# Experiment Snapshot 2026-04-13

## A. 本次已知结论（冻结口径）
1. S2/S5/S7 三个模型均已完成 stage0 与 stage1，且最终 checkpoint 健康可读。
2. 当前三者采用 repair-style stage0：
   - 不是代码原生 STAGE=0 语义。
   - 是 README3 风格 STAGE=1 repair 的第一阶段，用于产出 stage0-compatible resume 基座。
3. stage1 恢复语义（本次审计口径）：
   - 以 stage0-compatible checkpoint 为基础进入 stage1。
   - 实际恢复是否携带优化器/调度器状态，以当次 resume 源与训练脚本行为为准，不能仅按变量名推断。
4. corrected Brain Corr 已修复旧 run_debug 负值解释问题；旧负值不再作为正式结论。
5. 四被试总体判断：
   - S2/S5/S7 当前不建议立刻重训。
   - 当前最大系统不确定性是 S1 对照资产链不闭合。
   - last vs best 仍可能导致 S2/S5/S7 被低估。
6. 当前 corrected Brain Corr：
   - S2: nsd_general 0.3755, V1 0.3671, V2 0.3273, V3 0.3417, V4 0.3495, higher_vis 0.3736
   - S5: nsd_general 0.4024, V1 0.3337, V2 0.3394, V3 0.3201, V4 0.3083, higher_vis 0.4167
   - S7: nsd_general 0.296169, V1 0.289935, V2 0.283341, V3 0.268082, V4 0.250500, higher_vis 0.292982
7. run_debug / inference / corrected Brain Corr 注意事项：
   - run_debug 必须显式传入 model_name 与 all_recons_path，避免默认路由到错误被试。
   - inference 输出需按输出目录与 subject 双重核对，不可仅凭文件名。
   - corrected Brain Corr 以 aligned sidecar 结果为准，legacy 统计仅作排错线索。
8. 后续最优先工作：
   - 补齐 S1 同口径资产链。
   - 做 S2/S5/S7 checkpoint 口径敏感性分析（last vs best）。
   - 再决定是否重训。

## B. 关键路径（恢复优先）
- S2 stage0-compatible: train_logs/s2_textalign_stage0_repair_80G_resume_compat_epoch0/last.pth
- S2 stage1 final: train_logs/s2_textalign_stage1_FINAL_BEST_32/last.pth
- S5 stage0-compatible: train_logs/s5_textalign_stage0_repair_80G_resume_compat_epoch0/last.pth
- S5 stage1 final: train_logs/s5_textalign_stage1_FINAL_BEST_32/last.pth
- S7 stage0-compatible: train_logs/s7_textalign_stage0_repair_80G_resume_compat_epoch0/last.pth
- S7 stage1 final: train_logs/s7_textalign_stage1_FINAL_BEST_32/last.pth
- eval 导出: evals/s{2,5,7}_textalign_stage1_FINAL_BEST_32/
- corrected Brain Corr: tables/s{2,5,7}_textalign_stage1_FINAL_BEST_32_brain_corr_aligned.json

## C. 恢复流程（下次重建）
1. 从 GitHub 拉取 snapshot 分支/标签中的代码与文档。
2. 从 HF 仓库 snapshots/2026-04-13/ 拉取模型/日志/评测/审计产物。
3. 按 ENV_SNAPSHOT 还原环境：pip freeze + conda env export + conda explicit。
4. 补齐未上传的大型公共原始资产（见 UPLOAD_PLAN.json 的 Exclude 列表）。
5. 校验关键 checkpoint、eval 表、Brain Corr 文件与 SHA256。
