# Experiment Snapshot 2026-04-12

## 1. Scope and current state
This snapshot captures the recoverable experiment state in /mnt/work/repos/TextAlign-mindeye2 before machine teardown, with emphasis on S2/S5/S7 stage0-stage1 outputs, evaluation artifacts, and audit evidence.

## 2. Verified conclusions from this round
1. S2/S5/S7 all completed stage0 and stage1, and final stage1 checkpoints are loadable and healthy.
2. The three subjects used repair-style stage0, not native code STAGE=0 path.
3. Stage1 resume semantics in this round:
   - Resume base for S2/S5 used stage0-compatible checkpoint directory *_resume_compat_epoch0.
   - S7 stage1 resumed from existing stage1 checkpoint state.
   - Scheduler/optimizer restore semantics follow current script behavior and stage compatibility checks.
4. Corrected Brain Corr sidecar fixed legacy run_debug negative-value interpretation issue; legacy negative Brain Corr is not treated as final conclusion.
5. Cross-subject judgment for S1/S2/S5/S7:
   - S2/S5/S7 are not recommended for immediate retraining.
   - Largest systemic uncertainty is unresolved S1 baseline asset-chain closure.
   - last-vs-best checkpoint choice may still underestimate S2/S5/S7.
6. Corrected Brain Corr (official aligned sidecar outputs):
   - S2: nsd_general 0.3755, V1 0.3671, V2 0.3273, V3 0.3417, V4 0.3495, higher_vis 0.3736
   - S5: nsd_general 0.4024, V1 0.3337, V2 0.3394, V3 0.3201, V4 0.3083, higher_vis 0.4167
   - S7: nsd_general 0.296169, V1 0.289935, V2 0.283341, V3 0.268082, V4 0.250500, higher_vis 0.292982
7. Known inference/eval/corrected caveats:
   - run_debug default model_name can route to S1 if arguments are omitted.
   - run_debug legacy Brain Corr path is not ID-aligned; use corrected sidecar outputs.
   - recon_inference_run outputs are model-scoped and should be verified by output directory and ids.
8. Next priorities:
   - Close S1 same-protocol asset chain.
   - Run checkpoint sensitivity analysis for S2/S5/S7 with best/last proxies.
   - Decide retraining only after the above closure.

## 3. Effective training assets (S2/S5/S7)
- S2 stage0-compatible checkpoint:
  - train_logs/s2_textalign_stage0_repair_80G_resume_compat_epoch0/last.pth
- S2 stage1 final checkpoint:
  - train_logs/s2_textalign_stage1_FINAL_BEST_32/last.pth

- S5 stage0-compatible checkpoint:
  - train_logs/s5_textalign_stage0_repair_80G_resume_compat_epoch0/last.pth
- S5 stage1 final checkpoint:
  - train_logs/s5_textalign_stage1_FINAL_BEST_32/last.pth

- S7 stage0-compatible checkpoint:
  - train_logs/s7_textalign_stage0_repair_80G_resume_compat_epoch0/last.pth
- S7 stage1 final checkpoint:
  - train_logs/s7_textalign_stage1_FINAL_BEST_32/last.pth

## 4. Effective inference/eval assets (S2/S5/S7)
- Inference exports:
  - evals/s2_textalign_stage1_FINAL_BEST_32/
  - evals/s5_textalign_stage1_FINAL_BEST_32/
  - evals/s7_textalign_stage1_FINAL_BEST_32/
- Evaluation tables:
  - tables/s2_textalign_stage1_FINAL_BEST_32_*
  - tables/s5_textalign_stage1_FINAL_BEST_32_*
  - tables/s7_textalign_stage1_FINAL_BEST_32_*
- Corrected Brain Corr:
  - tables/s2_textalign_stage1_FINAL_BEST_32_brain_corr_aligned.json
  - tables/s5_textalign_stage1_FINAL_BEST_32_brain_corr_aligned.json
  - tables/s7_textalign_stage1_FINAL_BEST_32_brain_corr_aligned.json

## 5. Training semantics note (repair-style stage0)
- Repair-style stage0 here is operationally a STAGE=1-style compatibility phase used to produce resume-compatible checkpoints under current workflow conventions.
- It is not equivalent to native STAGE=0 branch semantics in code.
- Therefore, resuming stage1 must be interpreted using actual resume source directory and checkpoint metadata, not just environment labels.

## 6. Correct command usage notes
- Inference:
  - src/recon_inference_run.py with explicit model_name, output_dir, subj, and new_test flags.
- Evaluation:
  - src/run_debug.py must pass explicit model_name and all_recons_path to avoid default-route ambiguity.
- Corrected Brain Corr:
  - src/recompute_brain_corr_aligned.py using eval_dir/model_name/subj with aligned ids.

## 7. Recovery guide
1. Clone code snapshot from GitHub branch/tag in this snapshot report.
2. Pull artifact snapshot from Hugging Face path snapshots/2026-04-12/.
3. Restore environment from ENV_SNAPSHOT files:
   - conda_env_export_no_builds.yml
   - conda_list_explicit.txt
   - pip_freeze.txt
4. Re-provide public/raw assets that are not mirrored in snapshot (documented in UPLOAD_PLAN and manifest), including dataset-level sources.
5. Validate restore:
   - check presence of stage1 checkpoints for S2/S5/S7
   - check eval exports under evals/s2|s5|s7_textalign_stage1_FINAL_BEST_32
   - check corrected Brain Corr json files under tables/
6. Continue next workstream:
   - close S1 baseline chain, then run checkpoint sensitivity refinement, then decide retraining.

## 8. Risk notes
- S1 same-protocol chain remains unresolved in this workspace.
- last-vs-best uncertainty remains for S2/S5/S7 until explicit best checkpoint evidence is available.
