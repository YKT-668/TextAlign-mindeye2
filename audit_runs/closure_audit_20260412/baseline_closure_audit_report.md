# Baseline Closure Audit Report (2026-04-12)

## Evidence files generated in this audit
- s1_s2_s5_s7_asset_chain_audit.json
- checkpoint_sensitivity_stage1_logproxy_v2.json
- checkpoint_sensitivity_stage1_logproxy_v2.csv

## Core findings
1. S1 asset chain is not complete in current workspace for same-protocol comparison.
2. S1 corrected Brain Corr cannot be recomputed with current local assets due missing required inputs.
3. S2/S5/S7 stage1 directories only contain last.pth; no explicit best.pth.
4. Stage1 log proxies indicate last may underestimate for S2/S5/S7.

## S1 missing links
- Missing stage1 checkpoint directory: train_logs/s1_textalign_stage1_FINAL_BEST_32
- Missing eval export directory: evals/s1_textalign_stage1_FINAL_BEST_32
- Missing aligned brain corr outputs: tables/s1_textalign_stage1_FINAL_BEST_32_brain_corr_aligned.{json,tsv}
- Missing subj01 aligned resources for corrected sidecar in this workspace:
  - betas_all_subj01_fp32_renorm.hdf5
  - wds/subj01/new_test/0.tar
  - s1_*_all_ids.pt and s1_*_all_enhancedrecons.pt

## Checkpoint sensitivity (from stage1 logs)
- S2: blurry best-last gap = 0.072, bwd best-last gap = 0.017
- S5: blurry best-last gap = 0.111, bwd best-last gap = 0.060
- S7: blurry best-last gap = 0.016, bwd best-last gap = 0.057

Judgement: all three subjects show potential underestimation risk when using last only.
