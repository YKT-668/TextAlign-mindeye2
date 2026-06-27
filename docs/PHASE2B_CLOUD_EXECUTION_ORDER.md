# Phase 2B: Cloud Execution Order

## Overview

This document defines the phased execution plan for TextAlign-mindeye2 experiments on the cloud server. The plan is divided into 7 phases (Phase 0-6) to ensure safe, incremental validation before committing to full training runs.

**Commit:** `579ab6e`
**Date:** 2026-06-11

---

## Quick Reference: 12 Experiments

| ID | Config | Group | Training Mode | Est. GPU Hours |
|----|--------|-------|---------------|----------------|
| exp01 | exp01_frozen_backbone_feasibility_subj01 | phase1_ablation | frozen_backbone | ~6h |
| exp02 | exp02_hardneg_ablation_subj01 | phase1_ablation | frozen_backbone | ~6h |
| exp03 | exp03_scale_ablation_subj01 | phase1_ablation | frozen_backbone | ~6h |
| exp04 | exp04_tau_ablation_subj01 | phase1_ablation | frozen_backbone | ~6h |
| exp05 | exp05_loss_ablation_subj01 | phase1_ablation | frozen_backbone | ~6h |
| main01 | AAAI_MAIN01_projector_only_feasibility_subj01 | aaai_main_feasibility | projector_only | ~4h |
| main02 | AAAI_MAIN02_frozen_backbone_feasibility_subj01 | aaai_main_feasibility | frozen_backbone | ~6h |
| main03 | AAAI_MAIN03_end2end_reference_subj01 | aaai_main_feasibility | end_to_end | ~8h |
| main04 | AAAI_MAIN04_negative_source_ablation_subj01 | aaai_main_ablation | frozen_backbone | ~6h |
| main05 | AAAI_MAIN05_cross_llm_delta_eval_subj01 | aaai_main_cross_llm | frozen_backbone | ~6h |
| main06 | AAAI_MAIN06_lowdata_fair_baseline_subj01 | aaai_main_lowdata | frozen_backbone | ~6h |
| main07 | AAAI_MAIN07_loss_ablation_subj01 | aaai_main_ablation | frozen_backbone | ~6h |

---

## Phase 0: Cloud Environment Validation (NO TRAINING)

**Script:** `scripts/cloud_gpu/00_cloud_bootstrap.sh`

**Goal:** Validate that the cloud server has all prerequisites and the conda environment works.

**Steps:**
1. Clone repo
2. Create conda env from `environment_mindeye21.yml`
3. `pip install -r requirements_mindeye21.txt`
4. Run `python -c "import torch; print(torch.cuda.is_available())"`
5. Run `py_compile` on all `src/` Python files
6. Log results

**Expected output:** Clean compilation, CUDA available, all imports succeed.

---

## Phase 1: Asset Download & Path Verification (NO TRAINING)

**Script:** `scripts/cloud_gpu/01_download_assets_requires_approval.sh`

**Goal:** Download all required assets from HuggingFace and verify paths.

**Warnings:**
- This may download >5GB of data
- Requires HuggingFace token
- Does NOT download NSD restricted data

**Steps:**
1. Check disk space (>50GB free)
2. Check HuggingFace login
3. Download TextAlign-mindeye2 model snapshot
4. Verify all required files exist
5. Log asset manifest

---

## Phase 2: GPU Smoke Test (NO TRAINING)

**Script:** `scripts/cloud_gpu/02_gpu_smoke_test_no_training.sh`

**Goal:** Verify GPU compute, data loading, and config parsing.

**Constraints:**
- `max_steps=1` only
- Does NOT start full training
- Does NOT load large models (SDXL/unCLIP)

**Checks:**
1. `torch.cuda.is_available()` and device count
2. Small tensor GPU forward/backward test
3. `py_compile` on `Train_textalign.py` and `quick_eval.py`
4. Registry/config preflight check

---

## Phase 3: First Real Experiments

**Wrapper:** `scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh`

**Important:** Each experiment requires `USER_APPROVED_TRAINING=YES`.

**Recommended order:**
1. **main01** (projector_only, ~4h) - Lowest risk, fewest trainable params
2. **main02** (frozen_backbone, ~6h) - Default configuration
3. **main03** (end_to_end, ~8h) - Upper bound reference

**After each:** Check `outputs/cloud_runs/<experiment_id>/train.log` for convergence.

---

## Phase 4: Core Ablation Experiments

**Order:**
1. **main07** (loss_ablation, ~6h) - Determine optimal loss formulation
2. **main04** (negative_source_ablation, ~6h) - 4 variants
3. **exp01-exp05** (phase1_ablation) - Additional ablation studies

---

## Phase 5: Cross-LLM Delta Evaluation

**Experiment:** main05 (cross_llm_delta_eval, ~6h)

**Critical:**
- Report **Delta = Ours - Baseline** on same GPT-4o eval set
- Do NOT interpret absolute CCD alone
- Absolute CCD may drop due to OpenCLIP text embedding domain shift

---

## Phase 6: Fair Low-Data Baseline

**Experiment:** main06 (lowdata_fair_baseline, ~6h)

**Critical:**
- Baseline must be **tuned** fairly (same trainable params, weight decay, LR, early stopping)
- Do NOT compare untuned official full baseline vs ours
- Report **Delta = Ours - TunedBaseline** for each data regime

---

## Key Rules

1. **NO training** without explicit user approval
2. All training commands need `USER_APPROVED_TRAINING=YES`
3. Use `03_run_single_experiment_requires_approval.sh` wrapper
4. Do **NOT** run all 12 experiments at once
5. Check output path doesn't exist before training
6. Do **NOT** modify `src/` core model code
7. Changes to model code must be patch proposals only
8. Do **NOT** overwrite existing checkpoints
9. Cross-LLM: report Delta, not absolute
10. Low-data: baseline must be tuned fairly
