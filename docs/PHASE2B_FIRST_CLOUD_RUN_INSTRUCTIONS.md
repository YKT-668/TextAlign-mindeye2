# Phase 2B: First Cloud Run Instructions

## Overview

This document provides step-by-step instructions for running the TextAlign-mindeye2 experiments on a cloud GPU server for the first time. Follow the phases in order. Do NOT skip phases or run all experiments at once.

**Commit:** `579ab6e`
**Configs location:** `<CLOUD_PROJECT_ROOT>/configs/aaai_revision/`

---

## Prerequisites

- Cloud server with NVIDIA A100 (or equivalent) GPU
- At least 100GB free disk space
- Git, conda/mamba installed
- HuggingFace account with access to `minniemel/TextAlign-mindeye2` model repo
- HuggingFace token (`~/.huggingface/token`)

---

## Phase 0: Environment Validation (NO TRAINING)

**Script:** `scripts/cloud_gpu/00_cloud_bootstrap.sh`

```bash
bash scripts/cloud_gpu/00_cloud_bootstrap.sh
```

This script will:
1. Set up path variables (CLOUD_PROJECT_ROOT, DATA_ROOT, CHECKPOINT_ROOT, etc.)
2. Clone repo or cd into existing directory
3. Record git commit hash
4. Check `nvidia-smi` output
5. Create conda environment from `environment_mindeye21.yml`
6. `pip install -r requirements_mindeye21.txt`
7. Run import smoke tests (torch, transformers, open_clip, etc.)
8. `py_compile` all src/ files
9. Log everything to `docs/cloud_bootstrap_log.md`

**Validation check:**
```bash
# After successful bootstrap, verify:
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())"
# Expected: CUDA: True Devices: 1 (or more)
```

---

## Phase 1: Asset Download (REQUIRES APPROVAL)

**Script:** `scripts/cloud_gpu/01_download_assets_requires_approval.sh`

**WARNING:** This script downloads large files from HuggingFace (>5GB total).

```bash
# First review the script
cat scripts/cloud_gpu/01_download_assets_requires_approval.sh

# Then run with approval
USER_APPROVED_DOWNLOAD=YES bash scripts/cloud_gpu/01_download_assets_requires_approval.sh
```

This script will:
1. Check available disk space
2. Verify HuggingFace login
3. Download model snapshots (with resume support)
4. Do NOT download NSD restricted data (requires separate agreement)
5. Log to `logs/download_assets.log`

**After download, verify assets:**
```bash
ls -la <FEATURE_ROOT>/hf_textalign_mindeye2_model/snapshots/
ls -la <DATA_ROOT>/nsd/
```

---

## Phase 2: GPU Smoke Test (NO TRAINING)

**Script:** `scripts/cloud_gpu/02_gpu_smoke_test_no_training.sh`

```bash
bash scripts/cloud_gpu/02_gpu_smoke_test_no_training.sh
```

This will:
1. Verify `torch.cuda` availability
2. Run small tensor GPU forward/backward test
3. Compile-check `Train_textalign.py` and `quick_eval.py`
4. Run registry/config preflight check
5. Do NOT load large models (SDXL/unCLIP)
6. Do NOT start training (max_steps=0)

**Expected output:**
```
[SMOKE] torch.cuda.is_available() = True
[SMOKE] Device count: 1
[SMOKE] Small tensor GPU test: PASSED
[SMOKE] py_compile Train_textalign.py: OK
[SMOKE] py_compile quick_eval.py: OK
```

---

## Phase 3: First Real Experiments (REQUIRES APPROVAL)

**Wrapper:** `scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh`

**Run experiments in this order ONLY:**

### Step 3a: Projector Only (Lowest Risk)

```bash
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    main01_projector_only_feasibility_subj01
```

- Estimated time: ~4 hours
- Trainable params: ~2.5M
- Check `outputs/cloud_runs/main01_projector_only_feasibility_subj01/train.log`

### Step 3b: Frozen Backbone

```bash
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    main02_frozen_backbone_feasibility_subj01
```

- Estimated time: ~6 hours
- Default configuration

### Step 3c: End-to-End Reference

```bash
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    main03_end2end_reference_subj01
```

- Estimated time: ~8 hours
- Upper bound reference (~900M trainable params)

---

## Phase 4: Core Ablation Experiments (REQUIRES APPROVAL)

Run after Phase 3 completes successfully.

```bash
# Loss ablation (determine optimal loss formulation)
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    main07_loss_ablation_subj01

# Negative source ablation (4 variants)
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    main04_negative_source_ablation_subj01

# Phase 1 ablations (additional studies)
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    exp01_frozen_backbone_feasibility_subj01
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    exp02_hardneg_ablation_subj01
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    exp03_scale_ablation_subj01
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    exp04_tau_ablation_subj01
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    exp05_loss_ablation_subj01
```

---

## Phase 5: Cross-LLM Delta Evaluation (REQUIRES APPROVAL)

```bash
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    main05_cross_llm_delta_eval_subj01
```

**IMPORTANT - Delta Interpretation:**
- Do NOT interpret absolute CCD scores alone
- Report **Delta = Ours - Baseline** on same GPT-4o eval set
- Absolute CCD may drop due to OpenCLIP text embedding domain shift
- Use same-eval-set comparison for fair evaluation

---

## Phase 6: Fair Low-Data Baseline (REQUIRES APPROVAL)

```bash
USER_APPROVED_TRAINING=YES bash scripts/cloud_gpu/03_run_single_experiment_requires_approval.sh \
    main06_lowdata_fair_baseline_subj01
```

**IMPORTANT - Fairness Requirements:**
- Baseline must be **tuned** fairly (same trainable params, weight decay, LR, early stopping)
- Do NOT compare untuned official full baseline vs ours
- Report **Delta = Ours - TunedBaseline** for each data regime
- Tuning dimensions: weight decay=[1e-3, 1e-2, 1e-1], LR=[1e-4, 3e-4, 1e-3], early_stopping=patience_20

---

## Critical Rules Summary

| Rule | Description |
|------|-------------|
| 1 | NO training without explicit user approval |
| 2 | All training commands need `USER_APPROVED_TRAINING=YES` |
| 3 | Use `03_run_single_experiment_requires_approval.sh` wrapper |
| 4 | Do NOT run all 12 experiments at once |
| 5 | Check output path uniqueness before training |
| 6 | Do NOT modify `src/` core model code |
| 7 | Changes to model code must be patch proposals only |
| 8 | Do NOT overwrite existing checkpoints |
| 9 | Cross-LLM: report Delta, not absolute |
| 10 | Low-data: baseline must be tuned fairly |

---

## Troubleshooting

### Conda env creation fails
```bash
# Try mamba instead
mamba env create -f environment_mindeye21.yml
```

### CUDA out of memory
- Reduce batch_size in the config
- Use gradient accumulation
- Try mixed precision (already set in Train_textalign.py)

### HuggingFace download interrupted
- Script supports resume
- Delete partial files and re-run

### Config not found
- Configs are stored at `<CLOUD_PROJECT_ROOT>/configs/aaai_revision/`
- Verify path: `ls <CLOUD_PROJECT_ROOT>/configs/aaai_revision/`

### Output path already exists
- Remove or rename `outputs/cloud_runs/<experiment_id>/`
- Do NOT overwrite without explicit checkpoint review
