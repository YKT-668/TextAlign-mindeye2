#!/bin/bash
# DO NOT RUN WITHOUT USER APPROVAL - THIS SCRIPT STARTS TRAINING

set -euo pipefail

# =============================================================================
# 03_run_single_experiment_requires_approval.sh
#
# Safety-gated training wrapper for TextAlign MindEye2 cloud experiments.
# Reads experiment config from aaai_revision/, validates safety flags,
# checks output path uniqueness, and launches training.
#
# Usage:
#   USER_APPROVED_TRAINING=YES bash 03_run_single_experiment_requires_approval.sh <experiment_id>
#
# Placeholders (replace before uploading to cloud server):
#   <CLOUD_PROJECT_ROOT>  - Absolute path to project root on cloud server
#   <DATA_ROOT>           - Root directory for NSD/coco data
# <CHECKPOINT_ROOT>     - Directory for saving model checkpoints
#   <FEATURE_ROOT>        - Directory for pre-extracted features
#
# =============================================================================

# --- Argument check ---
if [ $# -lt 1 ]; then
    echo "[ERROR] Usage: USER_APPROVED_TRAINING=YES bash 03_run_single_experiment_requires_approval.sh <experiment_id>"
    echo "  Example: USER_APPROVED_TRAINING=YES bash 03_run_single_experiment_requires_approval.sh exp01_frozen_backbone_feasibility_subj01"
    exit 1
fi

EXPERIMENT_ID="$1"
echo "[INFO] Experiment: ${EXPERIMENT_ID}"

# --- Config paths ---
# Configs are stored at project ROOT (E:\From_F\Project\AAAI\Mindeye\configs\aaai_revision/)
CONFIGS_DIR="<CLOUD_PROJECT_ROOT>/configs/aaai_revision"
CONFIG_FILE="${CONFIGS_DIR}/${EXPERIMENT_ID}.json"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[ERROR] Config not found: ${CONFIG_FILE}"
    exit 1
fi
echo "[INFO] Config: ${CONFIG_FILE}"

# --- Output path uniqueness check ---
OUTPUT_DIR="<CLOUD_PROJECT_ROOT>/TextAlign-mindeye2/outputs/cloud_runs/${EXPERIMENT_ID}"
if [ -d "${OUTPUT_DIR}" ]; then
    echo "[ERROR] Output path already exists: ${OUTPUT_DIR}"
    echo "  Remove or rename existing output before re-running."
    exit 1
fi
echo "[INFO] Output: ${OUTPUT_DIR} (will be created)"

# --- Approval gate ---
if [ "${USER_APPROVED_TRAINING:-}" != "YES" ]; then
    echo "[ERROR] USER_APPROVED_TRAINING is not set to YES."
    echo "  This script starts actual training. Set USER_APPROVED_TRAINING=YES to confirm."
    echo "  You must also verify requires_user_approval_before_training=true in the config."
    exit 1
fi
echo "[INFO] USER_APPROVED_TRAINING=YES confirmed."

# --- Parse config for safety fields ---
# Check requires_user_approval_before_training
APPROVAL=$(python3 -c "
import json
with open('${CONFIG_FILE}') as f:
    cfg = json.load(f)
meta = cfg.get('_meta', {})
approval = meta.get('requires_user_approval_before_training', False)
print(str(approval).lower())
")
if [ "${APPROVAL}" != "true" ]; then
    echo "[ERROR] Config ${EXPERIMENT_ID}: requires_user_approval_before_training is not true."
    echo "  Aborting for safety."
    exit 1
fi
echo "[INFO] requires_user_approval_before_training=true confirmed."

# Check training_mode
TRAIN_MODE=$(python3 -c "
import json
with open('${CONFIG_FILE}') as f:
    cfg = json.load(f)
meta = cfg.get('_meta', {})
mode = meta.get('training_mode', 'unknown')
print(mode)
")
echo "[INFO] Training mode: ${TRAIN_MODE}"

# --- Create output directory ---
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"
echo "[INFO] Log: ${LOG_FILE}"

# --- Special case: main06 (lowdata) uses different params ---
NUM_EPOCHS=150
BATCH_SIZE=32
NUM_SESSIONS=40

case "${EXPERIMENT_ID}" in
    *lowdata*|*main06*)
        NUM_EPOCHS=200
        BATCH_SIZE=16
        NUM_SESSIONS=1
        echo "[INFO] Low-data config: epochs=${NUM_EPOCHS}, batch=${BATCH_SIZE}, sessions=${NUM_SESSIONS}"
        ;;
esac

# --- Map experiment_id to model_name ---
MODEL_NAME=""
case "${EXPERIMENT_ID}" in
    exp01*) MODEL_NAME="exp01_frozen_backbone_subj01" ;;
    exp02*) MODEL_NAME="exp02_hardneg_ablation_subj01" ;;
    exp03*) MODEL_NAME="exp03_scale_ablation_subj01" ;;
    exp04*) MODEL_NAME="exp04_tau_ablation_subj01" ;;
    exp05*) MODEL_NAME="exp05_loss_ablation_subj01" ;;
    main01*) MODEL_NAME="main01_projector_only_subj01" ;;
    main02*) MODEL_NAME="main02_frozen_backbone_subj01" ;;
    main03*) MODEL_NAME="main03_end2end_subj01" ;;
    main04*) MODEL_NAME="main04_neg_source_subj01" ;;
    main05*) MODEL_NAME="main05_crossllm_subj01" ;;
    main06*) MODEL_NAME="main06_lowdata_subj01" ;;
    main07*) MODEL_NAME="main07_loss_subj01" ;;
    *)
        echo "[ERROR] Unknown experiment_id: ${EXPERIMENT_ID}"
        exit 1
        ;;
esac
echo "[INFO] Model name: ${MODEL_NAME}"

# --- Environment setup ---
export DATA_ROOT="${DATA_ROOT:-<DATA_ROOT>}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-<CHECKPOINT_ROOT>}"
export FEATURE_ROOT="${FEATURE_ROOT:-<FEATURE_ROOT>}"

# --- Special env vars from config variants ---
# Extract env vars from config if present
EXTRA_ENV=$(python3 -c "
import json
with open('${CONFIG_FILE}') as f:
    cfg = json.load(f)
# Check for env settings in _notes
notes = cfg.get('_notes', {})
env_str = notes.get('env', '')
print(env_str)
" 2>/dev/null || echo "")
if [ -n "${EXTRA_ENV}" ]; then
    echo "[INFO] Extra env: ${EXTRA_ENV}"
    export ${EXTRA_ENV}
fi

# --- Set MINDEYE env vars for text alignment ---
export MINDEYE_TEXTALIGN=1
export MINDEYE_TEXTALIGN_SCALE=0.05

# --- Run training ---
echo ""
echo "============================================================"
echo "  Starting training: ${EXPERIMENT_ID}"
echo "  Model: ${MODEL_NAME}"
echo "  Mode: ${TRAIN_MODE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Batch: ${BATCH_SIZE}"
echo "  Sessions: ${NUM_SESSIONS}"
echo "============================================================"
echo ""

# Build training command
PYTHON_CMD="CUDA_VISIBLE_DEVICES=0 python <CLOUD_PROJECT_ROOT>/TextAlign-mindeye2/src/Train_textalign.py"
PYTHON_CMD="${PYTHON_CMD} --model_name ${MODEL_NAME}"
PYTHON_CMD="${PYTHON_CMD} --num_epochs ${NUM_EPOCHS}"
PYTHON_CMD="${PYTHON_CMD} --batch_size ${BATCH_SIZE}"
PYTHON_CMD="${PYTHON_CMD} --num_sessions ${NUM_SESSIONS}"
PYTHON_CMD="${PYTHON_CMD} --lr <LR_PLACEHOLDER>"
PYTHON_CMD="${PYTHON_CMD} --train_mode <TRAIN_MODE_PLACEHOLDER>"
PYTHON_CMD="${PYTHON_CMD} --output_dir ${OUTPUT_DIR}"
PYTHON_CMD="${PYTHON_CMD} --data_path ${DATA_ROOT}"
PYTHON_CMD="${PYTHON_CMD} --cache_dir ${FEATURE_ROOT}"

echo "[CMD] ${PYTHON_CMD}"
echo ""

# Execute training with log capture
set -o pipefail
eval "${PYTHON_CMD}" 2>&1 | tee "${LOG_FILE}"
TRAIN_EXIT=$?
set +o pipefail

# --- Post-run summary ---
echo ""
echo "============================================================"
echo "  Training completed for ${EXPERIMENT_ID}"
echo "  Exit code: ${TRAIN_EXIT}"
echo "  Log: ${LOG_FILE}"
echo "============================================================"
echo ""

# --- Special reminders ---
case "${EXPERIMENT_ID}" in
    *cross_llm*|*main05*)
        echo ">>> Cross-LLM experiment: Report Delta = Ours - Baseline on same eval set."
        echo ">>> Do NOT interpret absolute CCD alone."
        ;;
    *lowdata*|*main06*)
        echo ">>> Low-data experiment: Ensure baseline is tuned fairly"
        echo ">>> (same trainable params, weight decay, LR, early stopping)."
        ;;
esac

# --- Post-run instructions ---
echo ""
echo "--- Post-Run Instructions ---"
echo "1. Check ${LOG_FILE} for convergence"
echo "2. Run run_final_report.py after training completes"
echo "3. Do NOT overwrite existing checkpoints"
echo "4. Report metrics with delta interpretation for Cross-LLM experiments"
echo "5. For low-data: ensure baseline fairness (tuned, not untuned)"

exit ${TRAIN_EXIT}
