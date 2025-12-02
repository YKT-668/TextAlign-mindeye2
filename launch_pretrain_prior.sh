#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/pretrain_prior_full_${TS}.log"
export PRETRAIN_ENABLE_PRIOR=1
export PRETRAIN_BATCH_SIZE=4
export PRETRAIN_EXP_NAME=mindeye_v4_pretrain_prior_full
export HOLDOUT_SUBJ_ID=8
export PRETRAIN_EPOCHS=150
export ENV_NAME=mindeye21
nohup bash run_training.sh pretrain > "$LOG" 2>&1 &
PID=$!
echo $PID > logs/pretrain_prior_full.pid
echo "启动完成 PID=$PID LOG=$LOG"
