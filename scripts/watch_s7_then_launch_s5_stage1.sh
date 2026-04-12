#!/usr/bin/env bash
set -euo pipefail

WATCH_LOG="/mnt/work/repos/TextAlign-mindeye2/logs/watch_s7_then_launch_s5_stage1.log"
S7_LOG="/mnt/work/repos/TextAlign-mindeye2/logs/s7_stage0_repair.log"
REPO_DIR="/mnt/work/repos/TextAlign-mindeye2"

mkdir -p /mnt/work/repos/TextAlign-mindeye2/logs

# All watcher runtime logs go to the required log file.
exec >>"$WATCH_LOG" 2>&1

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

s7_finished() {
  grep -aEq '===Finished!===|===Finished!===' "$S7_LOG"
}

s7_running() {
  ps -eo cmd | grep -F 'src/train_textalign_bplan_fixed.py' | grep -F -- '--model_name s7_textalign_stage0_repair_80G' | grep -v grep >/dev/null
}

s5_stage1_running() {
  ps -eo cmd | grep -F 'src/train_textalign_bplan_fixed.py' | grep -F -- '--model_name s5_textalign_stage1_FINAL_BEST_32' | grep -v grep >/dev/null
}

launch_s5_stage1() {
  log "Launching s5 stage1 with README-style command."

  unset http_proxy https_proxy all_proxy
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export MINDEYE_DTYPE=bf16
  export MINDEYE_RESUME="/mnt/work/repos/TextAlign-mindeye2/train_logs/s5_textalign_stage0_repair_80G"

  export MINDEYE_TEXTALIGN_STAGE=1
  export MINDEYE_TEXTALIGN=1
  export MINDEYE_TEXTALIGN_HARDNEG=1
  export MINDEYE_TEXTALIGN_HARD_SCALE=0.3
  export MINDEYE_TEXTALIGN_MARGIN=0.1

  export MINDEYE_LR_BACKBONE=1e-5
  export MINDEYE_LR_PRIOR=1e-5
  export MINDEYE_LR_RIDGE=1e-4
  export MINDEYE_LR_TEXT=1e-4

  export HF_HOME=/mnt/work/.cache/huggingface
  export HUGGINGFACE_HUB_CACHE=/mnt/work/.cache/huggingface/hub
  export TRANSFORMERS_CACHE=/mnt/work/.cache/transformers
  export TORCH_HOME=/mnt/work/.cache/torch
  export XDG_CACHE_HOME=/mnt/work/.cache
  export TMPDIR=/mnt/work/tmp

  cd /mnt/work/repos/TextAlign-mindeye2

  nohup /mnt/work/conda_envs/mindeye21_fix2/bin/accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    --main_process_port 29500 \
    src/train_textalign_bplan_fixed.py \
    --model_name s5_textalign_stage1_FINAL_BEST_32 \
    --subj 5 \
    --num_sessions 40 \
    --batch_size 32 \
    --num_epochs 120 \
    --use_prior \
    --lr_scheduler_type linear \
    --max_lr 1e-4 \
    --ckpt_interval 1 \
    --no-wandb_log \
    --hidden_dim 4096 \
    --textalign_hardneg_path "data/nsd_text/s5_train_coco_captions_hard_negs_clip.pt" \
    --multisubject_ckpt "/mnt/work/checkpoints/mindeyev2_official/train_logs/final_multisubject_subj05" \
    > logs/s5_stage1_repair.log 2>&1 &

  local pid=$!
  log "s5 stage1 launch submitted, PID=$pid"
}

log "Watcher started. Monitoring s7 stage0 log: $S7_LOG"

if [ ! -r "$S7_LOG" ]; then
  log "ERROR: s7 stage0 log not readable: $S7_LOG"
  exit 1
fi

while true; do
  if s5_stage1_running; then
    log "s5 stage1 already running; watcher exits without duplicate launch."
    exit 0
  fi

  if s7_finished; then
    log "Detected s7 stage0 normal completion marker."
    launch_s5_stage1
    exit 0
  fi

  if ! s7_running; then
    log "ERROR: s7 stage0 process exited but completion marker not found. Not launching s5 stage1."
    exit 1
  fi

  log "s7 stage0 still running; sleep 60s."
  sleep 60
done
