#!/usr/bin/env bash
# P1 特征模型顺序执行（单模型串行，避免内存叠加）
set -uo pipefail
ROOT=/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114
export PYTHONPATH=/mnt/data/cpu_pkgs
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export PYTHONUNBUFFERED=1
export HF_HOME=$ROOT/hf_home
export HF_HUB_CACHE=$ROOT/hf_home/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
OUT=$ROOT/evaluation/cpu_mindeye2_p1
LOG=$ROOT/logs

for m in "$@"; do
    echo "===== START $m $(date +%H:%M:%S) ====="
    /usr/bin/time -v /mnt/data/gpu_venv/bin/python "$ROOT/scripts/cpu_p1_features.py" \
        --model "$m" --out "$OUT" > "$LOG/p1_$m.log" 2> "$LOG/p1_$m.time.log"
    rc=$?
    echo "===== END $m rc=$rc $(date +%H:%M:%S) ====="
    tail -3 "$LOG/p1_$m.log"
done

