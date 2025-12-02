#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# 读取预训练 PID 和最新日志
PID_FILE="$LOG_DIR/pretrain_prior_full.pid"
PRE_LOG=$(ls -1t "$LOG_DIR"/pretrain_prior_full_*.log 2>/dev/null | head -n1 || true)

# 读取或自动发现预训练 PID
if [[ -f "$PID_FILE" ]]; then
  PRE_PID=$(cat "$PID_FILE" | tr -d "\n\r") || PRE_PID=""
fi
if [[ -z "${PRE_PID:-}" ]]; then
  # 尝试自动发现包含目标 model_name 的 Train.py 进程
  PRE_PID=$(pgrep -af "python .*src/Train.py" | awk '/model_name mindeye_v4_pretrain_prior_full/ {print $1; exit}' || true)
fi
if [[ -z "${PRE_PID:-}" ]]; then
  echo "找不到预训练 PID（PID 文件缺失/为空且自动发现失败）。" >&2
  echo "请确认预训练是否已启动，或手动将 PID 写入 $PID_FILE。" >&2
  exit 1
fi
echo "监控预训练 PID=$PRE_PID 日志=${PRE_LOG:-N/A}"

# 等待预训练结束
while kill -0 "$PRE_PID" >/dev/null 2>&1; do
  sleep 60
done

# 粗略检查预训练是否异常（非严格，供提示用）
if [[ -n "${PRE_LOG:-}" ]]; then
  echo "预训练已结束，最后几行日志如下:"
  tail -n 50 "$PRE_LOG" || true
  if grep -E "(Traceback|Error|CUDA out of memory|Killed)" -i "$PRE_LOG" >/dev/null 2>&1; then
    echo "检测到疑似错误关键字，停止自动流程。"
    exit 2
  fi
fi

echo "开始自动执行后续全流程: extract_proj_data -> projection -> finetune -> peft"

# 固定环境与变量，确保一致性
export ENV_NAME=mindeye21
export PRETRAIN_ENABLE_PRIOR=1
export PRETRAIN_BATCH_SIZE=4
export PRETRAIN_EXP_NAME=mindeye_v4_pretrain_prior_full
export HOLDOUT_SUBJ_ID=8

# 2) 提取特征
TS=$(date +%Y%m%d_%H%M%S)
LOG_EXTRACT="$LOG_DIR/extract_proj_${TS}.log"
nohup bash run_training.sh extract_proj_data > "$LOG_EXTRACT" 2>&1 &
EPID=$!
echo $EPID > "$LOG_DIR/extract_proj.pid"
wait "$EPID"

# 3) 训练投影矩阵
TS=$(date +%Y%m%d_%H%M%S)
LOG_PROJ="$LOG_DIR/projection_${TS}.log"
nohup bash run_training.sh projection > "$LOG_PROJ" 2>&1 &
PPID=$!
echo $PPID > "$LOG_DIR/projection.pid"
wait "$PPID"

# 4) 微调（单被试: holdout subj）
TS=$(date +%Y%m%d_%H%M%S)
LOG_FT="$LOG_DIR/finetune_${TS}.log"
nohup bash run_training.sh finetune > "$LOG_FT" 2>&1 &
FPID=$!
echo $FPID > "$LOG_DIR/finetune.pid"
wait "$FPID"

# 5) PEFT 适配器
TS=$(date +%Y%m%d_%H%M%S)
LOG_PEFT="$LOG_DIR/peft_${TS}.log"
nohup bash run_training.sh peft > "$LOG_PEFT" 2>&1 &
APID=$!
echo $APID > "$LOG_DIR/peft.pid"
wait "$APID"

echo "全流程完成。日志:"
echo " - pretrain:  ${PRE_LOG:-<none>}"
echo " - extract:   $LOG_EXTRACT"
echo " - projection:$LOG_PROJ"
echo " - finetune:  $LOG_FT"
echo " - peft:      $LOG_PEFT"
