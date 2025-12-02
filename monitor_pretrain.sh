#!/bin/bash
# 简易监控脚本: 查看预训练日志末尾与显存峰值
LOG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)/logs"
LOG_FILE=$(ls -1t "$LOG_DIR"/pretrain_prior_full_*.log 2>/dev/null | head -n1)
if [ -z "$LOG_FILE" ]; then
  echo "找不到日志文件: $LOG_DIR/pretrain_prior_full_*.log" >&2
  exit 1
fi
echo "使用日志: $LOG_FILE"
echo "---- 最近 40 行 ----"
TailCmd="tail -n 40 \"$LOG_FILE\""
# 显示尾部
bash -c "$TailCmd"
# 提取显存峰值行（假设包含 peak_alloc_GB / peak_reserved_GB）
GPU_LINES=$(grep -E "peak_alloc_GB|peak_reserved_GB" "$LOG_FILE" | tail -n 5)
if [ -n "$GPU_LINES" ]; then
  echo "---- 显存峰值（最近匹配） ----"
  echo "$GPU_LINES"
fi
# 进程信息
PID_FILE="$LOG_DIR/pretrain_prior_full.pid"
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if ps -p "$PID" >/dev/null 2>&1; then
    echo "进程仍在运行: PID=$PID"
  else
    echo "进程已结束: PID=$PID"
  fi
fi
