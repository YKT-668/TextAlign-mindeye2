#!/bin/bash
# 统一查看当前阶段日志的简易脚本
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs"

pick_last() {
  ls -1t "$LOG_DIR"/$1 2>/dev/null | head -n1
}

PRE=$(pick_last 'pretrain_prior_full_*.log')
EXT=$(pick_last 'extract_proj_*.log')
PRO=$(pick_last 'projection_*.log')
FIN=$(pick_last 'finetune_*.log')
PEF=$(pick_last 'peft_*.log')

show() {
  local name=$1
  local file=$2
  if [[ -n "${file:-}" && -f "$file" ]]; then
    echo "===== $name: $file ====="
    tail -n 40 "$file"
    echo
  fi
}

show PRETRAIN "${PRE:-}"
show EXTRACT  "${EXT:-}"
show PROJECT  "${PRO:-}"
show FINETUNE "${FIN:-}"
show PEFT     "${PEF:-}"

# 也显示显存峰值行
for f in "$PRE" "$EXT" "$PRO" "$FIN" "$PEF"; do
  if [[ -n "${f:-}" && -f "$f" ]]; then
    GPU_LINES=$(grep -E "peak_alloc_GB|peak_reserved_GB" "$f" | tail -n 5 || true)
    if [[ -n "$GPU_LINES" ]]; then
      echo "---- GPU peaks from $(basename "$f") ----"
      echo "$GPU_LINES"
    fi
  fi
done
