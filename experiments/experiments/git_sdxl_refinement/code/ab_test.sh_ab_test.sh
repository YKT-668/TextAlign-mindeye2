#!/bin/bash
# A/B 下载方案测试：顺序执行，结果写入 reports/_ab_test.txt
ROOT=/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114
source "$ROOT/env.sh"
OUT="$ROOT/reports/_ab_test.txt"
REV="3f20ccbe614876c83894e148cc36530c4b04caa7"
FILE="zavychromaxl_v30.safetensors"
> "$OUT"

echo "=== A': HF + hf_xet HIGH_PERFORMANCE (27MB 公开文件) ===" >> "$OUT"
HF_XET_HIGH_PERFORMANCE=1 python "$ROOT/scripts/xet_hp_test.py" >> "$OUT" 2>&1
echo "---" >> "$OUT"

echo "=== B: aria2 8连接 + 官方 huggingface.co (zavy, 限时30s) ===" >> "$OUT"
mkdir -p "$ROOT/tmp/aria2test_official"
timeout 30 aria2c -x 8 -s 8 --file-allocation=none --summary-interval=5 \
  --console-log-level=warn --max-tries=1 -d "$ROOT/tmp/aria2test_official" \
  "https://huggingface.co/pscotti/mindeyev2/resolve/$REV/$FILE" >> "$OUT" 2>&1
echo "exit=$?" >> "$OUT"
echo "---" >> "$OUT"

echo "=== B': aria2 8连接 + hf-mirror.com (zavy, 限时30s) ===" >> "$OUT"
mkdir -p "$ROOT/tmp/aria2test_mirror"
timeout 30 aria2c -x 8 -s 8 --file-allocation=none --summary-interval=5 \
  --console-log-level=warn --max-tries=1 -d "$ROOT/tmp/aria2test_mirror" \
  "https://hf-mirror.com/pscotti/mindeyev2/resolve/$REV/$FILE" >> "$OUT" 2>&1
echo "exit=$?" >> "$OUT"
echo "---" >> "$OUT"

echo "DONE" >> "$OUT"
