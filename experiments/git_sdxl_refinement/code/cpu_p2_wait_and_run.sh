#!/usr/bin/env bash
# 等待 CLIP-B 权重与 tokenizer 小文件就绪，然后跑 P2 caption 全量评测
set -u
ROOT=/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114
D=/mnt/data/cpu_models/clipb
B=https://huggingface.co/openai/clip-vit-base-patch32/resolve/main
NEED=605247071

for f in config.json vocab.json merges.txt tokenizer_config.json special_tokens_map.json; do
  for a in $(seq 1 20); do
    if [ -s "$D/$f" ]; then break; fi
    aria2c -q -x4 -s4 -m2 --file-allocation=none --console-log-level=error -d "$D" -o "$f" "$B/$f" || sleep 3
  done
done

for i in $(seq 1 120); do
  sz=$(stat -c%s "$D/pytorch_model.bin" 2>/dev/null || echo 0)
  if [ "$sz" -ge "$NEED" ] && [ ! -e "$D/pytorch_model.bin.aria2" ]; then
    echo "weights ready size=$sz"
    break
  fi
  sleep 10
done

export PYTHONPATH=/mnt/data/cpu_pkgs CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=3 HF_HOME=$ROOT/hf_home
/mnt/data/gpu_venv/bin/python "$ROOT/scripts/cpu_p2_caption.py" > "$ROOT/logs/p2_caption.log" 2>&1
echo "RUN_EXIT=$?" >> "$ROOT/logs/p2_caption.log"
