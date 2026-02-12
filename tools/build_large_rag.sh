#!/usr/bin/env bash
set -eo pipefail

# build_large_rag.sh : One-stop script to prepare a large COCO-based caption corpus
# and generate a CLIP (ViT-H-14) text index for RAG.
#
# Prerequisites:
#   - COCO caption JSON files (captions_train2017.json, captions_val2017.json)
#     placed under ${COCO_ANN_DIR} (default: data/coco_ann)
#   - Python environment with torch + open_clip installed.
#
# Outputs:
#   - ${OUT_CAPS_PT} : combined cleaned caption list (.pt)
#   - ${OUT_CAPS_TXT} : optional text dump (one per line)
#   - ${OUT_INDEX_PT} : CLIP ViT-H text embedding index
#   - ${QUALITY_JSON} : quality check report (optional)
#
# Usage (typical):
#   bash tools/build_large_rag.sh \
#     --train data/coco_ann/captions_train2017.json \
#     --val data/coco_ann/captions_val2017.json \
#     --out-prefix data/coco_full \
#     --dedup --lower --batch-size 512 --quality
#
# After building, update run_inference.sh env vars:
#   export TEXT_INDEX_PT="/home/vipuser/MindEyeV2_Project/data/coco_full_index.pt"
#   export ALL_CAPTIONS_PT="/home/vipuser/MindEyeV2_Project/data/coco_full_captions.pt"

TRAIN_JSON=""
VAL_JSON=""
OUT_PREFIX="data/coco_full"
DEDUP=false
LOWER=false
BATCH_SIZE=512
DEVICE="cuda"
QUALITY=false
MAX_SAMPLES=0
KNN_SAMPLES=100
KNN_K=5

while [[ $# -gt 0 ]]; do
  case $1 in
    --train) TRAIN_JSON="$2"; shift 2;;
    --val) VAL_JSON="$2"; shift 2;;
    --out-prefix) OUT_PREFIX="$2"; shift 2;;
    --dedup) DEDUP=true; shift;;
    --lower) LOWER=true; shift;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --quality) QUALITY=true; shift;;
    --max-samples) MAX_SAMPLES="$2"; shift 2;;
    --knn-samples) KNN_SAMPLES="$2"; shift 2;;
    --knn-k) KNN_K="$2"; shift 2;;
    *) echo "[warn] Unknown arg: $1"; shift;;
  esac
done

if [[ -z "$TRAIN_JSON" || -z "$VAL_JSON" ]]; then
  echo "[error] --train and --val JSON paths are required" >&2
  exit 1
fi

CAPS_PT="${OUT_PREFIX}_captions.pt"
CAPS_TXT="${OUT_PREFIX}_captions.txt"
INDEX_PT="${OUT_PREFIX}_index.pt"
QUALITY_JSON="${OUT_PREFIX}_quality.json"

echo "[cfg] TRAIN_JSON=$TRAIN_JSON"
echo "[cfg] VAL_JSON=$VAL_JSON"
echo "[cfg] OUT_PREFIX=$OUT_PREFIX"
echo "[cfg] DEDUP=$DEDUP LOWER=$LOWER BATCH_SIZE=$BATCH_SIZE DEVICE=$DEVICE MAX_SAMPLES=$MAX_SAMPLES QUALITY=$QUALITY"

LOWER_FLAG=""; $LOWER && LOWER_FLAG="--lower"
DEDUP_FLAG=""; $DEDUP && DEDUP_FLAG="--dedup"

echo "[step1] Preparing combined captions (.pt/.txt)"
python tools/prepare_coco_captions.py \
  --ann "$TRAIN_JSON" "$VAL_JSON" \
  --out_pt "$CAPS_PT" \
  --out_txt "$CAPS_TXT" \
  --min_len 3 --max_len 140 $DEDUP_FLAG $LOWER_FLAG

echo "[step2] Building text index (OpenCLIP ViT-H-14)"
python tools/text_index.py \
  --captions_pt "$CAPS_PT" \
  --out_pt "$INDEX_PT" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --max_samples "$MAX_SAMPLES"

if $QUALITY; then
  echo "[step3] Running quality check"
  python tools/check_rag_store.py \
    --text_index "$INDEX_PT" \
    --captions "$CAPS_PT" \
    --num_samples "$KNN_SAMPLES" \
    --k "$KNN_K" \
    --out "$QUALITY_JSON"
  echo "[done] Quality report -> $QUALITY_JSON"
fi

echo "[done] Large RAG build complete. Artifacts:"
echo "  captions_pt = $CAPS_PT"
echo "  index_pt    = $INDEX_PT"
if $QUALITY; then echo "  quality_json = $QUALITY_JSON"; fi
echo "Next: export TEXT_INDEX_PT=$INDEX_PT and ALL_CAPTIONS_PT=$CAPS_PT before run_inference.sh"
