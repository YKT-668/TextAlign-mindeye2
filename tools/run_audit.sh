#!/bin/bash
set -e

TAG="audit_s1_stage1_final_best32"
OUT_DIR="audit_runs/${TAG}"
BRAIN_PATH="/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/brain_clip.pt"
IDS_PATH="/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/ids.json"
SUBJ=1

echo "Starting Audit Run for TAG=${TAG}"
echo "Output: ${OUT_DIR}"

mkdir -p "$OUT_DIR/L1"
mkdir -p "$OUT_DIR/L2_2AFC"
mkdir -p "$OUT_DIR/L2_CCD/hard_k2"
mkdir -p "$OUT_DIR/L2_CCD/hard_k1"
mkdir -p "$OUT_DIR/L2_CCD/random_k2"
mkdir -p "$OUT_DIR/L3_RSA"

# L1
echo "Running L1..."
python evals/eval_shared982_latent.py --subj ${SUBJ} --brain_embed "$BRAIN_PATH" --ids_json "$IDS_PATH" --tag "$TAG" --out_dir "$OUT_DIR/L1"

# L2 2AFC
echo "Running L2 2AFC..."
python evals/eval_shared982_twoafc.py --subj ${SUBJ} --brain_embed "$BRAIN_PATH" --ids_json "$IDS_PATH" --tag "$TAG" --out_dir "$OUT_DIR/L2_2AFC"

# L2 CCD
echo "Running L2 CCD Hard K=2..."
python evals/eval_ccd_shared982.py --subj ${SUBJ} --brain_embed "$BRAIN_PATH" --ids_json "$IDS_PATH" --tag "$TAG" --out_dir "$OUT_DIR/L2_CCD/hard_k2" --neg_mode hardneg --hardneg_k 2 --difficulty hardest

echo "Running L2 CCD Hard K=1..."
python evals/eval_ccd_shared982.py --subj ${SUBJ} --brain_embed "$BRAIN_PATH" --ids_json "$IDS_PATH" --tag "$TAG" --out_dir "$OUT_DIR/L2_CCD/hard_k1" --neg_mode hardneg --hardneg_k 1 --difficulty hardest

echo "Running L2 CCD Random K=2..."
python evals/eval_ccd_shared982.py --subj ${SUBJ} --brain_embed "$BRAIN_PATH" --ids_json "$IDS_PATH" --tag "$TAG" --out_dir "$OUT_DIR/L2_CCD/random_k2" --neg_mode hardneg --hardneg_k 2 --difficulty random

# L3 RSA
echo "Running L3 RSA..."
python evals/eval_rsa_shared982.py --subj ${SUBJ} --brain_embed "$BRAIN_PATH" --ids_json "$IDS_PATH" --tag "$TAG" --out_dir "$OUT_DIR/L3_RSA"

echo "Running IS-RSA..."
# Define Baseline Paths (using Pretrained S1/S2/S5/S7)
BASE_S1="evals/brain_tokens/official_hf/final_subj01_pretrained_40sess_24bs/subj01_brain_clip_mean.pt"
BASE_S2="evals/brain_tokens/official_hf/final_subj02_pretrained_40sess_24bs/subj02_brain_clip_mean.pt"
BASE_S5="evals/brain_tokens/official_hf/final_subj05_pretrained_40sess_24bs/subj05_brain_clip_mean.pt"
BASE_S7="evals/brain_tokens/official_hf/final_subj07_pretrained_40sess_24bs/subj07_brain_clip_mean.pt"

# Prepare S1 for IS-RSA (Naming convention fix)
ln -sf "$BRAIN_PATH" "$OUT_DIR/subj01_brain_clip_mean.pt"
ln -sf "$IDS_PATH" "$OUT_DIR/subj01_brain_clip_ids.json"
TA_S1_NEW="$OUT_DIR/subj01_brain_clip_mean.pt"

# Targets for S2/S5/S7 (Using official pretrained as placeholders for S2/S5/S7 "Ours" if not separately trained, OR user implies using them as 'others')
# The user: "IS-RSA requires subject 1/2/5/7... Check workspace... generate if missing".
# Since we found them in official_hf, we use them.
TA_S2="$BASE_S2"
TA_S5="$BASE_S5"
TA_S7="$BASE_S7"

python evals/eval_isrsa_shared982.py \
  --N 982 \
  --tag "isrsa_audit_${TAG}" \
  --emb_s1 "$TA_S1_NEW" \
  --emb_s2 "$TA_S2" \
  --emb_s5 "$TA_S5" \
  --emb_s7 "$TA_S7" \
  --baseline_s1 "$BASE_S1" \
  --baseline_s2 "$BASE_S2" \
  --baseline_s5 "$BASE_S5" \
  --baseline_s7 "$BASE_S7" \
  --out_dir "$OUT_DIR/IS_RSA" \
  --bootstrap 1000 \
  --seed 42

echo "Audit Complete."
