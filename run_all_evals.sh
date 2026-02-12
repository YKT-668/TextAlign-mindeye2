#!/bin/bash
set -e

NEW_TAG="ours_s1_stage1_final_best32_lastpth_v1"
INFER_OUT="/mnt/work/tmp_infer_out_${NEW_TAG}"
OUT_BRAIN_TOK_DIR="evals/brain_tokens/${NEW_TAG}"
SUBJ=1

# Wait for inference
echo "Waiting for brain_clip.pt in ${INFER_OUT}..."
# Wait up to ~3 hours (1000 * 10s)
for i in $(seq 1 1000); do
    if [ -f "${INFER_OUT}/brain_clip.pt" ]; then
        break
    fi
    sleep 10
done

if [ ! -f "${INFER_OUT}/brain_clip.pt" ]; then
    echo "Timeout waiting for brain_clip.pt"
    exit 1
fi
echo "Found brain_clip.pt"

# Add delay to ensure writing finished
sleep 5

# Step 2B: copying
mkdir -p "${OUT_BRAIN_TOK_DIR}"
ln -sf "${INFER_OUT}/brain_clip.pt" "${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_mean.pt"
ln -sf "${INFER_OUT}/ids.json" "${OUT_BRAIN_TOK_DIR}/subj01_ids.json"
# Also copy tokens if exist
if [ -f "${INFER_OUT}/brain_clip_tokens.pt" ]; then
    ln -sf "${INFER_OUT}/brain_clip_tokens.pt" "${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_tokens.pt"
fi

FOUND_IDS="${INFER_OUT}/ids.json"

# Step 3: L1
echo "Running L1..."
python evals/eval_shared982_latent.py \
  --subj ${SUBJ} \
  --brain_embed "${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_mean.pt" \
  --ids_json "$FOUND_IDS" \
  --tag "${NEW_TAG}" \
  --out_dir "cache/model_eval_results/shared982_latent/${NEW_TAG}" \
  --seed 42

python evals/eval_shared982_twoafc.py \
  --subj ${SUBJ} \
  --brain_embed "${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_mean.pt" \
  --ids_json "$FOUND_IDS" \
  --tag "${NEW_TAG}" \
  --out_dir "cache/model_eval_results/shared982_twoafc/${NEW_TAG}" \
  --seed 42

# Step 4: L2 CCD
echo "Running L2 CCD..."
# Main
python evals/eval_ccd_shared982.py \
  --subj ${SUBJ} \
  --tag "${NEW_TAG}" \
  --brain_embed "${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_mean.pt" \
  --ids_json "$FOUND_IDS" \
  --neg_mode hardneg \
  --hardneg_k 2 \
  --difficulty hardest \
  --bootstrap 1000 \
  --seed 42 \
  --out_dir "cache/model_eval_results/shared982_ccd/${NEW_TAG}/main_k2_hardest"

# Ablation K4
python evals/eval_ccd_shared982.py \
  --subj ${SUBJ} \
  --tag "${NEW_TAG}" \
  --brain_embed "${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_mean.pt" \
  --ids_json "$FOUND_IDS" \
  --neg_mode hardneg \
  --hardneg_k 4 \
  --difficulty hardest \
  --bootstrap 1000 \
  --seed 42 \
  --out_dir "cache/model_eval_results/shared982_ccd/${NEW_TAG}/ablation_k4"

# Ablation difficulty
python evals/eval_ccd_shared982.py \
  --subj ${SUBJ} \
  --tag "${NEW_TAG}" \
  --brain_embed "${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_mean.pt" \
  --ids_json "$FOUND_IDS" \
  --neg_mode hardneg \
  --hardneg_k 2 \
  --difficulty random \
  --bootstrap 1000 \
  --seed 0 \
  --out_dir "cache/model_eval_results/shared982_ccd/${NEW_TAG}/ablation_difficulty_random_k2"

# Step 5: L3 RSA
echo "Running L3 RSA..."
python evals/eval_rsa_shared982.py \
  --subj ${SUBJ} \
  --tag "${NEW_TAG}" \
  --brain_embed "${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_mean.pt" \
  --ids_json "$FOUND_IDS" \
  --out_dir "cache/model_eval_results/shared982_rsa/${NEW_TAG}" \
  --seed 42

# Step 6: IS-RSA
echo "Running L3 IS-RSA..."
BASE_S1="evals/brain_tokens/official_hf/final_subj01_pretrained_40sess_24bs/subj01_brain_clip_mean.pt"
BASE_S2="evals/brain_tokens/official_hf/final_subj02_pretrained_40sess_24bs/subj02_brain_clip_mean.pt"
BASE_S5="evals/brain_tokens/official_hf/final_subj05_pretrained_40sess_24bs/subj05_brain_clip_mean.pt"
BASE_S7="evals/brain_tokens/official_hf/final_subj07_pretrained_40sess_24bs/subj07_brain_clip_mean.pt"

TA_S1_NEW="${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_mean.pt"
TA_S2="evals/brain_tokens/ours_s2_v10/subj02_brain_clip_mean.pt"
TA_S5="evals/brain_tokens/ours_s5_v10/subj05_brain_clip_mean.pt"
TA_S7="evals/brain_tokens/ours_s7_v10/subj07_brain_clip_mean.pt"

python evals/eval_isrsa_shared982.py \
  --N 982 \
  --tag "textalign_llm_s1_${NEW_TAG}" \
  --emb_s1 "$TA_S1_NEW" \
  --emb_s2 "$TA_S2" \
  --emb_s5 "$TA_S5" \
  --emb_s7 "$TA_S7" \
  --baseline_s1 "$BASE_S1" \
  --baseline_s2 "$BASE_S2" \
  --baseline_s5 "$BASE_S5" \
  --baseline_s7 "$BASE_S7" \
  --out_dir "cache/model_eval_results/shared982_isrsa/textalign_llm_s1_${NEW_TAG}" \
  --bootstrap 1000 \
  --seed 42

echo "ALL DONE"

# Step 7: Final Report
echo "Generating Report..."
export NEW_TAG=""
python run_final_report.py
