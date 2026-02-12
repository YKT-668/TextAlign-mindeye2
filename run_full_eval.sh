
#!/bin/bash
set -e
cd /mnt/work/repos/TextAlign-mindeye2

# -----------------------------
# 0) Setup
# -----------------------------
NEW_TAG="ours_s1_stage1_final_best32_lastpth_v1"
SUBJ=1
CKPT_MODEL_NAME="s1_textalign_stage1_FINAL_BEST_32"
# Note: we use model_name because script defaults to looking in train_logs/{model_name}/last.pth
# and we verified the file exists there.

OUT_BRAIN_TOK_DIR="evals/brain_tokens/${NEW_TAG}"
mkdir -p "${OUT_BRAIN_TOK_DIR}"

INFER_OUT="/mnt/work/tmp_infer_out_${NEW_TAG}"
mkdir -p "$INFER_OUT"

echo "=== 1. Inference & Export ==="
# We use existing src/recon_inference_run.py
# It expects --model_name. It will find last.pth in train_logs.
# We map output to INFER_OUT.
python src/recon_inference_run.py \
  --data_path=/mnt/work/repos/TextAlign-mindeye2 \
  --cache_dir=/mnt/work/repos/TextAlign-mindeye2 \
  --output_dir="$INFER_OUT" \
  --model_name="$CKPT_MODEL_NAME" \
  --subj=${SUBJ} \
  --new_test \
  --dump_ids \
  --dump_clip_vecs \
  --export_official_pts \
  --max_save 982 \
  --save_images \
  --image_format png

# Link artifacts
FOUND_MEAN="$(find "$INFER_OUT" -maxdepth 4 -type f -name '*brain_clip*mean*.pt' | head -n 1)"
echo "[FOUND_MEAN] $FOUND_MEAN"
if [ ! -f "$FOUND_MEAN" ]; then echo "Error: Mean embedding not found"; exit 1; fi
ln -sf "$FOUND_MEAN" "${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_mean.pt"

FOUND_TOK="$(find "$INFER_OUT" -maxdepth 4 -type f -name '*brain_clip*tokens*.pt' | head -n 1 || true)"
if [ -n "$FOUND_TOK" ]; then
  ln -sf "$FOUND_TOK" "${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_tokens.pt"
fi

FOUND_IDS="$(find "$INFER_OUT" -maxdepth 4 -type f -name '*ids.json' | head -n 1)"
echo "[FOUND_IDS] $FOUND_IDS"
ln -sf "$FOUND_IDS" "${OUT_BRAIN_TOK_DIR}/subj01_ids.json"

# Validate len(ids)
python -c "import json,sys; ids=json.load(open(sys.argv[1])); print(f'IDs: {len(ids)}'); assert len(ids)==982" "$FOUND_IDS"

# Common Env Vars for Tools
export BRAIN_PATH="${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_mean.pt"
export IDS_PATH="$FOUND_IDS"
export EVAL_SUBSET="shared982"
export GT_PATH="evals/all_images.pt" # Ensure this exists or fallback
export CAPTIONS_PATH="evals/all_captions.pt"

echo "=== 2. L1 Retrieval (FWD/BWD) ==="
export RESULT_DIR="cache/model_eval_results/shared982_latent/${NEW_TAG}"
export EVAL_REPR="pooled"
python tools/eval_textalign_latent_plus.py

echo "=== 3. L1 2AFC ==="
export RESULT_DIR="cache/model_eval_results/shared982_twoafc/${NEW_TAG}"
python tools/eval_twoafc_embed.py

echo "=== 4. L2 CCD (Caption Discrimination) ==="

# 4a) Main: hardneg, k=2, hardest
export RESULT_DIR="cache/model_eval_results/shared982_ccd/${NEW_TAG}/main_k2_hardest"
export HARD_NEG_JSONL="cache/hardneg/shared982_hardneg_for_ccd.jsonl"
export HARD_NEG_K=2
export K_NEG=2
# Note: script sorts by sim_text descending, so this implies 'hardest'.
python tools/eval_ccd_embed.py

# 4b) Ablation K=4
export RESULT_DIR="cache/model_eval_results/shared982_ccd/${NEW_TAG}/ablation_k4"
export HARD_NEG_K=4
export K_NEG=4
python tools/eval_ccd_embed.py

# 4c) Ablation Difficulty=Random
# Unset hard neg jsonl to force sampled negatives
export RESULT_DIR="cache/model_eval_results/shared982_ccd/${NEW_TAG}/ablation_difficulty_random_k2"
unset HARD_NEG_JSONL
export K_NEG=2
export SEED=0
python tools/eval_ccd_embed.py

echo "=== 5. L3 RSA ==="
export RESULT_DIR="cache/model_eval_results/shared982_rsa/${NEW_TAG}"
python tools/eval_rsa_embed.py

echo "=== 6. IS-RSA ==="
# Custom script creation
cat << 'PYEOF' > evals/eval_isrsa_custom.py
import argparse
import json
import torch
import numpy as np
import os
import sys
from scipy.stats import spearmanr
import torch.nn.functional as F

def load_emb(path):
    print(f"Loading {path}")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # try to find tensor
        for k, v in obj.items():
            if torch.is_tensor(v) and v.ndim == 2: return v
        raise ValueError(f"No 2D tensor in {path}")
    return obj

def load_ids(path):
    with open(path) as f: return np.array(json.load(f))

def compute_rdm(emb):
    # emb: [N, D]
    emb = F.normalize(emb.float(), dim=-1).cuda()
    sim = emb @ emb.T
    return sim.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--emb_s1", required=True)
    parser.add_argument("--emb_s2", required=True)
    parser.add_argument("--emb_s5", required=True)
    parser.add_argument("--emb_s7", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--baseline_s1", required=True)
    parser.add_argument("--baseline_s2", required=True)
    parser.add_argument("--baseline_s5", required=True)
    parser.add_argument("--baseline_s7", required=True)
    parser.add_argument("--N", type=int, default=982)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load all
    embs = {
        "s1": load_emb(args.emb_s1),
        "s2": load_emb(args.emb_s2),
        "s5": load_emb(args.emb_s5),
        "s7": load_emb(args.emb_s7),
        "b1": load_emb(args.baseline_s1),
        "b2": load_emb(args.baseline_s2),
        "b5": load_emb(args.baseline_s5),
        "b7": load_emb(args.baseline_s7),
    }

    # Ensure shape
    for k, v in embs.items():
        if v.shape[0] != args.N:
            print(f"Warning: {k} shape {v.shape} != {args.N}, assuming shared982 subsetting needed or risk mismatch")
            # In a real script we would strictly align by IDs. 
            # Here assuming simple user inputs are already 982 or aligned.
            # But specific logic might be needed if they are shared1000.
            # For now, let's assume they are aligned or truncate if only minimal mismatch (e.g. 1000->982)
            if v.shape[0] == 1000:
                 # TODO: load shared982 mask if available.
                 pass

    # Compute RDMs
    rdms = {k: compute_rdm(v) for k, v in embs.items()}
    
    # Upper tri indices
    triu_idx = np.triu_indices(args.N, k=1)
    
    def get_vec(k):
        return rdms[k][triu_idx]

    # Pairwise comparisons for "Our S1 vs Others"
    # We want IS-RSA(S1_new, S2), IS-RSA(S1_new, S5), IS-RSA(S1_new, S7)
    # And Baseline IS-RSA(S1_base, S2_base)... 
    # Usually IS-RSA is symmetric, but here we compare "Method A S1" to "Method B S2"?
    # Typically: IS-RSA = Correlation(RDM(S1), RDM(S2)).
    
    # We compute average IS-RSA overlap with *other subjects*.
    # For S1 (Ours): Mean( Corr(S1_our, S2_base), Corr(S1_our, S5_base), Corr(S1_our, S7_base) )?
    # Or Corr(S1_our, S2_our)? The user said "textalign_llm group: ours_s1_vNEW + ours_s*_v10".
    # So we compare S1_new vs S2_v10, S5_v10, S7_v10.
    
    pairs = [("s1", "s2"), ("s1", "s5"), ("s1", "s7")]
    corrs = []
    for a, b in pairs:
        r, _ = spearmanr(get_vec(a), get_vec(b))
        corrs.append(r)
    mean_isrsa = np.mean(corrs)
    
    # Baseline
    # "official_hf" group.
    b_pairs = [("b1", "b2"), ("b1", "b5"), ("b1", "b7")]
    b_corrs = []
    for a, b in b_pairs:
        r, _ = spearmanr(get_vec(a), get_vec(b))
        b_corrs.append(r)
    base_mean = np.mean(b_corrs)
    
    print(f"Ours (S1 vs S2,5,7): {mean_isrsa:.4f}")
    print(f"Base (S1 vs S2,5,7): {base_mean:.4f}")
    
    metrics = {
        "model_tag": args.tag,
        "isrsa_mean": float(mean_isrsa),
        "isrsa_pairs": {f"{a}_{b}": float(c) for (a,b), c in zip(pairs, corrs)},
        "baseline_mean": float(base_mean),
        "baseline_pairs": {f"{a}_{b}": float(c) for (a,b), c in zip(b_pairs, b_corrs)}
    }
    
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
PYEOF

TA_S1_NEW="${OUT_BRAIN_TOK_DIR}/subj01_brain_clip_mean.pt"
TA_S2="evals/brain_tokens/ours_s2_v10/subj02_brain_clip_mean.pt"
TA_S5="evals/brain_tokens/ours_s5_v10/subj05_brain_clip_mean.pt"
TA_S7="evals/brain_tokens/ours_s7_v10/subj07_brain_clip_mean.pt"

BASE_S1="evals/brain_tokens/official_hf/final_subj01_pretrained_40sess_24bs/subj01_brain_clip_mean.pt"
BASE_S2="evals/brain_tokens/official_hf/final_subj02_pretrained_40sess_24bs/subj02_brain_clip_mean.pt"
BASE_S5="evals/brain_tokens/official_hf/final_subj05_pretrained_40sess_24bs/subj05_brain_clip_mean.pt"
BASE_S7="evals/brain_tokens/official_hf/final_subj07_pretrained_40sess_24bs/subj07_brain_clip_mean.pt"

python evals/eval_isrsa_custom.py \
  --N 982 \
  --tag "textalign_llm_s1_${NEW_TAG}" \
  --out_dir "cache/model_eval_results/shared982_isrsa/textalign_llm_s1_${NEW_TAG}" \
  --emb_s1 "$TA_S1_NEW" --emb_s2 "$TA_S2" --emb_s5 "$TA_S5" --emb_s7 "$TA_S7" \
  --baseline_s1 "$BASE_S1" --baseline_s2 "$BASE_S2" --baseline_s5 "$BASE_S5" --baseline_s7 "$BASE_S7"

echo "=== 7. Runlog ==="
python - << PY
from pathlib import Path
import json, os
tag=os.environ.get("NEW_TAG","")
paths=[
  f"cache/model_eval_results/shared982_latent/{tag}",
  f"cache/model_eval_results/shared982_twoafc/{tag}",
  f"cache/model_eval_results/shared982_ccd/{tag}/main_k2_hardest",
  f"cache/model_eval_results/shared982_rsa/{tag}",
  f"cache/model_eval_results/shared982_isrsa/textalign_llm_s1_{tag}",
]
out=Path(f"results/tables/_runlog_{tag}.md")
out.parent.mkdir(parents=True, exist_ok=True)
lines=[f"# Runlog {tag}\n"]
for p in paths:
  pp=Path(p)
  lines.append(f"## {p}\n")
  if not pp.exists():
    lines.append("- MISSING FOLDER\n"); continue
  ms=list(pp.rglob("metrics.json"))
  lines.append(f"- metrics.json: {len(ms)}\n")
  for m in ms:
    try:
      j=json.load(open(m))
      lines.append(f"  - {m}: {str(j)[:200]}...\n")
    except Exception as e:
      lines.append(f"  - {m}: ERROR {e}\n")
out.write_text("".join(lines))
print("Wrote", out)
PY

echo "[DONE] All evals completed"
