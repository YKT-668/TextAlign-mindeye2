#!/bin/bash
# restore_from_cloud.sh - ConceptAlign Final14 Archive Restore
set -euo pipefail
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[DRY-RUN]"

GITHUB_REPO="https://github.com/YKT-668/TextAlign-mindeye2.git"
GITHUB_TAG="${GITHUB_TAG:-archive/final14}"
HF_MODEL_REPO="ykt668/textalign-mindeye2-model"
HF_DATASET_REPO="ykt668/textalign-mindeye2-data"

echo "=== Clone GitHub ==="
if [ "$DRY_RUN" = true ]; then echo "[DRY-RUN] clone $GITHUB_REPO"; else git clone --depth 1 --branch "$GITHUB_TAG" "$GITHUB_REPO" TextAlign-mindeye2; cd TextAlign-mindeye2; fi

echo "=== Download HF Model ==="
if [ "$DRY_RUN" = true ]; then echo "[DRY-RUN] download $HF_MODEL_REPO"; else hf download "$HF_MODEL_REPO" final14_archive/ --repo-type=model --local-dir=./hf_model/; fi

echo "=== Download HF Dataset ==="
if [ "$DRY_RUN" = true ]; then echo "[DRY-RUN] download $HF_DATASET_REPO"; else hf download "$HF_DATASET_REPO" final14_archive/ --repo-type=dataset --local-dir=./hf_data/; fi

echo "=== Verify Checksums ==="
if [ "$DRY_RUN" = true ]; then echo "[DRY-RUN] sha256sum -c"; else sha256sum -c archive/manifests/CHECKSUMS.sha256; fi

echo "=== Restore Environment ==="
if [ -f hf_model/final14_archive/reproducibility/conceptalign_env_linux64.tar.zst ]; then
    mkdir -p ~/conda_envs/mindeye21
    tar -xaf hf_model/final14_archive/reproducibility/conceptalign_env_linux64.tar.zst -C ~/conda_envs/mindeye21
    echo "Environment unpacked to ~/conda_envs/mindeye21"
fi

echo "=== DONE ==="