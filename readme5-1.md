# README5-1 — Final Experiment Archive

## 1. Purpose

This is the **final14** complete experiment archive for the AAAI ConceptAlign paper. It provides a one-to-one reproducible snapshot of all main paper experiments, follow-up supplementary experiments, complete reproducibility environment, manifests, checksums, and restore scripts.

## 2. Existing Cloud Baseline

| Resource | Repository | Branch/Revision | Content |
|---|---|---|---|
| GitHub | `https://github.com/YKT-668/TextAlign-mindeye2.git` | `main` / `579ab6e` | Release code & scripts |
| HF Model | `ykt668/textalign-mindeye2-model` | `main` / `131a31bc` | checkpoints/, features/, snapshots/ |
| HF Dataset | `ykt668/textalign-mindeye2-data` | `main` | Existing dataset files |

## 3. Changes Added

| Category | Content | Destination |
|---|---|---|
| Large experiments | Final checkpoints (s1/s2/s5/s7/ss2/shared/smoke), 42G train_logs, 12G readme3 inference outputs | HF model/data |
| Follow-up | Human Challenge HC1/HC2, Fast Validation C1/C2/Final, FIG4 redo experiments, 4.3G evals, recovery metadata | HF data |
| Reproducibility | conda-pack environment (~4GB .tar.zst), pip/conda locks, system info, git state | HF model / GitHub |
| Manifests | CHECKSUMS.sha256, LOCAL_ASSET_INVENTORY.tsv, THIRD_PARTY_DEPENDENCIES, DATA_PATH_MAP, COMMAND_PROVENANCE, EXCLUDED_ASSETS | GitHub + HF |
| Restore | restore_from_cloud.sh, verify_restored_archive.py, bootstrap_environment.sh, smoke_test_cpu.sh | GitHub |
| Documentation | This readme5-1.md | GitHub root |

## 4. Difference from Existing Cloud

**Previously NOT on cloud:** full experiment checkpoints (only 2026-04-12/13 snapshots existed), readme3 inference outputs, Human-written challenge experiments, Fast validation series, FIG4 redo, full evals/ tensors, conda-pack binary environment.

**No modifications** to existing cloud code, configs, or history. All additions are incremental under `archive/` or `final14_archive/` paths.

## 5. Cloud Layout

### GitHub
```
archive/manifests/*.tsv, CHECKSUMS.sha256
archive/reproducibility/environment/*.txt, *.yml, *.json
archive/restore/*.sh, *.py
readme5-1.md (root)
```

### HF Model (`ykt668/textalign-mindeye2-model`)
```
final14_archive/large_experiments/checkpoints/, training_logs/
final14_archive/reproducibility/conceptalign_env_linux64.tar.zst
final14_archive/manifests/CHECKSUMS.sha256
```

### HF Dataset (`ykt668/textalign-mindeye2-data`)
```
final14_archive/large_experiments/inference_outputs/, paper_materials/
final14_archive/followup_experiments/human_challenge/, fast_validation/, fig4/, evaluation/, recovery_metadata/
final14_archive/manifests/CHECKSUMS.sha256
```

## 6. One-to-One Restore

```bash
git clone --depth 1 --branch archive/final14 https://github.com/YKT-668/TextAlign-mindeye2.git
cd TextAlign-mindeye2
pip install huggingface_hub hf_xet
export HF_TOKEN=<your_token>
hf download ykt668/textalign-mindeye2-model final14_archive/ --repo-type=model --local-dir=./hf_model/
hf download ykt668/textalign-mindeye2-data final14_archive/ --repo-type=dataset --local-dir=./hf_data/
sha256sum -c archive/manifests/CHECKSUMS.sha256
# See archive/restore/restore_from_cloud.sh for full procedure
```

## 7. Experiment Mapping

See `archive/manifests/LOCAL_ASSET_INVENTORY.tsv` for complete mapping of paper experiments to code/config/HF artifacts/result sources.

## 8. Verification

Verification performed via `archive/restore/restore_from_cloud.sh --dry-run`, checksum verification, and per-object HF round-trip SHA256 (recorded in `FINAL14_REMOTE_ROUNDTRIP.tsv`).

## 9. Exclusions

- **Raw NSD** — NSD license, download from [naturalscenesdataset.org](https://naturalscenesdataset.org/)
- **COCO 2017 images** — COCO license, download from [cocodataset.org](https://cocodataset.org/)
- **Third-party HF models** (CLIP-ViT-bigG, CLIP-ViT-H, GIT-large-coco, MindEye2) — redistributed by original authors under their licenses
- **Cache, core dumps, tokens, personal info** — security policy

## 10. Frozen Revisions

| Artifact | Value |
|---|---|
| GitHub commit | `d4aad858c935c6f70ba10d3b24f4c977dec1dade` |
| GitHub branch | `archive/final14` |
| HF Model main revision | `ceb32860e6a9e1c775e30da8f8e884e0f1926795` |
| HF Model master revision | `a127f9295fd5656ac63ae436f07e61b80bf4efce` |
| HF Dataset revision | `2c612c5c2ca3c5344854edbd7028769cb254ab5a` |
| Archive manifest SHA256 | See `archive/manifests/CHECKSUMS.sha256` |
| Conda-pack SHA256 | See `conceptalign_env_linux64.tar.zst.sha256` |
| Archive date | 2026-07-26 |

### Checkpoint Paths (HF Model, branch `main`)

| Model | Path | Size | SHA256 OID |
|-------|------|------|-----------|
| C1 positive-only | `checkpoints/final14_ablation/positive_only/last.pth` | 24.95 GiB | `95c3250676ce1c9ccaec` |
| C2 random-negative | `checkpoints/final14_ablation/random_negative/last.pth` | 24.95 GiB | `e8b1996e55accf4f855f` |
| C3 CLIP-nearest | `checkpoints/final14_ablation/clip_nearest/last.pth` | 24.95 GiB | `13fb861766996ee0778b` |
| Shared Stage0 | `checkpoints/final14_ablation/shared_stage0/last.pth` | 22.04 GiB | `2f3a101771739e34edbe` |

### Ours S1/S2/S5/S7 (existing, not re-uploaded)

| Model | Repo | Branch | Path | Revision |
|-------|------|--------|------|----------|
| Ours S1 | HF Model | `master` | `checkpoints/s1_textalign_stage1_FINAL_BEST_32/last.pth` | `a127f9295fd5656a` |
| Ours S2 | HF Model | `main` (snapshot) | `snapshots/2026-04-13/train_logs/s2_textalign_stage1_FINAL_BEST_32/last.pth` | `5b31f3aa` |
| Ours S5 | HF Model | `main` (snapshot) | `snapshots/2026-04-13/train_logs/s5_textalign_stage1_FINAL_BEST_32/last.pth` | `5b31f3aa` |
| Ours S7 | HF Model | `main` (snapshot) | `snapshots/2026-04-13/train_logs/s7_textalign_stage1_FINAL_BEST_32/last.pth` | `5b31f3aa` |

### ours_ss2

ours_ss2 (`train_logs/ss2_textalign_stage1_FINAL_BEST_32/`, 16.08 GiB) is classified as **P2_DIAGNOSTIC_OR_RECOVERY_ONLY**. It is a diagnostic/recovery checkpoint not used by any paper table, figure, evaluation, or inference result. It is NOT included in the minimal required archive. If needed for exact training reproduction, it is available locally at the path above.
---
|---|
| GitHub commit | To be recorded after final push |
| GitHub tag | `archive/final14` |
| HF Model revision | To be recorded after upload |
| HF Dataset revision | To be recorded after upload |
| Archive date | 2026-07-26 |