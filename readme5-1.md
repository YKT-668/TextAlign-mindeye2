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
| GitHub commit | To be recorded after final push |
| GitHub tag | `archive/final14` |
| HF Model revision | To be recorded after upload |
| HF Dataset revision | To be recorded after upload |
| Archive date | 2026-07-26 |