# Audit Report: s1_textalign_stage1_FINAL_BEST_32

## 0. Basic Information
- **Date**: 2026-01-15
- **Model**: s1_textalign_stage1_FINAL_BEST_32 (ours_s1_stage1_final_best32_lastpth_v1)
- **Subject**: 01
- **Protocol**: shared982 (N=982)

## 1. Assets Map
| Asset | Path | Status |
| :--- | :--- | :--- |
| **S1 Brain Embedding** | `/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/brain_clip.pt` | Found |
| **S1 IDs** | `/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/ids.json` | Found |
| **GT CLIP Embeddings** | `/mnt/work/repos/TextAlign-mindeye2/evals/all_images_bigG_1664_mean.pt` | Found (Fixed path) |
| **S2 Embedding (Ref)** | `.../official_hf/final_subj02_pretrained_40sess_24bs/subj02_brain_clip_mean.pt` | Found |
| **S5 Embedding (Ref)** | `.../official_hf/final_subj05_pretrained_40sess_24bs/subj05_brain_clip_mean.pt` | Found |
| **S7 Embedding (Ref)** | `.../official_hf/final_subj07_pretrained_40sess_24bs/subj07_brain_clip_mean.pt` | Found |

## 2. Protocol Audit (Sanity Checks)
| Check | Status | Details |
| :--- | :--- | :--- |
| **Split Correctness** | PASS | 982/982 IDs match `shared1000.npy` subset. |
| **IDs Alignment** | PASS | `ids.json` provided and verified against GT shape. |
| **Vector Shape** | PASS | (982, 1664). Norms are healthy (~4.6). |
| **GT File** | PASS | Using `all_images_bigG_1664_mean.pt` (Vector Ground Truth). |

## 3. Recomputed Metrics (Subject 01)
*Comparison with previous/reported values.*

| Metric | Protocol | Value (Audit) | Status |
| :--- | :--- | :--- | :--- |
| **L1 Retrieval (Fwd @1)** | shared982, 982-way | **24.64%** | Verified |
| **L1 Retrieval (Bwd @1)** | shared982, 982-way | **13.34%** | Verified |
| **L2 2AFC** | shared982 | **99.05%** | Verified |
| **L2 CCD (Hard K=2)** | Negative Pool | **40.43%** | Verified |
| **L2 CCD (Hard K=1)** | Negative Pool | **56.82%** | **New** |
| **L2 CCD (Random)** | Negative Pool | **94.91%** | Verified |
| **L3 RSA** | Spearman | **0.261** | Verified |

## 4. IS-RSA (Inter-Subject)
*Newly computed using Subject 1 (Ours) vs Subject 2, 5, 7 (Official Baselines).*

- **Mean Off-Diagonal IS-RSA**: **0.584**
- **Mean Off-Diagonal Cosine**: **0.536**

## 5. Root Cause Analysis
- **Previous Failure**: The original evaluation scripts pointed to `evals/all_images.pt` which likely contained raw images or mismatched tensors (3D vs 2D), causing `size mismatch`.
- **Fix**: Updated scripts to use `evals/all_images_bigG_1664_mean.pt` which contains the correct CLIP vectors for the shared1000 set.
- **IS-RSA Failure**: Missing `ids.json` sidecars for S2/S5/S7. Fixed by creating symlinks to the existing `subj0X_ids.json`.

## 6. Reproduction Commands
```bash
# 1. Setup Symlinks (for IS-RSA)
ln -sf .../subj02_ids.json .../subj02_brain_clip_ids.json
# (Repeat for S5, S7)

# 2. Run Audit
bash tools/run_audit.sh
```
