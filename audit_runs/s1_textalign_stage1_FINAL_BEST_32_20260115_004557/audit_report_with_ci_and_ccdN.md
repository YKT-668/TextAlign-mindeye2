# Audit Report (Supplement Aligned)
**Tag:** s1_textalign_stage1_FINAL_BEST_32
**Timestamp:** 20260115_004557
**Protocol:** shared982 (Strict)
**IS-RSA:** SKIP

### PASS/FAIL Checklist
| Item | Status | Details |
| :--- | :--- | :--- |
| **CI Bootstrap** | **PASS** | N=1000, Image-Level, Seed=42 |
| **CCD N=909** | **INFO** | Actual Used N=982 (Full 982). Dropped 0. |
| **Shared982 Map** | **PASS** | See `shared982_ids_manifest.csv` |



## 1. Sanity Checks
See `sanity_check.log`.
- Shape: Checked
- IDs: Checked (Unique & Shared1000 aligned)
- GT: Checked (1664-d)

## 2. Metrics Summary

### Retrieval (982-way)
Metric,Value
FWD@1,0.24643585085868835
FWD@5,0.585539698600769
BWD@1,0.13340121507644653
BWD@5,0.3523421585559845

### Retrieval (300-way x 30)
Metric,Value
FWD@1_mean,0.43444444437821705
FWD@1_std,0.02437869985117167
BWD@1_mean,0.25822222282489143
BWD@1_std,0.020953801528296543

### 2AFC (Image-Level)
Metric,Value
2AFC_B2I,0.9904748201370239
2AFC_I2B,0.9784458875656128
2AFC_RandomCheck,0.5054331421852112

### CCD
Metric,Value
CCD_Hard_K1,0.5682281059063137
CCD_Hard_K2,0.40427698574338083
CCD_Random_K2,0.9490835030549898

### RSA
Metric,Value
RSA_Pearson,0.35993707180023193
RSA_Spearman,0.2607363921673944

## 3. Assets
- Brain: /mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/brain_clip.pt
- IDs: /mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/ids.json
- GT: /mnt/work/repos/TextAlign-mindeye2/evals/all_images_bigG_1664_mean.pt



## 4. Supplementary Audit
### Bootstrap 95% CI (N=1000)
| Metric       |     Mean |   CI_Lower |   CI_Upper |
|:-------------|---------:|-----------:|-----------:|
| 2AFC_B2I     | 0.989482 |   0.988243 |   0.990641 |
| 2AFC_I2B     | 0.977483 |   0.974643 |   0.980096 |
| CCD_Hard_K1  | 0.581982 |   0.546843 |   0.618126 |
| CCD_Hard_K2  | 0.494816 |   0.460285 |   0.532587 |
| RSA_Pearson  | 0.378682 |   0.357008 |   0.401654 |
| RSA_Spearman | 0.263263 |   0.241348 |   0.286607 |

### CCD N Analysis
- **N_Used**: 982
- **Explanation**: All 982 shared items were used. No hard negatives were missing or filtered. The "N=909" target not applicable to this run's data availability; using max available N=982 is preferred.
