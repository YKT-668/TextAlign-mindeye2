# Audit Report (Final)
**Tag:** s1_textalign_stage1_FINAL_BEST_32
**Timestamp:** 20260115_004557
**Protocol:** shared982 (Strict)
**IS-RSA:** SKIP

### 1. Consistency Check (Bootstrap vs Point)
| Metric        |   Point_Est |   Boot_Mean |    Diff_Abs | Status   |
|:--------------|------------:|------------:|------------:|:---------|
| 2AFC_B2I      |    0.990475 |    0.989482 | 0.000992963 | PASS     |
| 2AFC_I2B      |    0.978446 |    0.977483 | 0.000962445 | PASS     |
| RSA_Pearson   |    0.359937 |    0.378682 | 0.0187453   | PASS     |
| RSA_Spearman  |    0.260736 |    0.263263 | 0.00252699  | PASS     |
| CCD_Hard_K1   |    0.568228 |    0.567407 | 0.000820778 | PASS     |
| CCD_Hard_K2   |    0.404277 |    0.403209 | 0.00106823  | PASS     |
| CCD_Random_K2 |    0.949083 |    0.948905 | 0.000178205 | PASS     |


### CCD Fix Explanation
- **Issue**: Previous CCD bootstrap mean (0.49) significantly deviated from point estimate (0.40).
- **Cause**: The previous bootstrap logic re-mined hard negatives from the *sub-sampled* batch, which effectively changed the "dataset difficulty" (smaller pool -> easier negatives -> higher accuracy).
- **Fix**: Changed to "Method B": Pre-calculating per-image 0/1 correctness scores using the global fixed negative pool, then bootstrapping these binary scores.
- **Result**: Bootstrap mean now aligns with point estimate (Diff < 0.002).


### 2. PASS/FAIL Checklist
| Item | Status | Details |
| :--- | :--- | :--- |
| **CI Bootstrap** | **PASS** | CCD Fixed. Consistency Verified. |
| **CCD N=909** | **INFO** | Actual Used N=982. Dropped 0. |
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




## 4. Supplementary Audit (Finalized)
### Bootstrap 95% CI (N=1000)
| Metric        |     Mean |   CI_Lower |   CI_Upper |
|:--------------|---------:|-----------:|-----------:|
| 2AFC_B2I      | 0.989482 |   0.988243 |   0.990641 |
| 2AFC_I2B      | 0.977483 |   0.974643 |   0.980096 |
| RSA_Pearson   | 0.378682 |   0.357008 |   0.401654 |
| RSA_Spearman  | 0.263263 |   0.241348 |   0.286607 |
| CCD_Hard_K1   | 0.567407 |   0.53666  |   0.600815 |
| CCD_Hard_K2   | 0.403209 |   0.373727 |   0.433809 |
| CCD_Random_K2 | 0.948905 |   0.934827 |   0.962322 |

### CCD N Analysis
- **N_Used**: 982
- **Explanation**: All 982 items used. Hard negative mining successful for all.
