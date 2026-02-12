import os

patch_path = "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32/_patched_generators/fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.py"

with open(patch_path, 'r') as f:
    code = f.read()

# The block to replace
old_block = """    p_brain = "/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/brain_clip.pt"
    p_gt = "/mnt/work/repos/TextAlign-mindeye2/evals/all_images_bigG_1664_mean.pt"
    print(f"Computing RSM from patched paths:\\n Brain: {p_brain}\\n GT: {p_gt}")
    
    pred_feats = torch.load(p_brain)
    gt_feats = torch.load(p_gt)
    
    # Ensure CPU and float
    if hasattr(pred_feats, 'cpu'): pred_feats = pred_feats.cpu()
    if hasattr(gt_feats, 'cpu'): gt_feats = gt_feats.cpu()
    
    rsm_pred = compute_rsm(pred_feats)
    rsm_gt = compute_rsm(gt_feats)
    
    return rsm_gt, rsm_pred, (rsm_pred - rsm_gt), ["patched_source"]"""

new_block = """    p_brain = "/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/brain_clip.pt"
    p_gt = "/mnt/work/repos/TextAlign-mindeye2/evals/all_images_bigG_1664_mean.pt"
    p_manifest = "/mnt/work/repos/TextAlign-mindeye2/audit_runs/s1_textalign_stage1_FINAL_BEST_32_20260115_004557/shared982_ids_manifest.csv"
    
    print(f"Computing RSM from patched paths (with slicing):\\n Brain: {p_brain}\\n GT: {p_gt}\\n Manifest: {p_manifest}")
    
    pred_feats = torch.load(p_brain)
    gt_feats_all = torch.load(p_gt)
    
    # Slice GT using manifest
    import pandas as pd
    df_m = pd.read_csv(p_manifest)
    # Ensure sorted by position to match pred
    indices = df_m.sort_values("position_in_shared982")["source_shared1000_index"].values
    
    # Check bounds
    if gt_feats_all.shape[0] != 1000:
        print(f"WARNING: Expected GT size 1000, got {gt_feats_all.shape}. Slicing anyway if indices valid.")

    gt_feats = gt_feats_all[indices]

    # Ensure CPU and float
    if hasattr(pred_feats, 'cpu'): pred_feats = pred_feats.cpu()
    if hasattr(gt_feats, 'cpu'): gt_feats = gt_feats.cpu()
    
    print(f"Shapes -- Pred: {pred_feats.shape}, GT: {gt_feats.shape}")
    
    # Final check
    if pred_feats.shape[0] != gt_feats.shape[0]:
         # Attempt to truncate pred if it has dropped ids? 
         # But usually pred is 982. 
         # If manifest has 982 rows, GT becomes 982.
         pass

    rsm_pred = compute_rsm(pred_feats)
    rsm_gt = compute_rsm(gt_feats)
    
    return rsm_gt, rsm_pred, (rsm_pred - rsm_gt), ["patched_source_sliced"]"""

if old_block in code:
    print("Found old block. Replacing...")
    code = code.replace(old_block, new_block)
    with open(patch_path, 'w') as f:
        f.write(code)
    print("File patched.")
else:
    print("Could not find exact block match. Creating fuzzy replacement...")
    # It might be whitespace issues. Let's try to match by the unique return line.
    marker = 'return rsm_gt, rsm_pred, (rsm_pred - rsm_gt), ["patched_source"]'
    if marker in code:
        parts = code.split(marker)
        # We assume the part before marker contains the variable defs we want to replace.
        # But this is dangerous.
        print("Manual check required or better matching needed.")
        # Alternatively, let's just write the whole file content since we read it from `read_file` output exactly earlier.
        print("Aborting safe patch. Attempting overwrite with constructed content.")
        
        # We know the structure from the read_file output.
        # Lines 95-116 in the file content we read match `old_block` visually.
        # Let's try to normalize whitespaces? No, risky.
        
        # Let's try replacing just the return line with the new logic if we can't match the block?
        # No, we need the logic BEFORE the return.
        pass
