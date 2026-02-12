import os

patch_path = "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32/_patched_generators/fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.py"

with open(patch_path, 'r') as f:
    code = f.read()

# I injected: return torch.load(".../brain_clip.pt"), torch.load(".../all_images_bigG_1664_mean.pt")
# I need to find this string and replace it with full logic.
# The previous patch was:
bad_string = 'return torch.load("/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/brain_clip.pt"), torch.load("/mnt/work/repos/TextAlign-mindeye2/evals/all_images_bigG_1664_mean.pt")'

# The replacement logic needs to call compute_rsm.
# compute_rsm is defined in the file.
new_logic = """
    p_brain = "/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/brain_clip.pt"
    p_gt = "/mnt/work/repos/TextAlign-mindeye2/evals/all_images_bigG_1664_mean.pt"
    print(f"Computing RSM from patched paths:\\n Brain: {p_brain}\\n GT: {p_gt}")
    
    pred_feats = torch.load(p_brain)
    gt_feats = torch.load(p_gt)
    
    # Ensure CPU and float
    if hasattr(pred_feats, 'cpu'): pred_feats = pred_feats.cpu()
    if hasattr(gt_feats, 'cpu'): gt_feats = gt_feats.cpu()
    
    rsm_pred = compute_rsm(pred_feats)
    rsm_gt = compute_rsm(gt_feats)
    
    return rsm_gt, rsm_pred, (rsm_pred - rsm_gt), ["patched_source"]
"""

if bad_string in code:
    code = code.replace(bad_string, new_logic)
    with open(patch_path, 'w') as f:
        f.write(code)
    print("Fixed fig06.")
else:
    print("Could not find the bad string in fig06. Manual check needed.")
    # Check if I used the alternative patch? 
    # In `patch_scripts.py` I had commented out `brain_pt = ...` block and put the one-liner in replace call.
    # Ah, I see in my tool use earlier:
    # `code = code.replace('    # Ours S1 V2 paths', f'    # Ours S1 V2 paths\n    # PATCHED: use new model\n    return torch.load("{new_brain_path}"), torch.load("/mnt/work/repos/TextAlign-mindeye2/evals/all_images_bigG_1664_mean.pt")')`
    # So the bad string should match exactly the replacement result.
    pass
