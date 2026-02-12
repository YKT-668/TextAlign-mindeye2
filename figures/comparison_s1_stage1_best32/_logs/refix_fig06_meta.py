patch_path = "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32/_patched_generators/fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.py"

with open(patch_path, 'r') as f:
    code = f.read()

bad_line = 'return rsm_gt, rsm_pred, (rsm_pred - rsm_gt), ["patched_source_sliced"]'
good_line = 'return rsm_gt, rsm_pred, (rsm_pred - rsm_gt), {"src": "patched_source_sliced"}'

if bad_line in code:
    code = code.replace(bad_line, good_line)
    with open(patch_path, 'w') as f:
        f.write(code)
    print("Fixed meta type.")
else:
    print("Could not find meta return line.")
    # debug print
    # print(code[-500:]) 
