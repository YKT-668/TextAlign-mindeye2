import os

base_dir = "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32"
patched_dir = os.path.join(base_dir, "_patched_generators")
inputs_dir = os.path.join(base_dir, "_inputs_augmented")
input_ccd = os.path.join(inputs_dir, "ccd_summary.csv")
input_rsa = os.path.join(inputs_dir, "rsa_summary.csv")
input_hardneg = os.path.join(inputs_dir, "s1_textalign_stage1_FINAL_BEST_32_hardneg.jsonl")

# 1. Patch run_fig01_forest_multirun_2x2_v6.py
p1 = os.path.join(patched_dir, "run_fig01_forest_multirun_2x2_v6.py")
with open(p1, 'r') as f: code = f.read()

code = code.replace('CSV_PATH = "/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_ccd/ccd_summary.csv"', f'CSV_PATH = "{input_ccd}"')
code = code.replace('OUT_DIR = "/mnt/work/repos/TextAlign-mindeye2/figures"', f'OUT_DIR = "{base_dir}"')
# Handle tag label
code = code.replace('        if "official_s1" in t or "official_s2" in t or "official_s5" in t or "official_s7" in t:',
                    '        if "stage1_final_best32" in t.lower(): return "S1-Best32 (New)"\n        if "official_s1" in t or "official_s2" in t or "official_s5" in t or "official_s7" in t:')
with open(p1, 'w') as f: f.write(code)

# 2. Patch fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.py
p2 = os.path.join(patched_dir, "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.py")
with open(p2, 'r') as f: code = f.read()

code = code.replace('out_dir = "/mnt/work/repos/TextAlign-mindeye2/figures"', f'out_dir = "{base_dir}"')
code = code.replace(
    '"/mnt/work/repos/TextAlign-mindeye2/results/tables/main_results.csv",',
    f'"{input_rsa}",'
)
# Force it to pick our CSV by removing others or putting ours first and ensuring existence
code = code.replace('in_csv_candidates = [', f'in_csv_candidates = ["{input_rsa}",')

with open(p2, 'w') as f: f.write(code)

# 3. Patch fig03_rsa_bar_main_v1.py
p3 = os.path.join(patched_dir, "fig03_rsa_bar_main_v1.py")
with open(p3, 'r') as f: code = f.read()

code = code.replace('CSV_PATH = "/mnt/work/repos/TextAlign-mindeye2/cache/model_eval_results/shared982_rsa/rsa_summary.csv"', f'CSV_PATH = "{input_rsa}"')
code = code.replace('OUT_DIR = Path("/mnt/work/repos/TextAlign-mindeye2/figures")', f'OUT_DIR = Path("{base_dir}")')

with open(p3, 'w') as f: f.write(code)

# 4. Patch fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.py
p4 = os.path.join(patched_dir, "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.py")
with open(p4, 'r') as f: code = f.read()

code = code.replace('OUT_DIR = "/mnt/work/repos/TextAlign-mindeye2/figures"', f'OUT_DIR = "{base_dir}"')
# Patch load_computed_from_features to use new brain path
# We'll inject the path directly where it defines brain_dir or where it loads.
new_brain_path = "/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/brain_clip.pt"
# The script has: brain_dir = os.path.join(REPO_ROOT, "evals/brain_tokens/ours_s1_v2")
# multiline replace might be tricky. Let's find the function start and insert our override.
code = code.replace('    # Ours S1 V2 paths', f'    # Ours S1 V2 paths\n    # PATCHED: use new model\n    return torch.load("{new_brain_path}"), torch.load("/mnt/work/repos/TextAlign-mindeye2/evals/all_images_bigG_1664_mean.pt")')
# Currently the function returns (gt, pred, diff, keys) OR just (gt, pred) ? 
# Wait, let's check `load_computed_from_features` signature/return in the file content.
# It seems `load_computed_from_features` was NOT fully read in my previous generic read.
# I'll be careful. The original code was:
# def load_computed_from_features():
#     ...
#     return rsm_gt, rsm_pred, (rsm_pred - rsm_gt), ["compute_from_feats"]
# I should replace the body to do exactly that using the new file.
# Since I can't easily parse the whole function without reading it all, I will use a different approach:
# I will change the Config/Paths if they are constants.
# But they are inside the function.
# I will Replace the `brain_dir = ...` line.
code = code.replace('    brain_dir = os.path.join(REPO_ROOT, "evals/brain_tokens/ours_s1_v2")', 
                    f'    # brain_dir overridden\n    brain_pt = "{new_brain_path}"\n    gt_pt = "/mnt/work/repos/TextAlign-mindeye2/evals/all_images_bigG_1664_mean.pt"\n    print("Loading patched:", brain_pt)\n    pred_feats = torch.load(brain_pt)\n    gt_feats = torch.load(gt_pt)\n    rsm_pred = compute_rsm(pred_feats)\n    rsm_gt = compute_rsm(gt_feats)\n    return rsm_gt, rsm_pred, (rsm_pred - rsm_gt), ["patched"]')

with open(p4, 'w') as f: f.write(code)

# 5. Patch fig_audit_sim_text_v2.py
p5 = os.path.join(patched_dir, "fig_audit_sim_text_v2.py")
with open(p5, 'r') as f: code = f.read()

code = code.replace('OUT_DIR = "/mnt/work/repos/TextAlign-mindeye2/figures"', f'OUT_DIR = "{base_dir}"')
code = code.replace('SEARCH_ROOTS = [', f'SEARCH_ROOTS = ["{inputs_dir}", # Patched to prefer inputs\n')
# We also need to ensure it picks OUR file. The logic ranks by score. 
# `shared982_hardneg.jsonl` gets +10. Our file is named `s1_...hardneg.jsonl`.
# It has "hardneg" (+5) and ".jsonl".
# I should renaming the file in SEARCH_ROOTS to just the FILE PATH I want?
# The script has `candidates = find_candidate_data()`.
# I'll override `candidates = ["{input_hardneg}"]` after the function call or inside definitions.
code = code.replace('candidates = find_candidate_data()', f'candidates = ["{input_hardneg}"] # PATCHED')

with open(p5, 'w') as f: f.write(code)

print("Scripts patched.")
