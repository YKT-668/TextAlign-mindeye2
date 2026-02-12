import os
import subprocess
import shutil
import json

base_dir = "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32"
patched_dir = os.path.join(base_dir, "_patched_generators")
logs_dir = os.path.join(base_dir, "_logs")

# 1. Rerun fig06
print("Rerunning fig06...")
try:
    subprocess.run(["python", "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.py"], cwd=patched_dir, check=True)
    print("Fig06 Success.")
except Exception as e:
    print(f"Fig06 Failed: {e}")

# 2. Rename fig03
src = os.path.join(base_dir, "fig03_rsa_bar_pooled_main.png")
dst = os.path.join(base_dir, "fig03_rsa_bar_pooled_main_v14.png")
if os.path.exists(src):
    shutil.move(src, dst)
    print("Renamed fig03 to v14.")
    # Also PDF if exists
    src_pdf = os.path.join(base_dir, "fig03_rsa_bar_pooled_main.pdf")
    if os.path.exists(src_pdf):
        shutil.move(src_pdf, os.path.join(base_dir, "fig03_rsa_bar_pooled_main_v14.pdf"))

# 3. Final Verification Log
expected = [
    "Fig_audit_sim_text_v2.png",
    "fig01_ccd_acc1_per_subj_dot_ci_2x2_v2.png",
    "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.png",
    "fig03_rsa_bar_pooled_main_v14.png",
    "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.png"
]

found = []
for f in expected:
    if os.path.exists(os.path.join(base_dir, f)):
        found.append(f)
    elif f == "fig03_rsa_bar_pooled_main_v14.png" and os.path.exists(os.path.join(base_dir, "fig03_rsa_bar_pooled_main.png")):
         # Handle case where rename didn't happen (should not happen if code above runs)
         found.append(f + "(as v1)")

print("FOUND:", found)

# 4. Generate Fig To Generator Map
fig_map = {
    "Fig_audit_sim_text_v2.png": "fig_audit_sim_text_v2.py",
    "fig01_ccd_acc1_per_subj_dot_ci_2x2_v2.png": "run_fig01_forest_multirun_2x2_v6.py",
    "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.png": "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.py",
    "fig03_rsa_bar_pooled_main_v14.png": "fig03_rsa_bar_main_v1.py",
    "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.png": "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.py"
}
with open(os.path.join(logs_dir, "fig_to_generator.json"), 'w') as f:
    json.dump(fig_map, f, indent=2)

# 5. Input Files Used
input_map = {
    "Fig_audit_sim_text_v2.png": "s1_textalign_stage1_FINAL_BEST_32_hardneg.jsonl",
    "fig01_ccd_acc1_per_subj_dot_ci_2x2_v2.png": "ccd_summary.csv",
    "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.png": "rsa_summary.csv",
    "fig03_rsa_bar_pooled_main_v14.png": "rsa_summary.csv",
    "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.png": ["brain_clip.pt", "all_images_bigG_1664_mean.pt"]
}
with open(os.path.join(logs_dir, "input_files_used.json"), 'w') as f:
    json.dump(input_map, f, indent=2)

print("Finalize Complete.")
