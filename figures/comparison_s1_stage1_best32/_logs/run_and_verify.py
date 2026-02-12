import os
import subprocess
import json
import glob

base_dir = "/mnt/work/repos/TextAlign-mindeye2/figures/comparison_s1_stage1_best32"
patched_dir = os.path.join(base_dir, "_patched_generators")
logs_dir = os.path.join(base_dir, "_logs")

scripts = [
    "fig_audit_sim_text_v2.py",
    "run_fig01_forest_multirun_2x2_v6.py",
    "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.py",
    "fig03_rsa_bar_main_v1.py",
    "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.py"
]

fig_map = {
    "fig_audit_sim_text_v2.py": "Fig_audit_sim_text_v2.png",
    "run_fig01_forest_multirun_2x2_v6.py": "fig01_ccd_acc1_per_subj_dot_ci_2x2_v2.png",
    "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.py": "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.png",
    "fig03_rsa_bar_main_v1.py": "fig03_rsa_bar_pooled_main_v14.png",
    "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.py": "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.png"
}

print("Running generators...")
results = {}
for s in scripts:
    path = os.path.join(patched_dir, s)
    if not os.path.exists(path):
        print(f"Skipping {s}: Not found")
        continue

    print(f"Running {s}...")
    try:
        # Run in the patched_dir to resolve relative imports if any
        subprocess.run(["python", path], check=True, cwd=patched_dir)
        results[s] = "SUCCESS"
    except subprocess.CalledProcessError as e:
        print(f"Error running {s}: {e}")
        results[s] = "FAILED"

print("Verification...")
generated_files = []
missing_files = []

for s, expected_png in fig_map.items():
    # fig03 might produce v14 or something else depending on safe_save
    # We search for the base name
    stem = os.path.splitext(expected_png)[0]
    # glob in base_dir
    matches = glob.glob(os.path.join(base_dir, f"{stem}*.png"))
    if matches:
        generated_files.append(matches[0])
    else:
        missing_files.append(expected_png)

# Generate logs
with open(os.path.join(logs_dir, "fig_to_generator.json"), 'w') as f:
    json.dump({v: k for k, v in fig_map.items()}, f, indent=2)

input_map = {
    "Fig_audit_sim_text_v2.png": "s1_textalign_stage1_FINAL_BEST_32_hardneg.jsonl",
    "fig01_ccd_acc1_per_subj_dot_ci_2x2_v2.png": "ccd_summary.csv",
    "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.png": "rsa_summary.csv",
    "fig03_rsa_bar_pooled_main_v14.png": "rsa_summary.csv",
    "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.png": ["brain_clip.pt", "rsm_gt_mean.pt"]
}
with open(os.path.join(logs_dir, "input_files_used.json"), 'w') as f:
    json.dump(input_map, f, indent=2)

print("\nSUMMARY:")
for f in generated_files:
    print(f"[FOUND] {f}")
for m in missing_files:
    print(f"[MISSING] {m}")

if missing_files:
    with open(os.path.join(logs_dir, "need_user_help.txt"), 'w') as f:
        f.write("Missing figures:\n")
        for m in missing_files:
            f.write(f"- {m}\n")
else:
    print("All figures generated successfully.")

