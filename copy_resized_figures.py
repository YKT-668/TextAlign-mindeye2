
import shutil
import os
from pathlib import Path

# Base paths
repo = Path("/mnt/work/repos/TextAlign-mindeye2")
dest = repo / "figures/resized_figures"
dest.mkdir(parents=True, exist_ok=True)

# Sources
eff_dir = repo / "results/figures_main"
ccd_dir = repo / "cache/model_eval_results/shared982_ccd/figures"
isrsa_dir = repo / "cache/model_eval_results/shared982_isrsa/figures"
comp_dir = repo / "figures/comparison_s1_stage1_best32"

# Mapping: Source Name -> Target Name (in dest)
# Source directory key: 1=eff, 3=ccd, 4=isrsa, 6=comp
files_map = [
    (1, "Fig_efficiency_ccd_acc1.png", "Fig_efficiency_ccd_acc1_v1.png"),
    (1, "Fig_efficiency_twoafc_hard.png", "Fig_efficiency_twoafc_hard_v2.png"),
    (6, "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.png", "fig01_rsa_pearson_pooled_by_subj_v2_dumbbell.png"),
    (6, "fig03_rsa_bar_pooled_main.png", "fig03_rsa_bar_pooled_main_v14.png"),
    (6, "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.png", "fig06_rsm_heatmaps_subj01_best_overall_pooled_v3.png"),
    (6, "fig01_ccd_acc1_per_subj_dot_ci_2x2_v2.png", "fig01_ccd_acc1_per_subj_dot_ci_2x2_v2.png"),
    (3, "Fig09_ccd_ablation_difficulty.png", "Fig09_ccd_ablation_difficulty_v3_diverging.png"),
    (4, "Fig_isrsa_heatmap_textalign_llm.png", "Fig_isrsa_heatmap_textalign_llm_v2.png"),
    (6, "Fig_audit_sim_text_v2.png", "Fig_audit_sim_text_v2.png"),
]

dirs = {1: eff_dir, 3: ccd_dir, 4: isrsa_dir, 6: comp_dir}

report = []

for dir_key, src_name, dst_name in files_map:
    src_dir = dirs[dir_key]
    src = src_dir / src_name
    dst = dest / dst_name

    if src.exists():
        shutil.copy2(src, dst)
        report.append(f"SUCCESS: Copied {src_name} to {dst.name}")
    else:
        # Fallback for fig03 if user had generated v versions we missed
        if "fig03" in src_name:
             potentials = sorted(list(src_dir.glob("fig03*.png")))
             if potentials:
                 # Check if we should pick the last one
                 best_m = potentials[-1]
                 shutil.copy2(best_m, dst)
                 report.append(f"SUCCESS: Copied {best_m.name} (fallback) to {dst.name}")
                 continue

        report.append(f"FAILURE: Could not find source {src}")

print("\n".join(report))
