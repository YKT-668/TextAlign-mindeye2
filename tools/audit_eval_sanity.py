import torch
import numpy as np
import json
import os
import sys

def load_ids(path):
    with open(path) as f:
        return np.asarray(json.load(f))

def main():
    print("=== Audit Sanity Check ===")
    
    # Paths
    brain_path = "/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/brain_clip.pt"
    ids_path = "/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/ids.json"
    gt_path = "/mnt/work/repos/TextAlign-mindeye2/evals/all_images_bigG_1664_mean.pt"
    shared1000_path = "/mnt/work/repos/TextAlign-mindeye2/src/shared1000.npy"
    
    # 1. Load Brain
    print(f"Loading Brain: {brain_path}")
    brains = torch.load(brain_path, map_location="cpu")
    print(f"Brain Shape: {brains.shape}")
    
    # 2. Load IDs
    print(f"Loading IDs: {ids_path}")
    ids = load_ids(ids_path)
    print(f"IDs Count: {len(ids)}")
    print(f"First 10 IDs: {ids[:10]}")
    
    # 3. Load GT
    print(f"Loading GT: {gt_path}")
    gt_obj = torch.load(gt_path, map_location="cpu")
    
    gt_feats = None
    gt_ids = None
    if isinstance(gt_obj, dict):
        # Infer
        for k in ["ids", "image_ids", "nsd_ids", "coco_ids"]:
            if k in gt_obj: 
                gt_ids = np.asarray(gt_obj[k])
                break
        for v in gt_obj.values():
            if torch.is_tensor(v) and v.ndim == 2:
                gt_feats = v
                break
    elif isinstance(gt_obj, torch.Tensor):
        gt_feats = gt_obj
        # Try finding IDs separately?
        pass
        
    if gt_feats is not None:
        print(f"GT Feats Shape: {gt_feats.shape}")
    else:
        print("FAIL: GT Features not found")
        
    if gt_ids is not None:
        print(f"GT IDs Count: {len(gt_ids)}")
    else:
        print("WARN: GT IDs not in .pt file")
        
    # 4. Check Shared1000
    if os.path.exists(shared1000_path):
        shared_mask = np.load(shared1000_path)
        if shared_mask.dtype == bool:
            shared_metrics = np.where(shared_mask)[0]
        else:
            shared_metrics = shared_mask
        print(f"Shared1000 Count: {len(shared_metrics)}")
        
        # Check Overlap
        overlap = np.intersect1d(ids, shared_metrics)
        print(f"Overlap with Shared1000: {len(overlap)}")
        
        if len(overlap) == 982:
            print("PASS: IDs match strict shared982 protocol.")
        else:
            print(f"FAIL: IDs match {len(overlap)}/982 of shared1000.")
            missing = np.setdiff1d(shared_metrics, ids)
            print(f"Missing IDs: {missing}")

    # 5. Check Norms
    brain_norms = torch.norm(brains, dim=1)
    print(f"Brain Norms: mean={brain_norms.mean().item():.4f}, std={brain_norms.std().item():.4f}")
    if brain_norms.mean() < 0.1:
        print("WARN: Brain norms very small. Scale issue?")
        
    # 6. Check S2/S5/S7
    subjects = [2, 5, 7]
    for subj in subjects:
        path = f"/mnt/work/repos/TextAlign-mindeye2/evals/brain_tokens/official_hf/final_subj0{subj}_pretrained_40sess_24bs/subj0{subj}_brain_clip_mean.pt"
        if os.path.exists(path):
            s_emb = torch.load(path, map_location="cpu")
            print(f"Subj {subj} Embedding: Found. Shape={s_emb.shape}")
        else:
            print(f"Subj {subj} Embedding: MISSING")

if __name__ == "__main__":
    main()
