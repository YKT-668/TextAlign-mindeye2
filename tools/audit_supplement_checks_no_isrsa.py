import torch
import numpy as np
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain_path", required=True)
    parser.add_argument("--ids_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--shared1000_path", required=True)
    parser.add_argument("--out_manifest", required=True)
    args = parser.parse_args()

    print("=== Audit Sanity Check (No IS-RSA) ===")
    
    # 1. Load Brain
    print(f"Loading Brain: {args.brain_path}")
    brains = torch.load(args.brain_path, map_location="cpu")
    print(f"Brain Shape: {brains.shape}")
    
    if brains.shape != (982, 1664):
        print(f"FAIL: Brain shape mismatch. Expected (982, 1664), got {brains.shape}")
        return
    else:
        print("PASS: Brain shape correct.")

    # Norm check
    norms = torch.norm(brains, dim=1)
    if (norms < 1e-4).any():
        print("FAIL: Found near-zero vectors in brain embeddings.")
    else:
        print(f"PASS: Norms check healthy. Mean={norms.mean():.4f}")

    # 2. Load IDs
    print(f"Loading IDs: {args.ids_path}")
    with open(args.ids_path) as f:
        ids = np.asarray(json.load(f))
    print(f"IDs Count: {len(ids)}")
    
    if len(ids) != 982:
        print(f"FAIL: IDs count mismatch. Expected 982, got {len(ids)}")
        return
        
    if len(np.unique(ids)) != 982:
        print("FAIL: Duplicate IDs found.")
        return
    else:
        print("PASS: IDs unique.")

    # 3. Split Check (Shared1000)
    print(f"Loading Shared1000: {args.shared1000_path}")
    shared_mask = np.load(args.shared1000_path)
    if shared_mask.dtype == bool:
        shared_metrics = np.where(shared_mask)[0]
    else:
        shared_metrics = shared_mask
        
    print(f"Shared1000 Size: {len(shared_metrics)}")
    
    overlap = np.intersect1d(ids, shared_metrics)
    if len(overlap) != 982:
        print(f"FAIL: Only {len(overlap)} IDs match shared1000.")
        missing = np.setdiff1d(ids, shared_metrics) # present in ids but not in shared
        if len(missing) > 0:
             print(f"IDs not in shared1000: {missing}")
    else:
        print("PASS: All 982 IDs belong to shared1000.")

    # 4. Save Manifest
    import pandas as pd
    df = pd.DataFrame({"id": ids})
    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)
    df.to_csv(args.out_manifest, index=False)
    print(f"Saved manifest to {args.out_manifest}")

    # 5. GT Check
    print(f"Loading GT: {args.gt_path}")
    gt_obj = torch.load(args.gt_path, map_location="cpu")
    if isinstance(gt_obj, dict):
         # Try to find tensor
         vals = [v for v in gt_obj.values() if torch.is_tensor(v) and v.ndim==2]
         if vals: gt_feats = vals[0]
         else: 
             print("FAIL: No GT tensor found in dict")
             return
    else:
        gt_feats = gt_obj
        
    if gt_feats.shape[1] != 1664:
        print(f"FAIL: GT dim mismatch. Expected 1664, got {gt_feats.shape[1]}")
    else:
         print(f"PASS: GT Dimension matches (1664). Count={gt_feats.shape[0]}")

if __name__ == "__main__":
    main()
