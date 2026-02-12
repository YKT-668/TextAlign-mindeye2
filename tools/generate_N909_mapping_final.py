
import torch
import json
import numpy as np
import os
import pandas as pd

def main():
    # Paths
    n909_path = "cache/model_eval_results/shared982_ccd_assets/captions_openclip_ViT-bigG-14_laion2b_s39b_b160k_shared982_n909.pt"
    # The master set we are auditing against (represented by the IDs used in the inference)
    ids_path = "/mnt/work/tmp_infer_out_ours_s1_stage1_final_best32_lastpth_v1/ids.json"
    captions_path = "evals/all_captions.pt"
    shared1000_path = "src/shared1000.npy"
    
    print(f"Loading N909 Asset: {n909_path}")
    n909_data = torch.load(n909_path)
    # n909_data is likely a dict or tensor.
    if isinstance(n909_data, dict):
        # usually 'captions' or similar key
        if 'image_features' in n909_data: # If it's CLIP features
             n909_emb = n909_data['image_features']
        elif 'text_features' in n909_data:
             n909_emb = n909_data['text_features']
        else:
             # Just grab the first tensor
             n909_emb = next(v for v in n909_data.values() if isinstance(v, torch.Tensor))
    else:
        n909_emb = n909_data
        
    print(f"N909 Embedding Shape: {n909_emb.shape}")
    
    # Load Master Data
    print(f"Loading IDs: {ids_path}")
    ids_982 = json.load(open(ids_path)) # List of integers (Global IDs)
    n_982 = len(ids_982)
    print(f"Master IDs count: {n_982}")
    
    # We need to compute embeddings for the master set to match against n909_emb
    # OR we can assume we have embeddings somewhere.
    # The 'extract_n909_ids.py' referenced 'captions_openclip...n982.pt'.
    n982_ref_path = "cache/model_eval_results/shared982_ccd_assets/captions_openclip_ViT-bigG-14_laion2b_s39b_b160k_shared982_n982.pt"
    
    print(f"Loading Ref N982 Embeddings: {n982_ref_path}")
    n982_data = torch.load(n982_ref_path)
    if isinstance(n982_data, dict):
        n982_emb = next(v for v in n982_data.values() if isinstance(v, torch.Tensor))
    else:
        n982_emb = n982_data
        
    print(f"Ref N982 Embedding Shape: {n982_emb.shape}")
    
    # Check dimensions
    if n909_emb.shape[1] != n982_emb.shape[1]:
        print("Dimension mismatch! Trying to transpose?")
        # Logic to handle if needed
        pass
        
    # Matching
    # Ideally, n909 is a subset of n982.
    # We compute distance matrix.
    # Move to CPU for safety
    t909 = n909_emb.float().cpu()
    t982 = n982_emb.float().cpu()
    
    # Normalize?
    t909 = t909 / t909.norm(dim=-1, keepdim=True)
    t982 = t982 / t982.norm(dim=-1, keepdim=True)
    
    # Cosine Similarity
    sim = t909 @ t982.T
    
    print("Computed similarity matrix.")
    
    # Find best match for each 909 item
    best_vals, best_idxs = sim.max(dim=1)
    
    # Threshold check?
    matches = best_idxs.numpy()
    vals = best_vals.numpy()
    
    # Check if matches are good
    min_score = vals.min()
    print(f"Minimum Match Score: {min_score}")
    
    # Logic: The 'n982' file might not be in the same order as 'ids.json'.
    # We need to double check the order of 'ids_982'.
    # But usually 'n982.pt' corresponds to the shared982 dataset.
    # The 'ids.json' is ALSO corresponding to shared982.
    # Assuming they are 1-1 aligned. If not, we have a problem.
    # Result: 'matches' contains indices in [0, 981].
    
    # De-duplication
    unique_indices = np.unique(matches)
    print(f"Total Matches (Trials): {len(matches)}")
    print(f"Unique Matches (Images): {len(unique_indices)}")
    
    # Identify duplicates
    counts = {}
    for idx in matches:
        counts[idx] = counts.get(idx, 0) + 1
        
    dupes = {k:v for k,v in counts.items() if v > 1}
    print(f"Duplicate Indices in Target: {list(dupes.keys())} (Count: {len(dupes)})")
    
    # Save N901 Unique
    out_dir = "tables"
    os.makedirs(out_dir, exist_ok=True)
    
    out_json = os.path.join(out_dir, "ccd_used_ids_N901_unique.json")
    # Sort them for consistency
    unique_indices_sorted = sorted(unique_indices.tolist())
    with open(out_json, "w") as f:
        json.dump(unique_indices_sorted, f)
    print(f"Saved Unique Indices to {out_json}")
    
    # Save Mapping CSV
    # Columns: Trial_ID (0-908), Target_Index (0-981), Global_ID (from ids.json), Is_Duplicate
    mapping_data = []
    
    # We need Global IDs to map accurately
    # Warning: If n982.pt is not aligned with ids.json, Global IDs will be wrong.
    # But presumably they are both "shared982".
    
    for i, match_idx in enumerate(matches):
        global_id = ids_982[match_idx]
        is_dupe = counts[match_idx] > 1
        mapping_data.append({
            "Trial_ID": i,
            "Target_Index": int(match_idx),
            "Global_ID": global_id,
            "Similarity": float(vals[i]),
            "Is_Duplicate": is_dupe
        })
        
    df_map = pd.DataFrame(mapping_data)
    csv_path = os.path.join(out_dir, "mapping_909_to_901.csv")
    df_map.to_csv(csv_path, index=False)
    print(f"Saved Mapping to {csv_path}")
    
if __name__ == "__main__":
    main()
