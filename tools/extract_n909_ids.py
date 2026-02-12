
import torch
import json
import os
import numpy as np

n909_path = "cache/model_eval_results/shared982_ccd_assets/captions_openclip_ViT-bigG-14_laion2b_s39b_b160k_shared982_n909.pt"
n982_path = "cache/model_eval_results/shared982_ccd_assets/captions_openclip_ViT-bigG-14_laion2b_s39b_b160k_shared982_n982.pt"

print(f"Loading {n909_path}...")
n909_data = torch.load(n909_path)
print(f"Type n909: {type(n909_data)}")
if isinstance(n909_data, torch.Tensor):
    print(f"Shape n909: {n909_data.shape}")
elif isinstance(n909_data, dict):
    print(f"Keys n909: {n909_data.keys()}")
    for k, v in n909_data.items():
        if hasattr(v, 'shape'):
             print(f"Shape n909[{k}]: {v.shape}")

print(f"Loading {n982_path}...")
n982_data = torch.load(n982_path)
print(f"Type n982: {type(n982_data)}")
if isinstance(n982_data, torch.Tensor):
    print(f"Shape n982: {n982_data.shape}")
elif isinstance(n982_data, dict):
    print(f"Keys n982: {n982_data.keys()}")
    for k, v in n982_data.items():
        if hasattr(v, 'shape'):
             print(f"Shape n982[{k}]: {v.shape}")

# If they are tensors (embeddings), we can try to match them.
# Handle dict case
tensor_n909 = None
tensor_n982 = None

if isinstance(n909_data, dict):
     # Pick the first tensor
     for v in n909_data.values():
         if isinstance(v, torch.Tensor):
             tensor_n909 = v
             break
else:
     tensor_n909 = n909_data

if isinstance(n982_data, dict):
     for v in n982_data.values():
         if isinstance(v, torch.Tensor):
             tensor_n982 = v
             break
else:
     tensor_n982 = n982_data

if isinstance(tensor_n909, torch.Tensor) and isinstance(tensor_n982, torch.Tensor):
    n909_np = tensor_n909.cpu().numpy()
    n982_np = tensor_n982.cpu().numpy()
    
    valid_indices = []
    
    for i in range(len(n909_np)):
        vec = n909_np[i]
        # Find this vec in n982
        # Use simple difference
        diff = np.sum(np.abs(n982_np - vec), axis=1)
        match_idx = np.argmin(diff)
        
        # Relaxed threshold
        if diff[match_idx] < 0.1:
            valid_indices.append(int(match_idx))
        else:
            print(f"Warning: No match found for n909 index {i}, min diff {diff[match_idx]}")
            
    # Check uniqueness
    from collections import Counter
    counts = Counter(valid_indices)
    duplicates = [k for k, v in counts.items() if v > 1]
    
    unique_indices = sorted(list(set(valid_indices)))
    print(f"Found {len(valid_indices)} matches, {len(unique_indices)} unique.")
    print(f"Duplicates in n982 mapping: {duplicates}")
    
    # Check meta if available
    if isinstance(n909_data, dict) and 'meta' in n909_data:
         # assuming meta is list or similar
         pass

    # If we have exactly 909 "valid" indices even if not unique (maybe some images are repeated?), 
    # we can define the subset as "indices in n982 that are used".
    # BUT, duplicate usage suggests 1 image used effectively twice?
    # Or strict matching failed and we mapped 2 different n909 to same n982 because of threshold.
    
    # Let's inspect the distance for duplicates
    # ...
    
    output_path = "tables/ccd_used_ids_N909.json"
    
    # Force saving uniqueness for now
    with open(output_path, "w") as f:
        json.dump(unique_indices, f)
    print(f"Saved {len(unique_indices)} unique indices to {output_path}")

