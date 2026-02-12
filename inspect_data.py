import json
import torch
import os
from pathlib import Path

def check_positives():
    print("--- 1. Checking Positive Captions ---")
    p = Path("data/nsd_text/train_coco_captions.json")
    if not p.exists():
        print(f"File not found: {p}")
        return
    
    try:
        obj = json.loads(p.read_text())
        assert isinstance(obj, dict), type(obj)
        image_ids = obj.get("image_ids", [])
        captions = obj.get("captions", [])
        print("len(image_ids) =", len(image_ids))
        print("len(captions)  =", len(captions))
        if len(image_ids) > 0:
            print("first 5 image_ids:", image_ids[:5])
        assert len(image_ids) == len(captions)
    except Exception as e:
        print(f"Error checking positives: {e}")

def check_hard_negatives():
    print("\n--- 2. Checking Hard Negatives (Explicit File) ---")
    # This file was found in the search and referenced in training script
    p = Path("data/nsd_text/train_coco_captions_hard_negs_clip.pt")
    
    if not p.exists():
        print(f"File not found: {p}")
        # Try JSONL version if PT doesn't exist
        p_json = Path("data/nsd_text/train_coco_captions_hard_negs.jsonl")
        if p_json.exists():
            print(f"Found JSONL instead: {p_json}")
            # Just count lines
            with open(p_json) as f:
                n = sum(1 for _ in f)
            print(f"Lines in JSONL: {n}")
        return

    try:
        print(f"Loading {p}...")
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict):
            print("Keys:", list(obj.keys()))
            
            # Helper to print info
            def stats(key):
                if key in obj:
                    val = obj[key]
                    if isinstance(val, (list, tuple)):
                        print(f"{key}: len={len(val)}")
                        if len(val) > 0: print(f"  sample: {val[0]}")
                    elif isinstance(val, torch.Tensor):
                        print(f"{key}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"{key}: type={type(val)}")
            
            # Likely structure?
            # From filename ...hard_negs_clip.pt, it might contain precomputed CLIP embeddings? 
            # Or tokenized text?
            # User wants to know "How many explicit hard negative pairs".
            
            for k in obj.keys():
                stats(k)
                
            # Logic to infer coverage
            # If it has 'image_ids'
            if "image_ids" in obj:
                ids = obj["image_ids"]
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                unique_ids = set(ids)
                print(f"\nUnique Image IDs in Hard Neg file: {len(unique_ids)}")
                
                # Check negatives count per image
                # Often hard neg files have [N, K] structure or list of lists
                if "caption_ids" in obj and isinstance(obj["caption_ids"], torch.Tensor): # Maybe?
                     pass
                elif "hard_negs" in obj: # Maybe?
                     pass
                
                # If there is a huge tensor like [N, K, D], then K is num hard negs.
                
        else:
            print(f"Loaded object is {type(obj)}, expected dict.")

    except Exception as e:
        print(f"Error loading {p}: {e}")

if __name__ == "__main__":
    check_positives()
    check_hard_negatives()
