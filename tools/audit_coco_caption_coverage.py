import argparse
import json
import os
import glob
import numpy as np
import webdataset as wds
import time

def main():
    parser = argparse.ArgumentParser(description="Audit COCO caption coverage for training data")
    parser.add_argument("--data_path", default="/mnt/work/mindeye_data_real", help="Path to data containing wds folder")
    parser.add_argument("--subjs", default="1,2,3,4,5,6,7,8", help="Comma separated subject IDs")
    parser.add_argument("--num_sessions", type=int, default=40, help="Number of sessions to look for (optional, uses glob if not robust)")
    parser.add_argument("--caption_json", default="data/nsd_text/train_coco_captions.json", help="Path to JSON containing annotated image_ids")
    parser.add_argument("--out", default="results/coco_caption_coverage.json", help="Output JSON path")
    args = parser.parse_args()

    # 1. Load Caption Map
    print(f"[INFO] Loading caption map from {args.caption_json}...")
    if not os.path.exists(args.caption_json):
        print(f"[ERROR] Caption file not found: {args.caption_json}")
        return

    with open(args.caption_json, 'r') as f:
        cap_data = json.load(f)
    
    # Adapt to structure: dict with 'image_ids' or list
    if isinstance(cap_data, dict) and "image_ids" in cap_data:
        caption_ids = set(cap_data["image_ids"])
    elif isinstance(cap_data, list):
         # Assuming list of dicts, or list of ids?
         # MindEye2 utils usually save as dict.
         print("[WARN] Unknown JSON structure, assuming keys or list of objects not supported directly. Trying to infer.")
         idx0 = cap_data[0]
         if isinstance(idx0, dict) and "image_id" in idx0:
             caption_ids = set(x["image_id"] for x in cap_data)
         else:
             print("[ERROR] Could not parse caption_json. Expected dict['image_ids'] or list.")
             return
    else:
        # Fallback for some MindEye generated jsons
        keys = list(cap_data.keys())
        if "image_ids" in keys:
            caption_ids = set(cap_data["image_ids"])
        else:
             print("[ERROR] Missing 'image_ids' key in JSON.")
             return

    print(f"[INFO] Found {len(caption_ids)} unique images with captions in JSON source.")

    # 2. Iterate Subjects
    subjects = [int(s) for s in args.subjs.split(",")]
    s_results = {}
    
    print("-" * 65)
    print(f"{'Subj':<6} {'N_Shards':<10} {'Unique_Train':<15} {'With_Caption':<15} {'Coverage':<10}")
    print("-" * 65)

    total_unique = 0
    total_matched = 0

    for subj in subjects:
        subj_str = f"subj{subj:02d}"
        # Flexible glob pattern for shards
        pattern = os.path.join(args.data_path, "wds", subj_str, "train", "*.tar")
        tar_files = sorted(glob.glob(pattern))

        if not tar_files:
            print(f"{subj:<6} {'0':<10} {'-':<15} {'-':<15} {'-':<10}")
            continue

        local_ids = set()

        # Iterate shards without decoding images
        # We assume common NPY format for behav
        # Using simple iteration to minimize memory
        try:
            # Use default decoder which handles .npy automatically
            ds = wds.WebDataset(tar_files).decode().to_tuple("behav.npy")
            count = 0
            for (behav,) in ds:
                # Shape logic: typically (1, 17) for single sample in MindEye wds
                if behav.ndim >= 2:
                    img_id = behav[0, 0]
                else:
                    img_id = behav[0]
                local_ids.add(int(img_id))
                count += 1
        except Exception as e:
            print(f"[WARN] Error reading shard for {subj_str}: {e}")
        
        n_unique = len(local_ids)
        n_match = len(local_ids.intersection(caption_ids))
        coverage = (n_match / n_unique * 100) if n_unique > 0 else 0.0

        print(f"{subj:<6} {len(tar_files):<10} {n_unique:<15} {n_match:<15} {coverage:.1f}%")
        
        s_results[subj_str] = {
            "n_shards": len(tar_files),
            "n_unique_train": n_unique,
            "n_with_caption": n_match,
            "coverage_pct": coverage
        }
        
        total_unique += n_unique
        total_matched += n_match

    # 3. Save
    print("-" * 65)
    print(f"TOTAL  {'-':<10} {total_unique:<15} {total_matched:<15} {(total_matched/total_unique*100 if total_unique else 0):.1f}%")
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(s_results, f, indent=2)
    print(f"\n[INFO] Saved detailed audit to {args.out}")

if __name__ == "__main__":
    main()
