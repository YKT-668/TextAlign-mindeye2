
import torch
import numpy as np
import os
import h5py

def check_intersection():
    # 1. Load Teacher IDs (Subj01 9000 set)
    teacher_path = "data/nsd_text/train_coco_text_clip.pt"
    if not os.path.exists(teacher_path):
        print(f"Teacher file not found: {teacher_path}")
        return

    teacher = torch.load(teacher_path, map_location="cpu")
    teacher_ids = set(teacher["image_ids"].tolist())
    print(f"Teacher (Subj01) IDs count: {len(teacher_ids)}")

    # 2. Get Shared IDs
    # shared1000.npy contains the 73k-indices of shared images
    shared_path = "/mnt/work/mindeye_data_real/shared1000.npy"
    if os.path.exists(shared_path):
        shared_bool = np.load(shared_path)
        print(f"Shared Boolean Mask Shape: {shared_bool.shape}")
        items_shared = np.where(shared_bool)[0]
        print(f"Shared Count (True): {len(items_shared)}")
        
        # Intersection between Shared and Teacher
        intersect_shared = teacher_ids.intersection(set(items_shared.tolist()))
        print(f"Intersection (Teacher & Shared): {len(intersect_shared)}")
    else:
        print("Shared IDs file not found.")

    # 3. Check hard negatives count
    hard_neg_path = "data/nsd_text/train_coco_captions_hard_negs_clip.pt"
    if os.path.exists(hard_neg_path):
        hn = torch.load(hard_neg_path, map_location="cpu")
        hn_ids = set(hn["image_ids"].tolist())
        print(f"Hard Neg IDs count: {len(hn_ids)}")
        print(f"Hard Neg matches Teacher exactly? {teacher_ids == hn_ids}")
    
    # 4. Check Subj02 specific IDs if possible?
    # Usually we need subject-specific stimulus info, but just checking Shared is a good proxy for S2 overlap 
    # since S2 only shares 'shared1000' with S1, unless S2 has its own unique COCO-annotated images which happen to be in S1's set (impossible)
    # or if S2 has COCO images that are NOT in shared but ARE in S1 unique? (unlikely, unique means unique).
    
    # So the overlap for S2 should be exactly (Teacher & Shared).

if __name__ == "__main__":
    check_intersection()
