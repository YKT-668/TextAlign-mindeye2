
import pickle
import numpy as np
import os
import torch
import glob
import webdataset as wds

def get_train_ids(subj):
    wds_root = f"/mnt/work/repos/TextAlign-mindeye2/wds/subj0{subj}/train"
    shards = sorted(glob.glob(os.path.join(wds_root, "*.tar")))
    ids = set()
    print(f"Subj{subj}: scanning {len(shards)} shards...")
    for shard in shards:
        try:
            ds = wds.WebDataset(shard).decode("torch").rename(behav="behav.npy").to_tuple("behav")
            dl = torch.utils.data.DataLoader(ds, batch_size=2048)
            for (behav,) in dl:
                # behav: [B, 1, 17], 73k-based ID is at [:, 0, 0]
                image_ids = behav[:, 0, 0].int().numpy()
                ids.update(image_ids.tolist())
        except Exception as e:
            print(f"Error reading shard {shard}: {e}")
            pass
    return ids

def main():
    # 1. Load Global StimInfo
    pkl_path = "/mnt/work/repos/TextAlign-mindeye2/nsd_stim_info_merged.pkl"
    print(f"Loading {pkl_path}...")
    obj = pickle.load(open(pkl_path, "rb"), encoding='latin1')
    
    # 2. Get Train IDs for S1, S2, S5, S7
    subs = [1, 2, 5, 7]
    train_sets = {}
    
    for s in subs:
        train_sets[s] = get_train_ids(s)
        print(f"Subj{s}: Found {len(train_sets[s])} unique training images.")

    # 3. Check Overlaps
    s1_ids = train_sets[1]
    
    for s in [2, 5, 7]:
        overlap = s1_ids.intersection(train_sets[s])
        print(f"Overlap S1 vs S{s}: {len(overlap)} images.")

    # 4. Check Shared1000 Overlap
    shared_path = "/mnt/work/mindeye_data_real/shared1000.npy"
    if os.path.exists(shared_path):
        shared_mask = np.load(shared_path)
        shared_ids = set(np.where(shared_mask)[0].tolist())
        print(f"Shared IDs Total: {len(shared_ids)}")
        
        for s in subs:
            overlap = train_sets[s].intersection(shared_ids)
            print(f"Subj{s} Train vs Shared1000 Overlap: {len(overlap)}")
    else:
        print("Shared1000 file not found.")

    # 5. Check actual S1 teacher file
    teacher_path = "data/nsd_text/train_coco_text_clip.pt"
    if os.path.exists(teacher_path):
        t = torch.load(teacher_path, map_location="cpu")
        t_ids = set(t["image_ids"].tolist())
        print(f"S1 Actual Teacher IDs: {len(t_ids)}")
        print(f"S1 Teacher vs S1 Train: {len(t_ids.intersection(s1_ids))} (Should be 9000)")
        print(f"S1 Teacher vs S2 Train: {len(t_ids.intersection(train_sets[2]))}")
    
if __name__ == "__main__":
    main()
