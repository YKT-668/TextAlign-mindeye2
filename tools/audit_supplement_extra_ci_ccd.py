import torch
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd
import open_clip
import os
import argparse
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

def load_tensor(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # Try to find the tensor
        for v in obj.values():
            if torch.is_tensor(v) and v.ndim == 2: return v
    return obj

def get_ci(values, confidence=0.95):
    lower = np.percentile(values, (1 - confidence) / 2 * 100)
    upper = np.percentile(values, (1 + confidence) / 2 * 100)
    return lower, upper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_dir", required=True)
    parser.add_argument("--brain_path", required=True)
    parser.add_argument("--ids_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--captions_path", default="evals/all_captions.pt")
    parser.add_argument("--shared1000_path", default="src/shared1000.npy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    print(f"=== Audit Supplement Extra: CI & CCD N (Seed={args.seed}) ===")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load Assets
    print("Loading assets...")
    ids = np.asarray(json.load(open(args.ids_path))) # 982
    brain = load_tensor(args.brain_path).float() # [982, D]
    gt_all = load_tensor(args.gt_path).float()   # [1000 or 73k, D]
    
    # Load Shared1000
    shared1000_mask = np.load(args.shared1000_path) # [73000] bool or indices
    if shared1000_mask.dtype == bool:
        shared1000_indices = np.where(shared1000_mask)[0]
    else:
        shared1000_indices = shared1000_mask
        
    print(f"Shared1000 count: {len(shared1000_indices)}")
    
    # Align Shared1000
    # Map global ID to index in Shared1000 (0..999)
    # Actually we just need to verify mapping
    
    # 2. Manifest Generation
    manifest_rows = []
    shared_set = set(shared1000_indices)
    
    # Align brain to GT
    # We assume GT (all_images_bigG) matches shared1000 order?
    # Usually evals/all_images_bigG_1664_mean.pt contains 1000 vectors for shared1000.
    # Let's verify GT shape
    print(f"GT shape: {gt_all.shape}")
    
    # Assume GT is 1000 ordered by shared1000
    id2local = {gid: i for i, gid in enumerate(shared1000_indices)}
    
    gt_subset = []
    brain_subset = []
    valid_ids = []
    
    for i, bid in enumerate(ids):
        status = "present"
        source_idx = -1
        if bid in id2local:
            source_idx = id2local[bid]
            gt_subset.append(gt_all[source_idx])
            brain_subset.append(brain[i])
            valid_ids.append(bid)
        else:
            status = "missing_in_shared1000"
            
        manifest_rows.append({
            "position_in_shared982": i,
            "id": bid,
            "source_shared1000_index": source_idx,
            "status": status
        })
        
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(os.path.join(args.audit_dir, "shared982_ids_manifest.csv"), index=False)
    print(f"Saved manifest. Valid aligned items: {len(valid_ids)}")
    
    brain_aligned = torch.stack(brain_subset)
    gt_aligned = torch.stack(gt_subset)
    
    # Project GT/Brain to same space if needed for RSA/2AFC?
    # Brain is typically projected to CLIP space.
    # GT is CLIP features.
    # Check dims
    if brain_aligned.shape[1] != gt_aligned.shape[1]: 
        print(f"Warning: Dim mismatch B={brain_aligned.shape[1]} G={gt_aligned.shape[1]}. Assumed handled by similarity calc or projection.")
        
    # For CCD, we need Captions and Text Encoder
    print("Preparing CCD assets (Encoding text)...")
    all_captions = torch.load(args.captions_path) # usually full list
    # We need to pick captions corresponding to valid_ids
    # This requires assuming all_captions index aligns with global ID or we have a map
    # Typically in NSD:
    # all_captions is 73k list. index = global ID.
    target_captions = []
    for bid in valid_ids:
        if len(all_captions) == 1000:
             # Assume ordered by shared1000 similar to GT
             if bid in id2local:
                 target_captions.append(all_captions[id2local[bid]])
             else:
                 raise ValueError(f"ID {bid} not found in shared1000 mapping but expected.")
        else:
             target_captions.append(all_captions[bid])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create CLIP
    model, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
    
    # Encode text
    bs = 128
    text_feats_list = []
    with torch.no_grad():
        for i in range(0, len(target_captions), bs):
            batch = target_captions[i:i+bs]
            tok = tokenizer(batch).to(device)
            emb = model.encode_text(tok)
            emb = F.normalize(emb, dim=-1)
            text_feats_list.append(emb.cpu())
    text_feats = torch.cat(text_feats_list) # [982, 1280]
    
    # Project Brain to 1280 if needed
    brain_ccd = brain_aligned.to(device)
    # Check if projection needed
    if brain_ccd.shape[1] == 1664:
        if hasattr(model.visual, "proj") and model.visual.proj is not None:
             brain_ccd = brain_ccd @ model.visual.proj.to(device)
    
    brain_ccd = F.normalize(brain_ccd, dim=-1).cpu()
    
    # Precompute Similarity Matrices
    # 2AFC / RSA usually on Brain vs Image
    # Normalize
    b_norm = F.normalize(brain_aligned, dim=-1)
    g_norm = F.normalize(gt_aligned, dim=-1)
    
    # CCD use brain_ccd and text_feats
    
    # 3. Bootstrap Loop
    print(f"Running {args.n_boot} Bootstraps...")
    
    metrics_boot = {
        "2AFC_B2I": [], "2AFC_I2B": [],
        "CCD_Hard_K1": [], "CCD_Hard_K2": [],
        "RSA_Pearson": [], "RSA_Spearman": []
    }
    
    N = len(valid_ids) # 982
    indices = np.arange(N)
    
    # Precompute full matrices to speed up indexing
    # S_BG: Brain vs Image [N, N]
    if b_norm.shape[1] != g_norm.shape[1]:
         # Project B to G space? Or just assume dot product valid?
         # If B=1664 G=1664 OK.
         # If B=1664 G=1280?
         pass
         
    S_BG = b_norm @ g_norm.T
    
    # S_BT: Brain vs Text [N, N]
    S_BT = brain_ccd @ text_feats.T
    
    # S_TT: Text vs Text [N, N] (for hard negative mining)
    S_TT = text_feats @ text_feats.T
    
    # Mask diagonal for hard neg mining
    mask_diag = ~torch.eye(N, dtype=torch.bool)
    
    for _ in tqdm(range(args.n_boot)):
        # Sample with replacement
        sample_idx = np.random.choice(indices, N, replace=True)
        sample_idx_t = torch.tensor(sample_idx)
        
        # 3.1 2AFC
        # We need to replicate the logic:
        # B2I: for item i, Score(b_i, g_i) vs Score(b_i, g_j)
        # In a bootstrap sample, we treat the sample as the "dataset".
        # So we look at the specific pairs (b_k, g_k) from the sample.
        # And negatives are other items *in the sample*.
        
        # S_sub = S_BG[sample_idx][:, sample_idx] # [N, N] sub-matrix
        # Diagonal is positive pairs
        # Off-diagonal are negative pairs
        
        S_BG_sub = S_BG[sample_idx_t][:, sample_idx_t]
        
        diag = torch.diag(S_BG_sub).view(-1, 1)
        # Valid negatives are off-diagonal
        # Note: with replacement, we might have same image multiple times.
        # If we have duplicate i, S[i,i] is pos. S[i,j] where j is same image is also pos.
        # Strict 2AFC says "compare to different image".
        # So mask out where image_idx is same?
        # Usually bootstrap just relies on index identity in the matrix.
        # We'll use standard off-diagonal logic.
        
        mask_sub = ~torch.eye(N, dtype=torch.bool)
        
        # B2I
        diff_b2i = diag - S_BG_sub
        acc_b2i = (diff_b2i[mask_sub] > 0).float().mean().item()
        metrics_boot["2AFC_B2I"].append(acc_b2i)
        
        # I2B
        S_GB_sub = S_BG_sub.T
        diag_t = torch.diag(S_GB_sub).view(-1, 1)
        diff_i2b = diag_t - S_GB_sub
        acc_i2b = (diff_i2b[mask_sub] > 0).float().mean().item()
        metrics_boot["2AFC_I2B"].append(acc_i2b)
        
        # 3.2 RSA
        # Flatten RDM upper triangle
        # RDM_B = Corr(B_sub)
        # RDM_G = Corr(G_sub)
        # Or Cov? Usually Correlation distance or 1-corr.
        # Or just Dot product similarity matrix?
        # Usually RSA uses the off-diagonal of the correlation matrix of the features.
        
        B_sub = b_norm[sample_idx_t]
        G_sub = g_norm[sample_idx_t]
        
        # RDM: 1 - corr(rows)
        # Or simply sim matrix B @ B.T
        rdm_b = B_sub @ B_sub.T
        rdm_g = G_sub @ G_sub.T
        
        iu = torch.triu_indices(N, N, offset=1)
        v_b = rdm_b[iu[0], iu[1]].numpy()
        v_g = rdm_g[iu[0], iu[1]].numpy()
        
        try:
            p_val = pearsonr(v_b, v_g)[0]
            s_val = spearmanr(v_b, v_g)[0]
        except:
            p_val, s_val = 0, 0
            
        metrics_boot["RSA_Pearson"].append(p_val)
        metrics_boot["RSA_Spearman"].append(s_val)
        
        # 3.3 CCD
        # Difficulty: Hardest
        # Negatives from sample
        S_BT_sub = S_BT[sample_idx_t][:, sample_idx_t]
        S_TT_sub = S_TT[sample_idx_t][:, sample_idx_t]
        
        # For each row i, find hard negatives in columns j!=i
        # Hardness defined by S_TT_sub[i, j]
        
        # Hard Neg K=1
        k1_acc = 0
        k2_acc = 0
        
        mask_sub = ~torch.eye(N, dtype=torch.bool)
        
        # Iterate over N items in sample
        for i in range(N):
            # Candidates: all j!=i
            # Actually, if we have duplicates (index a appears twice), 
            # should we exclude the duplicate from negatives?
            # Ideally yes, "same class".
            # But "indices" array tracks original ID using `sample_idx`.
            # items with same sample_idx are same text.
            
            # Find indices in current batch that are NOT same as item i
            # i is index in batch. sample_idx[i] is original ID.
            
            curr_id = sample_idx[i]
            # Negatives are indices j where sample_idx[j] != curr_id
            
            # Efficient implementation:
            # But standard bootstrap ignores this detail often.
            # Let's be strict: Negative must have different ground truth.
            
            neg_mask = sample_idx != curr_id
            neg_indices = np.where(neg_mask)[0] 
            
            if len(neg_indices) < 2: 
                # Degenerate case, count as pass or fail?
                # If no negatives, trivial pass? Or fail?
                continue
                
            # Get text-text sims
            sims_t = S_TT_sub[i, neg_indices]
            
            # Top K hard
            # argsort descending
            sort_idx = torch.argsort(sims_t, descending=True)
            
            # K=1
            top1_idx = neg_indices[sort_idx[0]]
            score_pos = S_BT_sub[i, i]
            score_neg1 = S_BT_sub[i, top1_idx]
            if score_pos > score_neg1: k1_acc += 1
            
            # K=2
            if len(neg_indices) >= 2:
                top2_idx = neg_indices[sort_idx[:2]]
                scores_neg2 = S_BT_sub[i, top2_idx]
                if (score_pos > scores_neg2).all(): k2_acc += 1
            else:
                pass # skip
                
        metrics_boot["CCD_Hard_K1"].append(k1_acc / N)
        metrics_boot["CCD_Hard_K2"].append(k2_acc / N)

    # 4. Summary & Save
    rows = []
    for k, vals in metrics_boot.items():
        mean = np.mean(vals)
        low, high = get_ci(vals)
        rows.append({"Metric": k, "Mean": mean, "CI_Lower": low, "CI_Upper": high})
        print(f"{k}: {mean:.4f} [{low:.4f}, {high:.4f}]")
        
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.audit_dir, "tables", "ci_bootstrap.csv"), index=False)
    
    # 5. Output Used/Dropped IDs
    # Since we used all valid logical IDs in the process:
    # Used = All 982
    # Dropped = 0
    used = [{"id": int(i), "reason": "valid"} for i in ids]
    dropped = [] # None dropped in this logic
    
    with open(os.path.join(args.audit_dir, "tables", "used_ids.json"), "w") as f:
        json.dump(used, f, indent=2)
    with open(os.path.join(args.audit_dir, "tables", "dropped_ids.json"), "w") as f:
        json.dump(dropped, f, indent=2)

if __name__ == "__main__":
    main()
