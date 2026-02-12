import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
import open_clip
import pandas as pd

def load_tensor(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
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
    parser.add_argument("--filter_path", default=None, help="JSON file with indices (0-981) to keep")
    args = parser.parse_args()

    print(f"=== Audit CCD Fix (Seed={args.seed}) ===")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load Assets
    ids = np.asarray(json.load(open(args.ids_path))) # 982
    brain = load_tensor(args.brain_path).float()
    
    # Load Shared1000 indices for mapping captions
    shared1000_mask = np.load(args.shared1000_path)
    if shared1000_mask.dtype == bool:
        shared1000_indices = np.where(shared1000_mask)[0]
    else:
        shared1000_indices = shared1000_mask
    
    id2local = {gid: i for i, gid in enumerate(shared1000_indices)}
    
    all_captions = torch.load(args.captions_path)
    # Align captions
    target_captions = []
    
    valid_ids_ordered = []
    
    # Verify alignment logic matches previous run
    for bid in ids:
        if len(all_captions) == 1000:
             if bid in id2local:
                 target_captions.append(all_captions[id2local[bid]])
                 valid_ids_ordered.append(bid)
             else:
                 raise ValueError(f"ID {bid} not found in shared1000 mapping.")
        else:
             target_captions.append(all_captions[bid])
             valid_ids_ordered.append(bid)

    # Check for Filter
    if args.filter_path:
        print(f"Loading filter from {args.filter_path}...")
        filter_indices = json.load(open(args.filter_path))
        print(f"Filtering dataset: {len(ids)} -> {len(filter_indices)} samples.")
        
        ids = ids[filter_indices]
        brain = brain[filter_indices]
        target_captions = [target_captions[i] for i in filter_indices]
        # valid_ids_ordered update is implicit via ids
        
    # Encode Text
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
    
    bs = 128
    text_feats_list = []
    with torch.no_grad():
        for i in range(0, len(target_captions), bs):
            batch = target_captions[i:i+bs]
            tok = tokenizer(batch).to(device)
            emb = model.encode_text(tok)
            emb = F.normalize(emb, dim=-1)
            text_feats_list.append(emb.cpu())
    text_feats = torch.cat(text_feats_list)
    
    # Project Brain (if needed)
    brain_ccd = brain.to(device)
    if brain_ccd.shape[1] == 1664 and text_feats.shape[1] == 1280:
        if hasattr(model.visual, "proj") and model.visual.proj is not None:
             brain_ccd = brain_ccd @ model.visual.proj.to(device)
    
    brain_ccd = F.normalize(brain_ccd, dim=-1).cpu()
    
    # Compute Matrices
    print("Computing Similarity Matrices...")
    S_BT = brain_ccd @ text_feats.T
    S_TT = text_feats @ text_feats.T
    
    N = len(ids)
    
    scores_k1 = np.zeros(N, dtype=int)
    scores_k2 = np.zeros(N, dtype=int)
    scores_rand = np.zeros(N, dtype=int)
    
    # Mining Loop
    mask = ~torch.eye(N, dtype=torch.bool)
    
    print("Mining Negatives...")
    for i in range(N):
        # Candidates (Global Pool excluding self)
        cands = np.where(mask[i].numpy())[0] # indices, not tensor
        
        # Hardest K (based on Text-Text similarity)
        sims_i = S_TT[i, cands]
        
        # Sort descending
        # Using numpy for stable sorting/behavior matching? Torch topk is fine
        topk_vals, topk_idx = torch.topk(sims_i, 2)
        hard_indices = cands[topk_idx.numpy()]
        
        # Random K
        # deterministic shuffle with seed control for checking?
        # Re-seed inside loop? No, just rely on global seed.
        # But to match 'CCD_Random_K2=0.9490', we might simply report new value or try to match.
        # The prompt says "output per-image scores" for Random too.
        rand_perm = torch.randperm(len(cands))[:2]
        rand_indices = cands[rand_perm.numpy()]
        
        score_pos = S_BT[i, i].item()
        
        # K=1 Hard
        neg1 = S_BT[i, hard_indices[0]].detach().item()
        if score_pos > neg1: scores_k1[i] = 1
        
        # K=2 Hard
        negs_hard2 = S_BT[i, hard_indices[:2]].detach().numpy()
        if (score_pos > negs_hard2).all(): scores_k2[i] = 1
        
        # K=2 Random
        negs_rand2 = S_BT[i, rand_indices].detach().numpy()
        if (score_pos > negs_rand2).all(): scores_rand[i] = 1

    # Save Check
    mean_k1 = scores_k1.mean()
    mean_k2 = scores_k2.mean()
    mean_rand = scores_rand.mean()
    
    print(f"Point Estimates:")
    print(f"CCD_Hard_K1: {mean_k1:.8f}")
    print(f"CCD_Hard_K2: {mean_k2:.8f}")
    print(f"CCD_Random_K2: {mean_rand:.8f}")
    
    out_table_dir = os.path.join(args.audit_dir, "tables")
    os.makedirs(out_table_dir, exist_ok=True)
    
    np.save(os.path.join(out_table_dir, "ccd_hard_k1_per_image.npy"), scores_k1)
    np.save(os.path.join(out_table_dir, "ccd_hard_k2_per_image.npy"), scores_k2)
    np.save(os.path.join(out_table_dir, "ccd_rand_k2_per_image.npy"), scores_rand)
    
    # Convert to python int list for JSON
    valid_ids_py = [int(x) for x in valid_ids_ordered]
    with open(os.path.join(out_table_dir, "ccd_ids.json"), "w") as f:
        json.dump(valid_ids_py, f)
        
    # Bootstrap
    print(f"Bootstrapping {args.n_boot} times...")
    boot_means = {"CCD_Hard_K1": [], "CCD_Hard_K2": [], "CCD_Random_K2": []}
    
    indices = np.arange(N)
    for _ in range(args.n_boot):
        sample = np.random.choice(indices, N, replace=True)
        boot_means["CCD_Hard_K1"].append(scores_k1[sample].mean())
        boot_means["CCD_Hard_K2"].append(scores_k2[sample].mean())
        boot_means["CCD_Random_K2"].append(scores_rand[sample].mean())
        
    ci_rows = []
    for k, vals in boot_means.items():
        mean_boot = np.mean(vals)
        low, high = get_ci(vals)
        ci_rows.append({"Metric": k, "Mean": mean_boot, "CI_Lower": low, "CI_Upper": high})
        print(f"{k}: Mean={mean_boot:.4f} CI=[{low:.4f}, {high:.4f}] Diff={abs(mean_boot - locals()['mean_k1' if 'K1' in k else ('mean_rand' if 'Random' in k else 'mean_k2')]):.6f}")

    pd.DataFrame(ci_rows).to_csv(os.path.join(out_table_dir, "ccd_bootstrap_ci_fixed.csv"), index=False)

if __name__ == "__main__":
    main()
