#!/usr/bin/env python
import os, json, argparse, torch
import torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="/home/vipuser/train_logs/s1_ps1p5_h512_e5_cycle/inference/brain_clip.pt")
    ap.add_argument("--ids",   default="/home/vipuser/train_logs/s1_ps1p5_h512_e5_cycle/inference/ids.json")
    ap.add_argument("--gt_emb", default="/home/vipuser/MindEyeV2_Project/src/evals/all_images_bigG1664.pt")
    ap.add_argument("--gt_caps", default="/home/vipuser/MindEyeV2_Project/src/evals/all_captions.pt")
    ap.add_argument("--out", default="/home/vipuser/train_logs/s1_ps1p5_h512_e5_cycle/inference/topk_bigG.json")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    B = torch.load(args.brain, map_location="cpu").float()      # [Q,1664]
    G = torch.load(args.gt_emb, map_location="cpu").float()     # [N,1664]
    CAPS = torch.load(args.gt_caps, map_location="cpu", weights_only=False)  # 修复：加 weights_only=False

    B = F.normalize(B, dim=-1); G = F.normalize(G, dim=-1)
    S = B @ G.t()   # [Q,N]
    vals, idxs = torch.topk(S, args.k, dim=1)

    try:
        ids = json.load(open(args.ids))
    except Exception:
        ids = list(range(B.shape[0]))

    def get_caption(i):
        cap = CAPS[i]
        if isinstance(cap, (list,tuple)): cap = cap[0]
        return str(cap)

    records = []
    for qi in range(B.shape[0]):
        topk = []
        for rank,(j,sc) in enumerate(zip(idxs[qi].tolist(), vals[qi].tolist()), start=1):
            topk.append({"rank": rank, "idx": j, "score": float(sc), "caption": get_caption(j)})
        records.append({"sample": int(ids[qi]) if qi < len(ids) else qi,
                        "query_idx": qi,
                        "topk": topk})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out,"w"), ensure_ascii=False, indent=2)
    print("WROTE:", args.out, f"({len(records)} queries, k={args.k})")
    print("sim mean:", float(S.mean()), "NN mean:", float(S.max(1).values.mean()))

if __name__ == "__main__":
    main()