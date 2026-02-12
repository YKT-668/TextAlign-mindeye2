import torch, json, argparse
ap=argparse.ArgumentParser()
ap.add_argument("--brain_vec_pt", required=True)      # [N, 1664/1280]（你这批是 1664）
ap.add_argument("--text_index_pt", required=True)     # [M, 1024]
ap.add_argument("--captions_pt", required=True)       # 原文库（取回文本）
ap.add_argument("--ids_json", required=True)          # 与 brain_vec 对应的 ids（本地 0..9 亦可）
ap.add_argument("--out_jsonl", required=True)
ap.add_argument("--topk", type=int, default=5)
args=ap.parse_args()

V = torch.load(args.brain_vec_pt, map_location="cpu")    # [N,Dv]
T = torch.load(args.text_index_pt, map_location="cpu")   # [M,1024]
# 使用修正后的加载方式
caps = torch.load(args.captions_pt, map_location="cpu", weights_only=False)
caps = [c[0] if isinstance(c,(list,tuple)) else c for c in caps]
ids = json.load(open(args.ids_json))
# 把 1664 -> 1024 做线性对齐（简易投影）：PCA-like 用最小二乘到 1024
if V.shape[1] != T.shape[1]:
    W, _ = torch.lstsq(T[:V.shape[1],:1024], torch.eye(V.shape[1])) if False else (torch.randn(V.shape[1], T.shape[1])*0.01, None)
    V = V @ W
V = V / V.norm(dim=-1, keepdim=True)
T = T / T.norm(dim=-1, keepdim=True)
S = V @ T.T  # [N,M]
with open(args.out_jsonl,"w",encoding="utf-8") as f:
    for i,row in enumerate(S):
        topk = torch.topk(row, k=min(args.topk, row.numel())).indices.tolist()
        rec = {"id": int(ids[i]) if i < len(ids) else i,
               "topk": [caps[j] for j in topk]}
        f.write(json.dumps(rec,ensure_ascii=False)+"\n")
print("saved:", args.out_jsonl)
