import sys, json, torch, open_clip
from pathlib import Path

cap_path = Path(sys.argv[1])      # captions_demo.txt
k        = int(sys.argv[2])       # Top-K
z_path   = Path(sys.argv[3])      # Z.pt (B,768) 先用 recons.pt 代替

caps = [l.strip() for l in open(cap_path, encoding='utf-8') if l.strip()]
model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
tok = open_clip.get_tokenizer("ViT-L-14")
with torch.inference_mode():
    T = model.encode_text(tok(caps)).float()
    T = T / T.norm(dim=-1, keepdim=True)
    Z = torch.load(z_path, "cpu").float()
    Z = Z / Z.norm(dim=-1, keepdim=True)
    S = Z @ T.T  # [B, |caps|]
    topv, topi = S.topk(k, dim=1)
out=[]
for i in range(topi.size(0)):
    items=[]
    for j, (ix, sc) in enumerate(zip(topi[i].tolist(), topv[i].tolist()), 1):
        items.append({"rank":j,"score":float(sc),"caption":caps[ix]})
    out.append({"sample":i,"topk":items})
print(json.dumps(out, ensure_ascii=False, indent=2))
