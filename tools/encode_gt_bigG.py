
#!/usr/bin/env python
import sys
sys.path.insert(0, '../src')  # 加 src 目录到路径（相对 tools/）

import torch
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
# ... 其余代码不变
import torch, os, argparse
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_pixels", default="/home/vipuser/MindEyeV2_Project/src/evals/all_images.pt")
    ap.add_argument("--out", default="/home/vipuser/MindEyeV2_Project/src/evals/all_images_bigG1664.pt")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    imgs = torch.load(args.gt_pixels, map_location="cpu").float()  # [N,3,224,224]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14", version="laion2b_s39b_b160k", output_tokens=True
    ).to(device).eval().requires_grad_(False)

    outs = []
    for i in range(0, imgs.shape[0], args.batch):
        x = imgs[i:i+args.batch].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            tok = enc(x)[0] if isinstance(enc(x), tuple) else enc(x)  # 取第一项 (tokens) 如果是 tuple          # [B,256,1664]
        vec = tok.mean(dim=1)     # -> [B,1664]
        outs.append(vec.float().cpu())
    emb = torch.cat(outs, dim=0)  # [N,1664]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(emb, args.out)
    print("WROTE:", args.out, "shape=", tuple(emb.shape))

if __name__ == "__main__":
    main()

