#!/usr/bin/env python3
# tools/quick_eval.py
import argparse, os, json, csv, sys
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

try:
    import open_clip
except Exception as e:
    print("[FATAL] open_clip not found. `pip install open-clip-torch`.", file=sys.stderr)
    raise

def list_images(d):
    exts = (".png",".jpg",".jpeg",".bmp",".webp")
    return sorted([p for p in glob(os.path.join(d, "*")) if p.lower().endswith(exts)])

def load_pairs_from_csv(csv_path):
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            pairs.append((row[0], row[1]))
    return pairs

def pil2tensor(preprocess, path_or_img):
    if isinstance(path_or_img, Image.Image):
        im = path_or_img
    else:
        im = Image.open(path_or_img).convert("RGB")
    return preprocess(im).unsqueeze(0)

def cosine_sim(a, b, eps=1e-8):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

def ssim_torch(x, y):
    # very light-weight SSIM (not exact to skimage); expects [1,3,H,W], [0,1]
    # For quick reference only.
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = torch.nn.functional.avg_pool2d(x, 7, 1, 0)
    mu_y = torch.nn.functional.avg_pool2d(y, 7, 1, 0)
    sigma_x  = torch.nn.functional.avg_pool2d(x*x, 7, 1, 0) - mu_x**2
    sigma_y  = torch.nn.functional.avg_pool2d(y*y, 7, 1, 0) - mu_y**2
    sigma_xy = torch.nn.functional.avg_pool2d(x*y, 7, 1, 0) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2))/((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))
    return ssim_map.mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", type=str, required=True, help="Directory of generated images")
    ap.add_argument("--prompts_json", type=str, default=None, help="JSON/JSONL of prompts to compute CLIPScore(img,text)")
    ap.add_argument("--gt_dir", type=str, default=None, help="Directory of GT images (same number & filename match)")
    ap.add_argument("--pairs_csv", type=str, default=None, help="Optional CSV with 2 columns: gen_path,gt_path")
    ap.add_argument("--out_json", type=str, required=True, help="Save metrics here")
    ap.add_argument("--do_retrieval", action="store_true", help="Compute image-image retrieval Top-K vs gt_dir")
    ap.add_argument("--topk", type=int, nargs="+", default=[1,5,10])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model", type=str, default="ViT-H-14")
    ap.add_argument("--pretrained", type=str, default="laion2b_s32b_b79k")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    gen_paths = list_images(args.gen_dir)
    if len(gen_paths) == 0:
        raise RuntimeError(f"No images in {args.gen_dir}")

    # ---- CLIPScore(img, text) ----
    clip_scores = None
    if args.prompts_json:
        texts = []
        raw = open(args.prompts_json, "r", encoding="utf-8").read().strip()
        if raw and raw[0] == "{":
            # JSONL
            for line in raw.splitlines():
                if not line.strip(): continue
                obj = json.loads(line)
                p = obj.get("positive") or obj.get("prompt") or obj.get("text") or obj.get("caption")
                if p: texts.append(p)
        else:
            # JSON Array
            arr = json.loads(raw) if raw else []
            for obj in arr:
                p = obj.get("positive") or obj.get("prompt") or obj.get("text") or obj.get("caption")
                if p: texts.append(p)
        
        if len(texts) < len(gen_paths):
            reps = (len(gen_paths) + len(texts) - 1)//max(1,len(texts))
            texts = (texts * reps)[:len(gen_paths)]
        elif len(texts) > len(gen_paths):
            texts = texts[:len(gen_paths)]

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            text_embs = []
            for i in tqdm(range(0, len(texts), 256), desc="Encoding texts"):
                toks = tokenizer(texts[i:i+256]).to(device)
                te = model.encode_text(toks)
                te = te / te.norm(dim=-1, keepdim=True)
                text_embs.append(te.float().cpu())
            text_embs = torch.cat(text_embs, dim=0)

            img_embs = []
            for p in tqdm(gen_paths, desc="Encoding generated images"):
                t = pil2tensor(preprocess, p).to(device)
                ie = model.encode_image(t)
                ie = ie / ie.norm(dim=-1, keepdim=True)
                img_embs.append(ie.float().cpu())
            img_embs = torch.cat(img_embs, dim=0)

        clip_scores = cosine_sim(img_embs, text_embs).numpy().tolist()

    # ---- PixCorr / SSIM & Retrieval ----
    pixcorr, ssim_vals = None, None
    retrieval = None

    pairs = []
    if args.pairs_csv:
        pairs = load_pairs_from_csv(args.pairs_csv)
    elif args.gt_dir:
        gt_paths = list_images(args.gt_dir)
        if len(gt_paths) != len(gen_paths):
            print(f"[WARN] gt images != gen images ({len(gt_paths)} vs {len(gen_paths)}), will align by sorted order length min.", file=sys.stderr)
        n = min(len(gt_paths), len(gen_paths))
        pairs = list(zip(gen_paths[:n], gt_paths[:n]))

    if pairs:
        pixcorr, ssim_vals = [], []
        for gp, tp in tqdm(pairs, desc="Pix/SSIM"):
            g = Image.open(gp).convert("RGB").resize((256,256), Image.BICUBIC)
            t = Image.open(tp).convert("RGB").resize((256,256), Image.BICUBIC)
            g_t = torch.from_numpy(np.array(g)).permute(2,0,1).unsqueeze(0).float()/255.0
            t_t = torch.from_numpy(np.array(t)).permute(2,0,1).unsqueeze(0).float()/255.0
            pc = torch.corrcoef(torch.stack([g_t.flatten(), t_t.flatten()]))[0,1].item()
            pixcorr.append(pc)
            ssim_vals.append(ssim_torch(g_t, t_t).item())

        if args.do_retrieval:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                G, T_embs = [], []
                for gp in tqdm([p[0] for p in pairs], desc="Enc gen for retrieval"):
                    t = pil2tensor(preprocess, gp).to(device)
                    e = model.encode_image(t); e = e / e.norm(dim=-1, keepdim=True)
                    G.append(e.float().cpu())
                for tp in tqdm([p[1] for p in pairs], desc="Enc gt for retrieval"):
                    t = pil2tensor(preprocess, tp).to(device)
                    e = model.encode_image(t); e = e / e.norm(dim=-1, keepdim=True)
                    T_embs.append(e.float().cpu())
            G = torch.cat(G, dim=0)
            T_embs = torch.cat(T_embs, dim=0)
            S = (F.normalize(G, dim=-1) @ F.normalize(T_embs, dim=-1).t())
            retrieval = {}
            for k in args.topk:
                topk_idx = torch.topk(S, k=k, dim=1).indices
                gt_idx = torch.arange(S.shape[0]).unsqueeze(1)
                hits = (topk_idx == gt_idx).any(dim=1).float().mean().item()
                retrieval[f"top{k}"] = hits

    out = {
        "N_gen": len(gen_paths),
        "CLIPScore_mean": float(np.mean(clip_scores)) if clip_scores else None,
        "CLIPScore_std": float(np.std(clip_scores)) if clip_scores else None,
        "PixCorr_mean": float(np.mean(pixcorr)) if pixcorr else None,
        "SSIM_mean": float(np.mean(ssim_vals)) if ssim_vals else None,
        "Retrieval": retrieval,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[saved]", args.out_json)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()