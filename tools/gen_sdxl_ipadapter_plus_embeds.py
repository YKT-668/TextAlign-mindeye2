#!/usr/bin/env python
import os
import json
import glob
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline

def main():
    # --- 参数解析 ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True, help="Path to the BASE IP-Adapter models directory (e.g., /home/vipuser/models/IP-Adapter).")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--ids_json", required=True)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--w", type=int, default=1024)
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    # --- 环境和模型设置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if (device == "cuda" and args.dtype == "fp16") else torch.float32
    g = torch.Generator(device=device).manual_seed(args.seed)

    print(f"[device] {device}, {torch_dtype}")
    
    print("[load] SDXL base 1.0")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if torch_dtype == torch.float16 else None
    ).to(device)

    print("[load] IP-Adapter (plus vit-h)")
    # 不使用 load_ip_adapter，改用手动加载避免自动加载错误的 encoder
    from diffusers.loaders import IPAdapterMixin
    from transformers import CLIPVisionModelWithProjection
    
    # 先加载正确的 CLIP ViT-H/14 image encoder
    print("[load] CLIP ViT-H/14 image encoder (laion/CLIP-ViT-H-14-laion2B-s32B-b79K)")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=torch_dtype,
    ).to(device)
    pipe.image_encoder = image_encoder
    
    # 再加载 IP-Adapter 权重
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
    )
    
    # 使用其他显存优化方法（不用 attention slicing，它与 IP-Adapter 不兼容）
    # pipe.enable_model_cpu_offload()  # 如果显存不够可以启用这个
    pipe.enable_vae_slicing()  # VAE slicing 是安全的

    # --- 数据加载 ---
    prompts = json.load(open(args.prompts, "r", encoding="utf-8"))
    ids = json.load(open(args.ids_json, "r"))
    img_paths = sorted(glob.glob(os.path.join(args.ref_dir, "*.png")) + glob.glob(os.path.join(args.ref_dir, "*.jpg")))
    print(f"[data] prompts={len(prompts)}  refs={len(img_paths)}  ids={len(ids)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- 图像生成循环 ---
    print("[run] generate with IP-Adapter images")
    for i, (rec, k) in enumerate(tqdm(zip(prompts, ids), total=min(len(prompts), len(ids)))):
        if k >= len(img_paths):
            print(f"[warn] id {k} out of range for refs")
            continue
        
        ref_img_path = img_paths[k]
        img = Image.open(ref_img_path).convert("RGB")
        
        pos_prompt = (rec.get("positive", "") + ", " + rec.get("style", "")).strip(", ")
        neg_prompt = rec.get("negative", "")

        # 直接传入 PIL Image，让 IP-Adapter 内部处理编码
        image = pipe(
            prompt=pos_prompt,
            negative_prompt=neg_prompt,
            ip_adapter_image=img,  # 直接传图像，不用手动编码
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.w,
            height=args.h,
            generator=g
        ).images[0]

        out_path = os.path.join(args.out_dir, f"{i:02d}.png")
        image.save(out_path)

    print(f"[done] All images saved to {args.out_dir}")

if __name__ == "__main__":
    main()