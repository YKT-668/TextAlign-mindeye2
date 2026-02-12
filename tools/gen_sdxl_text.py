#!/usr/bin/env python
import os, json, math, torch, random
from diffusers import StableDiffusionXLPipeline

PROMPTS = "/home/vipuser/train_logs/s1_ps1p5_h512_e5_cycle/inference/prompt_bigG.json"
OUTDIR  = "/home/vipuser/train_logs/s1_ps1p5_h512_e5_cycle/inference/gen_text"
MODEL  = "stabilityai/stable-diffusion-xl-base-1.0"  # 可换本地路径/缓存目录
SEED   = 1234
STEPS  = 30
GUIDANCE = 6.5
H, W = 1024, 1024   # 先 1024×1024，卡不够可降到 768/512

os.makedirs(OUTDIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
pipe.to(device)
pipe.set_progress_bar_config(disable=True)

def _txt(p):
    # 结构化 JSON 字段 → 正文 prompt
    main = p.get("positive","")
    style = p.get("style","")
    return (main + ((", " + style) if style else "")).strip()

with open(PROMPTS,"r",encoding="utf-8") as f:
    items = json.load(f)

g = torch.Generator(device=device).manual_seed(SEED)

for i,p in enumerate(items):
    prompt = _txt(p)
    negative = p.get("negative","")
    img = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        height=H, width=W,
        generator=g
    ).images[0]
    img.save(os.path.join(OUTDIR, f"text_{i:03d}.png"))

print(f"[text] done -> {OUTDIR}, n={len(items)}")