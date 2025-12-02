#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Brain vector → ViT-H token embed (1×1280) → SDXL + IP-Adapter-plus (ViT-H) generation

- Inputs
  --adapter_dir:      IP-Adapter repo local dir (expects sdxl_models/...)
  --prompts:          JSON file. Each item supports keys: {"prompt" or ["positive","style"], "negative"}
  --brain_vec_pt:     torch.save'd tensor, shape [N, D] (D ∈ {1664, 1024, 1280})
  --proj_pt:          optional linear proj (brain 1664→1024) checkpoint. Supports plain Tensor or nn.Linear.state_dict
  --out_dir:          output image dir

- Options
  --steps/--cfg/--w/--h/--seed/--dtype(fp16|fp32|bf16)/--ip_scale

Notes
- We normalize to (B, 1, 1280) before feeding ip_adapter_image_embeds.
- If brain D=1664 (OpenCLIP bigG), we map 1664→1024 via --proj_pt, then 1024→1280 using
  open_clip ViT-H visual.proj. If D=1024,仅做 1024→1280。如果 D=1280，直接用。
- Robust loader for proj_pt: supports keys {"weight"[,"bias"]} or raw 2D tensor.
"""

import os, json, argparse, math
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch import nn
from PIL import Image

from diffusers import StableDiffusionXLPipeline

# -------------------------- Args --------------------------
AP = argparse.ArgumentParser()
AP.add_argument("--adapter_dir", required=True, help="IP-Adapter local root (contains sdxl_models/...)")
AP.add_argument("--prompts", required=True, help="path to prompts json")
AP.add_argument("--brain_vec_pt", required=True, help="pt: [N,D] brain→CLIP vectors")
AP.add_argument("--proj_pt", default="", help="optional proj ckpt for 1664→1024 mapping")
AP.add_argument("--out_dir", required=True)

AP.add_argument("--steps", type=int, default=28)
AP.add_argument("--cfg", type=float, default=5.0)
AP.add_argument("--w", type=int, default=1024)
AP.add_argument("--h", type=int, default=1024)
AP.add_argument("--seed", type=int, default=42)
AP.add_argument("--dtype", choices=["fp16","fp32","bf16"], default="fp16")
AP.add_argument("--ip_scale", type=float, default=0.8, help="IP-Adapter conditioning scale")
AP.add_argument("--max_items", type=int, default=-1, help="Limit number of items to generate (quick eval subset)")

args = AP.parse_args()

# --------------------- Device / dtype ---------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if args.dtype == "fp16":
    torch_dtype = torch.float16
elif args.dtype == "bf16":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

print(f"[device] {device.type}, dtype={torch_dtype}")
if args.max_items > 0:
    print(f"[config] max_items={args.max_items}")

torch.manual_seed(args.seed)

# ------------------ Load SDXL + IP-Adapter ----------------
print("[load] SDXL base 1.0")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch_dtype if hasattr(StableDiffusionXLPipeline, "from_pretrained") else None,
    torch_dtype=torch_dtype,
    use_safetensors=True,
    variant="fp16" if torch_dtype==torch.float16 else None,
)
pipe.to(device)

# Lighter memory path on mismatched xformers
try:
    pipe.enable_attention_slicing()
except Exception:
    pass


def apply_sliced_attn_processor_patch(pipe=None):
    """Monkeypatch SlicedAttnProcessor to provide a default slice_size when
    the installed diffusers version requires it (compat shim).

    If `pipe` is provided, also patch the concrete classes referenced by the
    UNet's `attn_processors` mapping so loader code that calls
    `cls()` without args succeeds.
    """
    try:
        import inspect
        import diffusers.models.attention_processor as attn_proc_modules

        def _patch_class(cls):
            try:
                sig = inspect.signature(cls.__init__)
                if "slice_size" in sig.parameters and sig.parameters["slice_size"].default is inspect._empty:
                    orig_init = cls.__init__

                    def __init__(self, slice_size: int = 64, *args, **kwargs):
                        return orig_init(self, slice_size, *args, **kwargs)

                    cls.__init__ = __init__
                    return True
            except Exception:
                pass
            return False

        patched = False
        if hasattr(attn_proc_modules, "SlicedAttnProcessor"):
            patched = _patch_class(attn_proc_modules.SlicedAttnProcessor) or patched

        # If a pipeline/unet instance was passed, patch concrete classes used there too
        if pipe is not None and hasattr(pipe, "unet") and hasattr(pipe.unet, "attn_processors"):
            for v in pipe.unet.attn_processors.values():
                cls = v.__class__
                if cls.__name__ == "SlicedAttnProcessor":
                    patched = _patch_class(cls) or patched

        if patched:
            print("✓ Applied SlicedAttnProcessor patch successfully.")
    except Exception as e:
        print(f"⚠️  Could not apply SlicedAttnProcessor patch: {e}.")

print("[load] IP-Adapter (plus vit-h)")
adapter_root = args.adapter_dir.rstrip("/")
# Expect weights here:
#   {adapter_root}/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
#   {adapter_root}/sdxl_models/image_encoder_{vith|bigg}/ {config.json, model.safetensors}
apply_sliced_attn_processor_patch(pipe)

# Best-effort locate a CLIP image encoder folder to satisfy diffusers>=0.30 loader
enc_candidates = [
    os.path.join(adapter_root, "sdxl_models", "image_encoder"),
    os.path.join(adapter_root, "sdxl_models", "image_encoder_vith"),
    os.path.join(adapter_root, "sdxl_models", "image_encoder_bigg_bak"),
]
enc_folder = None
for c in enc_candidates:
    if os.path.isdir(c) and os.path.isfile(os.path.join(c, "config.json")) and (
        os.path.isfile(os.path.join(c, "model.safetensors")) or os.path.isfile(os.path.join(c, "pytorch_model.bin"))
    ):
        enc_folder = c
        break

if enc_folder is None:
    print("image_encoder folder not found; proceeding without it (will rely on ip_adapter_image_embeds)")

pipe.load_ip_adapter(
    pretrained_model_name_or_path_or_dict=adapter_root,
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
    image_encoder_folder=enc_folder,
)
pipe.set_ip_adapter_scale(args.ip_scale)

# ------------------ Prompts utilities ---------------------
with open(args.prompts, "r", encoding="utf-8") as f:
    raw_prompts = json.load(f)
if args.max_items > 0 and len(raw_prompts) > args.max_items:
    raw_prompts = raw_prompts[:args.max_items]
    print(f"[subset] prompts trimmed to {len(raw_prompts)}")

def _clamp77(s: str) -> str:
    # avoid CLIP 77 token overflow hot-words
    return s.replace(" 4 k", "").replace(" 4k", "").replace(" 4 K", "").strip().strip(",")

def _unpack_prompt(rec: Dict[str,Any]) -> Tuple[str,str]:
    if isinstance(rec, dict):
        if "prompt" in rec:  # unified field
            pos = rec.get("prompt", "")
        else:
            pos = ", ".join([rec.get("positive", ""), rec.get("style", "")])
        neg = rec.get("negative", "")
    else:
        pos = str(rec)
        neg = ""
    return _clamp77(pos), _clamp77(neg)

# ---------------- Brain vectors & projection ---------------
print("[load] brain vectors")
V = torch.load(args.brain_vec_pt, map_location="cpu")
if isinstance(V, dict) and "tensor" in V:
    V = V["tensor"]
V = torch.as_tensor(V)
assert V.ndim == 2, f"brain_vec must be 2D [N,D], got {tuple(V.shape)}"
N, D = V.shape
print(f"  brain_vec shape: {tuple(V.shape)}")
if args.max_items > 0 and V.shape[0] > args.max_items:
    V = V[:args.max_items]
    print(f"[subset] brain vectors trimmed to {V.shape[0]}")
    N, D = V.shape

# Helper: load 1664→1024 projection (optional)
def load_linear_proj(path: str) -> Optional[torch.Tensor]:
    if not path:
        return None
    ckpt = torch.load(path, map_location="cpu")
    W = None
    if isinstance(ckpt, dict):
        if "weight" in ckpt and isinstance(ckpt["weight"], torch.Tensor):
            W = ckpt["weight"]  # [out,in]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
            for k in ["weight", "linear.weight", "proj.weight", "W", "module.weight"]:
                if k in sd and sd[k].ndim == 2:
                    W = sd[k]
                    break
        elif "W" in ckpt and isinstance(ckpt["W"], torch.Tensor) and ckpt["W"].ndim==2:
            W = ckpt["W"]
    elif isinstance(ckpt, torch.Tensor) and ckpt.ndim==2:
        W = ckpt
    if W is None:
        raise ValueError(f"Unrecognized proj ckpt format: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    return W

W_1664_to_1024 = None
if D == 1664 and args.proj_pt:
    print("[proj] load 1664→1024 linear proj from", args.proj_pt)
    W_1664_to_1024 = load_linear_proj(args.proj_pt)  # [out,in] or [in,out]
    if W_1664_to_1024.shape[0] in (1024, 1664) and W_1664_to_1024.shape[1] in (1024,1664):
        # We want matmul: (N,1664) @ (1664,1024) => (N,1024)
        if W_1664_to_1024.shape == (1024,1664):
            W_1664_to_1024 = W_1664_to_1024.T
        elif W_1664_to_1024.shape == (1664,1024):
            pass
        else:
            raise ValueError(f"proj dims ambiguous: {tuple(W_1664_to_1024.shape)}")
    else:
        raise ValueError(f"proj dims must be 1664x1024 or 1024x1664, got {tuple(W_1664_to_1024.shape)}")

# Load open_clip ViT-H visual.proj (1024→1280)
print("[open_clip] ViT-H-14 (laion2b_s32b_b79k) for 1024→1280")
try:
    import open_clip
    oc_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k")
    # visual.proj: [embed_dim(1024)→width(1280)] stored as (out,in)
    W_1024_to_1280 = oc_model.visual.proj  # nn.Linear weight: [1280,1024]
    if isinstance(W_1024_to_1280, torch.Tensor):
        pass
    else:
        W_1024_to_1280 = W_1024_to_1280.weight
    W_1024_to_1280 = W_1024_to_1280.detach().to(torch.float32)
except Exception as e:
    raise RuntimeError(f"Failed to load open_clip ViT-H proj: {e}")

# Build final (N,1280)
with torch.no_grad():
    if D == 1280:
        E1280 = V.to(torch.float32)
    elif D == 1024:
        # (N,1024) @ (1024,1280)T? visual.proj is (1280,1024); we need (1024,1280)
        E1280 = V @ W_1024_to_1280.T
    elif D == 1664:
        if W_1664_to_1024 is None:
            raise ValueError("brain_vec is 1664-D (bigG). Provide --proj_pt for 1664→1024 mapping.")
        V1024 = V @ W_1664_to_1024  # (N,1664) @ (1664,1024) -> (N,1024)
        E1280 = V1024 @ W_1024_to_1280.T
    else:
        raise ValueError(f"Unsupported brain_vec dim D={D}. Expect one of {1664,1024,1280}.")

print(f"[embed] Final (N,1280) = {tuple(E1280.shape)}")

# -> (B, 1, 1280) and correct dtype/device
E1280 = E1280.to(device=device, dtype=pipe.text_encoder.dtype if hasattr(pipe, "text_encoder") else torch_dtype)
E_tokens = E1280.unsqueeze(1)  # (B,1,1280)

# --------------------- Generate loop ----------------------
os.makedirs(args.out_dir, exist_ok=True)

B = E_tokens.shape[0]
M = len(raw_prompts)
T = min(B, M)
print(f"[data] brain_vec={B}, prompts={M} -> will generate {T}")

@torch.no_grad()
def generate_one(i: int, pos: str, neg: str):
    emb = E_tokens[i:i+1]  # (1,1,1280)
    # prepare (uncond, cond) stacked along batch dim as the pipeline expects
    uncond = torch.zeros_like(emb)
    stacked = torch.cat([uncond, emb], dim=0)
    # ensure sequence dimension exists: (batch, num_images, seq_len, embed_dim)
    if stacked.ndim == 3:
        stacked = stacked.unsqueeze(2)
    # move to device and match unet dtype
    target_dtype = pipe.unet.dtype if hasattr(pipe, "unet") else torch_dtype
    stacked = stacked.to(device=device, dtype=target_dtype)

    images = pipe(
        prompt=pos,
        negative_prompt=neg,
        ip_adapter_image_embeds=[stacked],
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        width=args.w,
        height=args.h,
    ).images
    return images[0]

import sys, time
start_time = time.time()
for i in range(T):
    pos, neg = _unpack_prompt(raw_prompts[i])
    try:
        img = generate_one(i, pos, neg)
        out_path = os.path.join(args.out_dir, f"{i:02d}.png")
        img.save(out_path)
        elapsed = time.time() - start_time
        rate = (i+1)/elapsed if elapsed > 0 else 0
        print(f"[ok] {i:02d} -> {out_path} | elapsed={elapsed:,.1f}s | avg={rate:,.2f} img/s", flush=True)
    except Exception as e:
        import traceback
        print(f"[error] {i:02d} failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        sys.stdout.flush()

print(f"[done] -> {args.out_dir}")
