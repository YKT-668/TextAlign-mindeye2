#!/usr/bin/env python
import os, json, glob, argparse
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline

def e(msg): raise SystemExit(f"[FATAL] {msg}")

ap = argparse.ArgumentParser()
ap.add_argument("--adapter_dir", required=True, help="IP-Adapter 根目录（包含 sdxl_models/ ... ）")
ap.add_argument("--prompts", required=True, help="结构化提示 JSON（含 positive/style/negative）")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--ref_dir", default="", help="参考图目录（参考图模式）")
ap.add_argument("--brain_clip_pt", default="", help="向量模式：1664维 brain→CLIP 向量 .pt")
ap.add_argument("--ids_json", default="", help="与 brain_clip.pt 对齐的 ids.json（用于定位参考图文件名）")
ap.add_argument("--steps", type=int, default=28)
ap.add_argument("--cfg", type=float, default=5.0)
ap.add_argument("--w", type=int, default=1024)
ap.add_argument("--h", type=int, default=1024)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--dtype", default="auto", choices=["auto","fp16","fp32"])
ap.add_argument("--cpu_offload", action="store_true", help="显存紧张时开启")
args = ap.parse_args()

# ---------- 0) 路径与权重健壮性检查 ----------
sdxl_sub = "sdxl_models"
enc_sub  = os.path.join(sdxl_sub, "image_encoder")
w_name   = "ip-adapter_sdxl.safetensors"
adapter_weights = os.path.join(args.adapter_dir, sdxl_sub, w_name)
image_encoder_dir = os.path.join(args.adapter_dir, enc_sub)
image_encoder_cfg = os.path.join(image_encoder_dir, "config.json")
image_encoder_ckp = os.path.join(image_encoder_dir, "model.safetensors")

if not os.path.isfile(adapter_weights):
    e(f"未找到 IP-Adapter 权重: {adapter_weights}")
if not (os.path.isfile(image_encoder_cfg) and os.path.isfile(image_encoder_ckp)):
    e(f"缺少 image_encoder 配置或权重: {image_encoder_dir}/(config.json|model.safetensors)")

if not os.path.isfile(args.prompts):
    e(f"prompts 文件不存在: {args.prompts}")
os.makedirs(args.out_dir, exist_ok=True)

# ---------- 1) 设备/精度 ----------
if args.dtype == "fp16":
    torch_dtype = torch.float16
elif args.dtype == "fp32":
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}, dtype={torch_dtype}")

# ---------- 2) 载入 SDXL 主干 ----------
print("[load] Stable Diffusion XL base 1.0")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch_dtype,
    use_safetensors=True,
    variant="fp16" if torch_dtype == torch.float16 else None
)
pipe.to(device)
pipe.enable_attention_slicing()
if args.cpu_offload and device == "cuda":
    # 显存很紧时开启；否则注释掉以提升速度
    pipe.enable_model_cpu_offload()

# 可选：关闭安全检查
pipe.safety_checker = None

# ---------- 3) 加载 IP-Adapter ----------
print(f"[load] IP-Adapter @ {args.adapter_dir}")
pipe.load_ip_adapter(
    pretrained_model_name_or_path_or_dict=args.adapter_dir,
    subfolder=sdxl_sub,
    weight_name=w_name,
    image_encoder_folder=enc_sub
)
pipe.set_ip_adapter_scale(0.8)

# ---------- 4) 读 prompts ----------
prompts = json.load(open(args.prompts, "r", encoding="utf-8"))
if args.limit and len(prompts) > args.limit:
    prompts = prompts[:args.limit]
    print(f"[info] limit prompts -> {len(prompts)}")

def clamp77(s: str) -> str:
    return s.replace(" 4 k", "").replace(" 4k", "").replace(" 4 K", "").strip(", ")

# ---------- 5) 参考图/向量输入准备 ----------
use_vec = bool(args.brain_clip_pt)
ip_images = []
image_embeds = None
idx_to_name = None

if use_vec:
    print(f"[mode] 向量模式 (ip_adapter_image_embeds) -> {args.brain_clip_pt}")
    if not os.path.isfile(args.brain_clip_pt):
        e(f"brain_clip_pt 不存在: {args.brain_clip_pt}")
    brain_clip = torch.load(args.brain_clip_pt, map_location="cpu")
    if not isinstance(brain_clip, torch.Tensor):
        e("brain_clip_pt 加载后不是 Tensor")
    # 期望形状 [N, 1664]
    if brain_clip.ndim != 2 or brain_clip.shape[1] != 1664:
        e(f"brain_clip 形状不对，期望 [N,1664]，当前 {tuple(brain_clip.shape)}")
    image_embeds = brain_clip.to(device=device, dtype=torch_dtype)

    # 可选用于命名/对齐
    if args.ids_json and os.path.isfile(args.ids_json):
        with open(args.ids_json, "r", encoding="utf-8") as f:
            idx_to_name = json.load(f)  # list[int] 或 list[str]
else:
    print(f"[mode] 参考图模式 (ip_adapter_image) -> {args.ref_dir}")
    if not args.ref_dir or not os.path.isdir(args.ref_dir):
        e("参考图模式需要提供有效的 --ref_dir")
    # 支持两种命名：自然排序 或 按 ids.json 指定名字（如 0001.png）
    paths = sorted(glob.glob(os.path.join(args.ref_dir, "*.png")) +
                   glob.glob(os.path.join(args.ref_dir, "*.jpg")) +
                   glob.glob(os.path.join(args.ref_dir, "*.jpeg")))
    if args.ids_json and os.path.isfile(args.ids_json):
        with open(args.ids_json, "r", encoding="utf-8") as f:
            ids = json.load(f)
        # 尝试按 <id>.png / <id>.jpg 匹配
        cand = []
        for idv in ids:
            for ext in (".png",".jpg",".jpeg"):
                p = os.path.join(args.ref_dir, f"{idv}{ext}")
                if os.path.isfile(p):
                    cand.append(p); break
        if cand:
            paths = cand
            print(f"[align] 按 ids.json 对齐到 {len(paths)} 张")
    if not paths:
        e(f"在 {args.ref_dir} 未找到参考图")
    for p in paths:
        try:
            ip_images.append(Image.open(p).convert("RGB"))
        except Exception as ex:
            print(f"[warn] 加载参考图失败 {p}: {ex}")
            ip_images.append(None)
    print(f"[load] 参考图数量: {len(ip_images)}")

# ---------- 6) 生成 ----------
g = torch.Generator(device=device).manual_seed(args.seed)
print("[run] start generation ...")
for i, rec in enumerate(prompts):
    try:
        pos = clamp77((rec.get("positive","") + ", " + rec.get("style","")).strip(", "))
        neg = clamp77(rec.get("negative",""))
        kwargs = dict(
            prompt=pos,
            negative_prompt=neg,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.w, height=args.h,
            generator=g
        )

        if use_vec:
            # 取对应一条向量；若越界则循环利用
            emb = image_embeds[i % image_embeds.shape[0]].unsqueeze(0)
            kwargs["ip_adapter_image_embeds"] = emb
        else:
            ip_img = ip_images[i] if i < len(ip_images) else None
            if ip_img is None:
                print(f"[warn] 缺失参考图，第{i}条跳过")
                continue
            kwargs["ip_adapter_image"] = ip_img

        img = pipe(**kwargs).images[0]

        # 输出命名：优先 ids 名称
        if idx_to_name and i < len(idx_to_name):
            base = str(idx_to_name[i])
            out_path = os.path.join(args.out_dir, f"{base}.png")
        else:
            out_path = os.path.join(args.out_dir, f"{i:02d}.png")

        img.save(out_path)
        print(f"[ok] {i} -> {out_path}")
    except Exception as ex:
        print(f"[error] 生成 {i} 失败: {ex}")

print(f"[done] -> {args.out_dir}")
