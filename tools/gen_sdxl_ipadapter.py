#!/usr/bin/env python
import os, json, math, torch, torch.nn as nn

# ===== 输入/输出路径 =====
PROMPTS   = "/home/vipuser/train_logs/s1_ps1p5_h512_e5_cycle/inference/prompt_bigG.json"
BRAIN_VEC = "/home/vipuser/train_logs/s1_ps1p5_h512_e5_cycle/inference/brain_clip.pt"  # (N, 1664)
OUTDIR    = "/home/vipuser/train_logs/s1_ps1p5_h512_e5_cycle/inference/gen_ip"

# ===== SDXL + IP-Adapter 模型与权重 =====
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
# 这里请填你的 IP-Adapter SDXL 权重（safetensors 或者 HF Hub 名称）
IPADAPTER_WEIGHTS = os.environ.get("IP_SDXL_WEIGHTS", "")  # e.g. "/home/vipuser/models/ip-adapter/ip-adapter_sdxl.safetensors"

SEED, STEPS, GUIDANCE = 1234, 30, 6.0
H, W = 1024, 1024

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTDIR, exist_ok=True)

# ===== 读取数据 =====
items = json.load(open(PROMPTS, "r", encoding="utf-8"))
brain = torch.load(BRAIN_VEC, map_location="cpu").float()  # [N,1664]
assert brain.ndim==2 and brain.shape[1]==1664, brain.shape

# ===== 映射层：bigG(1664) -> IP-Adapter 期望维度(1024) =====
mapper = nn.Linear(1664, 1024, bias=True).to(device).eval()
# 初始化为单位映射近似：截断或随机都行，这里用 xavier，后续你可以微调它
nn.init.xavier_uniform_(mapper.weight); nn.init.zeros_(mapper.bias)

# ===== 尝试导入 IP-Adapter 管线 =====
try:
    from diffusers import StableDiffusionXLIPAdapterPipeline
    have_ip = True
except Exception as e:
    print("[warn] diffusers 未提供 StableDiffusionXLIPAdapterPipeline：", e)
    have_ip = False

if not have_ip or not IPADAPTER_WEIGHTS or not os.path.exists(IPADAPTER_WEIGHTS):
    print("[info] 未检测到 IP-Adapter 权重或管线不可用，先运行文本基线以避免阻塞。")
    # 文本基线兜底
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(SDXL_MODEL, torch_dtype=torch.float16)
    pipe.to(device).set_progress_bar_config(disable=True)

    g = torch.Generator(device=device).manual_seed(SEED)
    for i,p in enumerate(items):
        main = p.get("positive",""); style=p.get("style",""); negative=p.get("negative","")
        prompt = (main + ((", " + style) if style else "")).strip()
        img = pipe(prompt=prompt, negative_prompt=negative,
                   num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                   height=H, width=W, generator=g).images[0]
        img.save(os.path.join(OUTDIR, f"fallback_text_{i:03d}.png"))
    print(f"[fallback text-only] done -> {OUTDIR}")
    raise SystemExit(0)

# ===== 有 IP-Adapter：正式管线 =====
from diffusers import StableDiffusionXLIPAdapterPipeline
pipe = StableDiffusionXLIPAdapterPipeline.from_pretrained(
    SDXL_MODEL, torch_dtype=torch.float16
)
# 加载 IP-Adapter 权重（不同发行版 API 可能略异，这里采用通用 attach 方法）
pipe.load_ip_adapter(IPADAPTER_WEIGHTS)
pipe.set_ip_adapter_scale(0.7)  # 引导强度，0.5~0.9 之间调
pipe.to(device).set_progress_bar_config(disable=True)

g = torch.Generator(device=device).manual_seed(SEED)

# 归一化脑向量（cosine 空间更稳）
brain = brain / brain.norm(dim=-1, keepdim=True).clamp_min(1e-6)

for i,(p,vec) in enumerate(zip(items, brain)):
    main = p.get("positive",""); style=p.get("style",""); negative=p.get("negative","")
    prompt = (main + ((", " + style) if style else "")).strip()

    with torch.no_grad():
        z = mapper(vec.to(device))  # [1024]
        z = z.unsqueeze(0)          # [1,1024]
        # 有的实现要求 image_embeds 形状为 [B,1024] 或 dict，下面两种常见喂法二选一：
        image_embeds = z

    # 一些版本的 IP-Adapter 需要通过关键字 ip_adapter_image_embeds/ip_adapter_image
    # 这里优先尝试 embeds 接口，失败会降级到 image=None
    kwargs = dict(
        prompt=prompt, negative_prompt=negative,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
        height=H, width=W, generator=g
    )

    try:
        img = pipe(ip_adapter_image_embeds=image_embeds, **kwargs).images[0]
        tag = "ip"
    except Exception as e:
        print(f"[warn] embeds 接口失败（第{i}个）：{e}，尝试无脑图兜底")
        from diffusers import StableDiffusionXLPipeline
        txt_pipe = StableDiffusionXLPipeline.from_pretrained(SDXL_MODEL, torch_dtype=torch.float16).to(device)
        img = txt_pipe(**kwargs).images[0]
        tag = "text"

    img.save(os.path.join(OUTDIR, f"{tag}_{i:03d}.png"))

print(f"[ip-adapter] done -> {OUTDIR}, n={len(items)}")