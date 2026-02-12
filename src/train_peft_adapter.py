# =============================================
# train_text_adapter.py  (skeleton)
# =============================================
"""
Minimal, per-subject training script for **Soft-Prompt** and/or **LoRA-on-text-encoder**
that plugs into a Stable Diffusion pipeline (SD1.5 by default). It optimizes a
lightweight objective so that images generated *from our structured prompts*
match the GT images in a CLIP-image space (optionally + LPIPS).

Compute budget friendly:
- Low-res 512 inference
- Few denoising steps (e.g., 10–20)
- Freeze UNet/VAEs/Schedulers; only train soft tokens and/or text-encoder LoRA

Inputs (recommended):
- A CSV listing training pairs per subject: subject_id, prompt, neg_prompt (opt), gt_image_path (opt), ip_embed_path (opt)
- IP-Adapter image embedding (optional). If provided, we pass it to the pipe.

Outputs (per subject directory):
- soft_tokens.pt   (if --train_soft)
- peft_text_lora/  (if --train_lora)
- train_log.csv

NOTE: This is a **skeleton** intended for your project wiring. You may refine
losses (e.g., add CLIP-T alignment, perceptual terms), batch sizes, schedulers,
etc. The goal is to give you a ready-to-run starting point.
"""

import os, math, csv, argparse, random
import numpy as np
import h5py
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Diffusers / Transformers / PEFT
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
from transformers import AutoTokenizer

# OpenCLIP for CLIP-image loss
import open_clip

# Optional LPIPS
try:
    import lpips
    _has_lpips = True
except Exception:
    _has_lpips = False

# PEFT for LoRA on text encoder
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    _has_peft = True
except Exception:
    _has_peft = False


# ----------------------------
# Small helpers
# ----------------------------

def seed_everything(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pil_to_tensor(img: Image.Image):
    return torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                             .float().view(img.size[1], img.size[0], len(img.getbands()))/255.0).numpy()).permute(2,0,1)


def load_image(path: str, size: int = 512):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return img


def _pil_from_ndarray(arr: np.ndarray) -> Image.Image:
    """Convert a numpy array (HWC or CHW, float[0,1] or uint8) to PIL RGB image."""
    if arr.ndim == 3:
        # CHW -> HWC if needed
        if arr.shape[-1] != 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
    # Normalize dtype and range
    if np.issubdtype(arr.dtype, np.floating):
        # assume [0,1] or similar
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr, mode='RGB')


def load_hdf5_image_by_index(h5_path: str, index: int, resize_hw: tuple[int, int]) -> Image.Image:
    """Load one image from HDF5 dataset 'images' by index and return PIL resized to (W,H)."""
    with h5py.File(h5_path, 'r') as hf:
        if 'images' not in hf:
            raise KeyError(f"HDF5 文件 {h5_path} 不包含 'images' 数据集")
        ds = hf['images']
        if index < 0 or index >= len(ds):
            raise IndexError(f"索引超界: {index} (共有 {len(ds)} 张图像)")
        arr = ds[index]
    pil = _pil_from_ndarray(np.array(arr))
    W, H = resize_hw
    return pil.convert('RGB').resize((W, H), Image.BICUBIC)


# ----------------------------
# Dataset
# ----------------------------
class PromptImageDataset(Dataset):
    def __init__(self, csv_path: str, subject: Optional[str] = None, size: int = 512, hdf5_path: Optional[str] = None):
        rows = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if subject is None or r['subject_id'] == subject:
                    rows.append(r)
        self.rows = rows
        self.size = size
        self.hdf5_path = hdf5_path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = r.get('prompt', '')
        neg = r.get('neg_prompt', '')
        gt_path = r.get('gt_image_path', '')
        ip_path = r.get('ip_embed_path', '')
        sample = {
            'prompt': prompt,
            'neg_prompt': neg,
            'gt_image_path': gt_path,
            'ip_embed_path': ip_path,
            'subject_id': r.get('subject_id', 'unknown')
        }
        # 如果 gt_image_path 是 hdf5_index_X 形式，解析索引
        if self.hdf5_path and gt_path.startswith('hdf5_index_'):
            try:
                idx = int(gt_path.split('hdf5_index_')[-1])
                sample['hdf5_index'] = idx
            except Exception:
                sample['hdf5_index'] = None
        return sample


# ----------------------------
# Soft-Prompt module
# ----------------------------
class SoftPrompt(nn.Module):
    def __init__(self, n_tokens: int, hidden_size: int, init_std: float = 0.02):
        super().__init__()
        self.embeds = nn.Parameter(torch.randn(n_tokens, hidden_size) * init_std)

    def forward(self):
        return self.embeds


# ----------------------------
# CLIP utility (image encoder only, for loss)
# ----------------------------
class ClipImageEncoder(nn.Module):
    def __init__(self, name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k"):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, pil_images: List[Image.Image], device: torch.device):
        imgs = torch.stack([self.preprocess(im).to(device) for im in pil_images], dim=0)
        with torch.cuda.amp.autocast(enabled=False):
            feats = self.model.encode_image(imgs)
            feats = F.normalize(feats.float(), dim=-1)
        return feats


# ----------------------------
# Build prompt embeds with soft tokens
# ----------------------------
def build_prompt_embeds(pipe: StableDiffusionPipeline, tokenizer, text_encoder, prompt: str, neg: str,
                        soft_prompt: Optional[SoftPrompt] = None, device: torch.device = torch.device('cuda')):
    # Tokenize
    text_inputs = tokenizer(
        [prompt], padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt'
    )
    neg_inputs = tokenizer(
        [neg] if neg else [""], padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt'
    )

    text_input_ids = text_inputs.input_ids.to(device)
    neg_input_ids = neg_inputs.input_ids.to(device)

    # Get token embeddings
    text_hidden = text_encoder(text_input_ids, output_hidden_states=True).last_hidden_state  # [1, T, H]
    neg_hidden = text_encoder(neg_input_ids, output_hidden_states=True).last_hidden_state

    if soft_prompt is not None:
        sp = soft_prompt().to(device)  # [Nsp, H]
        sp = sp.unsqueeze(0)  # [1, Nsp, H]
        text_hidden = torch.cat([sp, text_hidden], dim=1)
        neg_hidden = torch.cat([sp, neg_hidden], dim=1)  # 可选：也可另外一套 soft-neg

    return text_hidden, neg_hidden


# ----------------------------
# Train loop
# ----------------------------
@dataclass
class TrainConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    csv_path: str = "train_pairs.csv"
    subject_id: Optional[str] = None
    out_dir: str = "outputs/subj01_text_adapter"
    images_hdf5_path: Optional[str] = None

    seed: int = 1234
    train_soft: bool = True
    n_soft: int = 8
    train_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    lr: float = 5e-4
    wd: float = 0.0
    bs: int = 1
    epochs: int = 1
    steps: int = 1000

    ddim_steps: int = 15
    guidance: float = 6.0
    height: int = 512
    width: int = 512

    clip_name: str = "ViT-B-16"
    clip_ckpt: str = "laion2b_s34b_b88k"
    use_lpips: bool = False

    mixed_precision: str = "fp16"


def attach_text_lora_if_needed(text_encoder: nn.Module, cfg: TrainConfig):
    if not cfg.train_lora:
        return text_encoder, None
    assert _has_peft, "PEFT not installed. pip install peft"
    target_modules = []
    for n, m in text_encoder.named_modules():
        if isinstance(m, nn.Linear):
            target_modules.append(n)
    lcfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=target_modules, bias='none', task_type='SEQ_CLS'  # task_type is not used by text encoders but required
    )
    te = get_peft_model(text_encoder, lcfg)
    return te, lcfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default=TrainConfig.model_id)
    parser.add_argument('--csv_path', type=str, default=TrainConfig.csv_path)
    parser.add_argument('--subject_id', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=TrainConfig.out_dir)
    parser.add_argument('--images_hdf5_path', type=str, default=None, help='可选：从该HDF5(数据集名为images)按索引读取GT图像')
    parser.add_argument('--seed', type=int, default=TrainConfig.seed)
    parser.add_argument('--train_soft', action='store_true')
    parser.add_argument('--n_soft', type=int, default=TrainConfig.n_soft)
    parser.add_argument('--train_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=TrainConfig.lora_r)
    parser.add_argument('--lora_alpha', type=int, default=TrainConfig.lora_alpha)
    parser.add_argument('--lora_dropout', type=float, default=TrainConfig.lora_dropout)
    parser.add_argument('--lr', type=float, default=TrainConfig.lr)
    parser.add_argument('--wd', type=float, default=TrainConfig.wd)
    parser.add_argument('--bs', type=int, default=TrainConfig.bs)
    parser.add_argument('--epochs', type=int, default=TrainConfig.epochs)
    parser.add_argument('--steps', type=int, default=TrainConfig.steps)
    parser.add_argument('--ddim_steps', type=int, default=TrainConfig.ddim_steps)
    parser.add_argument('--guidance', type=float, default=TrainConfig.guidance)
    parser.add_argument('--height', type=int, default=TrainConfig.height)
    parser.add_argument('--width', type=int, default=TrainConfig.width)
    parser.add_argument('--clip_name', type=str, default=TrainConfig.clip_name)
    parser.add_argument('--clip_ckpt', type=str, default=TrainConfig.clip_ckpt)
    parser.add_argument('--use_lpips', action='store_true')
    parser.add_argument('--train_objective', type=str, default='denoise', choices=['denoise','clip'],
                        help='训练目标：denoise 使用扩散噪声预测MSE（可反传）；clip 使用生成图像的CLIP图像损失（仅推理，无法反传）')
    parser.add_argument('--mixed_precision', type=str, default=TrainConfig.mixed_precision, choices=['no', 'fp16', 'bf16'])
    args = parser.parse_args()

    cfg = TrainConfig(
        model_id=args.model_id,
        csv_path=args.csv_path,
        subject_id=args.subject_id,
        out_dir=args.out_dir,
        images_hdf5_path=args.images_hdf5_path,
        seed=args.seed,
        train_soft=args.train_soft,
        n_soft=args.n_soft,
        train_lora=args.train_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        wd=args.wd,
        bs=args.bs,
        epochs=args.epochs,
        steps=args.steps,
        ddim_steps=args.ddim_steps,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
        clip_name=args.clip_name,
        clip_ckpt=args.clip_ckpt,
        use_lpips=args.use_lpips,
        mixed_precision=args.mixed_precision,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_everything(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Pipeline (SD1.5) — freeze all
    pipe = StableDiffusionPipeline.from_pretrained(cfg.model_id, torch_dtype=torch.float16 if cfg.mixed_precision=='fp16' else torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.vae.eval(); pipe.text_encoder.eval(); pipe.unet.eval()
    for p in pipe.vae.parameters(): p.requires_grad_(False)
    for p in pipe.unet.parameters(): p.requires_grad_(False)

    tokenizer: AutoTokenizer = pipe.tokenizer
    text_encoder: nn.Module = pipe.text_encoder

    # 2) Attach LoRA on text encoder if requested
    text_encoder, lcfg = attach_text_lora_if_needed(text_encoder, cfg)

    # 3) Soft prompt init (dim = hidden size of text encoder)
    soft = None
    if cfg.train_soft:
        hidden = text_encoder.get_input_embeddings().embedding_dim
        soft = SoftPrompt(cfg.n_soft, hidden)
        soft = soft.to(device)

    # 4) CLIP image encoder for loss（仅在 clip 目标时使用）
    clip_img_enc = None
    if getattr(args, 'train_objective', 'denoise') == 'clip':
        clip_img_enc = ClipImageEncoder(cfg.clip_name, cfg.clip_ckpt).to(device)

    # 5) LPIPS (optional)
    if cfg.use_lpips and _has_lpips:
        lp = lpips.LPIPS(net='vgg').to(device)
    else:
        lp = None

    # 6) Optimizer
    params = []
    if cfg.train_soft:
        params += list(soft.parameters())
    if cfg.train_lora and _has_peft:
        for n, p in text_encoder.named_parameters():
            if 'lora_' in n and p.requires_grad:
                params.append(p)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)

    # 7) Data
    ds = PromptImageDataset(cfg.csv_path, cfg.subject_id, size=cfg.height, hdf5_path=cfg.images_hdf5_path)
    # HDF5 多进程读取可能不稳定，这里若使用HDF5则将 num_workers 设为0 更稳妥
    num_workers = 0 if cfg.images_hdf5_path is not None else 2
    dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=num_workers, collate_fn=lambda b: b)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.mixed_precision=='fp16'))

    # 训练用的扩散调度器（用于加噪/去噪步骤的时间步）
    train_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

    global_step = 0
    for epoch in range(cfg.epochs):
        for batch in dl:
            if global_step >= cfg.steps:
                break

            # each item handled independently (bs=1 recommended for light compute)
            losses = []
            optimizer.zero_grad()

            for item in batch:
                prompt = item['prompt']
                neg = item['neg_prompt']
                gt_path = item['gt_image_path']

                # 读取GT图像（用于计算latents）；denoise目标必须有GT
                gt_img = None
                if isinstance(gt_path, str) and os.path.isfile(gt_path):
                    gt_img = load_image(gt_path, size=cfg.height)
                elif cfg.images_hdf5_path is not None:
                    h5_index = None
                    if isinstance(gt_path, str) and gt_path.startswith('hdf5_index_'):
                        try:
                            h5_index = int(gt_path.split('hdf5_index_')[-1])
                        except Exception:
                            h5_index = None
                    elif isinstance(gt_path, str) and gt_path.isdigit():
                        h5_index = int(gt_path)
                    elif 'hdf5_index' in item and isinstance(item.get('hdf5_index'), int):
                        h5_index = item['hdf5_index']
                    if h5_index is not None:
                        try:
                            gt_img = load_hdf5_image_by_index(
                                cfg.images_hdf5_path, h5_index, (cfg.width, cfg.height)
                            )
                        except Exception as e:
                            print(f"[WARN] 从HDF5读取图像失败 index={h5_index}: {e}")

                # 构造条件与无条件文本嵌入（允许梯度，以便反传到soft/LoRA）
                pos_embeds, neg_embeds = build_prompt_embeds(pipe, tokenizer, text_encoder, prompt, neg, soft, device)
                # UNet 期望的 dtype（通常 fp16 当 mixed_precision=fp16）
                unet_dtype = next(pipe.unet.parameters()).dtype
                pos_embeds = pos_embeds.to(dtype=unet_dtype)
                neg_embeds = neg_embeds.to(dtype=unet_dtype)

                # 训练目标：denoise（可反传）或 clip（仅推理）
                if getattr(args, 'train_objective', 'denoise') == 'denoise':
                    if gt_img is None:
                        continue
                    # PIL -> VAE latents（不需对VAE反传）
                    img_np = np.array(gt_img).astype(np.float32) / 255.0
                    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W], 0..1
                    img_t = img_t * 2.0 - 1.0
                    # 确保与 VAE 参数 dtype 一致（避免 float32 输入到 fp16 VAE）
                    img_t = img_t.to(device=device, dtype=pipe.vae.dtype)
                    with torch.no_grad():
                        posterior = pipe.vae.encode(img_t).latent_dist
                        latents = posterior.sample()
                        latents = latents * 0.18215

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, train_scheduler.num_train_timesteps, (1,), device=device).long()
                    noisy_latents = train_scheduler.add_noise(latents, noise, timesteps)

                    # CFG：拼接无条件与条件分支
                    latent_model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
                    t_cat = torch.cat([timesteps, timesteps], dim=0)
                    encoder_hidden_states = torch.cat([neg_embeds, pos_embeds], dim=0)

                    with torch.autocast(device_type='cuda', dtype=torch.float16 if cfg.mixed_precision=='fp16' else torch.float32, enabled=(cfg.mixed_precision!='no')):
                        noise_pred = pipe.unet(latent_model_input, t_cat, encoder_hidden_states=encoder_hidden_states).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_cfg = noise_pred_uncond + cfg.guidance * (noise_pred_text - noise_pred_uncond)
                        loss = F.mse_loss(noise_pred_cfg, noise)

                    losses.append(loss)

                else:  # 'clip'（仅推理，无法反传）
                    # 生成图像（推理）
                    with torch.autocast(device_type='cuda', dtype=torch.float16 if cfg.mixed_precision=='fp16' else torch.float32, enabled=(cfg.mixed_precision!='no')):
                        out = pipe(prompt_embeds=pos_embeds,
                                   negative_prompt_embeds=neg_embeds,
                                   num_inference_steps=cfg.ddim_steps,
                                   guidance_scale=cfg.guidance,
                                   height=cfg.height, width=cfg.width)
                        gen = out.images[0]

                    if gt_img is None or clip_img_enc is None:
                        continue
                    with torch.no_grad():
                        gt_feat = clip_img_enc.encode([gt_img], device=device)
                        gen_feat = clip_img_enc.encode([gen], device=device)
                        cos = (gen_feat * gt_feat).sum(dim=-1)
                        loss = 1.0 - cos.mean()
                        if lp is not None:
                            gen_t = pil_to_tensor(gen).unsqueeze(0).to(device)
                            gt_t  = pil_to_tensor(gt_img).unsqueeze(0).to(device)
                            loss = loss + 0.2 * lp(gen_t*2-1, gt_t*2-1).mean()
                    # 注意：clip 模式不反传梯度，这里不加入 losses 以避免 backward 报错
                    # 若需要将其当作评估指标，可在此记录到日志

            if not losses:
                continue

            loss_total = torch.stack(losses).mean()
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % 10 == 0:
                print(f"step {global_step} | loss {loss_total.item():.4f}")

            # Save periodically
            if global_step % 100 == 0:
                subj_dir = cfg.out_dir
                os.makedirs(subj_dir, exist_ok=True)
                if cfg.train_soft:
                    torch.save(soft.state_dict(), os.path.join(subj_dir, 'soft_tokens.pt'))
                if cfg.train_lora and _has_peft:
                    text_encoder.save_pretrained(os.path.join(subj_dir, 'peft_text_lora'))

        if global_step >= cfg.steps:
            break

    # Final save
    subj_dir = cfg.out_dir
    os.makedirs(subj_dir, exist_ok=True)
    if cfg.train_soft:
        torch.save(soft.state_dict(), os.path.join(subj_dir, 'soft_tokens.pt'))
    if cfg.train_lora and _has_peft:
        text_encoder.save_pretrained(os.path.join(subj_dir, 'peft_text_lora'))


if __name__ == "__main__":
    main()


