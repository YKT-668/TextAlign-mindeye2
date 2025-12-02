# =============================================
# apply_text_adapter.py  (skeleton)
# =============================================
"""
Load per-subject soft-prompt and/or text-encoder LoRA, then generate images with
our structured prompts. Optionally load IP-Adapter image embeddings (.pt) and
ControlNet hints (edge/depth) if your pipeline is extended accordingly.
"""

import os, argparse
import torch
import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import AutoTokenizer

# Reuse SoftPrompt from above
class SoftPrompt(nn.Module):
    def __init__(self, n_tokens: int, hidden_size: int):
        super().__init__()
        self.embeds = nn.Parameter(torch.zeros(n_tokens, hidden_size))
    def load(self, path: str, map_location='cpu'):
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd)
    def forward(self):
        return self.embeds

@torch.no_grad()
def build_prompt_embeds(pipe, tokenizer, text_encoder, prompt: str, neg: str, soft_prompt: SoftPrompt=None, device=torch.device('cuda')):
    text_inputs = tokenizer([prompt], padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    neg_inputs  = tokenizer([neg] if neg else [""], padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_hidden = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True).last_hidden_state
    neg_hidden  = text_encoder(neg_inputs.input_ids.to(device), output_hidden_states=True).last_hidden_state
    if soft_prompt is not None:
        sp = soft_prompt().to(device).unsqueeze(0)
        text_hidden = torch.cat([sp, text_hidden], dim=1)
        neg_hidden  = torch.cat([sp, neg_hidden], dim=1)
    return text_hidden, neg_hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    ap.add_argument('--out', type=str, default='gen.png')
    ap.add_argument('--prompt', type=str, required=True)
    ap.add_argument('--neg', type=str, default='')
    ap.add_argument('--adapter_dir', type=str, required=True, help='dir containing soft_tokens.pt and/or peft_text_lora')
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=7.5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.vae.eval(); pipe.unet.eval(); pipe.text_encoder.eval()

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Load text-encoder LoRA if present
    lora_dir = os.path.join(args.adapter_dir, 'peft_text_lora')
    if os.path.isdir(lora_dir):
        try:
            from peft import PeftModel
            text_encoder = PeftModel.from_pretrained(text_encoder, lora_dir)
            text_encoder.eval()
            print('[Info] Loaded text-encoder LoRA from', lora_dir)
        except Exception as e:
            print('[Warn] Failed to load PEFT LoRA:', e)

    # Load soft prompt if present
    soft_path = os.path.join(args.adapter_dir, 'soft_tokens.pt')
    soft = None
    if os.path.isfile(soft_path):
        # 推断 checkpoint 中 soft tokens 的形状
        sd = torch.load(soft_path, map_location='cpu')
        if isinstance(sd, dict) and 'embeds' in sd:
            n_tokens_ckpt, hidden_ckpt = sd['embeds'].shape
        else:
            raise RuntimeError(f'Invalid soft_tokens checkpoint format: {soft_path}')

        # 以 checkpoint 形状创建 SoftPrompt 并加载权重
        soft = SoftPrompt(n_tokens=n_tokens_ckpt, hidden_size=hidden_ckpt)
        soft.load_state_dict(sd)
        soft = soft.to(device)
        print(f'[Info] Loaded soft tokens from {soft_path} (n_tokens={n_tokens_ckpt}, hidden={hidden_ckpt})')

    pos_embeds, neg_embeds = build_prompt_embeds(pipe, tokenizer, text_encoder, args.prompt, args.neg, soft, device)
    # 确保与 UNet dtype 对齐
    unet_dtype = next(pipe.unet.parameters()).dtype
    pos_embeds = pos_embeds.to(dtype=unet_dtype)
    neg_embeds = neg_embeds.to(dtype=unet_dtype)

    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
        out = pipe(prompt_embeds=pos_embeds,
                   negative_prompt_embeds=neg_embeds,
                   num_inference_steps=args.steps,
                   guidance_scale=args.guidance,
                   height=args.height, width=args.width)
        img = out.images[0]
    img.save(args.out)
    print('Saved to', args.out)

if __name__ == '__main__':
    main()


# =============================================
# README (usage cheatsheet)
# =============================================
"""
1) 安装（建议新建 env）

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # 视CUDA改
pip install diffusers==0.30.0 transformers accelerate peft open_clip_torch lpips safetensors

2) 准备 CSV（最小列）
subject_id,prompt,neg_prompt,gt_image_path,ip_embed_path
subj01,"a brown dog running on grass","blurry, low quality","/path/to/gt1.jpg",
subj01,"a red car on a street","", "/path/to/gt2.jpg",
...

3) 训练（每被试一个输出目录）
python train_text_adapter.py \
  --model_id runwayml/stable-diffusion-v1-5 \
  --csv_path /path/to/train_pairs.csv \
  --subject_id subj01 \
  --out_dir outputs/subj01_text_adapter \
  --train_soft --n_soft 12 \
  --train_lora --lora_r 8 \
  --steps 800 --epochs 1 --bs 1 \
  --ddim_steps 15 --guidance 6.0 \
  --use_lpips

4) 推理
python apply_text_adapter.py \
  --model_id runwayml/stable-diffusion-v1-5 \
  --adapter_dir outputs/subj01_text_adapter \
  --prompt "<这里放RAG+LLM融合出的结构化主提示>" \
  --neg "<结构化负面提示>" \
  --out demo.png --steps 30 --guidance 7.5

5) 与 MindEye2 串接
- 前半段（fMRI→CLIP-image→RAG→LLM）保持你现有流程；
- 将 LLM 产出的主/负面提示传给 apply_text_adapter.py；
- 同时在你现有生成端加入 IP-Adapter（可与本脚本并行使用，代码留有挂点）。

注意：本骨架用“CLIP-image 距离 + 可选 LPIPS”作为轻量代理损失，避免端到端扩散反传的高算力消耗。后续可以：
- 用 GT caption 的 CLIP-I/T 对齐补充损失；
- 在少量步（e.g., 5 steps）做 UNet 低成本反传以进一步贴近数据分布；
- SDXL 替代 SD1.5（同样接口），或给 UNet Cross-Attn 再挂 LoRA 作 ablation。
"""
