#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_sdxl_with_peft.py

功能:
  一个集大成的SDXL生成脚本，融合了三大核心输入：
  1. 来自大脑的特征向量 (通过 IP-Adapter 注入)
  2. 来自LLM的结构化文本提示
  3. (可选) 针对特定被试的个性化PEFT适配器 (Soft-Prompt 和/或 Text-Encoder LoRA)

工作流程:
  - 加载 SDXL 基础模型和 IP-Adapter (Plus, ViT-H)。
  - (可选) 加载并应用针对文本编码器的 LoRA 权重。
  - 加载大脑解码出的特征向量 (e.g., 1664D) 并将其投影到 IP-Adapter 所需的 1280D 空间。
  - 在生成循环中，为每个样本：
    - (可选) 将 Soft-Prompt 嵌入与文本提示的嵌入进行拼接。
    - 将大脑特征向量作为图像嵌入传递给 IP-Adapter。
    - 执行扩散过程，生成最终图像。
"""

import os, json, argparse, math, gc
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch import nn
from PIL import Image
from diffusers import StableDiffusionXLPipeline
# We'll apply a compatibility shim for SlicedAttnProcessor if needed (see below)
from transformers import AutoTokenizer

# -------------------------- PEFT/Soft-Prompt 相关组件 --------------------------
try:
    from peft import PeftModel
    _has_peft = True
except ImportError:
    _has_peft = False

class SoftPrompt(nn.Module):
    def __init__(self, n_tokens: int, hidden_size: int):
        super().__init__()
        self.embeds = nn.Parameter(torch.zeros(n_tokens, hidden_size))
    def load(self, path: str, map_location='cpu'):
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd)
    def forward(self):
        return self.embeds

def build_prompt_embeds_with_peft(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str,
    soft_prompt: Optional[SoftPrompt] = None,
    device: torch.device = torch.device("cuda")
):
    """为SDXL构建文本嵌入，并可选地拼接Soft-Prompt。"""
    # SDXL 使用两个文本编码器
    text_encoder_one = pipe.text_encoder
    text_encoder_two = pipe.text_encoder_2
    tokenizer_one = pipe.tokenizer
    tokenizer_two = pipe.tokenizer_2

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]
    
    prompt_embeds_list = []
    
    # 获取正面提示的嵌入
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        
        if soft_prompt is not None:
            if soft_prompt.embeds.shape[1] == prompt_embeds.hidden_states[-1].shape[-1]:
                soft_embeds = soft_prompt().to(device).unsqueeze(0)
                prompt_embeds.hidden_states = list(prompt_embeds.hidden_states)
                prompt_embeds.hidden_states[-1] = torch.cat([soft_embeds, prompt_embeds.hidden_states[-1]], dim=1)
        
        prompt_embeds_list.append(prompt_embeds[0])

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    
    # 获取负面提示的嵌入 (通常不加soft-prompt)
    negative_prompt_embeds, pooled_negative_prompt_embeds = pipe.encode_prompt(
        prompt=negative_prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
    )
    
    # 从正面提示中提取 pooled embedding (使用第二个编码器的[CLS] token)
    pooled_prompt_embeds = prompt_embeds_list[1][:, 0]

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, pooled_negative_prompt_embeds


# -------------------------- 主逻辑 --------------------------
def main():
    AP = argparse.ArgumentParser(description="SDXL + IP-Adapter + PEFT 生成脚本 (显存优化版)")
    AP.add_argument("--adapter_dir", required=True, help="IP-Adapter local root")
    AP.add_argument("--prompts", required=True, help="Path to prompts json")
    AP.add_argument("--brain_vec_pt", required=True, help="Path to brain->CLIP vectors [N,D]")
    AP.add_argument("--proj_pt", default="", help="Optional projection ckpt for 1664->1024 mapping")
    AP.add_argument("--peft_adapter_dir", default="", help="[新增] 指向包含soft_tokens.pt和/或peft_text_lora的目录")
    AP.add_argument("--out_dir", required=True)
    AP.add_argument("--steps", type=int, default=28)
    AP.add_argument("--cfg", type=float, default=5.0)
    AP.add_argument("--w", type=int, default=1024)
    AP.add_argument("--h", type=int, default=1024)
    AP.add_argument("--seed", type=int, default=42)
    AP.add_argument("--dtype", choices=["fp16","fp32","bf16"], default="fp16")
    AP.add_argument("--ip_scale", type=float, default=0.8)
    AP.add_argument("--enable_cpu_offload", action="store_true", help="启用CPU Offload等一系列显存优化措施，会降低速度。")
    AP.add_argument("--limit", type=int, default=0, help="可选：只处理前N个样本，用于快速测试。") 
    args = AP.parse_args()

    # --- 设备 / 精度 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"[device] {device.type}, dtype={torch_dtype}")
    torch.manual_seed(args.seed)

    # --- 加载 SDXL + IP-Adapter ---
    print("[load] SDXL base 1.0")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if torch_dtype==torch.float16 else None,
    )
    
    if args.enable_cpu_offload and device.type == 'cuda':
        print("🟡 启用显存优化: Model CPU Offloading, VAE Slicing, Sequential Offloading...")
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)

    # 先仅启用 VAE slicing，注意力切片推迟到加载 IP-Adapter 之后
    pipe.enable_vae_slicing()

    # 兼容性补丁：为 diffusers 中可能缺少默认 slice_size 的 SlicedAttnProcessor 提供参数
    def apply_sliced_attn_processor_patch(pipe=None):
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

            # 如果传入了 pipe，则尝试对 UNet 中实际使用的类进行补丁
            if pipe is not None and hasattr(pipe, "unet") and hasattr(pipe.unet, "attn_processors"):
                for v in pipe.unet.attn_processors.values():
                    cls = v.__class__
                    if cls.__name__ == "SlicedAttnProcessor":
                        patched = _patch_class(cls) or patched

            if patched:
                print("✓ Applied SlicedAttnProcessor patch successfully.")
        except Exception as e:
            print(f"⚠️  Could not apply SlicedAttnProcessor patch: {e}.")

    apply_sliced_attn_processor_patch()

    print("[load] IP-Adapter (plus vit-h)")
    try:
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        )
    except TypeError as e:
        # 如果加载失败，尝试针对 pipe.unet 的子类做补丁后重试
        print(f"[warn] load_ip_adapter 失败({e}), 尝试应用 SlicedAttnProcessor 补丁并重试...")
        apply_sliced_attn_processor_patch(pipe)
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        )
    pipe.set_ip_adapter_scale(args.ip_scale)
    # 注意：不要在加载 IP-Adapter 之后覆盖 UNet 的 attention processor，也不要此时启用 attention slicing，
    # 以免把 IP-Adapter 安装的自定义 processor 覆盖掉（会导致 encoder_hidden_states 传入 tuple 时出错）。

    # --- 加载 PEFT 适配器 ---
    soft_prompt_instance = None
    soft_prompt_te1 = None  # embeds for text_encoder (dim ~1280)
    soft_prompt_te2 = None  # embeds for text_encoder_2 (dim ~768)
    if args.peft_adapter_dir and os.path.isdir(args.peft_adapter_dir):
        print(f"[peft] 正在从 {args.peft_adapter_dir} 加载适配器...")
        
        lora_dir = os.path.join(args.peft_adapter_dir, 'peft_text_lora')
        if _has_peft and os.path.isdir(lora_dir):
            try:
                print("  - 正在加载 Text-Encoder LoRA...")
                pipe.load_lora_weights(lora_dir)
                print("  ✓ 成功加载 LoRA 权重到管线")
            except Exception as e:
                print(f"  ✗ 加载LoRA失败: {e}")

        soft_prompt_path = os.path.join(args.peft_adapter_dir, 'soft_tokens.pt')
        if os.path.isfile(soft_prompt_path):
            print("  - 正在加载 Soft-Prompt...")
            sd = torch.load(soft_prompt_path, map_location='cpu')
            # 兼容多种保存格式：
            # - {'embeds': (n,dim)}
            # - {'embeds_te1': (n,1280), 'embeds_te2': (n,768)}
            if 'embeds_te1' in sd or 'embeds_te2' in sd:
                if 'embeds_te1' in sd:
                    E1 = torch.as_tensor(sd['embeds_te1'])
                    soft_prompt_te1 = nn.Parameter(E1.clone().detach())
                    print(f"  ✓ Soft-Prompt(te1) tokens={E1.shape[0]}, dim={E1.shape[1]}")
                if 'embeds_te2' in sd:
                    E2 = torch.as_tensor(sd['embeds_te2'])
                    soft_prompt_te2 = nn.Parameter(E2.clone().detach())
                    print(f"  ✓ Soft-Prompt(te2) tokens={E2.shape[0]}, dim={E2.shape[1]}")
                # 为了复用原有 SoftPrompt 结构，若某一侧未提供，则置为空
                if soft_prompt_te1 is None and soft_prompt_te2 is None:
                    print("  ✗ Soft-Prompt 文件未包含有效键，已跳过。")
                else:
                    # 标记存在 soft prompt
                    soft_prompt_instance = object()
            elif 'embeds' in sd:
                embeds = torch.as_tensor(sd['embeds'])
                n_tokens, hidden_size = embeds.shape
                # 暂存为通用 soft prompt，后续根据 encoder 维度匹配分配
                soft_prompt_instance = SoftPrompt(n_tokens, hidden_size)
                soft_prompt_instance.load_state_dict({'embeds': embeds})
                soft_prompt_instance = soft_prompt_instance.to(device)
                print(f"  ✓ 成功加载 Soft-Prompt (tokens={n_tokens}, dim={hidden_size})")
            else:
                print("  ✗ Soft-Prompt 文件格式不正确，已跳过。")

    # --- 加载并投影大脑向量 ---
    print("[load] brain vectors")
    V = torch.load(args.brain_vec_pt, map_location="cpu").float()
    N, D = V.shape
    print(f"  brain_vec shape: ({N}, {D})")

    # 投影逻辑
    if D == 1280:
        E1280 = V
    else:
        # 需要投影
        if D == 1664:
            if not args.proj_pt: raise ValueError("Brain vec is 1664D, requires --proj_pt for 1664->1024 mapping.")
            W_1664_to_1024 = torch.load(args.proj_pt, map_location="cpu")['W'].float()
            V = V @ W_1664_to_1024
            print(f"  ✓ Projected brain vec from 1664D to {V.shape[1]}D")
        
        if V.shape[1] == 1024:
            import open_clip
            oc_model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
            W_1024_to_1280 = oc_model.visual.proj.float()
            E1280 = V @ W_1024_to_1280.T
            print(f"  ✓ Projected brain vec from 1024D to 1280D")
        else:
            raise ValueError(f"Unsupported intermediate brain_vec dim: {V.shape[1]}")

    # SDXL IP-Adapter 期望的 image_embeds 为 [B, 1280]（后续内部会处理为需要的形状）
    E_tokens = E1280.to(device=device, dtype=torch_dtype)
    print(f"[embed] Final (N,1280) = {tuple(E1280.shape)}")

    # --- 加载Prompts ---
    with open(args.prompts, "r", encoding="utf-8") as f:
        raw_prompts = json.load(f)

    # --- 生成循环 ---
    os.makedirs(args.out_dir, exist_ok=True)
    B = E_tokens.shape[0]
    M = len(raw_prompts)
    T = min(B, M)
    if args.limit > 0:
        print(f"🟡 应用限制: 将只处理前 {args.limit} 个样本。")
        T = min(T, args.limit)
    print(f"[run] brain_vec={B}, prompts={M} -> 将生成 {T} 张图像")


    # 预检测两侧 text-encoder 的隐层维度与最大长度
    te1_hidden = None
    te2_hidden = None
    max_len_te1 = pipe.tokenizer.model_max_length if hasattr(pipe, 'tokenizer') else 77
    max_len_te2 = pipe.tokenizer_2.model_max_length if hasattr(pipe, 'tokenizer_2') else 77
    try:
        with torch.no_grad():
            ids1 = pipe.tokenizer([" "], padding="max_length", max_length=max_len_te1, return_tensors="pt").input_ids.to(device)
            out1 = pipe.text_encoder(ids1)
            te1_hidden = out1.last_hidden_state.shape[-1]
            ids2 = pipe.tokenizer_2([" "], padding="max_length", max_length=max_len_te2, return_tensors="pt").input_ids.to(device)
            out2 = pipe.text_encoder_2(ids2)
            te2_hidden = out2.last_hidden_state.shape[-1]
            print(f"[te] dims: te1={te1_hidden}, te2={te2_hidden}, max_len: te1={max_len_te1}, te2={max_len_te2}")
    except Exception as e:
        print(f"[warn] 读取 text-encoder 维度失败: {e}")

    def build_soft_prompt_embeds(pos: str, neg: str):
        """返回 dict，包含 prompt/negative 的 token 与 pooled。
        正样本的 token-embeds 会注入 soft-prompt；pooled 直接复用管线 encode_prompt 的结果。
        """
        # 先用管线获取标准 embedding（供负样本与 pooled 使用）
        enc_pos = pipe.encode_prompt(prompt=pos, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        enc_neg = pipe.encode_prompt(prompt=neg, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)

        def pick(enc, idx):
            if isinstance(enc, torch.Tensor):
                return enc
            if isinstance(enc, (list, tuple)) and len(enc) > idx and isinstance(enc[idx], torch.Tensor):
                return enc[idx]
            return None

        base_pos_tokens = pick(enc_pos, 0)
        base_pos_pooled = pick(enc_pos, 2)
        base_neg_tokens = pick(enc_neg, 0)
        base_neg_pooled = pick(enc_neg, 2)

        if base_pos_tokens is None or base_neg_tokens is None:
            # 回退：让管线自己编码
            return {
                'prompt': pos,
                'negative_prompt': neg,
            }

        B, L, Dtot = base_pos_tokens.shape  # Dtot 应为 te1_hidden+te2_hidden
        # 分拆两侧 hidden（若无法确定，则默认 te2_hidden 已读出）
        d1 = te1_hidden or (Dtot - (te2_hidden or 0))
        d2 = Dtot - d1
        pos_te1 = base_pos_tokens[:, :, :d1]
        pos_te2 = base_pos_tokens[:, :, d1:]

        # 根据 soft prompt 维度匹配拼接（截断到各自最大长度）
        def apply_soft(hs: torch.Tensor, sp_param: Optional[nn.Parameter], max_len: int) -> torch.Tensor:
            if sp_param is None:
                # 若提供了通用 SoftPrompt（soft_prompt_instance 是 SoftPrompt 实例），且维度匹配则使用
                if isinstance(soft_prompt_instance, SoftPrompt) and soft_prompt_instance.embeds.shape[1] == hs.shape[-1]:
                    sp = soft_prompt_instance().to(device=device, dtype=hs.dtype)  # (n,dim)
                else:
                    return hs
            else:
                sp = sp_param.to(device=device, dtype=hs.dtype)  # (n,dim)
            sp = sp.unsqueeze(0).expand(hs.shape[0], -1, -1)  # (B,n,dim)
            new_hs = torch.cat([sp, hs], dim=1)
            if new_hs.shape[1] > max_len:
                new_hs = new_hs[:, :max_len, :]
            return new_hs

        pos_te1_new = apply_soft(pos_te1, soft_prompt_te1, max_len_te1)
        pos_te2_new = apply_soft(pos_te2, soft_prompt_te2, max_len_te2)

        # 若两侧长度不同，按较小长度对齐（与 pipeline 习惯一致）
        L_new = min(pos_te1_new.shape[1], pos_te2_new.shape[1])
        pos_te1_new = pos_te1_new[:, :L_new, :]
        pos_te2_new = pos_te2_new[:, :L_new, :]
        pos_tokens_new = torch.cat([pos_te1_new, pos_te2_new], dim=-1)

        # 组装 kwargs（注意：pooled 直接用 encode_prompt 的结果，以保持稳定）
        return {
            'prompt_embeds': pos_tokens_new.to(device=device, dtype=torch_dtype),
            'negative_prompt_embeds': base_neg_tokens.to(device=device, dtype=torch_dtype),
            'pooled_prompt_embeds': base_pos_pooled.to(device=device, dtype=torch_dtype) if base_pos_pooled is not None else None,
            'negative_pooled_prompt_embeds': base_neg_pooled.to(device=device, dtype=torch_dtype) if base_neg_pooled is not None else None,
        }

    for i in range(T):
        rec = raw_prompts[i]
        pos = (rec.get("positive", "") + ", " + rec.get("style", "")).strip(", ")
        neg = rec.get("negative", "")
        
        try:
            # 构造 IP-Adapter embeddings: 需要 4D tensor [batch, num_images, seq_len, embed_dim]
            # 对于 IP-Adapter-plus，通常期望 [2, 1, 1, 1280] (uncond + cond)
            brain_embeds = E_tokens[i:i+1]  # [1, 1280]
            
            # 创建 negative (unconditioned) embeddings
            neg_embeds = torch.zeros_like(brain_embeds)  # [1, 1280]
            
            # 堆叠为 [neg, pos] 并调整为 4D: [batch=2, num_images=1, seq_len=1, embed_dim=1280]
            ip_embeds = torch.cat([neg_embeds, brain_embeds], dim=0)  # [2, 1280]
            ip_embeds = ip_embeds.unsqueeze(1).unsqueeze(2)  # [2, 1, 1, 1280]
            
            # 若存在 soft prompt，则构建注入后的 prompt_embeds；否则交给管线自行编码
            if soft_prompt_instance is not None:
                prompt_kwargs = build_soft_prompt_embeds(pos, neg)
            else:
                prompt_kwargs = {'prompt': pos, 'negative_prompt': neg}

            # Debug: inspect prompt kwargs and ip_embeds types/shapes to find tuple issues
            print("[debug] About to call pipe with the following prompt kwargs:")
            for k, v in prompt_kwargs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: Tensor shape={tuple(v.shape)}, dtype={v.dtype}")
                elif isinstance(v, (list, tuple)):
                    print(f"  {k}: {type(v)} len={len(v)}")
                    try:
                        for ii, vv in enumerate(v):
                            if isinstance(vv, torch.Tensor):
                                print(f"    [{ii}] Tensor shape={tuple(vv.shape)}, dtype={vv.dtype}")
                            else:
                                print(f"    [{ii}] type={type(vv)}")
                    except Exception:
                        pass
                else:
                    print(f"  {k}: type={type(v)}")

            if isinstance(ip_embeds, torch.Tensor):
                print(f"[debug] ip_embeds: Tensor shape={tuple(ip_embeds.shape)}, dtype={ip_embeds.dtype}")
            else:
                print(f"[debug] ip_embeds: type={type(ip_embeds)}")

            images = pipe(
                ip_adapter_image_embeds=[ip_embeds],
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                width=args.w,
                height=args.h,
                **prompt_kwargs
            ).images
            
            img = images[0]
            out_path = os.path.join(args.out_dir, f"{i:02d}.png")
            img.save(out_path)
            print(f"[ok] {i:02d} -> {out_path}")

        except Exception as e:
            import traceback
            print(f"[error] {i:02d} 生成失败: {type(e).__name__}: {e}")
            print(traceback.format_exc())
        
        finally:
            if device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
                # print(f"  - (mem) Cleared CUDA cache for iteration {i}")

    print(f"[done] -> {args.out_dir}")

if __name__ == "__main__":
    main()
