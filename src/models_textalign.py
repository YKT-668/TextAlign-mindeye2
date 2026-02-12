# models_textalign.py
# 兼容：train_textalign / eval_textalign_latent / MindEye2-style pipeline
# 目标：稳定、可复现、维度严格检查、optional 依赖友好报错


from __future__ import annotations

import os
import random
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# optional deps
try:
    import clip  # type: ignore
except Exception as e:  # pragma: no cover
    clip = None
    print(f"[WARN] optional dependency 'clip' not available: {type(e).__name__}: {e}")

try:
    import utils  # type: ignore
except Exception as e:  # pragma: no cover
    utils = None
    print(f"[WARN] optional dependency 'utils' not available: {type(e).__name__}: {e}")

# for prior（可选依赖）
try:
    from dalle2_pytorch import DiffusionPrior  # type: ignore
    from dalle2_pytorch.dalle2_pytorch import (  # type: ignore
        l2norm,
        default,
        exists,
        RotaryEmbedding,
        CausalTransformer,
        SinusoidalPosEmb,
        MLP,
        Rearrange,
        repeat,
        rearrange,
        prob_mask_like,
        LayerNorm,
        RelPosBias,
        Attention,
        FeedForward,
    )
    _DALLE2_AVAILABLE = True
except Exception:
    DiffusionPrior = None
    _DALLE2_AVAILABLE = False


# ============================================================
# TextAlign Head
# ============================================================
class TextAlignHead(nn.Module):
    """
    输入：
      - bigG image/brain tokens: [B, T, C] 例如 [B, 256, 1664]
      - 或 pooled 向量: [B, C]
    输出：
      - 文本向量 [B, d_text] 例如 d_text=768
    """

    def __init__(self, token_dim: int = 1664, hidden_dim: int = 2048, text_dim: int = 768):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.text_dim = int(text_dim)

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.text_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"TextAlignHead expects torch.Tensor, got {type(x)}")
        if x.dim() == 3:
            # [B, T, C] -> mean pool -> [B, C]
            x = x.mean(dim=1)
        elif x.dim() != 2:
            raise ValueError(f"TextAlignHead expects [B,T,C] or [B,C], got shape={tuple(x.shape)}")

        if x.shape[-1] != self.token_dim:
            raise ValueError(f"TextAlignHead token_dim mismatch: expected {self.token_dim}, got {x.shape[-1]}")
        return self.mlp(x)


# ============================================================
# BrainNetwork (MindEye2-style): ridge out -> tokens -> clip tokens + blurry latent
# ============================================================
class BrainNetwork(nn.Module):
    """
    约定输出：
      backbone_tokens: [B, n_tokens, clip_size]  (用于 prior / TextAlign)
      clip_tokens    : [B, n_tokens, clip_size]  (用于 clip contrastive)
      blurry_tuple   : (latent_pred [B,4,28,28], aux_tokens [B,49,512])
    """

    def __init__(
        self,
        h: int = 4096,
        in_dim: int = 15724,          # 为兼容旧接口保留，但不一定用到
        out_dim: int = 768,
        seq_len: int = 2,
        n_blocks: int = 4,
        drop: float = 0.15,
        clip_size: int = 768,
        blurry_recon: bool = True,
        clip_scale: float = 1.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.h = int(h)
        self.clip_size = int(clip_size)
        self.blurry_recon = bool(blurry_recon)
        self.clip_scale = float(clip_scale)

        if out_dim % self.clip_size != 0:
            raise ValueError(
                f"[BrainNetwork] out_dim ({out_dim}) must be divisible by clip_size ({self.clip_size}). "
                f"Otherwise reshape to tokens will silently break."
            )
        self.out_dim = int(out_dim)
        self.n_tokens = int(out_dim // self.clip_size)

        # Mixer blocks
        self.mixer_blocks1 = nn.ModuleList([self._mixer_block1(self.h, drop) for _ in range(n_blocks)])
        self.mixer_blocks2 = nn.ModuleList([self._mixer_block2(self.seq_len, drop) for _ in range(n_blocks)])

        # backbone -> tokens
        self.backbone_linear = nn.Linear(self.h * self.seq_len, self.out_dim, bias=True)

        # token->token projector (predict CLIP tokens)
        self.clip_proj = self._projector(self.clip_size, self.clip_size, h=self.clip_size)

        # Blurry recon head
        self._init_blurry_modules()

    def _projector(self, in_dim: int, out_dim: int, h: int = 2048) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim),
        )

    def _mlp(self, in_dim: int, out_dim: int, drop: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )

    def _mixer_block1(self, h: int, drop: float) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(h),
            self._mlp(h, h, drop),  # token-mixing (最后维度是 h)
        )

    def _mixer_block2(self, seq_len: int, drop: float) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self._mlp(seq_len, seq_len, drop),  # channel-mixing (最后维度是 seq_len)
        )

    def _init_blurry_modules(self) -> None:
        if not self.blurry_recon:
            self.blin1 = None
            self.bdropout = None
            self.bnorm = None
            self.bupsampler = None
            self.b_maps_projector = None
            return

        # 延迟导入：避免 blurry_recon=False 时强依赖 diffusers
        try:
            from diffusers.models.autoencoders.vae import Decoder  # type: ignore
        except Exception as e:
            raise ImportError(
                f"[BrainNetwork] blurry_recon=True requires diffusers. Import failed: {type(e).__name__}: {e}"
            )

        # 7x7x64 -> Decoder -> 4 x H x W -> 强制 resize 到 28x28
        self.blin1 = nn.Linear(self.h * self.seq_len, 4 * 28 * 28, bias=True)
        self.bdropout = nn.Dropout(0.3)
        self.bnorm = nn.GroupNorm(1, 64)

        self.bupsampler = Decoder(
            in_channels=64,
            out_channels=4,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[32, 64, 128],
            layers_per_block=1,
        )

        # aux token features (for cont_loss)
        self.b_maps_projector = nn.Sequential(
            nn.Conv2d(64, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: [B, seq_len, h]
        return:
          backbone_tokens: [B, n_tokens, clip_size]
          clip_tokens    : [B, n_tokens, clip_size]
          (latent_pred [B,4,28,28], aux_tokens [B,49,512])
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"BrainNetwork expects torch.Tensor, got {type(x)}")
        if x.dim() != 3:
            raise ValueError(f"BrainNetwork expects [B,seq_len,h], got shape={tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"BrainNetwork seq_len mismatch: expected {self.seq_len}, got {x.shape[1]}")
        if x.shape[2] != self.h:
            raise ValueError(f"BrainNetwork h mismatch: expected {self.h}, got {x.shape[2]}")

        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0, 2, 1)  # [B, h, seq_len]

        for block1, block2 in zip(self.mixer_blocks1, self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x

            x = x.permute(0, 2, 1)  # [B, h, seq_len]
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0, 2, 1)  # back to [B, seq_len, h]

        # flatten -> tokens
        flat = x.reshape(x.size(0), -1)  # [B, h*seq_len]
        tokens_flat = self.backbone_linear(flat)  # [B, out_dim]
        # 显式 reshape：避免 out_dim/clip_size 不一致时 silent 错
        backbone_tokens = tokens_flat.view(x.size(0), self.n_tokens, self.clip_size).contiguous()

        # predict clip tokens
        # 即便 clip_scale==0，也返回一个形状正确的 tensor，避免外部加法/平均炸掉
        if self.clip_scale > 0:
            clip_tokens = self.clip_proj(backbone_tokens)
        else:
            clip_tokens = backbone_tokens

        # blurry tuple：保持 (latent_pred, aux_tokens) 结构恒定
        if self.blurry_recon:
            assert self.blin1 is not None
            assert self.bdropout is not None
            assert self.bnorm is not None
            assert self.bupsampler is not None
            assert self.b_maps_projector is not None

            b = self.blin1(flat)          # [B, 3136]
            b = self.bdropout(b)
            b = b.view(b.shape[0], 64, 7, 7).contiguous()
            b = self.bnorm(b)

            # aux tokens: [B,49,512]
            b_aux = self.b_maps_projector(b).flatten(2).permute(0, 2, 1).contiguous()  # [B,49,512]
            if b_aux.shape[1] != 49 or b_aux.shape[2] != 512:
                # 极端情况下做一次强制修正（一般不会触发）
                b_aux = b_aux.view(b_aux.shape[0], 49, 512)

            # decoder output -> latent_pred: [B,4,H,W]
            b_dec = self.bupsampler(b)
            if hasattr(b_dec, "sample"):  # diffusers DecoderOutput(sample=...)
                b_dec = b_dec.sample
            if not isinstance(b_dec, torch.Tensor):
                raise RuntimeError(f"[BrainNetwork] bupsampler returned non-tensor: {type(b_dec)}")

            # 对齐到 28x28 latent
            if b_dec.dim() != 4 or b_dec.shape[1] != 4:
                raise RuntimeError(f"[BrainNetwork] decoder output must be [B,4,H,W], got {tuple(b_dec.shape)}")
            if b_dec.shape[-2:] != (28, 28):
                b_dec = F.interpolate(b_dec, size=(28, 28), mode="bilinear", align_corners=False)

            blurry_tuple = (b_dec, b_aux)

        else:
            # 占位输出：保持 tuple 结构不变（避免某些调试/导出代码直接崩）
            b_dec = torch.zeros((x.size(0), 4, 28, 28), device=x.device, dtype=x.dtype)
            b_aux = torch.zeros((x.size(0), 49, 512), device=x.device, dtype=x.dtype)
            blurry_tuple = (b_dec, b_aux)

        return backbone_tokens, clip_tokens, blurry_tuple


# ============================================================
# CLIP helper (optional)
# ============================================================
class Clipper(nn.Module):
    def __init__(
        self,
        clip_variant: str,
        clamp_embs: bool = False,
        norm_embs: bool = False,
        hidden_state: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        if clip is None:
            raise ImportError(
                "Clipper requires the 'clip' package (openai/CLIP). "
                "Install it or avoid using Clipper in your pipeline."
            )

        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64"

        self.device = torch.device(device)
        self.clip_variant = clip_variant
        self.hidden_state = bool(hidden_state)

        # hidden_state embeddings only works with ViT-L/14 right now
        self.image_encoder = None
        if clip_variant == "ViT-L/14" and hidden_state:
            from transformers import CLIPVisionModelWithProjection  # type: ignore

            sd_cache_dir = '/fsx/proj-fmri/shared/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_cache_dir, subfolder='image_encoder').eval()
            image_encoder = image_encoder.to(self.device)
            for p in image_encoder.parameters():
                p.requires_grad_(False)
            self.image_encoder = image_encoder
        elif hidden_state:
            raise ValueError("hidden_state embeddings only works with ViT-L/14 right now")

        clip_model, _ = clip.load(clip_variant, device=self.device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
        self.clip = clip_model

        if clip_variant == "RN50x64":
            self.clip_size = (448, 448)
        else:
            self.clip_size = (224, 224)

        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])

        self.preprocess = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=tuple(self.mean.tolist()), std=tuple(self.std.tolist())),
        ])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clamp_embs = bool(clamp_embs)
        self.norm_embs = bool(norm_embs)

    def resize_image(self, image: torch.Tensor) -> torch.Tensor:
        return transforms.Resize(self.clip_size)(image.to(self.device))

    def _versatile_normalize_embeddings(self, encoder_output) -> torch.Tensor:
        # 使用 self.image_encoder，而不是闭包里不存在的 image_encoder 变量
        assert self.image_encoder is not None
        embeds = encoder_output.last_hidden_state
        embeds = self.image_encoder.vision_model.post_layernorm(embeds)
        embeds = self.image_encoder.visual_projection(embeds)
        return embeds

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Expects images in -1..1 range (沿用你原注释；具体取决于你喂给它的图像范围)"""
        if self.hidden_state:
            if self.image_encoder is None:
                raise RuntimeError("hidden_state=True but image_encoder is None")
            clip_in = self.preprocess(image.to(self.device))
            out = self.image_encoder(clip_in)
            clip_emb = self._versatile_normalize_embeddings(out)
        else:
            clip_in = self.preprocess(image.to(self.device))
            clip_emb = self.clip.encode_image(clip_in)

        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5)

        if self.norm_embs:
            if self.hidden_state:
                clip_emb = clip_emb / torch.norm(clip_emb[:, 0], dim=-1).reshape(-1, 1, 1)
            else:
                clip_emb = F.normalize(clip_emb, dim=-1)
        return clip_emb

    def embed_text(self, text_samples) -> torch.Tensor:
        if clip is None:
            raise RuntimeError("clip is not available")
        clip_text = clip.tokenize(text_samples).to(self.device)
        clip_text = self.clip.encode_text(clip_text)
        if self.clamp_embs:
            clip_text = torch.clamp(clip_text, -1.5, 1.5)
        if self.norm_embs:
            clip_text = F.normalize(clip_text, dim=-1)
        return clip_text

    def embed_curated_annotations(self, annots) -> torch.Tensor:
        if utils is None:
            raise ImportError("utils is required for embed_curated_annotations")
        txt = []
        for b in annots:
            t = ""
            while t == "":
                rand = torch.randint(5, (1, 1))[0][0]
                t = b[0, rand]
            txt.append(np.array(t))
        txt = np.vstack(txt).flatten()
        return self.embed_text(txt)


# ============================================================
# Diffusion prior (optional) - robustness fixes
# ============================================================
class BrainDiffusionPrior(DiffusionPrior if _DALLE2_AVAILABLE else nn.Module):
    """
    修复点：
    - forward 的 XOR 断言（a^b^c）会允许 3 个都 True：改为 sum==1
    - generator 参数真正生效
    """

    def __init__(self, *args, **kwargs):
        if not _DALLE2_AVAILABLE:
            raise ImportError("dalle2_pytorch is required for BrainDiffusionPrior; install it or disable use_prior.")
        voxel2clip = kwargs.pop("voxel2clip", None)
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        text_cond=None,
        self_cond=None,
        clip_denoised=True,
        cond_scale=1.0,
        generator=None,
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=t, text_cond=text_cond, self_cond=self_cond,
            clip_denoised=clip_denoised, cond_scale=cond_scale
        )

        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps=None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps=timesteps)

        image_embed = normalized_image_embed
        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale=1.0, generator=None):
        batch, device = shape[0], self.device
        if generator is None:
            image_embed = torch.randn(shape, device=device)
        else:
            image_embed = torch.randn(shape, device=device, generator=generator)

        x_start = None
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in reversed(range(0, self.noise_scheduler.num_timesteps)):
            times = torch.full((batch,), i, device=device, dtype=torch.long)
            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(
                image_embed, times, text_cond=text_cond, self_cond=self_cond,
                cond_scale=cond_scale, generator=generator
            )

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise=None, generator=None):
        if noise is None:
            noise = torch.randn_like(image_embed) if generator is None else torch.randn(
                image_embed.size(), device=image_embed.device, dtype=image_embed.dtype, generator=generator
            )

        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond=self_cond,
            text_cond_drop_prob=self.text_cond_drop_prob,
            image_cond_drop_prob=self.image_cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = F.mse_loss(pred, target)
        return loss, pred

    def forward(
        self,
        text=None,
        image=None,
        voxel=None,
        text_embed=None,
        image_embed=None,
        text_encodings=None,
        *args,
        **kwargs
    ):
        if not _DALLE2_AVAILABLE:
            raise ImportError("dalle2_pytorch is required")

        # exactly-one checks (修复原来的 XOR bug)
        n_text_inputs = int(exists(text)) + int(exists(text_embed)) + int(exists(voxel))
        if n_text_inputs != 1:
            raise AssertionError("Exactly one of {text, text_embed, voxel} must be supplied.")
        if int(exists(image)) + int(exists(image_embed)) != 1:
            raise AssertionError("Exactly one of {image, image_embed} must be supplied.")
        if self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text)):
            raise AssertionError("text encodings must be present if condition_on_text_encodings=True")

        if exists(voxel):
            if self.voxel2clip is None:
                raise AssertionError("voxel2clip must be trained if you wish to pass in voxels")
            if exists(text_embed):
                raise AssertionError("cannot pass in both text_embed and voxels")

            if getattr(self.voxel2clip, "use_projector", False):
                clip_voxels_mse, clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse
            else:
                clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels  # 统一为 text_embed

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed=text_embed)
        if self.condition_on_text_encodings:
            if not exists(text_encodings):
                raise AssertionError("text_encodings must be present for diffusion prior if specified")
            text_cond = {**text_cond, "text_encodings": text_encodings}

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # forward loss
        loss, pred = self.p_losses(image_embed, times, text_cond=text_cond, *args, **kwargs)
        return loss, pred


class FlaggedCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_in=False,
        norm_out=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        final_proj=True,
        normformer=False,
        rotary_emb=True,
        causal=True,
    ):
        super().__init__()
        if not _DALLE2_AVAILABLE:
            raise ImportError("dalle2_pytorch is required for FlaggedCausalTransformer")

        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()
        self.rel_pos_bias = RelPosBias(heads=heads)
        rotary = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, causal=causal, dim_head=dim_head, heads=heads, dropout=attn_dropout, rotary_emb=rotary),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, post_activation_norm=normformer),
            ]))

        self.norm = LayerNorm(dim, stable=True) if norm_out else nn.Identity()
        self.project_out = nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, device = x.shape[1], x.device
        x = self.init_norm(x)
        attn_bias = self.rel_pos_bias(n, n + 1, device=device)
        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
        out = self.norm(x)
        return self.project_out(out)


class PriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps=None,
        num_time_embeds=1,
        num_tokens=257,
        causal=True,
        learned_query_mode="none",
        **kwargs
    ):
        super().__init__()
        if not _DALLE2_AVAILABLE:
            raise ImportError("dalle2_pytorch is required for PriorNetwork; install it or disable use_prior.")

        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode
        self.num_tokens = num_tokens
        self.self_cond = False

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds)
            if exists(num_timesteps)
            else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)),
            Rearrange("b (n d) -> b n d", n=num_time_embeds),
        )

        if learned_query_mode in ("token", "pos_emb", "all_pos_emb"):
            scale = dim ** -0.5
            if learned_query_mode == "all_pos_emb":
                self.learned_query = nn.Parameter(torch.randn(num_tokens * 2 + 1, dim) * scale)
            else:
                self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        else:
            self.learned_query = None

        self.causal_transformer = FlaggedCausalTransformer(dim=dim, causal=causal, **kwargs)

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, brain_cond_drop_prob=1.0, image_cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        self_cond=None,
        brain_embed=None,
        text_embed=None,
        brain_cond_drop_prob=0.0,
        text_cond_drop_prob=None,
        image_cond_drop_prob=0.0,
    ):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob
        if brain_embed is None:
            raise ValueError("brain_embed (or text_embed) must be provided")

        batch, _, dim = image_embed.shape[0], image_embed.shape[1], image_embed.shape[2]
        device, dtype = image_embed.device, image_embed.dtype

        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device=device)
        brain_keep_mask = rearrange(brain_keep_mask, "b -> b 1 1")

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device=device)
        image_keep_mask = rearrange(image_keep_mask, "b -> b 1 1")

        null_brain = self.null_brain_embeds.to(brain_embed.dtype)
        brain_embed = torch.where(brain_keep_mask, brain_embed, null_brain[None])

        null_image = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(image_keep_mask, image_embed, null_image[None])

        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        learned_queries = torch.empty((batch, 0, dim), device=device, dtype=brain_embed.dtype)
        if self.learned_query_mode == "token":
            learned_queries = repeat(self.learned_query, "n d -> b n d", b=batch)
        elif self.learned_query_mode == "pos_emb":
            pos_embs = repeat(self.learned_query, "n d -> b n d", b=batch)
            image_embed = image_embed + pos_embs
        elif self.learned_query_mode == "all_pos_emb":
            pos_embs = repeat(self.learned_query, "n d -> b n d", b=batch)

        tokens = torch.cat((brain_embed, time_embed, image_embed, learned_queries), dim=-2)
        if self.learned_query_mode == "all_pos_emb":
            tokens = tokens + pos_embs

        tokens = self.causal_transformer(tokens)
        pred_image_embed = tokens[..., -self.num_tokens:, :]
        return pred_image_embed


# ============================================================
# GNet8 parts (保留但加固：utils 缺失时报错更清晰)
# ============================================================
class TrunkBlock(nn.Module):
    def __init__(self, feat_in, feat_out):
        super().__init__()
        self.conv1 = nn.Conv2d(feat_in, int(feat_out * 1.0), kernel_size=3, stride=1, padding=1, dilation=1)
        self.drop1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(feat_in, eps=1e-5, momentum=0.25, affine=True, track_running_stats=True)

        torch.nn.init.xavier_normal_(self.conv1.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.constant_(self.conv1.bias, 0.0)

    def forward(self, x):
        return F.relu(self.conv1(self.drop1(self.bn1(x))))


class PreFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        c1 = self.conv1(x)
        y = self.conv2(c1)
        return y


class EncStage(nn.Module):
    def __init__(self, trunk_width=64, pass_through=64):
        super().__init__()
        self.conv3 = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=0)
        self.drop1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(192, eps=1e-5, momentum=0.25, affine=True, track_running_stats=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.tw = int(trunk_width)
        self.pt = int(pass_through)
        ss = (self.tw + self.pt)
        self.conv4a = TrunkBlock(128, ss)
        self.conv5a = TrunkBlock(ss, ss)
        self.conv6a = TrunkBlock(ss, ss)
        self.conv4b = TrunkBlock(ss, ss)
        self.conv5b = TrunkBlock(ss, ss)
        self.conv6b = TrunkBlock(ss, self.tw)

        torch.nn.init.xavier_normal_(self.conv3.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.constant_(self.conv3.bias, 0.0)

    def forward(self, x):
        c3 = F.relu(self.conv3(self.drop1(self.bn1(x))), inplace=False)
        c4a = self.conv4a(c3)
        c4b = self.conv4b(c4a)
        c5a = self.conv5a(self.pool1(c4b))
        c5b = self.conv5b(c5a)
        c6a = self.conv6a(c5b)
        c6b = self.conv6b(c6a)

        return [
            torch.cat([c3, c4a[:, : self.tw], c4b[:, : self.tw]], dim=1),
            torch.cat([c5a[:, : self.tw], c5b[:, : self.tw], c6a[:, : self.tw], c6b], dim=1),
        ], c6b


class GEncoder(nn.Module):
    def __init__(self, mu, trunk_width, pass_through=64):
        super().__init__()
        self.mu = nn.Parameter(torch.from_numpy(mu), requires_grad=False)
        self.pre = PreFilter()
        self.enc = EncStage(trunk_width, pass_through)

    def forward(self, x):
        fmaps, h = self.enc(self.pre(x - self.mu))
        return x, fmaps, h


class Torch_LayerwiseFWRF(nn.Module):
    def __init__(self, fmaps, nv=1, pre_nl=None, post_nl=None, dtype=np.float32):
        super().__init__()
        self.fmaps_shapes = [list(f.size()) for f in fmaps]
        self.nf = int(np.sum([s[1] for s in self.fmaps_shapes]))
        self.pre_nl = pre_nl
        self.post_nl = post_nl
        self.nv = int(nv)

        self.rfs = []
        self.sm = nn.Softmax(dim=1)
        for k, fm_rez in enumerate(self.fmaps_shapes):
            rf = nn.Parameter(torch.tensor(np.ones(shape=(self.nv, fm_rez[2], fm_rez[2]), dtype=dtype), requires_grad=True))
            self.register_parameter(f"rf{k}", rf)
            self.rfs += [rf]

        self.w = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(self.nv, self.nf)).astype(dtype), requires_grad=True))
        self.b = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(self.nv,)).astype(dtype), requires_grad=True))

    def forward(self, fmaps):
        phi = []
        for fm, rf in zip(fmaps, self.rfs):
            g = self.sm(torch.flatten(rf, start_dim=1))
            f = torch.flatten(fm, start_dim=2)
            if self.pre_nl is not None:
                f = self.pre_nl(f)
            phi.append(torch.tensordot(g, f, dims=[[1], [2]]))
        Phi = torch.cat(phi, dim=2)
        if self.post_nl is not None:
            Phi = self.post_nl(Phi)
        vr = torch.squeeze(torch.bmm(Phi, torch.unsqueeze(self.w, 2))).t() + torch.unsqueeze(self.b, 0)
        return vr


class GNet8_Encoder:
    def __init__(self, subject=1, device="cuda", model_path="gnet_multisubject.pt"):
        if utils is None:
            raise ImportError("GNet8_Encoder requires 'utils' (iterate_range/get_value/process_image).")
        self.device = torch.device(device)
        torch.backends.cudnn.enabled = True

        self.subject = int(subject)
        subject_sizes = [0, 15724, 14278, 15226, 13153, 13039, 17907, 12682, 14386]
        self.x_size = subject_sizes[self.subject]

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"GNet model_path not found: {model_path}")

        self.joined_checkpoint = torch.load(model_path, map_location=self.device)

        self.subjects = list(self.joined_checkpoint["voxel_mask"].keys())
        self.gnet8j_voxel_mask = self.joined_checkpoint["voxel_mask"]
        self.gnet8j_voxel_roi = self.joined_checkpoint["voxel_roi"]
        self.gnet8j_voxel_index = self.joined_checkpoint["voxel_index"]
        self.gnet8j_brain_nii_shape = self.joined_checkpoint["brain_nii_shape"]
        self.gnet8j_val_cc = self.joined_checkpoint["val_cc"]

    def _model_fn(self, _ext, _con, _x):
        _y, _fm, _h = _ext(_x)
        return _con(_fm)

    def _pred_fn(self, _ext, _con, xb):
        return self._model_fn(_ext, _con, torch.from_numpy(xb).to(self.device))

    def subject_pred_pass(self, _pred_fn, _ext, _con, x, batch_size):
        pred0 = _pred_fn(_ext, _con, x[:batch_size])
        pred = np.zeros(shape=(len(x), pred0.shape[1]), dtype=np.float32)
        for rb, _ in utils.iterate_range(0, len(x), batch_size):
            pred[rb] = utils.get_value(_pred_fn(_ext, _con, x[rb]))
        return pred

    def _mask_state_dict(self, params: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        masked = {}
        for k, v in params.items():
            if not isinstance(v, torch.Tensor):
                masked[k] = v
                continue
            if v.dim() == 0:
                masked[k] = v
                continue
            # 默认按第一维 mask（nv 维度）
            if v.size(0) == mask.numel():
                masked[k] = v[mask]
            else:
                masked[k] = v
        return masked

    def gnet8j_predictions(self, image_data, _pred_fn, trunk_width, pass_through, checkpoint, mask, batch_size, device):
        subjects = list(image_data.keys())

        if mask is None:
            subject_nv = {s: len(v) for s, v in checkpoint["val_cc"].items()}
        else:
            subject_nv = {s: len(v) for s, v in checkpoint["val_cc"].items()}
            subject_nv[subjects[0]] = int(torch.sum(mask == True))

        subject_image_pred = {s: np.zeros(shape=(len(image_data[s]), subject_nv[s]), dtype=np.float32) for s in subjects}
        _log_act_fn = lambda _x: torch.log(1 + torch.abs(_x)) * torch.tanh(_x)

        best_params = checkpoint["best_params"]
        shared_model = GEncoder(np.array(checkpoint["input_mean"]).astype(np.float32), trunk_width=trunk_width, pass_through=pass_through).to(device)
        shared_model.load_state_dict(best_params["enc"])
        shared_model.eval()

        rec, fmaps, h = shared_model(torch.from_numpy(image_data[list(image_data.keys())[0]][:20]).to(device))

        for s in subjects:
            sd = Torch_LayerwiseFWRF(fmaps, nv=subject_nv[s], pre_nl=_log_act_fn, post_nl=_log_act_fn, dtype=np.float32).to(device)
            params = best_params["fwrfs"][s]

            if mask is None:
                sd.load_state_dict(params)
            else:
                masked_params = self._mask_state_dict(params, mask)
                sd.load_state_dict(masked_params)

            sd.eval()
            subject_image_pred[s] = self.subject_pred_pass(_pred_fn, shared_model, sd, image_data[s], batch_size)

        return subject_image_pred

    def predict(self, images, mask=None):
        self.stim_data = {}
        data = []
        w, h = 227, 227

        if isinstance(images, list):
            for im in images:
                im_pil = im.convert("RGB").resize((w, h), resample=__import__("PIL").Image.Resampling.LANCZOS)
                arr = np.array(im_pil).astype(np.float32) / 255.0
                data.append(arr)
        elif isinstance(images, torch.Tensor):
            if utils is None:
                raise ImportError("utils is required to process torch.Tensor images")
            for i in range(images.shape[0]):
                im_pil = utils.process_image(images[i], w, h)
                arr = np.array(im_pil).astype(np.float32) / 255.0
                data.append(arr)
        else:
            raise TypeError(f"images must be list[PIL.Image] or torch.Tensor, got {type(images)}")

        self.stim_data[self.subject] = np.moveaxis(np.array(data), 3, 1)

        gnet8j_image_pred = self.gnet8j_predictions(
            self.stim_data,
            self._pred_fn,
            64,
            192,
            self.joined_checkpoint,
            mask,
            batch_size=100,
            device=self.device,
        )

        return torch.from_numpy(gnet8j_image_pred[self.subject])
