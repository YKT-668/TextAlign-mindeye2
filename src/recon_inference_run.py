#!/usr/bin/env python
# coding: utf-8

import os, torch
# ==== 兼容补丁 1：给旧版本 torch 补上兼容的新接口 register_pytree_node ====
try:
    import torch.utils._pytree as _pytree

    # 只有旧接口 _register_pytree_node，没有新接口时才打补丁
    if not hasattr(_pytree, "register_pytree_node") and hasattr(_pytree, "_register_pytree_node"):
        _old_register = _pytree._register_pytree_node

        def register_pytree_node(node_type, flatten_fn, unflatten_fn,
                                 *, serialized_type_name=None, serialized_attributes_fn=None):
            # 旧版 torch 不支持序列化参数，这里直接忽略，只调用旧函数
            return _old_register(node_type, flatten_fn, unflatten_fn)

        _pytree.register_pytree_node = register_pytree_node
except Exception:
    pass

# ==== 兼容补丁 2：给 accelerate 提供 torch.amp.GradScaler ====
try:
    from torch.cuda.amp import GradScaler as _CudaGradScaler
    import types as _types

    if not hasattr(torch, "amp"):
        torch.amp = _types.SimpleNamespace()

    if not hasattr(torch.amp, "GradScaler"):
        torch.amp.GradScaler = _CudaGradScaler
except Exception:
    pass
# =======================================================================
os.environ["XFORMERS_DISABLED"] = "1"       # 彻底关 xformers
os.environ["FLASH_ATTENTION_DISABLE"] = "1" # 防止走 flash-attn
# 强制 PyTorch 只用 math 实现
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
    torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
# Ensure we prefer the local `src/generative_models` implementation (it contains newer signatures )
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
# Insert at front so it takes precedence over other installed/adjacent copies
local_generative_models = os.path.join(script_dir, 'generative_models')
if os.path.isdir(local_generative_models):
    if local_generative_models not in sys.path:
        sys.path.insert(0, local_generative_models)
else:
    # fallback to a generative-models folder at repo root if present
    repo_root_gen = os.path.join(os.path.dirname(script_dir), 'generative-models')
    if os.path.isdir(repo_root_gen) and repo_root_gen not in sys.path:
        sys.path.insert(0, repo_root_gen)

import sgm

# ======================= ULTIMATE FIX: Globally disable xformers BEFORE any SGM import =======================
from sgm.modules.attention import CrossAttention
CrossAttention.use_xformers = False
# ==========================================================================================================

import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToPILImage
from accelerate import Accelerator

from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

import safetensors.torch
import deepspeed

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
from models import *

# ======================= MODIFICATION: Align with training-side BrainNetwork/TextAlign =======================
# We need the same BrainNetwork implementation as training (models_textalign.BrainNetwork)
try:
    from models_textalign import BrainNetwork as BrainNetwork_TextAlign
except Exception as e:
    BrainNetwork_TextAlign = None
    _MODELS_TEXTALIGN_IMPORT_ERR = e
# ============================================================================================================

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:", device)

# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "final_subj01_pretrained_40sess_24bs"
    print("model_name:", model_name)

    # other variables can be specified in the following string:
    jupyter_args = f"--data_path=/weka/proj-medarc/shared/mindeyev2_dataset \
                    --cache_dir=/weka/proj-medarc/shared/mindeyev2_dataset \
                    --model_name={model_name} --subj=1 \
                    --hidden_dim=4096 --n_blocks=4 --new_test"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()

    from IPython.display import clear_output  # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')

parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default=os.getcwd(),
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default=os.getcwd(),
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon", action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--n_blocks", type=int, default=4,
)
parser.add_argument(
    "--hidden_dim", type=int, default=512,
)
parser.add_argument(
    "--new_test", action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--seed", type=int, default=42,
)
parser.add_argument(
    "--output_dir", type=str, default=os.path.dirname(script_dir),
    help="Directory where outputs (recons, captions, etc.) will be saved. Defaults to repo root.",
)
parser.add_argument(
    "--save_images", action=argparse.BooleanOptionalAction, default=False,
    help="Also save individual reconstructions as image files (PNG/JPEG) into output_dir/images/",
)
parser.add_argument(
    "--image_format", type=str, default="png",
    choices=["png", "jpg", "jpeg"],
    help="Image file format to save (png or jpg).",
)
parser.add_argument(
    "--max_save", type=int, default=10,
    help="Maximum number of samples to process/save when outputs are enabled (default 10).",
)

parser.add_argument(
    "--ckpt_path", type=str, default=None,
    help="Explicit path to checkpoint .pth file. If set, overrides model_name search logic.",
)

# === dump brain->CLIP image vectors ===
parser.add_argument("--dump_clip_vecs", action="store_true",
                    help="If set, saves the brain-decoded CLIP vectors to a .pt file.")
parser.add_argument("--clip_out", type=str, default=None,
                    help="Path to save brain-decoded CLIP vectors. Defaults to <output_dir>/brain_clip.pt")
parser.add_argument("--dump_ids", action="store_true",
                    help="If set, saves the corresponding sample IDs to ids.json.")

# === NEW: export official notebook pt artifacts ===
parser.add_argument("--export_official_pts", action="store_true",
                    help="If set, export official notebook artifacts: *_all_enhancedrecons.pt, *_all_blurryrecons.pt, *_all_clipvoxels.pt, *_all_prior_out.pt, *_all_backbones.pt, *_all_predcaptions.pt, *_all_ids.pt")

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# seed all random functions
utils.seed_everything(seed)

# make output directory
os.makedirs("evals", exist_ok=True)
# create model-specific subdir under evals and also ensure output_dir exists
os.makedirs(f"evals/{model_name}", exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# ======================= MODIFICATION: Align ckpt path + read ckpt args early =======================
# Align with training outdir convention when possible
repo_root = os.path.dirname(script_dir)
candidate_outdir_1 = os.path.join(repo_root, "train_logs", model_name)
candidate_outdir_2 = os.path.join("/home/vipuser/train_logs", model_name)  # legacy fallback

outdir = candidate_outdir_1 if os.path.isdir(candidate_outdir_1) else candidate_outdir_2
tag = 'last'
if ckpt_path:
    pth_path = ckpt_path
else:
    pth_path = os.path.join(outdir, f"{tag}.pth")
print(f"[INFO] ckpt outdir: {outdir}")
print(f"[INFO] ckpt path  : {pth_path}")

ckpt_args = {}
ckpt_stage = None
ckpt_use_prior = True
ckpt_clip_scale = 1.0
if os.path.exists(pth_path):
    try:
        _ckpt_meta = torch.load(pth_path, map_location="cpu", weights_only=False)

        # --- IMPORTANT: infer hidden_dim from ckpt weights (more reliable than saved args) ---
        try:
            _sd = _ckpt_meta.get("model_state_dict", _ckpt_meta.get("state_dict", _ckpt_meta))
            _infer_hidden = int(_sd["ridge.linears.0.weight"].shape[0])  # e.g., 1024
            if int(hidden_dim) != _infer_hidden:
                print(f"[WARN] override hidden_dim {hidden_dim} -> {_infer_hidden} (inferred from ckpt weights)")
                hidden_dim = _infer_hidden
        except Exception as _e2:
            print(f"[WARN] Failed to infer hidden_dim from ckpt weights: {_e2}")

        if isinstance(_ckpt_meta, dict):
            ckpt_args = _ckpt_meta.get("args", {}) or {}
            ckpt_stage = _ckpt_meta.get("stage", None)
            ckpt_use_prior = bool(ckpt_args.get("use_prior", True))
            ckpt_clip_scale = float(ckpt_args.get("clip_scale", 1.0))
            # override key hyperparams to match training unless user explicitly wants mismatch
            hidden_dim = int(ckpt_args.get("hidden_dim", hidden_dim))
            n_blocks = int(ckpt_args.get("n_blocks", n_blocks))
            blurry_recon = bool(ckpt_args.get("blurry_recon", blurry_recon))
            new_test = bool(ckpt_args.get("new_test", new_test))

        del _ckpt_meta
        print(f"[INFO] Loaded ckpt meta args to align inference: hidden_dim={hidden_dim}, n_blocks={n_blocks}, "
              f"blurry_recon={blurry_recon}, new_test={new_test}, use_prior={ckpt_use_prior}, clip_scale={ckpt_clip_scale}")
    except Exception as _e:
        print(f"[WARN] Failed to read ckpt meta args from {pth_path}: {_e}")
else:
    print(f"[WARN] ckpt not found at {pth_path}. Will proceed and later try DeepSpeed dir fallback if available.")
# =====================================================================================================

voxels = {}
# Load hdf5 data for betas
f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
betas = f['betas'][:]
betas = torch.Tensor(betas).to("cpu")
num_voxels = betas[0].shape[-1]
voxels[f'subj0{subj}'] = betas
print(f"num_voxels for subj0{subj}: {num_voxels}")

if not new_test:  # using old test set from before full dataset released (used in original MindEye paper)
    if subj == 3:
        num_test = 2113
    elif subj == 4:
        num_test = 1985
    elif subj == 6:
        num_test = 2113
    elif subj == 8:
        num_test = 1985
    else:
        num_test = 2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
else:  # using larger test set from after full dataset released
    if subj == 3:
        num_test = 2371
    elif subj == 4:
        num_test = 2188
    elif subj == 6:
        num_test = 2371
    elif subj == 8:
        num_test = 2188
    else:
        num_test = 3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"

print(test_url)


def my_split_by_node(urls): return urls


test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node) \
    .decode("torch") \
    .rename(behav="behav.npy", past_behav="past_behav.npy",
            future_behav="future_behav.npy", olds_behav="olds_behav.npy") \
    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")

# Prep images but don't load them all to memory
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

# Prep test voxels and indices of test images
test_images_idx = []
test_voxels_idx = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = voxels[f'subj0{subj}'][behav[:, 0, 5].cpu().long()]
    test_voxels_idx = np.append(test_voxels_idx, behav[:, 0, 5].cpu().numpy())
    test_images_idx = np.append(test_images_idx, behav[:, 0, 0].cpu().numpy())
test_images_idx = test_images_idx.astype(int)
test_voxels_idx = test_voxels_idx.astype(int)

assert (test_i + 1) * num_test == len(test_voxels) == len(test_images_idx)
uniq_test_images = np.unique(test_images_idx)
print(test_i, len(test_voxels), len(test_images_idx), len(uniq_test_images))

# clip_img_embedder = FrozenOpenCLIPImageEmbedder(
#     arch="ViT-bigG-14",
#     version="laion2b_s39b_b160k",
#     output_tokens=True,
# )
# clip_img_embedder.to(device)
clip_seq_dim = 256
clip_emb_dim = 1664

if blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    # ======================= MODIFICATION: Align autoencoder weights with training =======================
    ae_path = f'{cache_dir}/sd_image_var_autoenc.pth'
    if not os.path.exists(ae_path):
        raise FileNotFoundError(
            f"[Autoenc] Expected training-aligned autoencoder weights not found: {ae_path}\n"
            f"Please place sd_image_var_autoenc.pth into cache_dir ({cache_dir}) or change --cache_dir."
        )
    ckpt = torch.load(ae_path, map_location="cpu")
    autoenc.load_state_dict(ckpt, strict=True)
    # ================================================================================================
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()

    def forward(self, x):
        return x


model = MindEyeModule()


class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_sizes, out_features):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(input_size, out_features) for input_size in input_sizes
        ])

    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
        return out


model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)

from diffusers.models.autoencoders.vae import Decoder

# ======================= MODIFICATION: Use training-aligned BrainNetwork implementation =======================
if BrainNetwork_TextAlign is None:
    raise RuntimeError(
        f"[BrainNetwork] Failed to import models_textalign.BrainNetwork: {_MODELS_TEXTALIGN_IMPORT_ERR}\n"
        f"Your inference env must have models_textalign.py (same as training)."
    )

model.backbone = BrainNetwork_TextAlign(
    h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks,
    clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim,
    blurry_recon=blurry_recon, clip_scale=float(ckpt_clip_scale)
)
# ================================================================================================
utils.count_params(model.ridge)
utils.count_params(model.backbone)
utils.count_params(model)

# setup diffusion prior network (only if ckpt trained with use_prior=True)
if ckpt_use_prior:
    out_dim = clip_emb_dim
    depth = 6
    dim_head = 52
    heads = clip_emb_dim // 52  # heads * dim_head = clip_emb_dim
    timesteps = 100

    prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=clip_seq_dim,
        learned_query_mode="pos_emb"
    )

    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
else:
    model.diffusion_prior = None
    print("[INFO] ckpt_use_prior=False -> diffusion_prior disabled in inference.")

model.to(device)

if model.diffusion_prior is not None:
    utils.count_params(model.diffusion_prior)
utils.count_params(model)

# Load pretrained model ckpt
tag = 'last'
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")

# First, try to load as a single .pth file
if os.path.exists(pth_path):
    try:
        import gc
        print(f"Loading checkpoint with mmap=True to save RAM...")
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False, mmap=True)

        # Try to find the state dictionary under common keys
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # If no common key is found, assume the checkpoint itself is the state_dict
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        del checkpoint
        del state_dict
        import gc; gc.collect()
        print(f"Loaded model state from single pth: {pth_path}")

    except Exception as e:
        print(f"Single .pth file load failed with error: {e}")
        print("Attempting to load as a DeepSpeed checkpoint instead...")
        # If single file load fails, proceed to DeepSpeed loading

        ckpt_dir = os.path.join(outdir, str(tag).replace(".pth", ""))
        if os.path.isdir(ckpt_dir):
            state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
                checkpoint_dir=outdir, tag=tag)
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            print("Loaded from DeepSpeed zero checkpoint after single file failure.")
        else:
            raise FileNotFoundError(
                f"Neither a valid .pth file nor a DeepSpeed checkpoint directory was found for tag '{tag}' in {outdir}")

# If the .pth file doesn't exist, assume it's a DeepSpeed checkpoint directory
elif os.path.isdir(os.path.join(outdir, str(tag).replace(".pth", ""))):
    print(f".pth file not found. Assuming DeepSpeed checkpoint directory structure.")
    ckpt_dir = os.path.join(outdir, str(tag).replace(".pth", ""))
    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    print("Loaded from DeepSpeed zero checkpoint.")

else:
    raise FileNotFoundError(
        f"Could not find a valid checkpoint. No .pth file found at '{pth_path}' and no DeepSpeed directory found either.")

print("ckpt loaded!")

# setup text caption networks
from transformers import AutoProcessor, AutoModelForCausalLM
from modeling_git import GitForCausalLMClipEmb

processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
clip_text_model.to(device)  # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
clip_text_model.eval().requires_grad_(False)
clip_text_seq_dim = 257
clip_text_emb_dim = 1024


class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super(CLIPConverter, self).__init__()
        self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0, 2, 1))
        return x


clip_convert = CLIPConverter()
state_dict = torch.load(f"{cache_dir}/bigG_to_L_epoch8.pth", map_location='cpu')['model_state_dict']
clip_convert.load_state_dict(state_dict, strict=True)
clip_convert.to(device)  # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
del state_dict

# prep unCLIP
def _resolve_config_path(*candidates):
    # Try several likely locations: cwd-relative, src/ prefix, script dir, and hyphen variant
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    checked = []

    # helper to check a path and return absolute if exists
    def _check(p):
        if os.path.isabs(p):
            cand = p
        else:
            cand = os.path.join(cwd, p)
        checked.append(cand)
        if os.path.exists(cand):
            return cand
        # try relative to script dir
        cand2 = os.path.join(script_dir, p)
        checked.append(cand2)
        if os.path.exists(cand2):
            return cand2
        return None

    for c in candidates:
        found = _check(c)
        if found:
            print(f"Using config: {found}")
            return found

    # try src/ prefix
    for c in candidates:
        found = _check(os.path.join('src', c))
        if found:
            print(f"Using config: {found}")
            return found

    # try replacing underscore folder with hyphen variant
    for c in candidates:
        if 'generative_models' in c:
            c2 = c.replace('generative_models', 'generative-models')
            found = _check(c2)
            if found:
                print(f"Using config: {found}")
                return found

    raise FileNotFoundError(
        f"Could not find config file. Checked the following paths:\n" + "\n".join(checked)
    )


config_path = _resolve_config_path("generative_models/configs/unclip6.yaml")
print(f"!!! ABSOLUTE CONFIG PATH BEING USED: {os.path.abspath(config_path)}")

config = OmegaConf.load(config_path)
config = OmegaConf.to_container(config, resolve=True)

unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

# ======================= ULTIMATE FIX: Force-override config values in code =======================
print("[INFO] Force-overriding config values to ensure compatibility...")

# 1. Fix adm_in_channels to match the actual vector dimension
network_config["params"]["adm_in_channels"] = 1024
print(f"  - Set network_config.params.adm_in_channels -> {network_config['params']['adm_in_channels']}")

# 2. Remove the problematic 'only_tokens' argument
if "only_tokens" in conditioner_config["params"]["emb_models"][0]["params"]:
    del conditioner_config["params"]["emb_models"][0]["params"]["only_tokens"]
    print("  - Removed 'only_tokens' from conditioner_config.")

# 3. Remove the 'Pruner' that causes size mismatch
if "ckpt_config" in unclip_params and "pre_adapters" in unclip_params["ckpt_config"]["params"]:
    adapters = unclip_params["ckpt_config"]["params"]["pre_adapters"]
    unclip_params["ckpt_config"]["params"]["pre_adapters"] = [
        adapter for adapter in adapters if adapter.get("target") != "sgm.modules.checkpoint.Pruner"
    ]
    print("  - Removed 'Pruner' from ckpt_config.")
# ==================================================================================================

first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 38

diffusion_engine = DiffusionEngine(
    network_config=network_config,
    denoiser_config=denoiser_config,
    first_stage_config=first_stage_config,
    conditioner_config=conditioner_config,
    sampler_config=sampler_config,
    scale_factor=scale_factor,
    disable_first_stage_autocast=disable_first_stage_autocast
)
# set to inference
diffusion_engine.eval().requires_grad_(False)

ckpt_path = os.path.join(cache_dir, 'unclip6_epoch0_step110000.ckpt')
if not os.path.exists(ckpt_path):
    # helpful error: list files in cache_dir and suggest fixes
    cache_dir_to_list = os.path.dirname(ckpt_path) or '.'
    try:
        files = os.listdir(cache_dir_to_list)
    except Exception:
        files = []
    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_path}\n"
        f"Searched cache dir: {cache_dir_to_list}\n"
        f"Files found there: {files}\n\n"
        "Fixes:\n"
        "  - Place `unclip6_epoch0_step110000.ckpt` into the cache dir and re-run, or\n"
        "  - Pass --cache_dir /path/to/dir-containing-checkpoint when running the script, or\n"
        "  - Download the checkpoint from its source (see project README) into the cache dir.\n"
    )
ckpt = torch.load(ckpt_path, map_location="cpu")
diffusion_engine.load_state_dict(ckpt["state_dict"])

# ✅ 关键：把整个 diffusion_engine（包含 denoiser/model 等）搬到 GPU
diffusion_engine.to(device).eval().requires_grad_(False)

# 你当前方案A：conditioner 用 fp16（省显存）
diffusion_engine.conditioner.to(device).eval()
diffusion_engine.conditioner.half()

# ✅ 关键：确保 denoiser 的 sigmas buffer 在同一张卡上
try:
    diffusion_engine.denoiser.to(device)
    if hasattr(diffusion_engine.denoiser, "sigmas"):
        diffusion_engine.denoiser.sigmas = diffusion_engine.denoiser.sigmas.to(device)
    print("[INFO] diffusion_engine moved to", device, "| sigmas device:", diffusion_engine.denoiser.sigmas.device)
except Exception as _e:
    print("[WARN] denoiser/sigmas move failed:", _e)

print("\n[INFO] Disabling xformers memory efficient attention due to hardware incompatibility.")

batch = {"jpg": torch.randn(1, 3, 1, 1).to(device),  # jpg doesnt get used, it's just a placeholder
         "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
         "crop_coords_top_left": torch.zeros(1, 2).to(device)}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device)
if vector_suffix.shape[-1] == 2304:  # 保险：如果还有 spatial，切前1024
    vector_suffix = vector_suffix[:, :1024]
print("vector_suffix (fixed)", vector_suffix.shape)  # 确认 (1, 1024)

# get all reconstructions
model.to(device)
model.eval().requires_grad_(False)

# ==================== dump containers ====================
saved_vecs, saved_ids = [], []
if args.dump_clip_vecs:
    print("[INFO] --dump_clip_vecs is active. Preparing to collect brain-decoded CLIP vectors.")

# ==================== OFFICIAL NOTEBOOK EXPORT (prealloc) ====================
official = None
official_imsize = 224
if args.export_official_pts:
    # will export at most `max_save` samples if user sets it, otherwise all unique test images
    _cap = int(max_save) if (max_save is not None) else int(len(uniq_test_images))
    _N = min(len(uniq_test_images), _cap)

    official = dict(
        N=_N,
        i=0,
        enhancedrecons=torch.empty((_N, 3, official_imsize, official_imsize), dtype=torch.float16, device="cpu"),
        blurryrecons=torch.empty((_N, 3, official_imsize, official_imsize), dtype=torch.float16, device="cpu") if blurry_recon else None,
        clipvoxels=torch.empty((_N, clip_seq_dim, clip_emb_dim), dtype=torch.float16, device="cpu"),
        prior_out=torch.empty((_N, clip_seq_dim, clip_emb_dim), dtype=torch.float16, device="cpu"),
        backbones=torch.empty((_N, clip_seq_dim, clip_emb_dim), dtype=torch.float16, device="cpu"),
        predcaptions=[None] * _N,
        ids=torch.empty((_N,), dtype=torch.int64, device="cpu"),
    )
    print(f"[INFO] --export_official_pts active. Prealloc official tensors for N={_N} (fp16, CPU).")
# ============================================================================

# 1. Initialize counters and tools
piler = ToPILImage()
saved_count = 0
imsize = 256  # Define image size for resizing early

# 2. Create output directories beforehand
image_dir = os.path.join(output_dir, "images")
if args.save_images:
    os.makedirs(image_dir, exist_ok=True)
if blurry_recon and args.save_images:
    blurry_dir = os.path.join(output_dir, "blurry_images")
    os.makedirs(blurry_dir, exist_ok=True)

# 3. Open the captions file for writing, if saving images
caps_file = None
if args.save_images:
    caps_path = os.path.join(output_dir, f"{model_name}_captions.txt")
    caps_file = open(caps_path, "w", encoding="utf-8")

# 4. Define other variables
minibatch_size = 1
num_samples_per_image = 1
assert num_samples_per_image == 1

# plotting flag: default False, enable when running interactively
plotting = False
if utils.is_interactive():
    plotting = True

# If we need any unCLIP outputs, diffusion prior must exist
need_unclip = bool(args.save_images or args.export_official_pts)
if need_unclip and (model.diffusion_prior is None):
    raise RuntimeError(
        "[FATAL] You requested --save_images and/or --export_official_pts, but model.diffusion_prior is None "
        "(ckpt_use_prior=False). Cannot generate prior_out / reconstructions."
    )

# Global cap for loop (apply to vectors/images/official export consistently)
cap = int(max_save) if (max_save is not None) else None

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    for batch_idx, uniq_img in enumerate(tqdm(uniq_test_images, desc="Reconstructing images")):

        # unified early exit
        if cap is not None:
            # stop if we've already produced enough samples for any enabled output
            if args.dump_clip_vecs and (len(saved_vecs) >= cap):
                break
            if args.save_images and (saved_count >= cap):
                break
            if (official is not None) and (official["i"] >= official["N"]):
                break

        # Find all occurrences of the current unique image index
        locs = np.where(test_images_idx == uniq_img)[0]

        # Handle cases with 1 or 2 repetitions by duplicating them to get 3
        if len(locs) == 1:
            locs = locs.repeat(3)
        elif len(locs) == 2:
            locs = np.concatenate((locs, locs[:1]))
        assert len(locs) == 3, f"Expected 3 repetitions, but found {len(locs)} for image index {uniq_img}"

        # Select corresponding voxels for the 3 repetitions
        voxel = test_voxels[None, locs].to(device)  # Shape: [1, 3, num_voxels]

        # Average the outputs over the 3 repetitions
        accum_clip_voxels = None
        accum_backbone = None
        accum_blurry_enc = None

        for rep in range(3):
            voxel_ridge = model.ridge(voxel[:, [rep]], 0)  # 0th index of subj_list
            backbone_out, clip_voxels_out, blurry_image_enc_out = model.backbone(voxel_ridge)

            if isinstance(blurry_image_enc_out, tuple):
                blurry_tensor = blurry_image_enc_out[0]
            else:
                blurry_tensor = blurry_image_enc_out

            if rep == 0:
                accum_clip_voxels = clip_voxels_out
                accum_backbone = backbone_out
                accum_blurry_enc = blurry_tensor
            else:
                accum_clip_voxels += clip_voxels_out
                accum_backbone += backbone_out
                accum_blurry_enc += blurry_tensor

        clip_voxels = accum_clip_voxels / 3
        backbone = accum_backbone / 3
        blurry_image_enc = accum_blurry_enc / 3

        # ==================== CAPTURE brain->CLIP vecs ====================
        if args.dump_clip_vecs:
            vec = clip_voxels
            if vec.dim() == 3:
                vec = vec.mean(dim=1)  # [1,1664]
            vec = vec.squeeze(0).detach().float().cpu()  # [1664]
            saved_vecs.append(vec)
            if args.dump_ids:
                saved_ids.append(int(uniq_img))
        # ================================================================

        # ==================== need prior_out/caption/recon? ====================
        if need_unclip:
            # --- Caption Generation ---
            prior_out = model.diffusion_prior.p_sample_loop(
                backbone.shape,
                text_cond=dict(text_embed=backbone),
                cond_scale=1., timesteps=20
            )
            pred_caption_emb = clip_convert(prior_out)
            generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
            cur_caption = str(generated_caption[0])

            # --- Image Reconstruction ---
            samples = utils.unclip_recon(
                prior_out,
                diffusion_engine,
                vector_suffix,
                num_samples=num_samples_per_image
            )

            # ==================== OFFICIAL export per-sample ====================
            if official is not None and official["i"] < official["N"]:
                oi = official["i"]
                official["ids"][oi] = int(uniq_img)
                official["predcaptions"][oi] = cur_caption

                # clipvoxels: want [256,1664]
                cv = clip_voxels
                if cv.dim() == 3:
                    cv = cv[0]
                official["clipvoxels"][oi].copy_(cv.detach().to("cpu", dtype=torch.float16))

                # prior_out: [1,256,1664] -> [256,1664]
                po = prior_out
                if po.dim() == 3:
                    po = po[0]
                official["prior_out"][oi].copy_(po.detach().to("cpu", dtype=torch.float16))

                # backbones: store tokens if shape matches
                bb = backbone
                if bb.dim() == 3:
                    bb = bb[0]
                official["backbones"][oi].copy_(bb.detach().to("cpu", dtype=torch.float16))

                # enhanced recon: samples[0] -> [3,224,224]
                recon = transforms.Resize((official_imsize, official_imsize))(samples[0]).detach().float().cpu()
                official["enhancedrecons"][oi].copy_(recon.to(torch.float16))

                # blurry recon
                if blurry_recon and (official["blurryrecons"] is not None):
                    blurred_image = (autoenc.decode(blurry_image_enc / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                    b = transforms.Resize((official_imsize, official_imsize))(blurred_image[0]).detach().float().cpu()
                    official["blurryrecons"][oi].copy_(b.to(torch.float16))

                official["i"] += 1
            # ===================================================================

            # ==================== save PNGs (optional) ====================
            if args.save_images and (saved_count < (cap if cap is not None else float('inf'))):
                print(f"\nSample {saved_count} (Image ID {uniq_img}): {cur_caption}")

                im = transforms.Resize((imsize, imsize))(samples[0]).float().cpu()
                pil_img = piler(im)
                fname = os.path.join(image_dir, f"{model_name}_recon_{saved_count}.{image_format}")
                pil_img.save(fname)

                if caps_file:
                    caps_file.write(f"{saved_count}\t{cur_caption}\n")

                if blurry_recon:
                    blurred_image = (autoenc.decode(blurry_image_enc / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                    im_blurry = transforms.Resize((imsize, imsize))(blurred_image[0]).float().cpu()
                    pil_blurry = piler(im_blurry)
                    fname_blurry = os.path.join(blurry_dir, f"{model_name}_blurry_{saved_count}.{image_format}")
                    pil_blurry.save(fname_blurry)

                saved_count += 1
            # =============================================================

# ==================== SAVE DUMPED VECTORS AND IDS ====================
if args.dump_clip_vecs:
    print("\n" + "=" * 60)
    print("🧠 Dumping brain-decoded CLIP vectors...")
    if saved_vecs:
        V = torch.stack(saved_vecs, dim=0)  # [N, 1664]
        clip_out_path = args.clip_out or os.path.join(args.output_dir, "brain_clip.pt")
        torch.save(V, clip_out_path)
        print(f"[dump] brain->CLIP vectors saved to: {clip_out_path}  shape={tuple(V.shape)}")

        if args.dump_ids:
            ids_path = os.path.join(args.output_dir, "ids.json")
            with open(ids_path, "w", encoding="utf-8") as f:
                json.dump([int(x) for x in saved_ids], f)
            print(f"[dump] ids saved to: {ids_path}  len={len(saved_ids)}")
    else:
        print("[dump] WARN: no vectors collected (nothing to save).")
# =====================================================================

# ==================== OFFICIAL NOTEBOOK EXPORT (save .pt files) ====================
if official is not None:
    # Slice to actual filled N (in case loop ended early)
    filled = int(official["i"])
    if filled <= 0:
        print("[official export] WARN: no samples exported.")
    else:
        # filenames follow official notebook conventions
        base = os.path.join(args.output_dir, f"{model_name}")
        torch.save(official["enhancedrecons"][:filled].contiguous(),
                   os.path.join(args.output_dir, f"{model_name}_all_enhancedrecons.pt"))

        if blurry_recon and (official["blurryrecons"] is not None):
            torch.save(official["blurryrecons"][:filled].contiguous(),
                       os.path.join(args.output_dir, f"{model_name}_all_blurryrecons.pt"))

        torch.save(official["clipvoxels"][:filled].contiguous(),
                   os.path.join(args.output_dir, f"{model_name}_all_clipvoxels.pt"))
        torch.save(official["prior_out"][:filled].contiguous(),
                   os.path.join(args.output_dir, f"{model_name}_all_prior_out.pt"))
        torch.save(official["backbones"][:filled].contiguous(),
                   os.path.join(args.output_dir, f"{model_name}_all_backbones.pt"))

        torch.save(official["predcaptions"][:filled],
                   os.path.join(args.output_dir, f"{model_name}_all_predcaptions.pt"))
        torch.save(official["ids"][:filled].contiguous(),
                   os.path.join(args.output_dir, f"{model_name}_all_ids.pt"))

        print("\n" + "=" * 60)
        print("[official export] saved:")
        print(" ", os.path.join(args.output_dir, f"{model_name}_all_enhancedrecons.pt"),
              tuple(official["enhancedrecons"][:filled].shape))
        if blurry_recon and (official["blurryrecons"] is not None):
            print(" ", os.path.join(args.output_dir, f"{model_name}_all_blurryrecons.pt"),
                  tuple(official["blurryrecons"][:filled].shape))
        print(" ", os.path.join(args.output_dir, f"{model_name}_all_clipvoxels.pt"),
              tuple(official["clipvoxels"][:filled].shape))
        print(" ", os.path.join(args.output_dir, f"{model_name}_all_prior_out.pt"),
              tuple(official["prior_out"][:filled].shape))
        print(" ", os.path.join(args.output_dir, f"{model_name}_all_backbones.pt"),
              tuple(official["backbones"][:filled].shape))
        print(" ", os.path.join(args.output_dir, f"{model_name}_all_predcaptions.pt"),
              f"len={len(official['predcaptions'][:filled])}")
        print(" ", os.path.join(args.output_dir, f"{model_name}_all_ids.pt"),
              f"len={int(official['ids'][:filled].numel())}")
# ================================================================================
