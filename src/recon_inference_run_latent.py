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


# In[1]:
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

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:",device)


# In[33]:


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
    
    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[34]:


parser = argparse.ArgumentParser(description="MindEyeV2 Latent Inference (brain -> CLIP image tokens)")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="Experiment folder name under ckpt_root containing checkpoint (e.g. s1_textalign_coco_train_long_v1)",
)
parser.add_argument(
    "--ckpt_root", type=str, default="/home/vipuser/MindEyeV2_Project/train_logs",
    help="Root directory holding experiment subfolders with last.pth or DeepSpeed zero checkpoints",
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
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=512,
)
parser.add_argument(
    "--new_test",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--seed",type=int,default=42,
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
    help="Maximum number of images to save when --save_images is set (default 10).",
)
parser.add_argument(
    "--latent_only",
    action="store_true",
    help="Only run MindEye2 encoder and save CLIP image latents, skip unCLIP/SDXL generation."
)


# ======================= MODIFICATION 1: ADD NEW ARGUMENTS =======================
# === dump brain->CLIP image vectors ===
parser.add_argument("--dump_clip_vecs", action="store_true", help="If set, saves the brain-decoded CLIP vectors to a .pt file.")
parser.add_argument("--clip_out", type=str, default=None, help="Path to save brain-decoded CLIP vectors. Defaults to <output_dir>/brain_clip.pt")
parser.add_argument("--dump_ids", action="store_true", help="If set, saves the corresponding sample IDs to ids.json.")
# pooling strategy for dumping vectors
parser.add_argument("--clip_pooling", type=str, default="mean", choices=["mean","cls"],
                    help="How to pool token sequence when dumping vectors: 'mean' over tokens or take 'cls' token (index 0). Default: mean")
# ===============================================================================

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
# seed all random functions
utils.seed_everything(seed)

# 是否启用 unCLIP / DiffusionEngine，如果 latent_only 就全部跳过
use_unclip = not args.latent_only


# make output directory
os.makedirs("evals",exist_ok=True)
# create model-specific subdir under evals and also ensure output_dir exists
os.makedirs(f"evals/{model_name}",exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


# In[35]:


voxels = {}
# Load hdf5 data for betas
f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
betas = f['betas'][:]
betas = torch.Tensor(betas).to("cpu")
num_voxels = betas[0].shape[-1]
voxels[f'subj0{subj}'] = betas
print(f"num_voxels for subj0{subj}: {num_voxels}")

if not new_test: # using old test set from before full dataset released (used in original MindEye paper)
    if subj==3:
        num_test=2113
    elif subj==4:
        num_test=1985
    elif subj==6:
        num_test=2113
    elif subj==8:
        num_test=1985
    else:
        num_test=2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
else: # using larger test set from after full dataset released
    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"
    
print(test_url)
def my_split_by_node(urls): return urls
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")


# In[36]:


# Prep images but don't load them all to memory
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

# Prep test voxels and indices of test images
test_images_idx = []
test_voxels_idx = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
    test_voxels_idx = np.append(test_voxels_idx, behav[:,0,5].cpu().numpy())
    test_images_idx = np.append(test_images_idx, behav[:,0,0].cpu().numpy())
test_images_idx = test_images_idx.astype(int)
test_voxels_idx = test_voxels_idx.astype(int)

assert (test_i+1) * num_test == len(test_voxels) == len(test_images_idx)
print(test_i, len(test_voxels), len(test_images_idx), len(np.unique(test_images_idx)))


# In[38]:



# In[38]:

# 对于我们现在的 latent_only 场景，其实不需要真正实例化 FrozenOpenCLIPImageEmbedder。
# 只保留维度定义，后续 BrainNetwork 会直接输出到 CLIP 图像 token 空间。
clip_seq_dim = 256
clip_emb_dim = 1664

clip_img_embedder = None
if not args.latent_only:
    # 只有在你未来真的想跑 “图像重建 + caption” 的完整流程时，
    # 才需要启用下面这段，会额外占用大量显存。
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
    )
    clip_img_embedder.to(device)


autoenc = None
if blurry_recon and use_unclip:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = safetensors.torch.load_file(f'{cache_dir}/vae-ft-mse-840000-ema-pruned.safetensors')
    autoenc.load_state_dict(ckpt, strict=False)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)


autoenc = None
if blurry_recon and not args.latent_only:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = safetensors.torch.load_file(f'{cache_dir}/vae-ft-mse-840000-ema-pruned.safetensors')
    autoenc.load_state_dict(ckpt, strict=False)
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
        out = self.linears[subj_idx](x[:,0]).unsqueeze(1)
        return out
        
model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)

from diffusers.models.autoencoders.vae import Decoder
from models import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, 
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim) 
utils.count_params(model.ridge)
utils.count_params(model.backbone)
utils.count_params(model)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
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
model.to(device)

utils.count_params(model.diffusion_prior)
utils.count_params(model)

# Load pretrained model ckpt
tag='last'
outdir = os.path.join(args.ckpt_root, model_name)
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
pth_path = f'{outdir}/{tag}.pth'

# First, try to load as a single .pth file
if os.path.exists(pth_path):
    try:
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
        
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
        print(f"Loaded model state from single pth: {pth_path}")
        
    except Exception as e:
        print(f"Single .pth file load failed with error: {e}")
        print("Attempting to load as a DeepSpeed checkpoint instead...")
        # If single file load fails, proceed to DeepSpeed loading
        
        ckpt_dir = os.path.join(outdir, str(tag).replace(".pth",""))
        if os.path.isdir(ckpt_dir):
            state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            print("Loaded from DeepSpeed zero checkpoint after single file failure.")
        else:
            raise FileNotFoundError(f"Neither a valid .pth file nor a DeepSpeed checkpoint directory was found for tag '{tag}' in {outdir}")

# If the .pth file doesn't exist, assume it's a DeepSpeed checkpoint directory
elif os.path.isdir(os.path.join(outdir, str(tag).replace(".pth",""))):
    print(f".pth file not found. Assuming DeepSpeed checkpoint directory structure.")
    ckpt_dir = os.path.join(outdir, str(tag).replace(".pth",""))
    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    print("Loaded from DeepSpeed zero checkpoint.")
    
else:
    raise FileNotFoundError(f"Could not find a valid checkpoint. No .pth file found at '{pth_path}' and no DeepSpeed directory found either.")

print("ckpt loaded!")


# In[30]:


# setup text caption networks
processor = None
clip_text_model = None
clip_convert = None

if args.save_images and use_unclip:
    from transformers import AutoProcessor, AutoModelForCausalLM
    from modeling_git import GitForCausalLMClipEmb

    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    clip_text_model.to(device)
    clip_text_model.eval().requires_grad_(False)
    clip_text_seq_dim = 257
    clip_text_emb_dim = 1024

    class CLIPConverter(torch.nn.Module):
        def __init__(self):
            super(CLIPConverter, self).__init__()
            self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
            self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)
        def forward(self, x):
            x = x.permute(0,2,1)
            x = self.linear1(x)
            x = self.linear2(x.permute(0,2,1))
            return x

    clip_convert = CLIPConverter()
    state_dict = torch.load(f"{cache_dir}/bigG_to_L_epoch8.pth", map_location='cpu')['model_state_dict']
    clip_convert.load_state_dict(state_dict, strict=True)
    clip_convert.to(device)
    del state_dict



# In[31]:


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

diffusion_engine = None
vector_suffix = None

if use_unclip:
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
    diffusion_engine.eval().requires_grad_(False)

    ckpt_path = os.path.join(cache_dir, 'unclip6_epoch0_step110000.ckpt')
    if not os.path.exists(ckpt_path):
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


    ckpt = torch.load(ckpt_path, map_location='cpu')
    diffusion_engine.load_state_dict(ckpt['state_dict'])

    print("\n[INFO] Disabling xformers memory efficient attention due to hardware incompatibility.")

    batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device)}
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    if vector_suffix.shape[-1] == 2304:  # 保险：如果还有 spatial，切前1024
        vector_suffix = vector_suffix[:, :1024]
    print("vector_suffix (fixed)", vector_suffix.shape)
else:
    print("[INFO] --latent_only enabled: skip unCLIP / DiffusionEngine initialization.")



# In[39]:


# get all reconstructions
model.to(device)

#from accelerate import cpu_offload
#cpu_offload(model, execution_device=device)
#print("[INFO] Enabled model CPU offloading for MindEyeModule to save VRAM.")

model.eval().requires_grad_(False)

# ==================== MODIFICATION 2: INITIALIZE CONTAINERS ====================
# This section replaces the old memory-intensive lists (all_recons, etc.)
# with file handlers and counters to save results one by one.

# Initialize containers for dumping vectors and IDs, only if requested
if args.dump_clip_vecs:
    saved_vecs, saved_ids = [], []
    print("[INFO] --dump_clip_vecs is active. Preparing to collect brain-decoded CLIP vectors.")
# ===============================================================================

# 1. Initialize counters and tools
piler = ToPILImage()
saved_count = 0
imsize = 256 # Define image size for resizing early

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

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    # The loop iterates through unique image indices. `batch` here is an index, not the data itself.
    for batch_idx, uniq_img in enumerate(tqdm(np.unique(test_images_idx), desc="Reconstructing images")):
        
        # Early exit if we are not saving anything and just dumping vectors
        if args.dump_clip_vecs and max_save is not None and len(saved_vecs) >= max_save:
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
        voxel = test_voxels[None, locs].to(device) # Shape: [1, 3, num_voxels]
        
                # Average the outputs over the 3 repetitions
        # Initialize accumulators
        accum_clip_voxels = None
        accum_backbone = None
        accum_blurry_enc = None

        for rep in range(3):
            voxel_ridge = model.ridge(voxel[:,[rep]], 0) # 0th index of subj_list
            backbone_out, clip_voxels_out, blurry_image_enc_out = model.backbone(voxel_ridge)
            
            # Ensure blurry_image_enc_out is a tensor, not a tuple
            # The model might return a tuple, e.g., (tensor, metadata). We take the first element.
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
        
        # Averaging the accumulated tensors
        clip_voxels = accum_clip_voxels / 3
        backbone = accum_backbone / 3
        blurry_image_enc = accum_blurry_enc / 3

        
        # Averaging the accumulated tensors
    
        
        # ==================== MODIFICATION 3: CAPTURE THE BRAIN VECTOR ====================
        if args.dump_clip_vecs:
            # clip_voxels: [1, 256, 1664] (token 序列) 或 [1, 1664]（若你的backbone已池化）
            vec = clip_voxels
            if vec.dim() == 3:
                if args.clip_pooling == "mean":
                    vec = vec.mean(dim=1)        # [1, 1664]
                else:  # cls
                    vec = vec[:, 0, :]
            vec = vec.squeeze(0).detach().float().cpu()  # [1664]
            saved_vecs.append(vec)
            print(f"[dump] Captured vec for img {uniq_img}: shape={tuple(vec.shape)}")
            if args.dump_ids:
                saved_ids.append(int(uniq_img))
        # ==================================================================================

        
        # --- Continue with original logic for image/caption generation if enabled ---
        if args.save_images and use_unclip and saved_count < (max_save if max_save is not None else float('inf')):
            
            # --- Caption Generation ---
            prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, 
                            text_cond = dict(text_embed = backbone), 
                            cond_scale = 1., timesteps = 20)
            pred_caption_emb = clip_convert(prior_out)
            generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"\nSample {saved_count} (Image ID {uniq_img}): {generated_caption[0]}")

            # --- Image Reconstruction and Saving ---
            # 1. Save the main reconstruction
            samples = utils.unclip_recon(prior_out,
                                diffusion_engine,
                                vector_suffix,
                                num_samples=num_samples_per_image)
            
            im = transforms.Resize((imsize, imsize))(samples[0]).float().cpu()
            pil_img = piler(im)
            fname = os.path.join(image_dir, f"{model_name}_recon_{saved_count}.{image_format}")
            pil_img.save(fname)

            # 2. Save the corresponding caption
            if caps_file:
                caps_file.write(f"{saved_count}\t{generated_caption[0]}\n")

            # 3. Save the blurry reconstruction (if enabled)
            if blurry_recon:
                blurred_image = (autoenc.decode(blurry_image_enc/0.18215).sample/ 2 + 0.5).clamp(0,1)
                im_blurry = transforms.Resize((imsize, imsize))(blurred_image[0]).float().cpu()
                pil_blurry = piler(im_blurry)
                fname_blurry = os.path.join(blurry_dir, f"{model_name}_blurry_{saved_count}.{image_format}")
                pil_blurry.save(fname_blurry)

            saved_count += 1

        # --- Early Exit ---
        # If we have saved the desired number of images (or vectors), break the main loop.
        if max_save is not None and (saved_count >= max_save):
            print(f"\nReached max_save limit of {max_save}. Stopping inference.")
            break

# ==================== MODIFICATION 4: SAVE DUMPED VECTORS AND IDS ====================
if args.dump_clip_vecs:
    print("\n" + "="*60)
    print("🧠 Dumping brain-decoded CLIP vectors...")
    if saved_vecs:
        import torch, os, json
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
# ===============================================================================
