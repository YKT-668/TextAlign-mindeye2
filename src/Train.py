#!/usr/bin/env python
# coding: utf-8

# # Import packages & functions

# In[1]:


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
from PIL import Image
import numpy as np
from accelerate import Accelerator
accelerator = Accelerator()  # 可按需传入 mixed_precision="fp16"/"bf16"

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
# 本仓库中对应路径为项目根目录下的 'generative-models/'，其中包含包 'sgm'
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_GEN_MODELS_DIR = os.path.join(_PROJ_ROOT, 'generative-models')
if _GEN_MODELS_DIR not in sys.path:
    sys.path.append(_GEN_MODELS_DIR)
import sgm
from sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder  # bigG embedder

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
# cuDNN benchmark can speed up on fixed input shapes
try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
except Exception:
    pass

# custom functions #
import utils


# h5py 数据集安全索引：
# - 支持任意顺序与重复索引
# - 通过 np.unique 获取升序唯一索引并用 inverse 还原原顺序
def _h5_take(ds, idx):
    idx = np.asarray(idx)
    uniq, inverse = np.unique(idx, return_inverse=True)
    vals_uniq = ds[uniq]
    return vals_uniq[inverse]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def pil_to_tensor(img: Image.Image):
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(arr)


def normalize_tensor(t: torch.Tensor, mean, std):
    mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device)[:, None, None]
    std_t = torch.tensor(std, dtype=t.dtype, device=t.device)[:, None, None]
    return (t - mean_t) / std_t


# In[2]:


### Multi-GPU & dtype config (via Accelerate) ###
# 使用 Accelerate 提供的 rank / world size / device
local_rank = accelerator.local_process_index
print("LOCAL RANK", local_rank)

# 根据 Accelerate 的混合精度选择 PyTorch dtype（默认 float32），允许通过环境变量覆盖
mp = accelerator.mixed_precision
env_dtype = os.environ.get("MINDEYE_DTYPE", "").lower()
if env_dtype in ("bf16", "bfloat16"):
    data_type = torch.bfloat16
elif env_dtype in ("fp16", "float16", "half"):
    data_type = torch.float16
else:
    if mp == "bf16":
        data_type = torch.bfloat16
    elif mp == "fp16":
        data_type = torch.float16
    else:
        data_type = torch.float32
print(f"Mixed precision override: env={env_dtype or 'none'}, accelerate={mp}, using {data_type}")

# 设备/并行信息
print("PID of this process =", os.getpid())
device = accelerator.device
world_size = accelerator.state.num_processes
distributed = accelerator.state.distributed_type != 'NO'

# 设备数：优先用 world_size，其次回退到 torch.cuda.device_count()
num_devices = world_size if world_size and world_size > 0 else torch.cuda.device_count()
if num_devices == 0:
    num_devices = 1
num_workers = num_devices

# 查看 Accelerate 的详细状态
print(accelerator.state)

# 只在主进程打印
print = accelerator.print
print("distributed =", distributed,
      "num_devices =", num_devices,
      "local rank =", local_rank,
      "world size =", world_size,
      "data_type =", data_type)


# # Configurations

# In[4]:




# In[5]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
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
    "--multisubject_ckpt", type=str, default=None,
    help="Path to pre-trained multisubject model to finetune a single subject from. multisubject must be False.",
)
parser.add_argument(
    "--num_sessions", type=int, default=1,
    help="Number of training sessions to include",
)
parser.add_argument(
    "--use_prior",action=argparse.BooleanOptionalAction,default=True,
    help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
)
parser.add_argument(
    "--batch_size", type=int, default=16,
    help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
)
parser.add_argument(
    "--wandb_log",action=argparse.BooleanOptionalAction,default=False,
    help="whether to log to wandb",
)
parser.add_argument(
    "--wandb_project",type=str,default="stability",
    help="wandb project name",
)
parser.add_argument(
    "--mixup_pct",type=float,default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
    help="whether to output blurry reconstructions",
)
parser.add_argument(
    "--blur_scale",type=float,default=.5,
    help="multiply loss from blurry recons by this number",
)
parser.add_argument(
    "--clip_scale",type=float,default=1.,
    help="multiply contrastive loss by this number",
)
parser.add_argument(
    "--prior_scale",type=float,default=30,
    help="multiply diffusion prior loss by this",
)
parser.add_argument(
    "--use_image_aug",action=argparse.BooleanOptionalAction,default=False,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--num_epochs",type=int,default=150,
    help="number of epochs of training",
)
parser.add_argument(
    "--multi_subject",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--train_subjs", type=str, default=None,
    help="逗号分隔的训练被试列表，如 '1,2'；若为空则沿用默认逻辑（multi_subject: 1..8 去除 --subj；单被试: [--subj]）",
)
parser.add_argument(
    "--test_subjs", type=str, default=None,
    help="逗号分隔的测试被试列表；若为空则默认与 train_subjs 相同（若 train_subjs 也为空则回落到单被试 subj）",
)
parser.add_argument(
    "--new_test",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=1024,
)
parser.add_argument(
    "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
)
parser.add_argument(
    "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--ckpt_interval",type=int,default=5,
    help="save backup ckpt and reconstruct every x epochs",
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--max_lr",type=float,default=3e-4,
)
parser.add_argument(
    "--train_split_ratio", type=float, default=None,
    help="仅单被试(subj01)时启用：按比例切分 subj01 的 train 分片为训练/保留(测试)；例如 0.9 表示 90% 训练、10% 留作测试(训练中不使用)。不影响原有 new_test 测试集加载。",
)

if utils.is_interactive():
    args = parser.parse_args([])
else:
    args = parser.parse_args()
global_batch_size = args.batch_size * accelerator.num_processes


# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
# seed all random functions
utils.seed_everything(seed)

# 始终将输出目录定位到项目根目录下的 train_logs，避免受工作目录影响
outdir = os.path.join(_PROJ_ROOT, 'train_logs', model_name)
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir,exist_ok=True)
    
if use_image_aug or blurry_recon:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
if use_image_aug:
    img_augment = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
        same_on_batch=False,
        data_keys=["input"],
    )
    
if multi_subject:
    if train_subjs is not None and str(train_subjs).strip():
        _ts = [int(x) for x in str(train_subjs).replace(' ', '').split(',') if x]
        subj_list = np.array(sorted(list(set(_ts))))
    else:
        subj_list = np.arange(1,9)
        subj_list = subj_list[subj_list != subj]
else:
    subj_list = [subj]

# 统一为 numpy 数组，建立 subj -> 索引 的映射，供评估阶段使用
subj_list = np.array(subj_list)
subj_to_idx = {int(s): i for i, s in enumerate(subj_list)}

print("subj_list", subj_list, "num_sessions", num_sessions)


# # Prep data, models, and dataloaders

# ### Creating wds dataloader, preload betas and all 73k possible images

# In[6]:


def my_split_by_node(urls): return urls
num_voxels_list = []

if multi_subject:
    nsessions_allsubj=np.array([40, 40, 32, 30, 40, 32, 40, 30])
    num_samples_per_epoch = (750*40) // num_devices 
else:
    num_samples_per_epoch = (750*num_sessions) // num_devices 

# FAST 模式：通过环境变量 MINDEYE_FAST 控制，显著降低每个 epoch 的样本数，避免预加载占用内存
_fast = os.environ.get("MINDEYE_FAST", "0") == "1"
if _fast:
    # 令总样本数为约 10 个迭代（后续按 batch 和 subj 数计算迭代数）
    # 14 = 2(batch) * 7(subj)，10*14=140
    num_samples_per_epoch = min(num_samples_per_epoch, 140)
    print("[FAST] override num_samples_per_epoch ->", num_samples_per_epoch)

# 可选：缩短每个 epoch 的样本数比例（0<frac<=1）
try:
    _epoch_frac = float(os.environ.get("MINDEYE_EPOCH_FRACTION", "1.0"))
    if _epoch_frac > 0 and _epoch_frac < 1.0:
        _ns = max(1, int(num_samples_per_epoch * _epoch_frac))
        print(f"[EPOCH_FRACTION] scale num_samples_per_epoch {num_samples_per_epoch} -> {_ns} (frac={_epoch_frac})")
        num_samples_per_epoch = _ns
except Exception:
    pass

print("dividing batch size by subj_list, which will then be concatenated across subj during training...") 
# 防止 batch_size 被 subj 数量整除后为 0，最小为 1
batch_size = max(1, batch_size // len(subj_list))

num_iterations_per_epoch = num_samples_per_epoch // (batch_size*len(subj_list))

# 可选：限制每个 epoch 的最大步数（>0 生效）
try:
    _max_steps = int(os.environ.get("MINDEYE_MAX_STEPS_PER_EPOCH", "0"))
    if _max_steps > 0 and num_iterations_per_epoch > _max_steps:
        print(f"[EPOCH_CAP] cap num_iterations_per_epoch {num_iterations_per_epoch} -> {_max_steps}")
        num_iterations_per_epoch = _max_steps
        # 使展示的 num_samples_per_epoch 与步数一致
        num_samples_per_epoch = num_iterations_per_epoch * (batch_size*len(subj_list))
except Exception:
    pass

print("batch_size =", batch_size, "num_iterations_per_epoch =",num_iterations_per_epoch, "num_samples_per_epoch =",num_samples_per_epoch)


# In[7]:


train_data = {}
train_dl = {}
num_voxels = {}
voxels = {}
_betas_files_keepalive = []  # 防止 h5py 文件被提前关闭

# 可选：单被试 subj01 训练/测试分片切分（仅影响训练所用的 train 分片；测试仍使用 new_test/0.tar）
_SUBJ01_TRAIN_SHARDS_OVERRIDE = None
if (not multi_subject) and (int(subj) == 1) and (getattr(args, "train_split_ratio", None) is not None):
    try:
        split_ratio = float(getattr(args, "train_split_ratio"))
        train_dir = os.path.join(data_path, f"wds/subj01/train")
        # 依据 num_sessions 生成 0..num_sessions-1 的分片列表（存在才纳入）
        shards_all = []
        for i in range(max(0, int(num_sessions))):
            p = os.path.join(train_dir, f"{i}.tar")
            if os.path.isfile(p):
                shards_all.append(p)
        # 若按 num_sessions 未找到，回退到 glob 全部分片
        if len(shards_all) == 0:
            import glob as _glob
            shards_all = sorted(_glob.glob(os.path.join(train_dir, "*.tar")))
        if len(shards_all) >= 2:
            k = int(round(len(shards_all) * max(0.0, min(1.0, split_ratio))))
            k = min(max(k, 1), len(shards_all) - 1)  # 至少留1个分片不参与训练
            _SUBJ01_TRAIN_SHARDS_OVERRIDE = shards_all[:k]
            print(f"[SPLIT] subj01 train shards override: use {len(_SUBJ01_TRAIN_SHARDS_OVERRIDE)}/{len(shards_all)} shards for training; holdout {len(shards_all)-k} shards")
        else:
            print("[SPLIT] subj01 shards < 2，跳过切分；沿用默认行为")
            _SUBJ01_TRAIN_SHARDS_OVERRIDE = None
    except Exception as _e:
        print(f"[SPLIT] 解析/应用 --train_split_ratio 失败: {_e}; 沿用默认行为")
for s in subj_list:
    print(f"Training with {num_sessions} sessions")
    if multi_subject:
        train_src = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
    else:
        # 单被试 subj01 可选切分：只在 s==1 且 override 可用时替换为明确的分片列表
        if int(s) == 1 and _SUBJ01_TRAIN_SHARDS_OVERRIDE is not None:
            train_src = list(_SUBJ01_TRAIN_SHARDS_OVERRIDE)
        else:
            train_src = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
    print(train_src)

    train_data[f'subj0{s}'] = wds.WebDataset(train_src,resampled=True,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=1500, rng=random.Random(42))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    f = h5py.File(f'{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
    _betas_files_keepalive.append(f)
    betas_ds = f['betas']  # h5py Dataset，按需切片，避免一次性载入到内存
    num_voxels_list.append(betas_ds.shape[-1])
    num_voxels[f'subj0{s}'] = betas_ds.shape[-1]
    voxels[f'subj0{s}'] = betas_ds
    print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")

print("Loaded all subj train dls and betas!\n")

# 测试集：支持多被试
def _get_test_cfg(_subj:int):
    if not new_test:
        if _subj==3:
            _num=2113
        elif _subj==4:
            _num=1985
        elif _subj==6:
            _num=2113
        elif _subj==8:
            _num=1985
        else:
            _num=2770
        _url = f"{data_path}/wds/subj0{_subj}/test/0.tar"
    else:
        if _subj==3:
            _num=2371
        elif _subj==4:
            _num=2188
        elif _subj==6:
            _num=2371
        elif _subj==8:
            _num=2188
        else:
            _num=3000
        _url = f"{data_path}/wds/subj0{_subj}/new_test/0.tar"
    return _num, _url

if test_subjs is not None and str(test_subjs).strip():
    _tests = [int(x) for x in str(test_subjs).replace(' ', '').split(',') if x]
    test_subj_list = np.array(sorted(list(set(_tests))))
elif train_subjs is not None and str(train_subjs).strip():
    test_subj_list = np.array(sorted(list(set([int(x) for x in str(train_subjs).replace(' ', '').split(',') if x]))))
else:
    # 回落：单被试 subj；若 multi_subject 且未显式给出，则用训练列表的第一个
    test_subj_list = np.array([subj_list[0] if len(subj_list)>0 else subj])

test_dl_map = {}
test_num_map = {}
for _s in test_subj_list:
    _num_test, _url = _get_test_cfg(int(_s))
    print(_url)
    _data = wds.WebDataset(_url,resampled=False,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=1500, rng=random.Random(42))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    test_dl_map[int(_s)] = torch.utils.data.DataLoader(_data, batch_size=_num_test, shuffle=False, drop_last=True, pin_memory=True)
    test_num_map[int(_s)] = _num_test
    print(f"Loaded test dl for subj{_s}!\n")


# In[8]:


# Load NSD COCO images lazily to avoid preloading 73k into RAM
class LazyH5Images:
    def __init__(self, h5_path: str):
        self._f = h5py.File(h5_path, "r")
        key = "images" if "images" in self._f else None
        if key is None:
            for k, ds in self._f.items():
                try:
                    if hasattr(ds, "shape") and len(ds.shape) == 4 and ds.shape[1:] == (3, 224, 224):
                        key = k
                        break
                except Exception:
                    pass
        if key is None:
            raise RuntimeError(f"未在 {h5_path} 找到形如 (N,3,224,224) 的数据集")
        self._ds = self._f[key]

    def __len__(self):
        return self._ds.shape[0]

    def get(self, idxs):
        import numpy as _np
        import torch as _torch
        if isinstance(idxs, _torch.Tensor):
            idxs = idxs.detach().cpu().numpy()
        idxs = _np.asarray(idxs, dtype=_np.int64)
        order = _np.argsort(idxs)
        rev = _np.argsort(order)
        data = self._ds[idxs[order]]
        data = data[rev]
        return _torch.from_numpy(_np.asarray(data))

coco_h5_path = f'{data_path}/coco_images_224_float16.hdf5'
lazy_coco = LazyH5Images(coco_h5_path)
print(f"Using lazy COCO images: total {len(lazy_coco)} (no full preloading)")


# ## Load models

# ### CLIP image embeddings  model

# In[9]:


clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
)
clip_img_embedder.to(device)
clip_img_embedder.eval()                             # ← 新增
for p in clip_img_embedder.parameters():             # ← 新增
    p.requires_grad_(False)    

clip_seq_dim = 256
clip_emb_dim = 1664


# ### SD VAE

# In[10]:


if blurry_recon:
    from diffusers import AutoencoderKL    
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)
    
    from autoencoder.convnext import ConvnextXL
    cnx = ConvnextXL(f'{cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth')
    cnx.requires_grad_(False)
    cnx.eval()
    cnx.to(device)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
    std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1)
    
    blur_augs = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop((224,224), scale=(.9,.9), ratio=(1,1), p=1.0),
        data_keys=["input"],
    )


# ### MindEye modules

# In[11]:


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x
        
model = MindEyeModule()
model


# In[12]:


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
        
model.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)
utils.count_params(model.ridge)
utils.count_params(model)

# test on subject 1 with fake data
b = torch.randn((2,1,num_voxels_list[0]))
print(b.shape, model.ridge(b,0).shape)


# In[13]:


from models import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=blurry_recon, clip_scale=clip_scale)
utils.count_params(model.backbone)
utils.count_params(model)

# test that the model works on some fake data
b = torch.randn((2,1,hidden_dim))
print("b.shape",b.shape)

backbone_, clip_, blur_ = model.backbone(b)
print(backbone_.shape, clip_.shape, blur_[0].shape, blur_[1].shape)


# ### Adding diffusion prior + unCLIP if use_prior=True

# In[14]:


if use_prior:
    from models import *

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
    
    utils.count_params(model.diffusion_prior)
    utils.count_params(model)


# ### Setup optimizer / lr / ckpt saving

# In[15]:


no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

opt_grouped_parameters = [
    {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
if use_prior:
    opt_grouped_parameters.extend([
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ])

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(np.floor(num_epochs*num_iterations_per_epoch)),
        last_epoch=-1
    )
elif lr_scheduler_type == 'cycle':
    total_steps=int(np.floor(num_epochs*num_iterations_per_epoch))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=min(0.5, 2/num_epochs)
    )
    
def save_ckpt(tag):
    ckpt_path = outdir+f'/{tag}.pth'
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")

def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,outdir=outdir,multisubj_loading=False): 
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir+'/last.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading: # remove incompatible ridge layer that will otherwise error
        state_dict.pop('ridge.linears.0.weight',None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint['epoch']
        print("Epoch",epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint

print("\nDone with model preparations!")
num_params = utils.count_params(model)


# # Weights and Biases

# In[16]:


if local_rank==0 and wandb_log: # only use main process for wandb logging
    import wandb
    wandb_project = 'mindeye'
    print(f"wandb {wandb_project} run {model_name}")
    # need to configure wandb beforehand in terminal with "wandb init"!
    wandb_config = {
      "model_name": model_name,
      "global_batch_size": global_batch_size,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "num_sessions": num_sessions,
      "num_params": num_params,
      "clip_scale": clip_scale,
      "prior_scale": prior_scale,
      "blur_scale": blur_scale,
      "use_image_aug": use_image_aug,
      "max_lr": max_lr,
      "mixup_pct": mixup_pct,
      "num_samples_per_epoch": num_samples_per_epoch,
      "num_test": num_test,
      "ckpt_interval": ckpt_interval,
      "ckpt_saving": ckpt_saving,
      "seed": seed,
      "distributed": distributed,
      "num_devices": num_devices,
      "world_size": world_size,
      "train_url": train_url,
      "test_url": test_url,
    }
    print("wandb_config:\n",wandb_config)
    print("wandb_id:",model_name)
    wandb.init(
        id=model_name,
        project=wandb_project,
        name=model_name,
        config=wandb_config,
        resume="allow",
    )
else:
    wandb_log = False


# # Main

# In[17]:


epoch = 0
losses, test_losses, lrs = [], [], []
best_test_loss = 1e9
torch.cuda.empty_cache()
if torch.cuda.is_available():
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# In[18]:


# load multisubject stage1 ckpt if set
if multisubject_ckpt is not None:
    load_ckpt("last",outdir=multisubject_ckpt,load_lr=False,load_optimizer=False,load_epoch=False,strict=False,multisubj_loading=True)


# In[19]:


train_dls = [train_dl[f'subj0{s}'] for s in subj_list]

model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)
# leaving out test_dl since we will only have local_rank 0 device do evals


# In[20]:


print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
test_image, test_voxel = None, None
mse = nn.MSELoss()
l1 = nn.L1Loss()
soft_loss_temps = torch.full((num_epochs - int(mixup_pct * num_epochs),), 0.02)

for epoch in progress_bar:
    model.train()

    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    test_fwd_percent_correct = 0.
    test_bwd_percent_correct = 0.
    
    recon_cossim = 0.
    test_recon_cossim = 0.
    recon_mse = 0.
    test_recon_mse = 0.

    loss_clip_total = 0.
    loss_blurry_total = 0.
    loss_blurry_cont_total = 0.
    test_loss_clip_total = 0.
    
    loss_prior_total = 0.
    test_loss_prior_total = 0.

    blurry_pixcorr = 0.
    test_blurry_pixcorr = 0. # needs >.456 to beat low-level subj01 results in mindeye v1

    # 流式按需加载，避免整 epoch 预加载导致内存爆炸
    train_iterators = [iter(dl) for dl in train_dls]
    # 可选：降低 prior 计算频率以提升吞吐（1 表示每步都算）
    try:
        _prior_interval = int(os.environ.get("MINDEYE_PRIOR_INTERVAL", "1"))
    except Exception:
        _prior_interval = 1

    for train_i in range(num_iterations_per_epoch):
        with torch.cuda.amp.autocast(dtype=data_type):
            optimizer.zero_grad()
            loss = 0.

            voxel_list = []
            image_chunks = []
            mix_perm_list, mix_betas_list, mix_select_list = [], [], []

            for si, s in enumerate(subj_list):
                # 逐个被试取到一个“去重后的可用 batch”
                while True:
                    try:
                        behav0, past_behav0, future_behav0, old_behav0 = next(train_iterators[si])
                    except StopIteration:
                        train_iterators[si] = iter(train_dls[si])
                        behav0, past_behav0, future_behav0, old_behav0 = next(train_iterators[si])

                    image_idx = behav0[:, 0, 0].cpu().long().numpy()
                    image0, image_sorted_idx = np.unique(image_idx, return_index=True)
                    if len(image0) != len(image_idx):
                        # 出现重复索引，换下一个 batch
                        continue

                    # 懒加载 HDF5 图像并在后续统一转设备/精度
                    img_tensor = lazy_coco.get(image0)
                    image_chunks.append(img_tensor)

                    voxel_idx = behav0[:, 0, 5].cpu().long().numpy()
                    voxel_sorted_idx = voxel_idx[image_sorted_idx]
                    voxel0_np = _h5_take(voxels[f'subj0{s}'], voxel_sorted_idx)
                    voxel0 = torch.tensor(voxel0_np).unsqueeze(1)

                    if epoch < int(mixup_pct * num_epochs):
                        voxel0, perm, betas, select = utils.mixco(voxel0)
                        mix_perm_list.append(perm)
                        mix_betas_list.append(betas)
                        mix_select_list.append(select)

                    voxel_list.append(voxel0)
                    break

            # 组合跨被试 batch
            image = torch.cat(image_chunks, dim=0)
            if image.dtype != torch.float32 and data_type == torch.float32:
                image = image.to(torch.float32)
            image = image.to(device, non_blocking=True)
            voxel_list = [v.detach().to(device) for v in voxel_list]

            if use_image_aug: 
                image = img_augment(image)

            with torch.no_grad():
                force_clip_fp32 = os.environ.get("CLIP_FP32", "1") == "1"
                clip_amp_dtype = torch.float32 if force_clip_fp32 else data_type
                with torch.cuda.amp.autocast(dtype=clip_amp_dtype):
                    _img = image.float() if force_clip_fp32 else image
                    _out = clip_img_embedder(_img)
                    clip_target = _out[0] if isinstance(_out, tuple) else _out
            assert not torch.any(torch.isnan(clip_target))

            if epoch < int(mixup_pct * num_epochs):
                perm = torch.cat([t.detach().to(device) for t in mix_perm_list], dim=0)
                betas = torch.cat([t.detach().to(device) for t in mix_betas_list], dim=0)
                select = torch.cat([t.detach().to(device) for t in mix_select_list], dim=0)

            voxel_ridge_list = [model.ridge(voxel_list[si],si) for si,s in enumerate(subj_list)]
            voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

            backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)

            if clip_scale>0:
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1, eps=1e-6)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1, eps=1e-6)

            if use_prior:
                _step_id = epoch * num_iterations_per_epoch + train_i
                _do_prior = (_prior_interval <= 1) or ((_step_id % _prior_interval) == 0)
                if _do_prior:
                    loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                    loss_prior_total += loss_prior.item()
                    # 当降低频率时，按间隔放大以保持期望梯度尺度不变
                    loss_prior = loss_prior * prior_scale * max(1, _prior_interval)
                    loss += loss_prior

                    recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                    recon_mse += mse(prior_out, clip_target).item()

            if clip_scale>0:
                if epoch < int(mixup_pct * num_epochs):                
                    loss_clip = utils.mixco_nce(
                        clip_voxels_norm.float(),
                        clip_target_norm.float(),
                        temp=0.02,
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm.float(),
                        temp=epoch_temp)

                loss_clip_total += loss_clip.item()
                loss_clip *= clip_scale
                loss += loss_clip

            if blurry_recon:     
                image_enc_pred, transformer_feats = blurry_image_enc_

                image_enc = autoenc.encode(2*image-1).latent_dist.mode() * 0.18215
                loss_blurry = l1(image_enc_pred, image_enc)
                loss_blurry_total += loss_blurry.item()

                if epoch < int(mixup_pct * num_epochs):
                    image_enc_shuf = image_enc[perm]
                    betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                    image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                        image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)

                image_norm = (image - mean)/std
                image_aug = (blur_augs(image) - mean)/std
                _, cnx_embeds = cnx(image_norm)
                _, cnx_aug_embeds = cnx(image_aug)

                cont_loss = utils.soft_cont_loss(
                    nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1, eps=1e-6),
                    nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1, eps=1e-6),
                    nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1, eps=1e-6),
                    temp=0.2)
                loss_blurry_cont_total += cont_loss.item()

                loss += (loss_blurry + 0.1*cont_loss) * blur_scale #/.18215

            if clip_scale>0:
                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm.float(), clip_voxels_norm), labels, k=1).item()

            if blurry_recon:
                with torch.no_grad():
                    # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                    random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
                    blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    blurry_pixcorr += pixcorr.item()

            utils.check_loss(loss)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if lr_scheduler_type is not None:
                lr_scheduler.step()

        # 每隔若干步在主进程打印一次进度，便于长周期监控
        if local_rank == 0:
            try:
                _interval = int(os.environ.get("LOG_STEP_INTERVAL", "200"))
            except Exception:
                _interval = 200
            if _interval > 0 and ((train_i + 1) % _interval == 0 or (train_i + 1) == num_iterations_per_epoch):
                try:
                    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
                    peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                    print(f"[epoch {epoch}/{num_epochs} step {train_i+1}/{num_iterations_per_epoch}] loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e} peak_alloc_GB={peak_alloc:.2f} peak_reserved_GB={peak_reserved:.2f}")
                except Exception:
                    print(f"[epoch {epoch}/{num_epochs} step {train_i+1}/{num_iterations_per_epoch}] loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

    model.eval()
    if local_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
            # 多被试测试循环
            for _eval_subj in list(test_dl_map.keys()):
                test_dl = test_dl_map[_eval_subj]
                num_test = test_num_map[_eval_subj]
                test_image, test_voxel = None, None
                test_i = -1
                for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):  
                    # 全量一次性载入
                    assert len(behav) == num_test

                ## Average same-image repeats ##
                if test_image is None:
                    voxel_idx = behav[:,0,5].cpu().long().numpy()
                    voxel_np = _h5_take(voxels[f'subj0{_eval_subj}'], voxel_idx)
                    voxel = torch.tensor(voxel_np).unsqueeze(1)

                    image = behav[:,0,0].cpu().long()

                    unique_image, sort_indices = torch.unique(image, return_inverse=True)
                    for im in unique_image:
                        locs = torch.where(im == image)[0]
                        if len(locs)==1:
                            locs = locs.repeat(3)
                        elif len(locs)==2:
                            locs = locs.repeat(2)[:3]
                        assert len(locs)==3
                        if test_image is None:
                            test_image = lazy_coco.get([im])
                            test_voxel = voxel[locs][None]
                        else:
                            test_image = torch.vstack((test_image, lazy_coco.get([im])))
                            test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                loss=0.
                            
                test_indices = torch.arange(len(test_voxel))[:300]
                voxel = test_voxel[test_indices].to(device)
                image = test_image[test_indices]
                if image.dtype != torch.float32:
                    image = image.to(torch.float32)
                image = image.to(device, non_blocking=True)
                assert len(image) == 300

                _out = clip_img_embedder(image.float())
                clip_target = _out[0] if isinstance(_out, tuple) else _out

                # 使用正确的被试线性层索引，确保维度匹配
                _subj_idx = subj_to_idx.get(int(_eval_subj), 0)
                for rep in range(3):
                    voxel_ridge = model.ridge(voxel[:,rep], _subj_idx)
                    backbone0, clip_voxels0, blurry_image_enc_ = model.backbone(voxel_ridge)
                    if rep==0:
                        clip_voxels = clip_voxels0
                        backbone = backbone0
                    else:
                        clip_voxels += clip_voxels0
                        backbone += backbone0
                clip_voxels /= 3
                backbone /= 3

                if clip_scale>0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1, eps=1e-6)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1, eps=1e-6)

                # for some evals, only doing a subset of the samples per batch because of computational cost
                random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
                
                if use_prior:
                    loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                    test_loss_prior_total += loss_prior.item()
                    loss_prior *= prior_scale
                    loss += loss_prior
                    
                if clip_scale>0:
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm.float(),
                        temp=0.02)

                    test_loss_clip_total += loss_clip.item()
                    loss_clip = loss_clip * clip_scale
                    loss += loss_clip

                if blurry_recon:
                    image_enc_pred, _ = blurry_image_enc_
                    blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample / 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    test_blurry_pixcorr += pixcorr.item()

                if clip_scale>0:
                    # forward and backward top 1 accuracy        
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                    test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm.float(), clip_voxels_norm), labels, k=1).item()
                
                utils.check_loss(loss)                
                test_losses.append(loss.item())

                # 每个被试一次性迭代
                assert (test_i+1) == 1

            logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                "test/loss": np.mean(test_losses[-(test_i+1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "test/num_steps": len(test_losses),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "test/test_fwd_pct_correct": test_fwd_percent_correct / max(1,(test_i + 1)),
                "test/test_bwd_pct_correct": test_bwd_percent_correct / max(1,(test_i + 1)),
                "train/loss_clip_total": loss_clip_total / (train_i + 1),
                "train/loss_blurry_total": loss_blurry_total / (train_i + 1),
                "train/loss_blurry_cont_total": loss_blurry_cont_total / (train_i + 1),
                "test/loss_clip_total": test_loss_clip_total / max(1,(test_i + 1)),
                "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                "train/recon_cossim": recon_cossim / (train_i + 1),
                "test/recon_cossim": test_recon_cossim / max(1,(test_i + 1)),
                "train/recon_mse": recon_mse / (train_i + 1),
                "test/recon_mse": test_recon_mse / max(1,(test_i + 1)),
                "train/loss_prior": loss_prior_total / (train_i + 1),
                "test/loss_prior": test_loss_prior_total / max(1,(test_i + 1)),
                }

            # if finished training, save jpg recons if they exist
            if (epoch == num_epochs-1) or (epoch % ckpt_interval == 0):
                if blurry_recon:    
                    image_enc = autoenc.encode(2*image[:4]-1).latent_dist.mode() * 0.18215
                    # transform blurry recon latents to images and plot it
                    fig, axes = plt.subplots(1, 8, figsize=(10, 4))
                    jj=-1
                    for j in [0,1,2,3]:
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc_pred[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')

                    if wandb_log:
                        logs[f"test/blur_recons"] = wandb.Image(fig, caption=f"epoch{epoch:03d}")
                        plt.close()
                    else:
                        plt.show()

            progress_bar.set_postfix(**logs)

            # 记录本轮峰值显存（便于评估 prior 分支显存需求）
            if torch.cuda.is_available():
                try:
                    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
                    peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                    print(f"[GPU] peak_alloc_GB={peak_alloc:.2f} peak_reserved_GB={peak_reserved:.2f}")
                except Exception:
                    pass

            if wandb_log: wandb.log(logs)
            
    # Save model checkpoint and reconstruct
    if (ckpt_saving) and (epoch % ckpt_interval == 0):
        save_ckpt(f'last')

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()

print("\n===Finished!===\n")
if ckpt_saving:
    save_ckpt(f'last')


# In[ ]:


plt.plot(losses)
plt.show()
plt.plot(test_losses)
plt.show()

