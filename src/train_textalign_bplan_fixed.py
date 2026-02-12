#!/usr/bin/env python
# coding: utf-8

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
from accelerate import Accelerator
import torch.nn.functional as F

# -------------------------
# Accelerator
# -------------------------
accelerator = Accelerator()  # 可按需传入 mixed_precision="fp16"/"bf16"
device = accelerator.device

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_GEN_MODELS_DIR = os.path.join(_PROJ_ROOT, 'generative-models')
if _GEN_MODELS_DIR not in sys.path:
    sys.path.append(_GEN_MODELS_DIR)

try:
    import sgm  # noqa: F401
    from sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder  # bigG embedder
except Exception as e:
    sgm = None
    FrozenOpenCLIPImageEmbedder = None
    _SGM_IMPORT_ERROR = e

torch.backends.cuda.matmul.allow_tf32 = True
try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
except Exception:
    pass

import utils


# -------------------------
# Helpers
# -------------------------
def _h5_take(ds, idx):
    idx = np.asarray(idx)
    uniq, inverse = np.unique(idx, return_inverse=True)
    vals_uniq = ds[uniq]
    return vals_uniq[inverse]


def set_requires_grad(m, flag: bool):
    if m is None:
        return
    for p in m.parameters():
        p.requires_grad_(flag)


def _split_decay(named_params, no_decay_keys):
    decay, nodecay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay_keys):
            nodecay.append(p)
        else:
            decay.append(p)
    return decay, nodecay


def _get_lr_from_optimizer(optim, prefer_tag=None):
    """返回当前学习率（优先某个tag）。"""
    if prefer_tag is not None:
        for g in optim.param_groups:
            if g.get("tag") == prefer_tag:
                return float(g.get("lr", 0.0))
    # 兜底：取最大 lr
    lrs = [float(g.get("lr", 0.0)) for g in optim.param_groups]
    return max(lrs) if len(lrs) else 0.0


# -------------------------
# Multi-GPU & dtype config
# -------------------------
local_rank = accelerator.local_process_index
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

world_size = accelerator.state.num_processes
distributed = accelerator.state.distributed_type != 'NO'

num_devices = world_size if world_size and world_size > 0 else torch.cuda.device_count()
if num_devices == 0:
    num_devices = 1
num_workers = num_devices

acc_print = accelerator.print
acc_print("LOCAL RANK", local_rank)
acc_print(f"Mixed precision override: env={env_dtype or 'none'}, accelerate={mp}, using {data_type}")
acc_print("PID of this process =", os.getpid())
acc_print(accelerator.state)
acc_print("distributed =", distributed,
          "num_devices =", num_devices,
          "local rank =", local_rank,
          "world size =", world_size,
          "data_type =", data_type)


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument("--model_name", type=str, default="testing",
                    help="name of model, used for ckpt saving and wandb logging (if enabled)")
parser.add_argument("--data_path", type=str, default=os.getcwd(),
                    help="Path to where NSD data is stored / where to download it to")
parser.add_argument("--cache_dir", type=str, default=os.getcwd(),
                    help="Path to where misc. files downloaded from huggingface are stored.")
parser.add_argument("--subj", type=int, default=1, choices=[1,2,3,4,5,6,7,8],
                    help="Validate on which subject?")
parser.add_argument("--multisubject_ckpt", type=str, default=None,
                    help="Path to pre-trained multisubject model to finetune a single subject from.")
# 控制加载 ckpt 时是否丢弃 ridge.*（仅在“真正从多被试 ckpt 迁移到单被试”时才需要 True）
parser.add_argument("--drop_ridge_on_load", action=argparse.BooleanOptionalAction, default=False,
                    help="If True, drop ridge.* params when loading --multisubject_ckpt (for multisubject->single transfer).")
parser.add_argument("--num_sessions", type=int, default=1,
                    help="Number of training sessions to include")
parser.add_argument("--use_prior", action=argparse.BooleanOptionalAction, default=True,
                    help="whether to train diffusion prior (True) or just rely on retrieval part (False)")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size")
parser.add_argument("--wandb_log", action=argparse.BooleanOptionalAction, default=False,
                    help="whether to log to wandb")
parser.add_argument("--wandb_project", type=str, default="stability",
                    help="wandb project name")
parser.add_argument("--mixup_pct", type=float, default=.33,
                    help="proportion of way through training when to switch from BiMixCo to SoftCLIP")
parser.add_argument("--blurry_recon", action=argparse.BooleanOptionalAction, default=True,
                    help="whether to output blurry reconstructions")
parser.add_argument("--blur_scale", type=float, default=.5,
                    help="multiply loss from blurry recons by this number")
parser.add_argument("--clip_scale", type=float, default=1.,
                    help="multiply contrastive loss by this number")
parser.add_argument("--prior_scale", type=float, default=30,
                    help="multiply diffusion prior loss by this")
parser.add_argument("--use_image_aug", action=argparse.BooleanOptionalAction, default=False,
                    help="whether to use image augmentation")
parser.add_argument("--num_epochs", type=int, default=150,
                    help="number of epochs of training")
parser.add_argument("--multi_subject", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--train_subjs", type=str, default=None,
                    help="逗号分隔的训练被试列表，如 '1,2'")
parser.add_argument("--test_subjs", type=str, default=None,
                    help="逗号分隔的测试被试列表")
parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--lr_scheduler_type", type=str, default='cycle', choices=['cycle','linear'])
parser.add_argument("--ckpt_saving", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--ckpt_interval", type=int, default=5,
                    help="save backup ckpt and reconstruct every x epochs")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_lr", type=float, default=3e-4)
parser.add_argument("--train_split_ratio", type=float, default=None,
                    help="仅单被试(subj01)时启用：按比例切分 subj01 的 train 分片为训练/保留(测试)")

# TextAlign
parser.add_argument("--textalign_teacher_path", type=str, default=None,
                    help="Optional: override teacher path .pt (dict: image_ids, text_feats)")
parser.add_argument("--textalign_hardneg_path", type=str, default=None,
                    help="Optional: override hard-neg path .pt (dict: image_ids, neg_text_feats or text_feats)")
parser.add_argument("--textalign_scale", type=float, default=None,
                    help="Optional: override alpha_text (else use env MINDEYE_TEXTALIGN_SCALE or default)")
parser.add_argument("--textalign_tau", type=float, default=None,
                    help="Optional: override tau (else use env MINDEYE_TEXTALIGN_TAU or default 0.07)")
parser.add_argument("--textalign_margin", type=float, default=None,
                    help="Optional: override margin (else use env MINDEYE_TEXTALIGN_MARGIN or default 0.1)")
parser.add_argument("--textalign_hard_scale", type=float, default=None,
                    help="Optional: override hard extra scale (else env MINDEYE_TEXTALIGN_HARD_SCALE or default 1.0)")

if utils.is_interactive():
    args = parser.parse_args([])
else:
    args = parser.parse_args()

# stage control
STAGE = int(os.environ.get("MINDEYE_TEXTALIGN_STAGE", "1"))
# 0: stage0 train text_head only
# 1: stage1 finetune ridge+backbone(+prior) + text_head

utils.seed_everything(args.seed)

global_batch_size = args.batch_size * accelerator.num_processes

outdir = os.path.join(_PROJ_ROOT, 'train_logs', args.model_name)
if not os.path.exists(outdir) and args.ckpt_saving:
    os.makedirs(outdir, exist_ok=True)

# Aug
if args.use_image_aug or args.blurry_recon:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
if args.use_image_aug:
    img_augment = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
        same_on_batch=False,
        data_keys=["input"],
    )

# Subject list
if args.multi_subject:
    if args.train_subjs is not None and str(args.train_subjs).strip():
        _ts = [int(x) for x in str(args.train_subjs).replace(' ', '').split(',') if x]
        subj_list = np.array(sorted(list(set(_ts))))
    else:
        subj_list = np.arange(1, 9)
        subj_list = subj_list[subj_list != args.subj]
else:
    subj_list = [args.subj]

subj_list = np.array(subj_list)
subj_to_idx = {int(s): i for i, s in enumerate(subj_list)}
acc_print("subj_list", subj_list, "num_sessions", args.num_sessions, "STAGE", STAGE)

# -------------------------
# TextAlign: load teacher & hardneg
# -------------------------
teacher_path = args.textalign_teacher_path or os.path.join(_PROJ_ROOT, "data/nsd_text/train_coco_text_clip.pt")
hard_neg_path = args.textalign_hardneg_path or os.path.join(_PROJ_ROOT, "data/nsd_text/train_coco_captions_hard_negs_clip.pt")

text_feats_teacher = None
neg_text_feats_teacher = None
id2row = None
USE_TEXT_ALIGN = False
USE_HARD_NEG = False
pos_image_ids = None
neg_image_ids = None

if os.path.isfile(teacher_path):
    acc_print(f"[TextAlign] Loading teacher text features from: {teacher_path}")
    state = torch.load(teacher_path, map_location="cpu")
    if (not isinstance(state, dict)) or ("image_ids" not in state) or ("text_feats" not in state):
        raise RuntimeError(f"[TextAlign] {teacher_path} 必须是 dict 且包含 image_ids/text_feats")
    pos_image_ids = state["image_ids"].long()
    text_feats_teacher = state["text_feats"].float().to(device)  # teacher feats float32
    id2row = {int(img_id): idx for idx, img_id in enumerate(pos_image_ids.tolist())}
    USE_TEXT_ALIGN = True
    acc_print(f"[TextAlign] teacher feats shape={tuple(text_feats_teacher.shape)} #ids={len(pos_image_ids)}")
else:
    acc_print(f"[TextAlign] teacher file NOT found at {teacher_path}, TextAlign disabled.")

if os.path.isfile(hard_neg_path):
    acc_print(f"[TextAlign] Loading HARD-NEG text features from: {hard_neg_path}")
    state_neg = torch.load(hard_neg_path, map_location="cpu")
    if (not isinstance(state_neg, dict)) or ("image_ids" not in state_neg):
        raise RuntimeError(f"[TextAlign] {hard_neg_path} 必须是 dict 且包含 image_ids")
    neg_image_ids = state_neg["image_ids"].long()
    if "neg_text_feats" in state_neg:
        neg_text_feats_teacher = state_neg["neg_text_feats"].float().to(device)
    elif "text_feats" in state_neg:
        neg_text_feats_teacher = state_neg["text_feats"].float().to(device)
    else:
        raise RuntimeError(f"[TextAlign] {hard_neg_path} 缺少 neg_text_feats 或 text_feats")

    if (pos_image_ids is not None) and torch.equal(pos_image_ids.cpu(), neg_image_ids.cpu()):
        USE_HARD_NEG = True
        acc_print("[TextAlign] HARD-NEG enabled (image_ids 与正样本一致)")
    else:
        acc_print("[TextAlign] WARNING: hard-neg image_ids 与正样本不一致，禁用 HARD-NEG")
else:
    acc_print(f"[TextAlign] HARD-NEG file NOT found at {hard_neg_path}, hard negatives disabled.")

acc_print(f"[TextAlign] USE_TEXT_ALIGN={USE_TEXT_ALIGN} USE_HARD_NEG={USE_HARD_NEG}")

TEXT_ALIGN_SCALE_DEFAULT = 0.05
HARD_NEG_EXTRA_SCALE_DEFAULT = 1.0
TEXT_ALIGN_TAU_DEFAULT = 0.07
TEXT_ALIGN_MARGIN_DEFAULT = 0.1

if STAGE == 0:
    # Stage0 必须有 head/teacher，否则没意义
    if not USE_TEXT_ALIGN:
        raise RuntimeError("[Stage0] 你选择了 STAGE=0 但 teacher 不存在/未加载，无法只训 head。请检查 teacher_path。")

# -------------------------
# Prep data
# -------------------------
def my_split_by_node(urls): 
    return urls

num_voxels_list = []

if args.multi_subject:
    nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
    num_samples_per_epoch = (750 * 40) // num_devices
else:
    num_samples_per_epoch = (750 * args.num_sessions) // num_devices

_fast = os.environ.get("MINDEYE_FAST", "0") == "1"
if _fast:
    num_samples_per_epoch = min(num_samples_per_epoch, 140)
    acc_print("[FAST] override num_samples_per_epoch ->", num_samples_per_epoch)

try:
    _epoch_frac = float(os.environ.get("MINDEYE_EPOCH_FRACTION", "1.0"))
    if 0 < _epoch_frac < 1.0:
        _ns = max(1, int(num_samples_per_epoch * _epoch_frac))
        acc_print(f"[EPOCH_FRACTION] scale num_samples_per_epoch {num_samples_per_epoch} -> {_ns} (frac={_epoch_frac})")
        num_samples_per_epoch = _ns
except Exception:
    pass

acc_print("dividing batch size by subj_list, which will then be concatenated across subj during training...")
per_subj_batch = max(1, args.batch_size // len(subj_list))
num_iterations_per_epoch = num_samples_per_epoch // (per_subj_batch * len(subj_list))

try:
    _max_steps = int(os.environ.get("MINDEYE_MAX_STEPS_PER_EPOCH", "0"))
    if _max_steps > 0 and num_iterations_per_epoch > _max_steps:
        acc_print(f"[EPOCH_CAP] cap num_iterations_per_epoch {num_iterations_per_epoch} -> {_max_steps}")
        num_iterations_per_epoch = _max_steps
        num_samples_per_epoch = num_iterations_per_epoch * (per_subj_batch * len(subj_list))
except Exception:
    pass

acc_print("batch_size(per_subj) =", per_subj_batch,
          "num_iterations_per_epoch =", num_iterations_per_epoch,
          "num_samples_per_epoch =", num_samples_per_epoch)

train_data = {}
train_dl = {}
num_voxels = {}
voxels = {}
_betas_files_keepalive = []

_SUBJ01_TRAIN_SHARDS_OVERRIDE = None
if (not args.multi_subject) and (int(args.subj) == 1) and (getattr(args, "train_split_ratio", None) is not None):
    try:
        split_ratio = float(getattr(args, "train_split_ratio"))
        train_dir = os.path.join(args.data_path, f"wds/subj01/train")
        shards_all = []
        for i in range(max(0, int(args.num_sessions))):
            p = os.path.join(train_dir, f"{i}.tar")
            if os.path.isfile(p):
                shards_all.append(p)
        if len(shards_all) == 0:
            import glob as _glob
            shards_all = sorted(_glob.glob(os.path.join(train_dir, "*.tar")))
        if len(shards_all) >= 2:
            k = int(round(len(shards_all) * max(0.0, min(1.0, split_ratio))))
            k = min(max(k, 1), len(shards_all) - 1)
            _SUBJ01_TRAIN_SHARDS_OVERRIDE = shards_all[:k]
            acc_print(f"[SPLIT] subj01 train shards override: use {len(_SUBJ01_TRAIN_SHARDS_OVERRIDE)}/{len(shards_all)} shards for training; holdout {len(shards_all)-k} shards")
        else:
            acc_print("[SPLIT] subj01 shards < 2，跳过切分；沿用默认行为")
            _SUBJ01_TRAIN_SHARDS_OVERRIDE = None
    except Exception as _e:
        acc_print(f"[SPLIT] 解析/应用 --train_split_ratio 失败: {_e}; 沿用默认行为")

for s in subj_list:
    acc_print(f"Training with {args.num_sessions} sessions")
    if args.multi_subject:
        train_src = f"{args.data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
    else:
        if int(s) == 1 and _SUBJ01_TRAIN_SHARDS_OVERRIDE is not None:
            train_src = list(_SUBJ01_TRAIN_SHARDS_OVERRIDE)
        else:
            train_src = f"{args.data_path}/wds/subj0{s}/train/" + "{0.." + f"{args.num_sessions-1}" + "}.tar"
    acc_print("train_src:", train_src)

    train_data[f'subj0{s}'] = (
        wds.WebDataset(train_src, resampled=True, nodesplitter=my_split_by_node)
        .shuffle(750, initial=1500, rng=random.Random(42))
        .decode("torch")
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")
        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    )
    train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(
        train_data[f'subj0{s}'],
        batch_size=per_subj_batch,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    f = h5py.File(f'{args.data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
    _betas_files_keepalive.append(f)
    betas_ds = f['betas']
    num_voxels_list.append(betas_ds.shape[-1])
    num_voxels[f'subj0{s}'] = betas_ds.shape[-1]
    voxels[f'subj0{s}'] = betas_ds
    acc_print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")

acc_print("Loaded all subj train dls and betas!\n")


def _get_test_cfg(_subj: int):
    if not args.new_test:
        if _subj == 3:
            _num = 2113
        elif _subj == 4:
            _num = 1985
        elif _subj == 6:
            _num = 2113
        elif _subj == 8:
            _num = 1985
        else:
            _num = 2770
        _url = f"{args.data_path}/wds/subj0{_subj}/test/0.tar"
    else:
        if _subj == 3:
            _num = 2371
        elif _subj == 4:
            _num = 2188
        elif _subj == 6:
            _num = 2371
        elif _subj == 8:
            _num = 2188
        else:
            _num = 3000
        _url = f"{args.data_path}/wds/subj0{_subj}/new_test/0.tar"
    return _num, _url


if args.test_subjs is not None and str(args.test_subjs).strip():
    _tests = [int(x) for x in str(args.test_subjs).replace(' ', '').split(',') if x]
    test_subj_list = np.array(sorted(list(set(_tests))))
elif args.train_subjs is not None and str(args.train_subjs).strip():
    test_subj_list = np.array(sorted(list(set([int(x) for x in str(args.train_subjs).replace(' ', '').split(',') if x]))))
else:
    test_subj_list = np.array([subj_list[0] if len(subj_list) > 0 else args.subj])

test_dl_map = {}
test_num_map = {}
for _s in test_subj_list:
    _num_test, _url = _get_test_cfg(int(_s))
    acc_print("test_src:", _url)
    _data = (
        wds.WebDataset(_url, resampled=False, nodesplitter=my_split_by_node)
        .shuffle(750, initial=1500, rng=random.Random(42))
        .decode("torch")
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")
        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    )
    test_dl_map[int(_s)] = torch.utils.data.DataLoader(_data, batch_size=_num_test, shuffle=False, drop_last=True, pin_memory=True)
    test_num_map[int(_s)] = _num_test
    acc_print(f"Loaded test dl for subj{_s} (N={_num_test})!\n")


# -------------------------
# Lazy COCO images
# -------------------------
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


coco_h5_path = f'{args.data_path}/coco_images_224_float16.hdf5'
lazy_coco = LazyH5Images(coco_h5_path)
acc_print(f"Using lazy COCO images: total {len(lazy_coco)} (no full preloading)")


# -------------------------
# Models
# -------------------------
if FrozenOpenCLIPImageEmbedder is None:
    raise RuntimeError(f"FrozenOpenCLIPImageEmbedder import failed: {_SGM_IMPORT_ERROR}")

clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
)
clip_img_embedder.to(device)
clip_img_embedder.eval()
for p in clip_img_embedder.parameters():
    p.requires_grad_(False)

clip_seq_dim = 256
clip_emb_dim = 1664

autoenc = None
cnx = None
mean = None
std = None
blur_augs = None

if args.blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{args.cache_dir}/sd_image_var_autoenc.pth', map_location="cpu")
    autoenc.load_state_dict(ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)

    from autoencoder.convnext import ConvnextXL
    cnx = ConvnextXL(f'{args.cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth')
    cnx.requires_grad_(False)
    cnx.eval()
    cnx.to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.228, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

    blur_augs = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop((224, 224), scale=(.9, .9), ratio=(1, 1), p=1.0),
        data_keys=["input"],
    )


class MindEyeModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class RidgeRegression(nn.Module):
    def __init__(self, input_sizes, out_features):
        super().__init__()
        self.out_features = out_features
        self.linears = nn.ModuleList([nn.Linear(input_size, out_features) for input_size in input_sizes])

    def forward(self, x, subj_idx):
        # x: [B,1,V]
        out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)  # [B,1,H]
        return out


model = MindEyeModule()
model.ridge = RidgeRegression(num_voxels_list, out_features=args.hidden_dim)

# TextAlign backbone/head
from models_textalign import BrainNetwork, TextAlignHead
model.backbone = BrainNetwork(
    h=args.hidden_dim, in_dim=args.hidden_dim, seq_len=1, n_blocks=args.n_blocks,
    clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim,
    blurry_recon=args.blurry_recon, clip_scale=args.clip_scale
)

if USE_TEXT_ALIGN:
    _text_dim = int(text_feats_teacher.shape[-1]) if text_feats_teacher is not None else 768
    model.text_head = TextAlignHead(token_dim=clip_emb_dim, hidden_dim=2048, text_dim=_text_dim)
else:
    model.text_head = None

# Prior
if args.use_prior:
    from models import PriorNetwork, BrainDiffusionPrior

    out_dim = clip_emb_dim
    depth = 6
    dim_head = 52
    heads = clip_emb_dim // 52
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

# -------------------------
# Stage: freeze/unfreeze
# -------------------------
if STAGE == 0:
    acc_print("[Stage0] train text_head only; freeze ridge/backbone/prior")
    set_requires_grad(model.ridge, False)
    set_requires_grad(model.backbone, False)
    set_requires_grad(model.diffusion_prior, False)
    if model.text_head is None:
        raise RuntimeError("[Stage0] model.text_head is None. 请确认 teacher 加载成功且 USE_TEXT_ALIGN=True。")
    set_requires_grad(model.text_head, True)
else:
    acc_print("[Stage1] finetune ridge+backbone(+prior) and text_head")
    set_requires_grad(model.ridge, True)
    set_requires_grad(model.backbone, True)
    set_requires_grad(model.diffusion_prior, True)
    if model.text_head is not None:
        set_requires_grad(model.text_head, True)

num_params = utils.count_params(model)
acc_print("num_params =", num_params)

# -------------------------
# Optimizer / Scheduler
# -------------------------
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

opt_grouped_parameters = []

# ridge
ridge_decay, ridge_nodecay = _split_decay(model.ridge.named_parameters(), no_decay)
if len(ridge_decay) > 0:
    opt_grouped_parameters.append({"params": ridge_decay, "weight_decay": 1e-2, "tag": "ridge"})
if len(ridge_nodecay) > 0:
    opt_grouped_parameters.append({"params": ridge_nodecay, "weight_decay": 0.0, "tag": "ridge"})

# backbone
backbone_decay, backbone_nodecay = _split_decay(model.backbone.named_parameters(), no_decay)
if len(backbone_decay) > 0:
    opt_grouped_parameters.append({"params": backbone_decay, "weight_decay": 1e-2, "tag": "backbone"})
if len(backbone_nodecay) > 0:
    opt_grouped_parameters.append({"params": backbone_nodecay, "weight_decay": 0.0, "tag": "backbone"})

# text_head
if model.text_head is not None:
    head_decay, head_nodecay = _split_decay(model.text_head.named_parameters(), no_decay)
    if len(head_decay) > 0:
        opt_grouped_parameters.append({"params": head_decay, "weight_decay": 1e-2, "tag": "text_head"})
    if len(head_nodecay) > 0:
        opt_grouped_parameters.append({"params": head_nodecay, "weight_decay": 0.0, "tag": "text_head"})

# prior
if model.diffusion_prior is not None:
    prior_decay, prior_nodecay = _split_decay(model.diffusion_prior.named_parameters(), no_decay)
    if len(prior_decay) > 0:
        opt_grouped_parameters.append({"params": prior_decay, "weight_decay": 1e-2, "tag": "prior"})
    if len(prior_nodecay) > 0:
        opt_grouped_parameters.append({"params": prior_nodecay, "weight_decay": 0.0, "tag": "prior"})

if len(opt_grouped_parameters) == 0:
    raise RuntimeError("No trainable parameters found. Check STAGE / requires_grad flags.")

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)

# Stage-specific LR policy
HEAD_LR_STAGE0 = float(os.environ.get("MINDEYE_LR_HEAD_STAGE0", "1e-3"))

LR_TEXT  = float(os.environ.get("MINDEYE_LR_TEXT", "1e-3"))
LR_BACKB = float(os.environ.get("MINDEYE_LR_BACKBONE", "1e-4"))
LR_RIDGE = float(os.environ.get("MINDEYE_LR_RIDGE", "3e-4"))
LR_PRIOR = float(os.environ.get("MINDEYE_LR_PRIOR", "1e-4"))

lr_scheduler = None

def _print_param_groups(title):
    if not accelerator.is_main_process:
        return
    acc_print(f"\n[{title}] optimizer param_groups summary:")
    for i, g in enumerate(optimizer.param_groups):
        n_params = sum(p.numel() for p in g["params"])
        acc_print(f"  - group{i:02d} tag={g.get('tag')} n_params={n_params/1e6:.3f}M lr={g.get('lr')} wd={g.get('weight_decay')}")

if STAGE == 0:
    # 固定 lr；非 text_head 的组 lr=0（安全）
    for g in optimizer.param_groups:
        if g.get("tag") == "text_head":
            g["lr"] = HEAD_LR_STAGE0
        else:
            g["lr"] = 0.0
    acc_print(f"[Stage0] fixed lr: head={HEAD_LR_STAGE0} (others=0); scheduler=OFF")
    _print_param_groups("Stage0")
else:
    total_steps = int(np.floor(args.num_epochs * num_iterations_per_epoch))
    acc_print("total_steps", total_steps)

    # per-group max_lr
    max_lrs = []
    for g in optimizer.param_groups:
        tag = g.get("tag", "")
        if tag == "text_head":
            max_lrs.append(LR_TEXT)
        elif tag == "backbone":
            max_lrs.append(LR_BACKB)
        elif tag == "ridge":
            max_lrs.append(LR_RIDGE)
        elif tag == "prior":
            max_lrs.append(LR_PRIOR)
        else:
            max_lrs.append(args.max_lr)

    if args.lr_scheduler_type == "cycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1,
            pct_start=min(0.5, 2 / args.num_epochs),
        )
        acc_print(f"[Stage1] OneCycle max_lr: text={LR_TEXT} backbone={LR_BACKB} ridge={LR_RIDGE} prior={LR_PRIOR}")
    else:
        # linear：用每组当前 lr 作为起点线性衰减到 0
        for g, lr0 in zip(optimizer.param_groups, max_lrs):
            g["lr"] = lr0
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=total_steps,
            last_epoch=-1
        )
        acc_print(f"[Stage1] LinearLR start_lr: text={LR_TEXT} backbone={LR_BACKB} ridge={LR_RIDGE} prior={LR_PRIOR}")

    _print_param_groups("Stage1")


# -------------------------
# CKPT save/load (scheduler None-safe + stage mismatch safe)
# -------------------------
def save_ckpt(tag, epoch, losses, test_losses, lrs):
    ckpt_path = os.path.join(outdir, f'{tag}.pth')
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        payload = {
            'epoch': epoch,
            'stage': STAGE,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            'args': vars(args),
        }
        if lr_scheduler is not None:
            payload['lr_scheduler'] = lr_scheduler.state_dict()
        torch.save(payload, ckpt_path)
    acc_print(f"\n---saved {ckpt_path} ckpt!---\n")


def load_ckpt(tag, outdir_override=None,
              load_lr=True, load_optimizer=True, load_epoch=True,
              strict=True, multisubj_loading=False):
    _out = outdir if outdir_override is None else outdir_override
    ckpt_path = os.path.join(_out, f"{tag}.pth")
    acc_print(f"\n---loading {ckpt_path} ckpt---\n")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    if multisubj_loading:
        drop_keys = [k for k in state_dict.keys() if k.startswith("ridge.")]
        for k in drop_keys:
            state_dict.pop(k, None)

    model.load_state_dict(state_dict, strict=strict)

    ckpt_stage = checkpoint.get("stage", None)
    if ckpt_stage is not None and int(ckpt_stage) != int(STAGE):
        acc_print(f"[CKPT] WARNING: ckpt_stage={ckpt_stage} != current STAGE={STAGE}. "
                  f"Skip optimizer/scheduler load to avoid param_group mismatch.")
        load_optimizer = False
        load_lr = False

    loaded_epoch = 0
    if load_epoch:
        loaded_epoch = int(checkpoint.get('epoch', 0))
        acc_print("Epoch in ckpt:", loaded_epoch)

    if load_optimizer and ('optimizer_state_dict' in checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if load_lr and (lr_scheduler is not None) and ('lr_scheduler' in checkpoint):
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    del checkpoint
    return loaded_epoch


# -------------------------
# W&B
# -------------------------
wandb_log = bool(args.wandb_log)
if local_rank == 0 and wandb_log:
    import wandb
    wandb_project = args.wandb_project or 'mindeye'
    acc_print(f"wandb {wandb_project} run {args.model_name}")

    wandb_config = {
        "model_name": args.model_name,
        "global_batch_size": global_batch_size,
        "batch_size_per_subj": per_subj_batch,
        "num_epochs": args.num_epochs,
        "num_sessions": args.num_sessions,
        "num_params": num_params,
        "clip_scale": args.clip_scale,
        "prior_scale": args.prior_scale,
        "blur_scale": args.blur_scale,
        "use_image_aug": args.use_image_aug,
        "mixup_pct": args.mixup_pct,
        "num_samples_per_epoch": num_samples_per_epoch,
        "num_iterations_per_epoch": num_iterations_per_epoch,
        "ckpt_interval": args.ckpt_interval,
        "ckpt_saving": args.ckpt_saving,
        "seed": args.seed,
        "distributed": distributed,
        "num_devices": num_devices,
        "world_size": world_size,
        "stage": STAGE,
        "use_textalign": USE_TEXT_ALIGN,
        "use_hardneg": USE_HARD_NEG,
        "teacher_path": teacher_path,
        "hardneg_path": hard_neg_path,
    }
    acc_print("wandb_config:\n", wandb_config)
    wandb.init(
        id=args.model_name,
        project=wandb_project,
        name=args.model_name,
        config=wandb_config,
        resume="allow",
    )
else:
    wandb_log = False


# -------------------------
# Load multisubject ckpt if set
# -------------------------
epoch = 0
losses, test_losses, lrs = [], [], []
best_test_loss = 1e9

torch.cuda.empty_cache()
if torch.cuda.is_available():
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

# [Auto-Patch] 断点续训逻辑
resume_path = os.environ.get('MINDEYE_RESUME', '')
if resume_path and os.path.exists(os.path.join(resume_path, 'last.pth')):
    acc_print(f'\n[RESUME] ⚠️ 检测到续训信号！正在从 {resume_path} 恢复进度...')
    # 强制加载 optimizer, scheduler 和 epoch，且 strict=False 兼容旧权重
    epoch = load_ckpt('last', outdir_override=resume_path, load_lr=True, load_optimizer=True, load_epoch=True, strict=False)
    acc_print(f'[RESUME] 成功恢复！将从 Epoch {epoch} 继续训练\n')
elif args.multisubject_ckpt is not None:
    # 复用 ckpt：strict=False（TextAlign head 新增参数 ckpt 里没有）
    _ = load_ckpt(
        "last",
        outdir_override=args.multisubject_ckpt,
        load_lr=False,
        load_optimizer=False,
        load_epoch=False,
        strict=False,
        multisubj_loading=bool(args.drop_ridge_on_load)
    )

# -------------------------
# Prepare with accelerator (scheduler None-safe)
# -------------------------
train_dls = [train_dl[f'subj0{s}'] for s in subj_list]

if lr_scheduler is None:
    model, optimizer, *train_dls = accelerator.prepare(model, optimizer, *train_dls)
else:
    model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)

acc_print(f"{args.model_name} starting with epoch {epoch} / {args.num_epochs}")

progress_bar = tqdm(range(epoch, args.num_epochs), ncols=120, disable=(local_rank != 0))

mse = nn.MSELoss()
l1 = nn.L1Loss()
soft_loss_temps = torch.full((args.num_epochs - int(args.mixup_pct * args.num_epochs),), 0.02)


# -------------------------
# TextAlign losses
# -------------------------
def text_align_loss(t_pred, t_teacher, tau=0.07):
    t_pred = F.normalize(t_pred, dim=-1)
    t_teacher = F.normalize(t_teacher, dim=-1)
    logits = (t_pred @ t_teacher.t()) / tau
    labels = torch.arange(t_pred.size(0), device=t_pred.device)
    return F.cross_entropy(logits, labels)


def text_align_hardneg_loss(t_pred, t_pos, t_neg, margin=0.1):
    t_pred_n = F.normalize(t_pred, dim=-1)
    t_pos_n = F.normalize(t_pos, dim=-1)
    t_neg_n = F.normalize(t_neg, dim=-1)
    cos_pos = (t_pred_n * t_pos_n).sum(dim=-1)
    cos_neg = (t_pred_n * t_neg_n).sum(dim=-1)
    return F.relu(margin - cos_pos + cos_neg).mean()


# -------------------------
# Train
# -------------------------
try:
    _prior_interval = int(os.environ.get("MINDEYE_PRIOR_INTERVAL", "1"))
except Exception:
    _prior_interval = 1

STAGE0_DO_EVAL = os.environ.get("MINDEYE_STAGE0_EVAL", "0") == "1"

for epoch in progress_bar:
    model.train()

    # meters
    fwd_percent_correct = 0.0
    bwd_percent_correct = 0.0
    test_fwd_percent_correct = 0.0
    test_bwd_percent_correct = 0.0

    recon_cossim = 0.0
    test_recon_cossim = 0.0
    recon_mse = 0.0
    test_recon_mse = 0.0

    loss_clip_total = 0.0
    loss_blurry_total = 0.0
    loss_blurry_cont_total = 0.0
    test_loss_clip_total = 0.0

    loss_prior_total = 0.0
    test_loss_prior_total = 0.0

    blurry_pixcorr = 0.0
    test_blurry_pixcorr = 0.0
    loss_text_total = 0.0

    train_iterators = [iter(dl) for dl in train_dls]

    for train_i in range(num_iterations_per_epoch):
        optimizer.zero_grad()
        loss = 0.0

        voxel_list = []
        image_chunks = []
        global_id_chunks = []
        mix_perm_list, mix_betas_list, mix_select_list = [], [], []

        # ---- fetch a batch per subject ----
        for si, s in enumerate(subj_list):
            while True:
                try:
                    behav0, past_behav0, future_behav0, old_behav0 = next(train_iterators[si])
                except StopIteration:
                    train_iterators[si] = iter(train_dls[si])
                    behav0, past_behav0, future_behav0, old_behav0 = next(train_iterators[si])

                image_idx = behav0[:, 0, 0].cpu().long().numpy()
                image0, image_sorted_idx = np.unique(image_idx, return_index=True)
                if len(image0) != len(image_idx):
                    continue

                img_tensor = lazy_coco.get(image0)
                image_chunks.append(img_tensor)
                global_id_chunks.append(torch.from_numpy(image0))

                voxel_idx = behav0[:, 0, 5].cpu().long().numpy()
                voxel_sorted_idx = voxel_idx[image_sorted_idx]
                voxel0_np = _h5_take(voxels[f'subj0{s}'], voxel_sorted_idx)
                voxel0 = torch.tensor(voxel0_np).unsqueeze(1)  # [B,1,V]

                if epoch < int(args.mixup_pct * args.num_epochs) and STAGE == 1:
                    voxel0, perm, betas, select = utils.mixco(voxel0)
                    mix_perm_list.append(perm)
                    mix_betas_list.append(betas)
                    mix_select_list.append(select)

                voxel_list.append(voxel0)
                break

        image = torch.cat(image_chunks, dim=0)
        global_ids = torch.cat(global_id_chunks, dim=0).to(device)

        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image = image.to(device, non_blocking=True)

        voxel_list = [v.to(device) for v in voxel_list]

        if args.use_image_aug and STAGE == 1:
            image = img_augment(image)

        # ---- CLIP image target (only Stage1 needs it) ----
        clip_target = None
        if STAGE == 1 and (args.clip_scale > 0 or args.use_prior):
            with torch.no_grad():
                force_clip_fp32 = os.environ.get("CLIP_FP32", "1") == "1"
                clip_amp_dtype = torch.float32 if force_clip_fp32 else data_type
                with torch.cuda.amp.autocast(dtype=clip_amp_dtype):
                    _img = image.float() if force_clip_fp32 else image
                    _out = clip_img_embedder(_img)
                    clip_target = _out[0] if isinstance(_out, tuple) else _out
            assert clip_target is not None
            assert not torch.any(torch.isnan(clip_target))

        # ---- mixco vars (Stage1 only) ----
        if STAGE == 1 and epoch < int(args.mixup_pct * args.num_epochs):
            perm = torch.cat([t.detach().to(device) for t in mix_perm_list], dim=0)
            betas = torch.cat([t.detach().to(device) for t in mix_betas_list], dim=0)
            select = torch.cat([t.detach().to(device) for t in mix_select_list], dim=0)

        # ---- forward ridge/backbone ----
        with accelerator.autocast():
            if STAGE == 0:
                # Stage0: no_grad for frozen modules to save memory
                with torch.no_grad():
                    voxel_ridge_list = [model.ridge(voxel_list[si], si) for si, _ in enumerate(subj_list)]
                    voxel_ridge = torch.cat(voxel_ridge_list, dim=0)
                    backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)
            else:
                voxel_ridge_list = [model.ridge(voxel_list[si], si) for si, _ in enumerate(subj_list)]
                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)
                backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)

            # ---------------- TextAlign loss (Stage0 & Stage1 both) ----------------
            use_textalign_flag = (
                USE_TEXT_ALIGN
                and (model.text_head is not None)
                and (id2row is not None)
                and (len(id2row) > 0)
                and os.environ.get("MINDEYE_TEXTALIGN", "1") == "1"
            )
            use_hard_neg_flag = (
                USE_HARD_NEG
                and os.environ.get("MINDEYE_TEXTALIGN_HARDNEG", "1") == "1"
            )

            alpha_text = (
                float(args.textalign_scale) if (args.textalign_scale is not None)
                else float(os.environ.get("MINDEYE_TEXTALIGN_SCALE", TEXT_ALIGN_SCALE_DEFAULT))
            )
            tau = (
                float(args.textalign_tau) if (args.textalign_tau is not None)
                else float(os.environ.get("MINDEYE_TEXTALIGN_TAU", TEXT_ALIGN_TAU_DEFAULT))
            )
            margin = (
                float(args.textalign_margin) if (args.textalign_margin is not None)
                else float(os.environ.get("MINDEYE_TEXTALIGN_MARGIN", TEXT_ALIGN_MARGIN_DEFAULT))
            )
            hard_extra_scale = (
                float(args.textalign_hard_scale) if (args.textalign_hard_scale is not None)
                else float(os.environ.get("MINDEYE_TEXTALIGN_HARD_SCALE", HARD_NEG_EXTRA_SCALE_DEFAULT))
            )

            if use_textalign_flag and alpha_text > 0:
                ids_np = global_ids.detach().cpu().tolist()
                rows = [id2row.get(int(gid), -1) for gid in ids_np]
                rows = torch.tensor(rows, device=device, dtype=torch.long)
                valid_mask = rows >= 0

                if epoch == 0 and train_i < 3 and accelerator.is_main_process:
                    acc_print(
                        f"[TextAlign] epoch {epoch} step {train_i}: "
                        f"batch={len(ids_np)} hits={int(valid_mask.sum())} use_hard_neg={use_hard_neg_flag}"
                    )

                if valid_mask.any():
                    rows_valid = rows[valid_mask]
                    backbone_tokens_valid = backbone[valid_mask]     # [Bv, 256, 1664]
                    t_pos = text_feats_teacher[rows_valid]           # float32 on device

                    t_pred = model.text_head(backbone_tokens_valid)  # [Bv, text_dim]

                    # loss in fp32 for stability
                    t_pred_f = t_pred.float()
                    t_pos_f = t_pos.float()
                    L_pos = text_align_loss(t_pred_f, t_pos_f, tau=tau)

                    L_combined = L_pos
                    if use_hard_neg_flag and (neg_text_feats_teacher is not None) and hard_extra_scale > 0:
                        t_neg = neg_text_feats_teacher[rows_valid]
                        L_hard = text_align_hardneg_loss(t_pred_f, t_pos_f, t_neg.float(), margin=margin)
                        L_combined = L_combined + hard_extra_scale * L_hard

                    loss_text_total += float(L_combined.detach().cpu().item())
                    loss = loss + alpha_text * L_combined

            # ---------------- Stage1 extra losses ----------------
            if STAGE == 1:
                if args.clip_scale > 0:
                    clip_voxels_norm = F.normalize(clip_voxels.flatten(1), dim=-1, eps=1e-6)
                    clip_target_norm = F.normalize(clip_target.flatten(1), dim=-1, eps=1e-6)

                # prior
                if model.diffusion_prior is not None:
                    _step_id = epoch * num_iterations_per_epoch + train_i
                    _do_prior = (_prior_interval <= 1) or ((_step_id % _prior_interval) == 0)
                    if _do_prior:
                        loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                        loss_prior_total += float(loss_prior.detach().cpu().item())
                        loss_prior = loss_prior * args.prior_scale * max(1, _prior_interval)
                        loss = loss + loss_prior

                        recon_cossim += float(F.cosine_similarity(prior_out, clip_target).mean().detach().cpu().item())
                        recon_mse += float(mse(prior_out, clip_target).detach().cpu().item())

                # clip contrastive
                if args.clip_scale > 0:
                    if epoch < int(args.mixup_pct * args.num_epochs):
                        loss_clip = utils.mixco_nce(
                            clip_voxels_norm.float(),
                            clip_target_norm.float(),
                            temp=0.02,
                            perm=perm, betas=betas, select=select
                        )
                    else:
                        epoch_temp = soft_loss_temps[epoch - int(args.mixup_pct * args.num_epochs)]
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm.float(),
                            temp=epoch_temp
                        )
                    loss_clip_total += float(loss_clip.detach().cpu().item())
                    loss = loss + (loss_clip * args.clip_scale)

                # blurry
                if args.blurry_recon and autoenc is not None and cnx is not None:
                    image_enc_pred, transformer_feats = blurry_image_enc_
                    image_enc = autoenc.encode(2 * image - 1).latent_dist.mode() * 0.18215
                    loss_blurry = l1(image_enc_pred, image_enc)
                    loss_blurry_total += float(loss_blurry.detach().cpu().item())

                    if epoch < int(args.mixup_pct * args.num_epochs):
                        image_enc_shuf = image_enc[perm]
                        betas_shape = [-1] + [1] * (len(image_enc.shape) - 1)
                        # [BF16 Fix] 强制转换数据类型以匹配 destination
                        mixed_val = image_enc[select] * betas[select].reshape(*betas_shape) + \
                                    image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)
                        image_enc[select] = mixed_val.to(image_enc.dtype)

                    image_norm = (image - mean) / std
                    image_aug = (blur_augs(image) - mean) / std
                    _, cnx_embeds = cnx(image_norm)
                    _, cnx_aug_embeds = cnx(image_aug)

                    cont_loss = utils.soft_cont_loss(
                        F.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1, eps=1e-6),
                        F.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1, eps=1e-6),
                        F.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1, eps=1e-6),
                        temp=0.2
                    )
                    loss_blurry_cont_total += float(cont_loss.detach().cpu().item())

                    loss = loss + (loss_blurry + 0.1 * cont_loss) * args.blur_scale

                # train retrieval metrics (Stage1 only)
                if args.clip_scale > 0:
                    labels = torch.arange(len(clip_voxels_norm), device=clip_voxels_norm.device)
                    fwd_percent_correct += float(utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).detach().cpu().item())
                    bwd_percent_correct += float(utils.topk(utils.batchwise_cosine_similarity(clip_target_norm.float(), clip_voxels_norm), labels, k=1).detach().cpu().item())

                if args.blurry_recon and autoenc is not None:
                    with torch.no_grad():
                        random_samps = np.random.choice(np.arange(len(image)), size=max(1, len(image)//5), replace=False)
                        blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps] / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                        pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                        blurry_pixcorr += float(pixcorr.detach().cpu().item())

        # backward / step
        utils.check_loss(loss)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # scheduler step
        if lr_scheduler is not None:
            lr_scheduler.step()

        losses.append(float(loss.detach().cpu().item()))
        lrs.append(_get_lr_from_optimizer(optimizer, prefer_tag="text_head"))

        # logging interval
        if local_rank == 0:
            try:
                _interval = int(os.environ.get("LOG_STEP_INTERVAL", "200"))
            except Exception:
                _interval = 200
            if _interval > 0 and ((train_i + 1) % _interval == 0 or (train_i + 1) == num_iterations_per_epoch):
                try:
                    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
                    peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                    acc_print(f"[epoch {epoch}/{args.num_epochs} step {train_i+1}/{num_iterations_per_epoch}] "
                              f"loss={loss.item():.4f} lr={_get_lr_from_optimizer(optimizer, 'text_head'):.2e} "
                              f"peak_alloc_GB={peak_alloc:.2f} peak_reserved_GB={peak_reserved:.2f}")
                except Exception:
                    acc_print(f"[epoch {epoch}/{args.num_epochs} step {train_i+1}/{num_iterations_per_epoch}] "
                              f"loss={loss.item():.4f} lr={_get_lr_from_optimizer(optimizer, 'text_head'):.2e}")

    # -------------------------
    # Eval
    # -------------------------
    model.eval()
    do_eval = (STAGE == 1) or (STAGE0_DO_EVAL and STAGE == 0)

    if local_rank == 0 and do_eval:
        # Stage0 的 eval 意义很弱（因为不算 clip/prior/blurry），这里默认只跑 Stage1；
        # 如果你确实想 Stage0 也跑，设置 env: MINDEYE_STAGE0_EVAL=1
        with torch.no_grad():
            for _eval_subj in list(test_dl_map.keys()):
                test_dl = test_dl_map[_eval_subj]
                num_test = test_num_map[_eval_subj]
                test_image, test_voxel = None, None

                for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
                    assert len(behav) == num_test
                    # 只会迭代一次（drop_last=True, batch_size=num_test）

                # build (image, voxel) for unique images with 3 repeats
                voxel_idx = behav[:, 0, 5].cpu().long().numpy()
                voxel_np = _h5_take(voxels[f'subj0{_eval_subj}'], voxel_idx)
                voxel = torch.tensor(voxel_np).unsqueeze(1)

                image_ids = behav[:, 0, 0].cpu().long()
                unique_image = torch.unique(image_ids)

                for im in unique_image:
                    im_int = int(im.item())
                    locs = torch.where(im == image_ids)[0]
                    if len(locs) == 1:
                        locs = locs.repeat(3)
                    elif len(locs) == 2:
                        locs = locs.repeat(2)[:3]
                    assert len(locs) == 3

                    if test_image is None:
                        test_image = lazy_coco.get([im_int])
                        test_voxel = voxel[locs][None]
                    else:
                        test_image = torch.vstack((test_image, lazy_coco.get([im_int])))
                        test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                # keep first 300
                test_indices = torch.arange(len(test_voxel))[:300]
                voxel = test_voxel[test_indices].to(device)
                image = test_image[test_indices].to(torch.float32).to(device, non_blocking=True)

                # Stage0: skip expensive eval losses
                if STAGE == 0:
                    continue

                # Stage1 eval
                with torch.cuda.amp.autocast(dtype=data_type):
                    _out = clip_img_embedder(image.float())
                    clip_target = _out[0] if isinstance(_out, tuple) else _out

                    _subj_idx = subj_to_idx.get(int(_eval_subj), 0)
                    for rep in range(3):
                        voxel_ridge = model.ridge(voxel[:, rep], _subj_idx)
                        backbone0, clip_voxels0, blurry_image_enc_ = model.backbone(voxel_ridge)
                        if rep == 0:
                            clip_voxels = clip_voxels0
                            backbone = backbone0
                        else:
                            clip_voxels += clip_voxels0
                            backbone += backbone0
                    clip_voxels /= 3
                    backbone /= 3

                    if args.clip_scale > 0:
                        clip_voxels_norm = F.normalize(clip_voxels.flatten(1), dim=-1, eps=1e-6)
                        clip_target_norm = F.normalize(clip_target.flatten(1), dim=-1, eps=1e-6)

                    random_samps = np.random.choice(np.arange(len(image)), size=max(1, len(image)//5), replace=False)

                    eval_loss = 0.0

                    if model.diffusion_prior is not None:
                        loss_prior, contaminated_prior_out = model.diffusion_prior(
                            text_embed=backbone[random_samps], image_embed=clip_target[random_samps]
                        )
                        test_loss_prior_total += float(loss_prior.detach().cpu().item())
                        test_recon_cossim += float(F.cosine_similarity(contaminated_prior_out, clip_target[random_samps]).mean().detach().cpu().item())
                        test_recon_mse += float(mse(contaminated_prior_out, clip_target[random_samps]).detach().cpu().item())
                        eval_loss = eval_loss + loss_prior * args.prior_scale

                    if args.clip_scale > 0:
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm.float(),
                            temp=0.02
                        )
                        test_loss_clip_total += float(loss_clip.detach().cpu().item())
                        eval_loss = eval_loss + loss_clip * args.clip_scale

                        labels = torch.arange(len(clip_voxels_norm), device=clip_voxels_norm.device)
                        test_fwd_percent_correct += float(utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).detach().cpu().item())
                        test_bwd_percent_correct += float(utils.topk(utils.batchwise_cosine_similarity(clip_target_norm.float(), clip_voxels_norm), labels, k=1).detach().cpu().item())

                    if args.blurry_recon and autoenc is not None:
                        image_enc_pred, _ = blurry_image_enc_
                        blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps] / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                        pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                        test_blurry_pixcorr += float(pixcorr.detach().cpu().item())

                utils.check_loss(eval_loss)
                test_losses.append(float(eval_loss.detach().cpu().item()))

        # logs (Stage1 meaningful; Stage0 minimal)
        denom_train = max(1, num_iterations_per_epoch)
        denom_test = max(1, len(test_losses)) if STAGE == 1 else 1

        logs = {
            "stage": STAGE,
            "train/loss": float(np.mean(losses[-denom_train:])),
            "train/lr_head": float(_get_lr_from_optimizer(optimizer, "text_head")),
            "train/loss_text": float(loss_text_total / denom_train),
        }

        if STAGE == 1:
            logs.update({
                "test/loss": float(np.mean(test_losses[-1:])) if len(test_losses) else 0.0,
                "train/fwd_pct_correct": float(fwd_percent_correct / denom_train),
                "train/bwd_pct_correct": float(bwd_percent_correct / denom_train),
                "test/fwd_pct_correct": float(test_fwd_percent_correct / max(1, len(test_dl_map))),
                "test/bwd_pct_correct": float(test_bwd_percent_correct / max(1, len(test_dl_map))),
                "train/loss_clip_total": float(loss_clip_total / denom_train),
                "train/loss_blurry_total": float(loss_blurry_total / denom_train),
                "train/loss_blurry_cont_total": float(loss_blurry_cont_total / denom_train),
                "test/loss_clip_total": float(test_loss_clip_total / max(1, len(test_dl_map))),
                "train/blurry_pixcorr": float(blurry_pixcorr / denom_train),
                "test/blurry_pixcorr": float(test_blurry_pixcorr / max(1, len(test_dl_map))),
                "train/recon_cossim": float(recon_cossim / denom_train),
                "test/recon_cossim": float(test_recon_cossim / max(1, len(test_dl_map))),
                "train/recon_mse": float(recon_mse / denom_train),
                "test/recon_mse": float(test_recon_mse / max(1, len(test_dl_map))),
                "train/loss_prior": float(loss_prior_total / denom_train),
                "test/loss_prior": float(test_loss_prior_total / max(1, len(test_dl_map))),
            })

        progress_bar.set_postfix(**{k: (v if isinstance(v, (int, float)) else 0) for k, v in logs.items()})

        if torch.cuda.is_available():
            try:
                peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
                peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                acc_print(f"[GPU] peak_alloc_GB={peak_alloc:.2f} peak_reserved_GB={peak_reserved:.2f}")
            except Exception:
                pass

        if wandb_log:
            import wandb
            wandb.log(logs)

    # ckpt
    if args.ckpt_saving and (epoch % args.ckpt_interval == 0):
        save_ckpt('last', epoch, losses, test_losses, lrs)

    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()

acc_print("\n===Finished!===\n")
if args.ckpt_saving:
    save_ckpt('last', epoch, losses, test_losses, lrs)

if accelerator.is_main_process:
    plt.plot(losses); plt.title("train losses"); plt.show()
    if len(test_losses):
        plt.plot(test_losses); plt.title("test losses"); plt.show()
