#!/usr/bin/env python3
"""预检：核验 Ours S1 checkpoint + MindEye2 数据文件（远端元数据，不下载大文件）。"""
import os
from huggingface_hub import HfApi

api = HfApi()

# ---- 1. checkpoint ----
print("=" * 60)
print("[1] Ours S1 checkpoint @ HF Model master")
REPO = "ykt668/textalign-mindeye2-model"
REV = "a127f9295fd5656ac63ae436f07e61b80bf4efce"
try:
    info = api.model_info(REPO, revision=REV, files_metadata=True, token=True)
    print("revision sha:", info.sha, "| files:", len(info.siblings))
    for s in info.siblings:
        print("  ", s.rfilename, s.size, (s.lfs.sha256 if s.lfs else None))
    # 找 checkpoint
    ck = [s for s in info.siblings if "s1_textalign_stage1_FINAL_BEST_32" in s.rfilename]
    for s in ck:
        print("CHECKPOINT:", s.rfilename, "size=", s.size,
              "sha=", (s.lfs.sha256 if s.lfs else None))
except Exception as e:
    print("ERR:", type(e).__name__, str(e)[:200])

# ---- 2. MindEye2 数据文件 ----
print("=" * 60)
print("[2] pscotti/mindeyev2 @ 3f20ccbe614876c83894e148cc36530c4b04caa7")
DREV = "3f20ccbe614876c83894e148cc36530c4b04caa7"
targets = [
    "betas_all_subj01_fp32_renorm.hdf5",
    "wds/subj01/new_test/0.tar",
    "unclip6_epoch0_step110000.ckpt",
    "bigG_to_L_epoch8.pth",
    "vox_include_subj01.npy",
    "brain_region_masks.hdf5",
    "COCO_73k_subj_indices.hdf5",
    "shared1000.npy",
    "subj01_annots.npy",
    "test_73k_images.npy",
]
try:
    info = api.dataset_info("pscotti/mindeyev2", revision=DREV, files_metadata=True)
    print("revision sha:", info.sha, "| files:", len(info.siblings))
    byname = {s.rfilename: s for s in info.siblings}
    for t in targets:
        s = byname.get(t)
        if s:
            print(f"  FOUND {t}: size={s.size} sha={s.lfs.sha256 if s.lfs else None}")
        else:
            print(f"  MISSING {t}")
except Exception as e:
    print("ERR:", type(e).__name__, str(e)[:200])
