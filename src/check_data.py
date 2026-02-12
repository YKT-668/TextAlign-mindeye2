# 二次实验时拿来检测数据集是否完全下载好

import h5py
import os
import numpy as np
import glob

def check_subject(subj):
    # 1. 检查 Beta 文件 (fMRI 数据)
    beta_file = f"betas_all_subj{subj:02d}_fp32_renorm.hdf5"
    if not os.path.exists(beta_file):
        print(f"❌ [Subj{subj}] 缺失 Beta 文件: {beta_file}")
        return
    
    try:
        with h5py.File(beta_file, 'r') as f:
            # 尝试读取 dataset 的形状
            keys = list(f.keys())
            print(f"✅ [Subj{subj}] Beta 文件正常 (Keys: {keys})")
    except Exception as e:
        print(f"❌ [Subj{subj}] Beta 文件损坏! Error: {e}")

    # 2. 检查 WDS (图像数据)
    wds_path = f"wds/subj{subj:02d}/train"
    tars = glob.glob(f"{wds_path}/*.tar")
    if len(tars) > 0:
        print(f"✅ [Subj{subj}] WDS 正常 (发现 {len(tars)} 个 tar 包)")
    else:
        print(f"⚠️ [Subj{subj}] WDS 警告: 在 {wds_path} 下没有找到 .tar 文件 (如果你没下这个人的wds则忽略)")

print("--- 开始数据完整性校验 ---")
# 检查你下载的 01, 02, 05, 07
for s in [1, 2, 5, 7]:
    check_subject(s)

# 检查共享文件
if os.path.exists("shared1000.npy"):
    try:
        data = np.load("shared1000.npy")
        print(f"✅ shared1000.npy 正常 (Shape: {data.shape})")
    except:
        print("❌ shared1000.npy 损坏")
else:
    print("❌ shared1000.npy 缺失")