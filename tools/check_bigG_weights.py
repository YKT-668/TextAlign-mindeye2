#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查本地 HuggingFace 缓存中的 ViT-bigG-14 权重 blob 是否可用。
策略：
1. 指定候选 blob 路径（9.9GB 大文件）列表。
2. 逐个尝试 torch.load（open_clip 的权重是一个普通 PyTorch state_dict 二进制）。
3. 打印是否成功、包含的键数量、若可能打印一个典型层的形状。
4. 成功的候选只保留一个，输出建议删除其它副本命令。

注意：
- 如果 torch.load 报 pickle/EOF 错误，则说明文件不完整或损坏。
- 如果没有网络，要避免触发额外下载，可在运行前导出 HF_HUB_OFFLINE=1。
"""
import os
import sys
import torch

CANDIDATES = [
    # hub 结构下的 blob
    "/data/huggingface_cache/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/blobs/db71bd27925917762c506c1b8a08e191d236e5586b5cd99f77d11aced578ec34",
    # legacy 镜像结构
    "/data/huggingface_cache/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/blobs/db71bd27925917762c506c1b8a08e191d236e5586b5cd99f77d11aced578ec34",
]

def try_load(path):
    print(f"\n[TEST] 加载: {path}")
    if not os.path.isfile(path):
        print("[FAIL] 文件不存在")
        return False
    size_gb = os.path.getsize(path) / (1024**3)
    print(f"[INFO] 文件大小: {size_gb:.2f} GB")
    try:
        # PyTorch >=2.6 默认 weights_only=True，会阻止正常的 state_dict 反序列化；显式关闭
        sd = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[FAIL] torch.load 失败: {e}")
        return False
    if not isinstance(sd, dict):
        print(f"[WARN] 加载结果不是 dict，类型: {type(sd)}")
        return False
    keys = list(sd.keys())
    print(f"[OK] state_dict 键数量: {len(keys)}")
    # 选一个常见的投影层权重看看形状
    sample_key = None
    for k in keys:
        if any(x in k.lower() for x in ["proj", "conv", "attn", "mlp"]):
            sample_key = k
            break
    if sample_key:
        t = sd[sample_key]
        if torch.is_tensor(t):
            print(f"[INFO] 示例张量: {sample_key} shape={tuple(t.shape)} dtype={t.dtype}")
    return True

valid = []
for p in CANDIDATES:
    ok = try_load(p)
    if ok:
        valid.append(p)

print("\n================ SUMMARY ================")
if not valid:
    print("没有任何候选成功加载，可能需重新下载。")
    sys.exit(1)
print("可用权重文件数量:", len(valid))
for i, p in enumerate(valid):
    print(f"  [{i}] {p}")
# 建议保留第一个，其它给出删除命令
keep = valid[0]
print(f"\n建议保留: {keep}")
if len(valid) > 1:
    print("可以删除其余副本：")
    for p in valid[1:]:
        print(f"  rm -f {p}")
print("==========================================")
