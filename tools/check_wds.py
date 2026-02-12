#第二次实验时自动侦测目录并打印 数据里包含哪些keys
import webdataset as wds
import os
import glob

# 自动判断到底是 new_test 还是 test 目录
base_dir = "src/wds/subj01"
possible_dirs = ["new_test", "test"]
target_tar = None

for d in possible_dirs:
    # 检查 0.tar
    path = os.path.join(base_dir, d, "0.tar")
    if not os.path.exists(path):
        # 检查 000000.tar
        path = os.path.join(base_dir, d, "000000.tar")
    
    if os.path.exists(path):
        target_tar = path
        break

if target_tar is None:
    print(f"❌ 错误：在 {base_dir} 下没找到 {possible_dirs} 里的 tar 包。")
    if os.path.exists(base_dir):
        print(f"当前目录结构: {os.listdir(base_dir)}")
    else:
        print(f"目录 {base_dir} 不存在！")
else:
    print(f"✅ 锁定目标数据包: {target_tar}")
    
    # 读取第一个样本看 Keys
    try:
        # 使用 WebDataset 读取
        ds = wds.WebDataset(target_tar).decode()
        sample = next(iter(ds))
        
        print("\n========= 📋 样本 KEYS 列表 (请复制这部分) =========")
        print(sorted(sample.keys()))
        
        # 顺便检查一下 ID 的样子
        for k in ['nsdId', 'cocoId', 'image_id', 'id', '__key__']:
            if k in sample:
                print(f"👉 发现 ID 字段 [{k}]: {sample[k]}")
                
        # 顺便检查一下是否有现成的 CLIP
        for k in sample.keys():
            if 'clip' in k.lower():
                print(f"💎 发现潜在 CLIP 缓存: {k}")
                
        print("===================================================")
    except Exception as e:
        print(f"❌ 读取 WebDataset 失败: {e}")