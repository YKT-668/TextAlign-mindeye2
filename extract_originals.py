# extract_originals.py

import os
import h5py
import numpy as np
import torch
import webdataset as wds
from torchvision.transforms import ToPILImage
from tqdm import tqdm

# ======================= 1. 配置参数 (请确保与你运行推理时一致) =======================
# 这些参数需要和你运行 recon_inference_run.py 时使用的完全一样！
subj = 1
data_path = '/home/vipuser/MindEyeV2_Project/src' # 你的数据路径
output_dir = '/home/vipuser/MindEyeV2_Project/original_images' # 我们将把原始图片保存在这里
num_to_extract = 10 # 我们要提取前10张
# =====================================================================================

print("Starting script to extract original images...")

# --- 步骤1: 重新加载测试集元数据，获取图片索引顺序 ---
# 这段代码逻辑完全复制自 recon_inference_run.py，以确保顺序一致
print(f"Loading test set metadata for subj{subj}...")
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

def my_split_by_node(urls): return urls
test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)\
                .decode("torch")\
                .rename(behav="behav.npy")\
                .to_tuple("behav")
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False)

test_images_idx = []
# The DataLoader returns a tuple, even with one item. We unpack it here.
for (behav_tensor,) in test_dl: 
    # behav_tensor[:,0] 存储的是原始COCO图片的索引
    test_images_idx.extend(behav_tensor[:,0].cpu().numpy())


# 获取与重建顺序完全一致的、唯一的图片索引
unique_test_image_indices = np.unique(np.array(test_images_idx).astype(int))
print(f"Found {len(unique_test_image_indices)} unique images in the test set.")

# --- 步骤2: 从HDF5文件中提取并保存图片 ---
print(f"Opening image database: {data_path}/coco_images_224_float16.hdf5")
with h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r') as f:
    images_dataset = f['images']
    
    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Will save original images to: {os.path.abspath(output_dir)}")
    
    piler = ToPILImage()

    # 循环提取前 num_to_extract 张图片
    for i in tqdm(range(num_to_extract), desc="Extracting images"):
        # 获取第 i 个重建图对应的原始图片索引
        original_image_index = unique_test_image_indices[i]
        
        # 从HDF5数据集中读取图片数据
        # 数据是 float16, [0,1]范围, (3, 224, 224) 格式
        image_data = torch.from_numpy(images_dataset[original_image_index]).float()
        
        # 转换为PIL Image对象并保存
        pil_image = piler(image_data)
        save_path = os.path.join(output_dir, f"original_{i}.png")
        pil_image.save(save_path)

print("\n✅ Extraction complete!")
print(f"You can now compare 'recon_{i}.png' with 'original_{i}.png'.")


