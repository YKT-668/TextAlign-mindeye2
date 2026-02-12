#这个脚本是二次实验时，利用移除投影层的 ViT-bigG-14 模型，将 1000 张测试集图片重新编码为 1664 维向量，从而生成与你脑特征维度完全对齐的评测标准答案（Ground Truth）
import torch
import numpy as np
import open_clip
from PIL import Image
import os
from tqdm import tqdm

# === 配置区 ===
IMAGES_PT = "src/evals/all_images.pt"
OUTPUT_NPY = "/mnt/work/data_cache/clip_img_gt.npy"

# 关键修正 1: 使用正确的模型
MODEL_NAME = "ViT-bigG-14"
PRETRAINED = "laion2b_s39b_b160k" 

if not os.path.exists(IMAGES_PT):
    print(f"❌ 找不到 {IMAGES_PT}")
    exit()

print(f"🚀 正在加载 OpenCLIP ({MODEL_NAME})...")
# 加载模型
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model.eval()

# 关键修正 2: 阉割掉投影层 (Projection Layer)
# 标准 bigG 输出是 1280，但我们需要 transformer 原始宽度 1664
if hasattr(model.visual, 'proj'):
    print(f"✂️  检测到投影层 (shape={model.visual.proj.shape})，正在移除以获取 1664 维特征...")
    model.visual.proj = None
else:
    print("⚠️ 警告：未找到投影层，模型可能已经是无投影版本。")

# 加载图片
print("📂 正在加载测试集图片 Tensor...")
images_tensor = torch.load(IMAGES_PT)
if images_tensor.max() > 1.0:
    images_tensor = images_tensor.float() / 255.0

# 归一化参数 (OpenCLIP 标准)
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

batch_size = 16
all_embs = []

print("⚡ 开始计算 1664 维特征 (CPU模式)...")
with torch.no_grad():
    for i in tqdm(range(0, len(images_tensor), batch_size)):
        batch = images_tensor[i : i + batch_size]
        
        # Resize 到 224x224 (bigG 可能支持更高分辨率，但 MindEye 默认为 224)
        if batch.shape[-1] != 224:
             import torch.nn.functional as F
             batch = F.interpolate(batch, size=(224, 224), mode='bicubic')

        # 归一化
        batch_norm = (batch - mean) / std
        
        # 编码 (因为 proj=None，这里会自动输出 1664 维)
        embs = model.encode_image(batch_norm)
        
        # 归一化 embedding (虽然没有投影，但通常还是做个 L2 norm 比较安全，或者保持原始)
        # MindEye2 这里通常直接用原始特征做 MSE，或者 Norm 后做 Cosine
        # 为了 Retrieve，我们通常做 Norm
        embs = embs / embs.norm(dim=-1, keepdim=True)
        
        all_embs.append(embs.cpu().numpy())

final_arr = np.concatenate(all_embs, axis=0)
np.save(OUTPUT_NPY, final_arr)
print(f"✅ 成功生成修正版 clip_img_gt.npy")
print(f"📊 最终形状: {final_arr.shape} (预期应该是 1000, 1664)")