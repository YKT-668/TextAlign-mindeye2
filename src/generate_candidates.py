import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm
from math import ceil

# --- 配置 ---
# 你的模型名
MODEL_NAME = "s1_textalign_stage1_FINAL_BEST_32"
# 你的结果路径
RECONS_PATH = f"evals/{MODEL_NAME}/{MODEL_NAME}_all_enhancedrecons.pt"
IMAGES_PATH = "evals/all_images.pt"
OUTPUT_DIR = "candidates_view"  # 结果保存的文件夹

# 每一页显示多少张图 (推荐 20，即 5行4列)
BATCH_SIZE = 20 
COLS = 5 # 每行显示几组

# --- 1. 加载数据 ---
print(f"[INFO] Loading data from {MODEL_NAME}...")
# 强制使用 CPU 加载以节省显存，反正只是画图
all_images = torch.load(IMAGES_PATH, map_location="cpu")
all_recons = torch.load(RECONS_PATH, map_location="cpu")

# 确保数量一致
assert len(all_images) == len(all_recons)
total_images = len(all_images)
print(f"[INFO] Total images: {total_images}")

# --- 2. 准备工具 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
imsize = 256
resize = transforms.Resize((imsize, imsize))
to_pil = transforms.ToPILImage()

def draw_text(img, text):
    """在图片左上角绘制红色编号"""
    draw = ImageDraw.Draw(img)
    # 默认字体，如果太小可以尝试加载大字体，但在服务器上通常只有默认
    # 为了清晰，我们画一个黑色背景框，再写白色字
    draw.rectangle([0, 0, 50, 20], fill="black") 
    draw.text((5, 5), text, fill="white")
    return img

# --- 3. 循环生成分页图 ---
num_pages = ceil(total_images / BATCH_SIZE)

print(f"[INFO] Generating {num_pages} pages of candidates...")

for page in tqdm(range(num_pages)):
    start_idx = page * BATCH_SIZE
    end_idx = min((page + 1) * BATCH_SIZE, total_images)
    
    # 获取当前页的数据
    batch_images = all_images[start_idx:end_idx]
    batch_recons = all_recons[start_idx:end_idx]
    
    # 准备当前页的 PIL 图片列表
    page_pil_imgs = []
    
    for i in range(len(batch_images)):
        global_idx = start_idx + i # 这是这组图在整个数据集中的真实编号
        
        # 处理原图 (Ground Truth)
        img_gt = resize(batch_images[i]).float()
        pil_gt = to_pil(img_gt.clamp(0,1))
        # !!! 关键步骤：印上编号 !!!
        pil_gt = draw_text(pil_gt, str(global_idx))
        
        # 处理重建图 (Recon)
        img_rec = resize(batch_recons[i]).float()
        pil_rec = to_pil(img_rec.clamp(0,1))
        
        # 拼在一起：左边是原图(带编号)，右边是重建图
        pair_img = Image.new('RGB', (imsize * 2, imsize))
        pair_img.paste(pil_gt, (0, 0))
        pair_img.paste(pil_rec, (imsize, 0))
        
        page_pil_imgs.append(pair_img)
    
    # --- 4. 将这一页的20组拼成网格 ---
    # 计算行数
    rows = ceil(len(page_pil_imgs) / COLS)
    
    # 创建大画布 (宽度 = 256*2 * 列数, 高度 = 256 * 行数)
    # 256*2 是因为每组包含原图和重建图
    page_canvas = Image.new('RGB', (imsize * 2 * COLS, imsize * rows))
    
    for i, pair in enumerate(page_pil_imgs):
        r = i // COLS
        c = i % COLS
        x = c * (imsize * 2)
        y = r * imsize
        page_canvas.paste(pair, (x, y))
        
    # 保存这一页
    save_name = f"{OUTPUT_DIR}/page_{page+1:02d}_ids_{start_idx}-{end_idx-1}.jpg"
    # 保存为 JPG 稍微压缩一下体积，方便下载，quality=90 保证质量
    page_canvas.save(save_name, quality=90)

print(f"[INFO] Done! Please download the folder '{OUTPUT_DIR}' to select your best images.")
