import torch
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from math import ceil

# --- 用户精选的 ID ---
SELECTED_IDS = [110, 115, 149, 153, 210, 230, 337, 483, 701, 720, 832, 858]

# --- 配置 ---
MODEL_NAME = "s1_textalign_stage1_FINAL_BEST_32"
# 路径配置 (确保与之前一致)
RECONS_PATH = f"evals/{MODEL_NAME}/{MODEL_NAME}_all_enhancedrecons.pt"
IMAGES_PATH = "evals/all_images.pt"

# 图片参数
IMSIZE = 256
GRID_COLS = 3  # 设置为 3 列 (4行x3列布局)
PADDING = 10   # 图片之间的间距 (像素)
BG_COLOR = (255, 255, 255) # 背景颜色：白色

def main():
    print(f"[INFO] Loading data for {len(SELECTED_IDS)} selected images...")
    
    # 加载数据 (CPU)
    try:
        all_images = torch.load(IMAGES_PATH, map_location="cpu")
        all_recons = torch.load(RECONS_PATH, map_location="cpu")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find data files: {e}")
        print("Please make sure you are running this script from the project root directory.")
        return

    # 准备转换工具
    resize = transforms.Resize((IMSIZE, IMSIZE))
    to_pil = transforms.ToPILImage()

    # 收集 PIL 图片对象
    pil_pairs = []
    
    print("[INFO] Processing images...")
    for idx in SELECTED_IDS:
        # 1. 提取
        raw_gt = all_images[idx]
        raw_rec = all_recons[idx]
        
        # 2. 转换
        img_gt = to_pil(resize(raw_gt).float().clamp(0, 1))
        img_rec = to_pil(resize(raw_rec).float().clamp(0, 1))
        
        # 3. 拼接单组 (左GT 右Rec)
        # 宽度 = 256*2 + 间距
        pair_width = IMSIZE * 2 + PADDING
        pair_height = IMSIZE
        
        pair_img = Image.new('RGB', (pair_width, pair_height), BG_COLOR)
        pair_img.paste(img_gt, (0, 0))
        pair_img.paste(img_rec, (IMSIZE + PADDING, 0))
        
        pil_pairs.append(pair_img)

    # --- 拼大图 ---
    num_items = len(pil_pairs)
    num_rows = ceil(num_items / GRID_COLS)
    
    # 计算大画布尺寸
    # 宽度 = 列数 * 单组宽 + (列数-1)*间距 + 2*边距
    # 高度 = 行数 * 单组高 + (行数-1)*间距 + 2*边距
    
    margin = 20 # 边缘留白
    
    canvas_w = GRID_COLS * (IMSIZE * 2 + PADDING) + (GRID_COLS - 1) * PADDING + 2 * margin
    canvas_h = num_rows * IMSIZE + (num_rows - 1) * PADDING + 2 * margin
    
    final_canvas = Image.new('RGB', (canvas_w, canvas_h), BG_COLOR)
    
    for i, pair in enumerate(pil_pairs):
        row = i // GRID_COLS
        col = i % GRID_COLS
        
        x = margin + col * (IMSIZE * 2 + PADDING * 2)
        y = margin + row * (IMSIZE + PADDING)
        
        final_canvas.paste(pair, (x, y))

    # --- 保存 ---
    save_name = f"best_{len(SELECTED_IDS)}_grid.png"
    final_canvas.save(save_name, quality=95)
    print(f"\n[SUCCESS] Final grid saved to: {os.path.abspath(save_name)}")
    print("You can now download this file!")

if __name__ == "__main__":
    main()
