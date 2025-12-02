import h5py
import numpy as np
import os
import csv
from tqdm import tqdm

# ==================================
# 1. 配置参数
# ==================================
# 输入文件路径（请确保这些文件与脚本在同一目录，或使用绝对路径）
IMAGES_HDF5_PATH = 'coco_images_224_float16.hdf5'
ANNOTS_NPY_PATH = 'subj01_annots.npy'

# 输出文件名
OUTPUT_CSV_PATH = 'train_pairs_subj01.csv'

# 被试ID
SUBJECT_ID = 'subj01'

# ==================================
# 2. 主逻辑：生成CSV文件
# ==================================
def create_training_csv():
    """
    加载数据，并生成一个用于训练文本适配器的CSV文件。
    """
    print(f"\n{'='*60}")
    print(f"🚀 开始生成训练数据CSV文件...")
    print(f"{'='*60}")

    # --- 检查输入文件是否存在 ---
    if not os.path.exists(IMAGES_HDF5_PATH):
        print(f"❌ 错误: 图像文件未找到 -> {IMAGES_HDF5_PATH}")
        return
    if not os.path.exists(ANNOTS_NPY_PATH):
        print(f"❌ 错误: 标注文件未找到 -> {ANNOTS_NPY_PATH}")
        return
    
    print("✅ 输入文件检查通过。")

    # --- 加载数据 ---
    print("📦 正在加载标注文件...")
    captions = np.load(ANNOTS_NPY_PATH, allow_pickle=True)
    num_captions = len(captions)
    print(f"   - 成功加载 {num_captions} 条文本描述。")

    print("📦 正在打开图像HDF5文件...")
    with h5py.File(IMAGES_HDF5_PATH, 'r') as hf:
        num_images = len(hf['images'])
        print(f"   - HDF5文件中包含 {num_images} 张图像。")

    # --- 决定数据集大小 ---
    dataset_size = min(num_captions, num_images)
    if num_captions != num_images:
        print(f"⚠️ 警告: 文本和图像数量不匹配。将使用较小的值: {dataset_size}")
    else:
        print(f"✅ 文本和图像数量匹配: {dataset_size}")

    # --- 写入CSV文件 ---
    print(f"\n✍️ 正在将 {dataset_size} 条记录写入到 {OUTPUT_CSV_PATH}...")
    
    # 定义CSV文件的表头
    fieldnames = ['subject_id', 'prompt', 'neg_prompt', 'gt_image_path', 'ip_embed_path']
    
    try:
        with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 使用tqdm创建进度条，逐行写入数据
            for i in tqdm(range(dataset_size), desc="生成CSV中"):
                # 获取当前行的文本描述
                prompt_text = captions[i]
                
                # 构造一行数据
                # 注意：gt_image_path 和 ip_embed_path 我们暂时留空或使用占位符
                # 因为我们的训练脚本可以直接从HDF5中按索引读取图像，
                # 但为了与你的骨架代码兼容，我们先创建这个列。
                row = {
                    'subject_id': SUBJECT_ID,
                    'prompt': prompt_text,
                    'neg_prompt': '',  # 负面提示暂时留空
                    'gt_image_path': f'hdf5_index_{i}', # 使用索引作为占位符
                    'ip_embed_path': '' # IP-Adapter嵌入路径暂时留空
                }
                
                # 写入这一行
                writer.writerow(row)
                
        print(f"\n🎉 成功！CSV文件已生成: {OUTPUT_CSV_PATH}")
        print(f"   - 总计写入 {dataset_size} 行数据。")
        print(f"   - 文件格式: {', '.join(fieldnames)}")

    except Exception as e:
        print(f"\n❌ 在写入CSV文件时发生错误: {e}")

    print(f"\n{'='*60}")
    print(f"✅ 数据准备工作完成！")
    print(f"{'='*60}\n")


# ==================================
# 3. 程序入口
# ==================================
if __name__ == '__main__':
    create_training_csv()

