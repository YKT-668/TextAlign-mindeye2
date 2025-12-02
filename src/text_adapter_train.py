import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open_clip
import h5py
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

# ==================================
# 1. 定义我们的核心组件：MLP适配器
# ==================================

class TextAdapter(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=2048, output_dim=1280):
        super(TextAdapter, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_embedding):
        return self.model(text_embedding)

# ==================================
# 2. 定义数据加载器
# ==================================

class COCOCaptionsDataset(Dataset):
    """
    一个自定义的PyTorch数据集，用于加载COCO图像及其对应的文本描述。
    """
    def __init__(self, images_path, annots_path, tokenizer):
        """
        初始化数据集。
        
        参数:
        - images_path: 指向 'coco_images_224_float16.hdf5' 文件的路径。
        - annots_path: 指向 'subj01_annots.npy' 文件的路径。
        - tokenizer: open_clip 的分词器，用于将文本转换为token。
        """
        self.images_path = images_path
        self.tokenizer = tokenizer
        
        # 加载标注文件
        self.captions = np.load(annots_path, allow_pickle=True)
        
        print("\n--- 数据加载报告 ---")
        print(f"标注数组形状: {self.captions.shape}")
        print(f"标注数组类型: {self.captions.dtype}")
        print(f"总共有 {len(self.captions)} 条文本描述")
        print(f"示例文本: '{self.captions[0]}'")
        
        # 检查HDF5文件中的图像数量
        with h5py.File(self.images_path, 'r') as hf:
            num_images = len(hf['images'])
            print(f"HDF5文件中有 {num_images} 张图像")
        
        # 确保文本数量和图像数量匹配
        if len(self.captions) != num_images:
            print(f"⚠️ 警告: 文本数量({len(self.captions)})和图像数量({num_images})不匹配！")
            self.dataset_size = min(len(self.captions), num_images)
            print(f"将使用前 {self.dataset_size} 个样本")
        else:
            self.dataset_size = len(self.captions)
            print(f"✅ 文本和图像数量匹配！")
        
        print("--- 报告结束 ---\n")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """获取单个数据点"""
        caption = self.captions[idx]
        text_tokens = self.tokenizer(caption)
        
        with h5py.File(self.images_path, 'r') as hf:
            image_data = hf['images'][idx]
            image_tensor = torch.from_numpy(image_data.astype(np.float32))

        return {
            "image": image_tensor,
            "text_tokens": text_tokens.squeeze(),
            "caption": caption
        }

# ==================================
# 3. 训练函数
# ==================================

def train_one_epoch(text_adapter, clip_model, data_loader, optimizer, device, epoch):
    """
    训练一个epoch
    
    参数:
    - text_adapter: 我们要训练的文本适配器
    - clip_model: 预训练的CLIP模型（冻结）
    - data_loader: 数据加载器
    - optimizer: 优化器
    - device: 设备（cuda或cpu）
    - epoch: 当前epoch编号
    
    返回:
    - 平均损失
    """
    text_adapter.train()  # 设置为训练模式
    clip_model.eval()     # CLIP模型保持评估模式（我们不训练它）
    
    total_loss = 0.0
    num_batches = 0
    
    # 使用tqdm创建进度条
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        # 1. 获取数据并移到设备上
        images = batch["image"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        
        # 2. 使用CLIP提取"黄金标准"的特征向量
        with torch.no_grad():  # 不计算梯度，节省内存
            # 提取图像特征（我们的目标）
            image_features = clip_model.encode_image(images)
            # 归一化（CLIP的标准做法）
            image_features = F.normalize(image_features, dim=-1)
            
            # 提取原始文本特征
            text_features = clip_model.encode_text(text_tokens)
            # 归一化
            text_features = F.normalize(text_features, dim=-1)
        
        # 3. 通过我们的适配器转换文本特征
        adapted_text_features = text_adapter(text_features)
        # 归一化适配后的特征
        adapted_text_features = F.normalize(adapted_text_features, dim=-1)
        
        # 4. 计算余弦相似度损失
        # 余弦相似度范围[-1, 1]，我们希望它接近1
        # 损失 = 1 - 余弦相似度，使得相似度越高，损失越低
        cosine_sim = F.cosine_similarity(adapted_text_features, image_features, dim=-1)
        loss = (1 - cosine_sim).mean()
        
        # 5. 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新权重
        
        # 6. 记录统计信息
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条显示
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}',
            'cos_sim': f'{cosine_sim.mean().item():.4f}'
        })
    
    avg_loss = total_loss / num_batches
    return avg_loss

# ==================================
# 4. 主训练流程
# ==================================

def main():
    # ==================== 配置参数 ====================
    # 数据路径
    IMAGES_HDF5_PATH = 'coco_images_224_float16.hdf5'
    ANNOTS_NPY_PATH = 'subj01_annots.npy'
    
    # 训练超参数
    BATCH_SIZE = 32          # 批次大小，根据GPU内存调整
    NUM_EPOCHS = 10          # 训练轮数
    LEARNING_RATE = 1e-4     # 学习率
    
    # 模型参数
    CLIP_DIM = 1280          # ViT-bigG-14的特征维度（实际是1280）
    HIDDEN_DIM = 2048        # 适配器隐藏层维度
    
    # 保存路径
    CHECKPOINT_DIR = 'checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"🚀 训练配置")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"模型维度: {CLIP_DIM} -> {HIDDEN_DIM} -> {CLIP_DIM}")
    print(f"注意: ViT-bigG-14 的实际特征维度是 1280")
    print(f"检查点保存目录: {CHECKPOINT_DIR}")
    print(f"{'='*60}\n")
    
    # ==================== 加载CLIP模型 ====================
    print("📦 正在加载预训练的CLIP模型...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-bigG-14', 
        pretrained='laion2b_s39b_b160k'
    )
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    
    # 将CLIP移到设备上并冻结参数
    clip_model = clip_model.to(device)
    for param in clip_model.parameters():
        param.requires_grad = False  # 冻结CLIP，不训练它
    
    print("✅ CLIP模型加载成功并已冻结参数\n")
    
    # ==================== 创建数据集和加载器 ====================
    print("📊 正在创建数据集...")
    dataset = COCOCaptionsDataset(
        images_path=IMAGES_HDF5_PATH,
        annots_path=ANNOTS_NPY_PATH,
        tokenizer=tokenizer
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # 多进程加载，加速训练
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✅ 数据加载器创建成功")
    print(f"   - 总样本数: {len(dataset)}")
    print(f"   - 每epoch批次数: {len(data_loader)}\n")
    
    # ==================== 创建文本适配器 ====================
    print("🧠 正在初始化文本适配器...")
    text_adapter = TextAdapter(
        input_dim=CLIP_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=CLIP_DIM
    ).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in text_adapter.parameters())
    trainable_params = sum(p.numel() for p in text_adapter.parameters() if p.requires_grad)
    print(f"✅ 文本适配器初始化成功")
    print(f"   - 总参数量: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}\n")
    
    # ==================== 创建优化器 ====================
    optimizer = torch.optim.AdamW(
        text_adapter.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01  # L2正则化，防止过拟合
    )
    
    print(f"⚙️ 优化器: AdamW (lr={LEARNING_RATE}, weight_decay=0.01)\n")
    
    # ==================== 训练循环 ====================
    print(f"{'='*60}")
    print(f"🎯 开始训练！")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    start_time = datetime.now()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n📅 Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # 训练一个epoch
        avg_loss = train_one_epoch(
            text_adapter=text_adapter,
            clip_model=clip_model,
            data_loader=data_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        print(f"\n✨ Epoch {epoch+1} 完成！")
        print(f"   - 平均损失: {avg_loss:.4f}")
        
        # 保存检查点
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, 
            f'text_adapter_epoch{epoch+1}.pth'
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': text_adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"   - 检查点已保存: {checkpoint_path}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, 'text_adapter_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': text_adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, best_model_path)
            print(f"   - 🏆 新的最佳模型已保存！(loss: {best_loss:.4f})")
    
    # ==================== 训练完成 ====================
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"🎉 训练完成！")
    print(f"{'='*60}")
    print(f"总训练时间: {training_time}")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"最佳模型保存在: {os.path.join(CHECKPOINT_DIR, 'text_adapter_best.pth')}")
    print(f"{'='*60}\n")

# ==================================
# 5. 程序入口
# ==================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()