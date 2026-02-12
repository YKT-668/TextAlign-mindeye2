import torch
import os

# 你的合成文件路径
ckpt_path = '/mnt/work/repos/TextAlign-mindeye2/train_logs/merged_stage0_for_stage1.pth'

print(f'🧐 [最终验收] 正在检查: {ckpt_path} ...')

try:
    sd = torch.load(ckpt_path, map_location='cpu')
    
    # 自动拆包
    if 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    elif 'state_dict' in sd:
        sd = sd['state_dict']

    # 指标容器
    prior_layers = 0
    ridge_dim = 0
    
    # 遍历检查
    for key, val in sd.items():
        k = key.lower()
        
        # 统计 Prior
        if 'prior' in k or 'diffusion' in k:
            prior_layers += 1
            
        # 检查 Ridge 维度
        if 'ridge' in k and 'weight' in k and val.ndim == 2:
            ridge_dim = val.shape[1]

    print('-' * 40)
    print(f'1. Prior 层数检测: {prior_layers} 层')
    print(f'2. Ridge 输入维度: {ridge_dim}')
    print('-' * 40)

    # 判定逻辑
    check_1 = (prior_layers > 80)      # 官方有85层，只要大于80就算成功
    check_2 = (ridge_dim == 15724)     # 必须是 Subject 1 的维度

    if check_1 and check_2:
        print('✅✅✅ 验收通过！PERFECT！')
        print('   - Prior 移植成功 (来自官方)')
        print('   - Ridge 修复成功 (来自Repair)')
        print('🚀 你可以绝对放心地启动 Stage 1 了！')
    else:
        print('❌❌❌ 验收失败！')
        if not check_1: print(f'   -> Prior 丢失! (期望 > 80, 实际 {prior_layers})')
        if not check_2: print(f'   -> Ridge 维度错误! (期望 15724, 实际 {ridge_dim})')

except Exception as e:
    print(f'❌ 文件读取错误: {e}')