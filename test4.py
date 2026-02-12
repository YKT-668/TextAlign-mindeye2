import torch
import os

# === 配置路径 ===
# 1. 官方权重 (有好的 Prior，坏的 Ridge)
path_official = '/mnt/work/repos/mindeyev2_ckpts/train_logs/final_multisubject_subj01/last.pth'
# 2. Repair权重 (有好的 Ridge/Head，没 Prior)
path_repair   = '/mnt/work/repos/TextAlign-mindeye2/train_logs/s1_textalign_stage0_repair_80G/last.pth'
# 3. 输出路径
path_out      = '/mnt/work/repos/TextAlign-mindeye2/train_logs/merged_stage0_for_stage1.pth'

print('🏥 [手术室] 准备进行最终权重缝合...')

def load_sd(path, name):
    print(f'   正在加载 {name}: {path} ...')
    sd = torch.load(path, map_location='cpu')
    # 自动拆包逻辑
    if 'model_state_dict' in sd:
        print(f'   📦 {name} 发现 [model_state_dict] 包装，正在拆开...')
        return sd['model_state_dict']
    elif 'state_dict' in sd:
        print(f'   📦 {name} 发现 [state_dict] 包装，正在拆开...')
        return sd['state_dict']
    elif 'model' in sd:
        print(f'   📦 {name} 发现 [model] 包装，正在拆开...')
        return sd['model']
    return sd

try:
    # 1. 加载
    sd_off = load_sd(path_official, "官方源")
    sd_rep = load_sd(path_repair, "Repair源")

    # 2. 准备底座 (以 Repair 为主，因为它有正确的 Ridge 和 Head)
    sd_final = sd_rep.copy()
    
    # 3. 移植 Prior
    count = 0
    print('\n💉 开始移植 Prior 权重 (从官方 -> 合成版)...')
    
    for key, val in sd_off.items():
        # 只要 key 里包含 'prior'，就强行覆盖
        if 'prior' in key.lower():
            sd_final[key] = val
            count += 1
            
    print(f'✅ 成功移植了 {count} 个 Prior 相关层！')
    
    # 4. 验证
    if count == 0:
        raise RuntimeError("怎么还是 0？检查逻辑！")

    # 5. 保存
    torch.save({'model_state_dict': sd_final}, path_out) # 保持和官方一样的包装习惯，稳一点
    print(f'\n💾 手术成功！合成模型已保存至: {path_out}')
    print('🚀 这是一个包含 [好Ridge + 好Backbone + 好Head + 官方Prior] 的完美模型！')

except Exception as e:
    print(f'\n❌ 手术失败: {e}')