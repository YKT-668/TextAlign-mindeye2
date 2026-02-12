'''把 “官方的 Prior” 移植到你 “修好的模型” 上，合成一个完美的 Frankenstein (缝合怪)。

原料 1：官方 Checkpoint (有好的 Prior，但 Ridge 是错的)。

原料 2：你的 Repair Checkpoint (有修好的 Ridge/Backbone/Head，但没有 Prior)。

手术：把 1 的 Prior 挖出来，塞进 2 里。'''

import torch
import os

# 1. 定义路径
path_official = '/mnt/work/repos/mindeyev2_ckpts/train_logs/final_multisubject_subj01/last.pth'
path_repair   = '/mnt/work/repos/TextAlign-mindeye2/train_logs/s1_textalign_stage0_repair_80G/last.pth'
path_out      = '/mnt/work/repos/TextAlign-mindeye2/train_logs/merged_stage0_for_stage1.pth'

print('💉 开始进行权重手术...')

# 2. 加载两个模型
print(f'Loading Official: {path_official}')
sd_off = torch.load(path_official, map_location='cpu')
# 处理嵌套
if 'state_dict' in sd_off: sd_off = sd_off['state_dict']
elif 'model' in sd_off: sd_off = sd_off['model']

print(f'Loading Repair:   {path_repair}')
sd_rep = torch.load(path_repair, map_location='cpu')
if 'state_dict' in sd_rep: sd_rep = sd_rep['state_dict']
elif 'model' in sd_rep: sd_rep = sd_rep['model']

# 3. 移植手术
# 以 Repair 为底座（因为它有正确的 Ridge 和 Head）
sd_final = sd_rep.copy()
count = 0

print('🔍 正在寻找并移植 Prior 权重...')
for key, val in sd_off.items():
    # 只要是 diffusion_prior 相关的权重，全部从官方覆盖过来
    if 'prior' in key or 'diffusion' in key:
        sd_final[key] = val
        count += 1

print(f'✅ 成功移植了 {count} 个 Prior 层权重！')

# 4. 保存
torch.save(sd_final, path_out)
print(f'💾 合成模型已保存至: {path_out}')
print('🚀 现在你可以用这个文件跑 Stage 1 了！')
