import torch
ckpt_path = '/mnt/work/repos/mindeyev2_ckpts/train_logs/final_multisubject_subj01/last.pth'
print(f'正在扫描: {ckpt_path} ...')
try:
    sd = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in sd: sd = sd['state_dict']
    elif 'model' in sd: sd = sd['model']
    
    found = False
    print('\n👇 找到以下 Ridge 相关参数:')
    for key, val in sd.items():
        # 只要键名里包含 'ridge' 且是权重(weight)，就打印出来
        if 'ridge' in key and 'weight' in key and val.ndim > 1:
            print(f'🔑 Key: {key}')
            print(f'   Shape: {val.shape}')
            found = True
            
            # 判定逻辑
            if val.shape[1] == 14278:
                print('   🚨 【铁证】: 形状对应 Subject 02 (14278)')
            elif val.shape[1] == 15724:
                print('   ✅ 【匹配】: 形状对应 Subject 01 (15724)')
            else:
                print(f'   ❓ 【未知】: 形状是 {val.shape[1]}')
    
    if not found:
        print('❌ 居然真的没找到 Ridge 相关参数？那说明这可能是纯 Backbone 权重。')

except Exception as e:
    print(e)