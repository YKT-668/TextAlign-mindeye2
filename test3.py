import torch
import os

# === 这里填官方权重路径 ===
ckpt_path = '/mnt/work/repos/mindeyev2_ckpts/train_logs/final_multisubject_subj01/last.pth'

print(f'🕵️‍♂️ [终极扫描] 正在加载文件: {ckpt_path} ...')

try:
    # 1. 加载文件
    raw_sd = torch.load(ckpt_path, map_location='cpu')
    sd = raw_sd
    
    # 2. 智能循环拆包 (剥洋葱)
    unpacked_layers = 0
    while True:
        if isinstance(sd, dict):
            keys = list(sd.keys())
            # 如果只包含一个 keys 且名字像包装壳，就往里拆
            if len(keys) == 1 and keys[0] in ['model_state_dict', 'state_dict', 'model']:
                print(f'   📦 拆开外包装: [{keys[0]}]')
                sd = sd[keys[0]]
                unpacked_layers += 1
                continue
            # 另外一种情况：DDP保存时可能有 'module.' 前缀，这个在遍历时处理
            break
        else:
            break

    print(f'   ✅ 拆包完成，共剥离 {unpacked_layers} 层包装。')
    print('-' * 50)

    # 3. 分类统计容器
    stats = {
        'backbone': {'count': 0, 'samples': []},
        'ridge':    {'count': 0, 'samples': [], 'input_dim': None},
        'prior':    {'count': 0, 'samples': []},
        'text':     {'count': 0, 'samples': []},
        'other':    {'count': 0, 'samples': []}
    }

    # 4. 遍历所有参数
    total_params = 0
    for key, val in sd.items():
        # 跳过非 Tensor 数据 (比如 step 计数)
        if not torch.is_tensor(val):
            continue
            
        shape_str = str(list(val.shape))
        num_params = val.numel()
        total_params += num_params
        
        k_low = key.lower()
        
        # 分类逻辑
        category = 'other'
        if 'backbone' in k_low or 'visual' in k_low:
            category = 'backbone'
        elif 'ridge' in k_low:
            category = 'ridge'
            # 抓取 Ridge 的输入维度 (关键!)
            if 'weight' in k_low and val.ndim == 2:
                stats['ridge']['input_dim'] = val.shape[1]
        elif 'prior' in k_low or 'diffusion' in k_low:
            category = 'prior'
        elif 'text' in k_low or 'head' in k_low:
            category = 'text'
            
        # 记录
        stats[category]['count'] += 1
        if len(stats[category]['samples']) < 3: # 每个类别只存前3个样本用于展示
            stats[category]['samples'].append(f"{key} \t {shape_str}")

    # 5. 输出详细报告
    print(f'\n📊 【最终体检报告】 (总参数量: {total_params / 1e6:.2f} M)')
    print('=' * 50)
    
    # --- Prior 部分 ---
    p = stats['prior']
    print(f"🎨 [Prior / 生成模型]")
    print(f"   - 包含层数: {p['count']}")
    if p['count'] > 0:
        print(f"   - ✅ 状态: 存在 (实锤！)")
        print(f"   - 抽样查看:")
        for s in p['samples']: print(f"     * {s}")
    else:
        print(f"   - ❌ 状态: 缺失")
    print('-' * 50)

    # --- Ridge 部分 ---
    r = stats['ridge']
    print(f"👀 [Ridge / 接口]")
    print(f"   - 包含层数: {r['count']}")
    if r['input_dim']:
        print(f"   - 📏 输入维度: {r['input_dim']}")
        if r['input_dim'] == 14278:
            print(f"   - 🚨 身份验证: 这是 Subject 02 的参数！(需要扔掉)")
        elif r['input_dim'] == 15724:
            print(f"   - ✅ 身份验证: 这是 Subject 01 的参数。")
        else:
            print(f"   - ❓ 身份验证: 未知被试")
    print('-' * 50)

    # --- Backbone 部分 ---
    b = stats['backbone']
    print(f"🧠 [Backbone / 主干]")
    print(f"   - 包含层数: {b['count']}")
    if b['count'] > 0:
        print(f"   - ✅ 状态: 正常")
    print('=' * 50)

    # 6. 最终建议
    if stats['prior']['count'] > 20: # 一般 Prior 至少几十层
        print("💡 决策建议: 检测到完整的 Prior 权重。")
        print("   👉 请立刻运行 [merge_final.py] 进行移植手术！")
    else:
        print("💡 决策建议: 未检测到 Prior 或层数过少。")
        print("   👉 请直接从零训练。")

except Exception as e:
    print(f'❌ 扫描出错: {e}')