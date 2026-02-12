import os

file_path = 'src/train_textalign_bplan_fixed.py'

print(f"🔧 正在修复 BF16 类型不匹配问题: {file_path} ...")

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
fixed = False
i = 0
while i < len(lines):
    line = lines[i]
    # 定位出错的那行 Mixup 代码
    if "image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) +" in line:
        print(f"   ✅ 找到目标代码 (Line {i+1})，正在应用补丁...")
        
        # 获取缩进
        indent = line[:line.find("image_enc")]
        
        # 构建修复后的代码：先计算出 mixed_val，然后强制转为 image_enc.dtype
        # 注意：这里我们把原来的两行逻辑重写为安全的逻辑
        
        # 1. 这一行是前半部分
        new_lines.append(f"{indent}# [BF16 Fix] 强制转换数据类型以匹配 destination\n")
        new_lines.append(f"{indent}mixed_val = image_enc[select] * betas[select].reshape(*betas_shape) + \\\n")
        
        # 2. 检查下一行是否是后半部分 (通常以 image_enc_shuf 开头)
        if i + 1 < len(lines):
            next_line = lines[i+1]
            # 提取下一行的核心计算逻辑
            val_part = next_line.strip().replace('\\', '') # 去掉换行符
            new_lines.append(f"{indent}            {val_part}\n")
            
            # 3. 添加赋值行，关键在于 .to(image_enc.dtype)
            new_lines.append(f"{indent}image_enc[select] = mixed_val.to(image_enc.dtype)\n")
            
            i += 2 # 跳过原来的两行
            fixed = True
            continue
    
    new_lines.append(line)
    i += 1

if fixed:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("🚀 修复成功！代码已更新。")
else:
    print("⚠️ 未找到目标代码，可能已经修复过或代码版本不匹配。请检查文件。")