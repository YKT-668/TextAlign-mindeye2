import os

file_path = 'src/train_textalign_bplan_fixed.py'
print(f"🔧 正在为 {file_path} 添加【断点续训】功能 ...")

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
patched = False

for line in lines:
    # 寻找最佳插入点：在加载 multisubject_ckpt 的逻辑之前
    if "if args.multisubject_ckpt is not None:" in line and not patched:
        indent = line[:line.find("if")]
        
        print("✅ 找到插入点，正在植入续训逻辑...")
        
        # 插入一段优先检查环境变量 MINDEYE_RESUME 的代码
        new_lines.append(f"{indent}# [Auto-Patch] 断点续训逻辑\n")
        new_lines.append(f"{indent}resume_path = os.environ.get('MINDEYE_RESUME', '')\n")
        new_lines.append(f"{indent}if resume_path and os.path.exists(os.path.join(resume_path, 'last.pth')):\n")
        new_lines.append(f"{indent}    acc_print(f'\\n[RESUME] ⚠️ 检测到续训信号！正在从 {{resume_path}} 恢复进度...')\n")
        new_lines.append(f"{indent}    # 强制加载 optimizer, scheduler 和 epoch，且 strict=False 兼容旧权重\n")
        new_lines.append(f"{indent}    epoch = load_ckpt('last', outdir_override=resume_path, load_lr=True, load_optimizer=True, load_epoch=True, strict=False)\n")
        new_lines.append(f"{indent}    acc_print(f'[RESUME] 成功恢复！将从 Epoch {{epoch}} 继续训练\\n')\n")
        new_lines.append(f"{indent}el") # 变成 elif
    
    new_lines.append(line)
    
    # 修正紧接着的 if 为 elif (字符串替换)
    if patched and "if args.multisubject_ckpt is not None:" in line:
        pass # 上面已经处理了连接词

    if "if args.multisubject_ckpt is not None:" in line:
        patched = True

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("🚀 续训功能已植入！现在可以通过设置 MINDEYE_RESUME 环境变量来接着跑了。")
