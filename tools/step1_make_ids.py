# 这个脚本用于二次实验时从布尔掩码文件中提取测试集图片的真实 ID 并保存为新的 NumPy 文件
import numpy as np
import os

shared_file = "src/shared1000.npy"
output_file = "/mnt/work/data_cache/test1000_ids.npy"

# 读取布尔掩码
mask = np.load(shared_file)
print(f"原始掩码形状: {mask.shape} (Type: {mask.dtype})")

# 关键修正：将 Boolean Mask 转换为 Integer Indices
# np.where(mask)[0] 会返回所有为 True 的位置的索引
ids = np.where(mask)[0]

print(f"✅ 提取到 {len(ids)} 个测试集图片的 ID")
print(f"   前 5 个真实 ID: {ids[:5]}")

# 保存
np.save(output_file, ids.astype(np.int64))
print(f"💾 已覆盖保存到: {output_file}")