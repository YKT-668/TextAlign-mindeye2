import numpy as np
p="/mnt/work/repos/TextAlign-mindeye2/test_73k_images.npy"
a=np.load(p, mmap_mode="r")
print("PATH:", p)
print("shape:", a.shape, "dtype:", a.dtype)
print("min/max:", int(a.min()), int(a.max()))
print("head20:", a[:20].tolist())
print("tail20:", a[-20:].tolist())
