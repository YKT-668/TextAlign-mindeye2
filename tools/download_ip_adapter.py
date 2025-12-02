import os, shutil
from huggingface_hub import hf_hub_download

save_dir = "/home/vipuser/models/ip-adapter"
os.makedirs(save_dir, exist_ok=True)

def safe_put(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.abspath(src) == os.path.abspath(dst):
        print("[ok] already at", dst); return dst
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        print("[copy2 fail]", e)
        shutil.copyfile(src, dst)
    print("[saved]", dst)
    return dst

downloaded = {}

# 1) 基础 SDXL 适配器
try:
    p = hf_hub_download("h94/IP-Adapter", "sdxl_models/ip-adapter_sdxl.safetensors")
    downloaded["ip-adapter_sdxl"] = safe_put(p, os.path.join(save_dir, "ip-adapter_sdxl.safetensors"))
except Exception as e:
    print("[miss] ip-adapter_sdxl.safetensors ->", e)

# 2) plus 版
plus_candidates = [
    "sdxl_models/ip-adapter-plus_sdxl.safetensors",
    "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
]
for fn in plus_candidates:
    if "ip-adapter-plus_sdxl" in downloaded: break
    try:
        p = hf_hub_download("h94/IP-Adapter", fn)
        downloaded["ip-adapter-plus_sdxl"] = safe_put(p, os.path.join(save_dir, os.path.basename(fn)))
        break
    except Exception as e:
        print("[skip plus]", fn, "->", e)

# 3) image encoder
enc_candidates = [
    "sdxl_models/image_encoder/model.safetensors",
    "image_encoder/model.safetensors",
]
for fn in enc_candidates:
    if "image_encoder" in downloaded: break
    try:
        p = hf_hub_download("h94/IP-Adapter", fn)
        downloaded["image_encoder"] = safe_put(p, os.path.join(save_dir, "image_encoder.safetensors"))
        break
    except Exception as e:
        print("[skip enc]", fn, "->", e)

print("\n== FINAL PATHS ==")
for k, v in downloaded.items():
    print(f"{k}: {v}")
if "ip-adapter_sdxl" not in downloaded:
    print("\n[WARN] 没有拿到基础 ip-adapter_sdxl（必须）。")
if "image_encoder" not in downloaded:
    print("[WARN] 没有拿到 image_encoder（建议有，但不一定阻塞文本基线）。")
