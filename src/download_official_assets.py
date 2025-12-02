import os
from huggingface_hub import hf_hub_download

repo_id = "pscotti/mindeyev2"
revision = "main"
repo_type = "dataset"

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1) subj01 官方 40sess 完整模型 ckpt
paths = [
    "train_logs/final_subj01_pretrained_40sess_24bs/last.pth",

    # 2) 官方 eval 资产（以后评测会用到）
    "evals/all_images.pt",
    "evals/all_captions.pt",
    "evals/all_git_generated_captions.pt",

    # 3) 官方 subj01 baseline 重建结果（以后写论文对比）
    "evals/final_subj01_pretrained_40sess_24bs/final_subj01_pretrained_40sess_24bs_all_recons.pt",
    "evals/final_subj01_pretrained_40sess_24bs/final_subj01_pretrained_40sess_24bs_all_enhancedrecons.pt",
]

for rel in paths:
    local_dir = root  # 直接下到项目根目录
    print(f"Downloading {rel} ...")
    hf_hub_download(
        repo_id=repo_id,
        filename=rel,
        repo_type=repo_type,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f" -> saved under {os.path.join(local_dir, rel)}")

print("Done.")