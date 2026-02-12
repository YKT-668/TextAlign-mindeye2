# Reproduction Guide

## Commit Hash
*To be filled after release*

## Model Weights
Available at [HuggingFace: ykt668/textalign-mindeye2-model](https://huggingface.co/ykt668/textalign-mindeye2-model)

Path: `checkpoints/s1_textalign_stage1_FINAL_BEST_32/last.pth`

## Minimum Reproduction Steps

1.  **Environment**
    ```bash
    conda activate mindeye21
    ```

2.  **Download Weights**
    Download `last.pth` from HF to `checkpoints/s1_textalign_stage1_FINAL_BEST_32/last.pth`.
    Download `features/` from HF to `data/nsd_text/` (optional, for training).

3.  **Inference Code**
    ```bash
    python src/recon_inference_run.py \
        --subject 1 \
        --ckpt_path checkpoints/s1_textalign_stage1_FINAL_BEST_32/last.pth \
        --eval_only
    ```
