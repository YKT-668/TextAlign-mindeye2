#!/usr/bin/env bash
set -euo pipefail

cd /mnt/work/repos/TextAlign-mindeye2

# Make sure no old stage1 process is running.
pkill -f "train_textalign_bplan_fixed.py.*s5_textalign_stage1_FINAL_BEST_32" || true
pkill -f "accelerate launch.*s5_textalign_stage1_FINAL_BEST_32" || true

nvidia-smi -i 0 --gpu-reset || true

mkdir -p /mnt/work/.cache/huggingface /mnt/work/.cache/huggingface/hub /mnt/work/.cache/huggingface/transformers /mnt/work/.cache/torch /mnt/work/.cache /mnt/work/tmp logs

nohup env -i \
	PATH=/mnt/work/conda_envs/mindeye21_fix2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
	HOME=/root \
	LANG=C.UTF-8 \
	HF_HOME=/mnt/work/.cache/huggingface \
	HUGGINGFACE_HUB_CACHE=/mnt/work/.cache/huggingface/hub \
	TRANSFORMERS_CACHE=/mnt/work/.cache/huggingface/transformers \
	TORCH_HOME=/mnt/work/.cache/torch \
	XDG_CACHE_HOME=/mnt/work/.cache \
	TMPDIR=/mnt/work/tmp \
	MINDEYE_DTYPE=bf16 \
	MINDEYE_TEXTALIGN_STAGE=1 \
	MINDEYE_TEXTALIGN=1 \
	MINDEYE_TEXTALIGN_HARDNEG=1 \
	MINDEYE_TEXTALIGN_HARD_SCALE=0.3 \
	MINDEYE_TEXTALIGN_MARGIN=0.1 \
	MINDEYE_RESUME=/mnt/work/repos/TextAlign-mindeye2/train_logs/s5_textalign_stage0_repair_80G_resume_compat_epoch0 \
	MINDEYE_LR_BACKBONE=1e-5 \
	MINDEYE_LR_PRIOR=1e-5 \
	MINDEYE_LR_RIDGE=1e-4 \
	MINDEYE_LR_TEXT=1e-4 \
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	/mnt/work/conda_envs/mindeye21_fix2/bin/python3.10 src/train_textalign_bplan_fixed.py \
		--model_name s5_textalign_stage1_FINAL_BEST_32 \
		--subj 5 \
		--num_sessions 40 \
		--batch_size 32 \
		--num_epochs 120 \
		--use_prior \
		--lr_scheduler_type linear \
		--max_lr 1e-4 \
		--ckpt_interval 1 \
		--no-wandb_log \
		--hidden_dim 4096 \
		--textalign_teacher_path data/nsd_text/s5_train_coco_text_clip.pt \
		--textalign_hardneg_path data/nsd_text/s5_train_coco_captions_hard_negs_clip.pt \
		--multisubject_ckpt /mnt/work/checkpoints/mindeyev2_official/train_logs/final_multisubject_subj05 \
	> logs/s5_stage1_envfix_retry7_epoch0.log 2>&1 &

echo "LAUNCHED_PID=$!"
