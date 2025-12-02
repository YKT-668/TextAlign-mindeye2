nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  MINDEYE_TEXTALIGN=1 \
  MINDEYE_TEXTALIGN_SCALE=0.05 \
  LOG_STEP_INTERVAL=50 \
  python src/Train_textalign_v1_backup.py \
    --model_name s1_textalign_coco_train_long_v2 \
    --data_path /home/vipuser/MindEyeV2_Project/src \
    --cache_dir "$HF_HOME" \
    --subj 1 \
    --num_sessions 40 \
    --num_epochs 10 \
    --batch_size 8 \
    --hidden_dim 1024 \
    --n_blocks 4 \
    --no-use_prior \
    --no-blurry_recon \
    --no-use_image_aug \
  > train_logs/s1_textalign_coco_train_long_v2.log 2>&1 &
目前最好的模块的命令


快速测试命令
ACCELERATE_MIXED_PRECISION=bf16 \
env \
  CUDA_VISIBLE_DEVICES=0 \
  MINDEYE_TEXTALIGN=1 \
  MINDEYE_TEXTALIGN_SCALE=0.05 \
  MINDEYE_FAST=1 \
  MINDEYE_EPOCH_FRACTION=0.2 \
  MINDEYE_MAX_STEPS_PER_EPOCH=50 \
  LOG_STEP_INTERVAL=10 \
  MINDEYE_HARDNEG_PATH=/home/vipuser/MindEyeV2_Project/data/nsd_text/train_coco_text_clip_hardneg.pt \
  python src/Train_textalign_v3.py \
    --model_name s1_textalign_coco_from_final_subj01_full_v7_fast \
    --data_path /home/vipuser/MindEyeV2_Project/src \
    --cache_dir "$HF_HOME" \
    --subj 1 \
    --num_sessions 40 \
    --num_epochs 2 \
    --batch_size 8 \
    --hidden_dim 4096 \
    --n_blocks 4 \
    --no-use_prior \
    --no-blurry_recon \
    --multisubject_ckpt /home/vipuser/train_logs/final_subj01_pretrained_40sess_24bs
v3 + FAST + 只训 ridge + text_head + 关 prior / blurry 重建

上吗那个显存不够
ACCELERATE_MIXED_PRECISION=bf16 \
env \
  CUDA_VISIBLE_DEVICES=0 \
  MINDEYE_TEXTALIGN=1 \
  MINDEYE_TEXTALIGN_SCALE=0.05 \
  MINDEYE_FAST=1 \
  MINDEYE_EPOCH_FRACTION=0.2 \
  MINDEYE_MAX_STEPS_PER_EPOCH=50 \
  LOG_STEP_INTERVAL=10 \
  MINDEYE_HARDNEG_PATH=/home/vipuser/MindEyeV2_Project/data/nsd_text/train_coco_captions_hard_negs_clip.pt \
  python src/Train_textalign_v3.py \
    --model_name s1_textalign_coco_from_final_subj01_full_v7_fast_cf \
    --data_path /home/vipuser/MindEyeV2_Project/src \
    --cache_dir "$HF_HOME" \
    --subj 1 \
    --num_sessions 40 \
    --num_epochs 2 \
    --batch_size 8 \
    --hidden_dim 4096 \
    --n_blocks 4 \
    --no-use_prior \
    --no-blurry_recon \
    --multisubject_ckpt /home/vipuser/train_logs/final_subj01_pretrained_40sess_24bs

按照ai说明，把中间共享模块冻结了，只跑其他部分的训练。
nohup env \
  ACCELERATE_MIXED_PRECISION=bf16 \
  CLIP_FP32=1 \
  CUDA_VISIBLE_DEVICES=0 \
  MINDEYE_TEXTALIGN=1 \
  MINDEYE_TEXTALIGN_SCALE=0.05 \
  LOG_STEP_INTERVAL=200 \
  MINDEYE_HARDNEG_PATH=/home/vipuser/MindEyeV2_Project/data/nsd_text/train_coco_captions_hard_negs_clip.pt \
  python src/Train_textalign_v3.py \
    --model_name s1_textalign_coco_from_final_subj01_full_cf_e50 \
    --data_path /home/vipuser/MindEyeV2_Project/src \
    --cache_dir "$HF_HOME" \
    --subj 1 \
    --num_sessions 40 \
    --num_epochs 10 \
    --batch_size 8 \
    --hidden_dim 4096 \
    --n_blocks 4 \
    --no-use_prior \
    --no-blurry_recon \
    --multisubject_ckpt /home/vipuser/train_logs/final_subj01_pretrained_40sess_24bs \
  > train_logs/s1_textalign_coco_from_final_subj01_full_cf_e50.log 2>&1 &



  sub1 s1_textalign_coco_train_long_v10  最终版


  nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  MINDEYE_TEXTALIGN=1 \
  MINDEYE_TEXTALIGN_SCALE=0.05 \
  LOG_STEP_INTERVAL=50 \
  python src/Train_textalign_v1_backup.py \
    --model_name s3_textalign_coco_train_long_v10 \
    --data_path /home/vipuser/MindEyeV2_Project/src \
    --cache_dir "$HF_HOME" \
    --subj 3 \
    --num_sessions 32 \
    --num_epochs 10 \
    --batch_size 8 \
    --hidden_dim 1024 \
    --n_blocks 4 \
    --no-use_prior \
    --no-blurry_recon \
    --no-use_image_aug \
  > train_logs/s3_textalign_coco_train_long_v2.log 2>&1 &

echo "subj03 已启动（完全公平对比版），PID: $!"

GT] CLIP 特征编码完成, shape = torch.Size([1000, 1664])
[GT] 未检测到单独的 ids 字段，假设顺序与 brain_ids 一致
[INFO] gt_sel: torch.Size([930, 1664])
====================================================
TextAlign latent eval (subj01 new_test 1000):
  FWD  Top-1: 0.11%   Top-5: 0.65%
  BWD  Top-1: 0.22%   Top-5: 1.08%
====================================================


/home/vipuser/MindEyeV2_Project/train_logs/s5_textalign_coco_train_long_v10/last ckpt!---'
[GT] 未检测到单独的 ids 字段，假设顺序与 brain_ids 一致
[INFO] gt_sel: torch.Size([1000, 1664])
====================================================
TextAlign latent eval (subj01 new_test 1000):
  FWD  Top-1: 5.20%   Top-5: 15.30%
  BWD  Top-1: 4.10%   Top-5: 15.00%
====================================================

tail -f train_logs/s7_textalign_coco_train_long_v10.log

[INFO] gt_sel: torch.Size([1000, 1664])
====================================================
TextAlign latent eval (subj01 new_test 1000):
  FWD  Top-1: 2.30%   Top-5: 10.00%
  BWD  Top-1: 1.70%   Top-5: 8.10%
====================================================


  nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  MINDEYE_TEXTALIGN=1 \
  MINDEYE_TEXTALIGN_SCALE=0.05 \
  LOG_STEP_INTERVAL=50 \
  python src/Train_textalign_v1_backup.py \
    --model_name s4_textalign_coco_train_long_v10 \
    --data_path /home/vipuser/MindEyeV2_Project/src \
    --cache_dir "$HF_HOME" \
    --subj 4 \
    --num_sessions 30 \
    --num_epochs 10 \
    --batch_size 8 \
    --hidden_dim 1024 \
    --n_blocks 4 \
    --no-use_prior \
    --no-blurry_recon \
    --no-use_image_aug \
  > train_logs/s4_textalign_coco_train_long_v10.log 2>&1 &

  ### Environment (recommended)

```bash
# Create conda env from exported YAML
conda env create -f environment_mindeye21.yml
conda activate mindeye21

# (Optional) ensure pip deps are synced
pip install -r requirements_mindeye21.txt
