TextAlign-mindeye2/src/re_train_v1_1.py 第三次实验，把textalign head挂到原模型的loss里面，然后在他们已经训练好的ckpt上训练（01）。这次是全模型训练。
'我具体改了哪里（确保符合你们当前想法）

TextAlign head 是否创建：改成“只有 teacher 存在才挂 head”

原来你是无论 teacher 有没有都 model.text_head = TextAlignHead(...)。

现在：USE_TEXT_ALIGN==True 才创建，否则 model.text_head=None。

这样不会“假训练 head”，也避免 teacher 缺失时误触发。

optimizer 里加入了 model.text_head 的参数组（很关键）

否则 head 不会更新，TextAlign loss 等于白加。

我按你 backbone 的 no_decay 规则也给 head 做了同样的 weight_decay 分组。

load_ckpt() 修复

你原版有 bug：不管传什么 tag 都会读 outdir+'/last.pth'。

我改成 os.path.join(outdir, f'{tag}.pth')，tag 生效。

同时 multisubj_loading=True 时：删除所有 ridge.linears.* keys（不是只删 .0.weight），避免多被试 ckpt 与单被试 ridge 不匹配。

TextAlign 的超参可以被 args/env 覆盖（不破坏你现有 env 用法）

alpha_text/tau/margin/hard_scale：优先 args，否则走 env，否则 default。

你继续用 MINDEYE_TEXTALIGN_SCALE / MINDEYE_TEXTALIGN_TAU / MINDEYE_TEXTALIGN_MARGIN / MINDEYE_TEXTALIGN_HARD_SCALE 都能生效。

TextAlign loss 用 fp32 计算更稳

在 autocast 下 t_pred 可能是 fp16/bf16，我改为 t_pred.float()、t_pos.float() 做 InfoNCE/hinge，防止数值不稳。

Wandb 配置修复

你原来 wandb_config 里用了 num_test/train_url/test_url 这些未定义变量，一旦开 wandb 就直接崩。

我删掉这些，保留必要信息（并加了 textalign/hardneg 开关和路径）。

Eval 阶段小修：im tensor 转 int

lazy_coco.get([im]) 里 im 是 tensor，某些环境下会报索引类型问题；我加了 im_int = int(im.item())。'

gpt第一次的修改



下面是我把你这份训练脚本完整改好的一版（按你同意的 B 方案：Stage0 只训 text_head；Stage1 解冻 ridge+backbone(+prior) 并用 OneCycle + per-group max_lr）。我把你脚本里所有我能预见会出错/会跑偏/会浪费大量算力的点都一起修了，核心包括：

✅ 彻底修掉 text_head 被加入 optimizer 两次（这是硬 bug）

✅ Stage0：只算 TextAlign loss，ridge/backbone/prior/blurry/clip 全都不算（省显存/省时/逻辑正确）

✅ Stage0：固定 lr，scheduler=OFF

✅ Stage1：OneCycle per-group max_lr（text_head/backbone/ridge/prior 各自不同峰值 lr）

✅ accelerator.prepare 兼容 lr_scheduler=None

✅ ckpt 存取兼容 lr_scheduler=None，并且 stage 不一致时自动跳过 optimizer/scheduler 恢复（避免 param_group 不匹配炸掉）

✅ 增加启动时的 param_group 摘要打印（一眼看你 Stage0 是否真的只有 head 在训练）

你直接把下面整段保存成一个新脚本（比如 train_textalign_bplan_fixed.py），不要覆盖旧的，避免回滚困难。


# 得到WebDataset (WDS) 索引、NSD 映射表以及 COCO Caption 库三者实现了完美的闭环匹配。
(base) root@ubuntu22:/mnt/work/repos/TextAlign-mindeye2# python tools/prepare_train_coco_captions_from_stiminfo.py \
  --subj 1 \
  --data_path /mnt/work/repos/TextAlign-mindeye2 \
  --out data/nsd_text/train_coco_captions.json
[STIMINFO] cocoId array: (73000,) min/max: 9 581929
[COCO] id2cap size: 123287
/home/vipuser/miniconda3/lib/python3.12/site-packages/webdataset/compat.py:379: UserWarning: WebDataset(shardshuffle=...) is None; set explicitly to False or a number
  warnings.warn("WebDataset(shardshuffle=...) is None; set explicitly to False or a number")
[WDS] behav.shape= (512, 1, 17) dtype= float64
[WDS] unique train local_ids: 9000 (min=13 max=72999)
[SAVE] data/nsd_text/train_coco_captions.json
[MAP] matched 9000/9000; missed=0


/mnt/work/repos/TextAlign-mindeye2/tools/gen_hard_neg_captions_from_json_v2.py 
从deepseek拉取hard negative（更新了拉取规则和提示词）
# 1. 确保用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 2. 创建一个下载脚本文件 (比直接 python - 更稳)
cat <<EOF > download_script.py
from huggingface_hub import snapshot_download
print("开始后台下载...")
snapshot_download(
    repo_id="pscotti/mindeyev2",
    repo_type="dataset",
    allow_patterns=["train_logs/final_multisubject_subj01/*"],
    local_dir="mindeyev2_ckpts",
    resume_download=True,
    max_workers=8
)
print("下载完成！")
EOF

# 3. 后台运行，并将日志输出到 download.log
nohup python download_script.py > download.log 2>&1 &


**stage0训练命令，只训练textalign head/ridge，冻结其他所有模块（backbone/prior）**
# 1. 先设置环境变量
unset http_proxy https_proxy all_proxy
export MINDEYE_TEXTALIGN_STAGE=1
export MINDEYE_TEXTALIGN=1
export MINDEYE_TEXTALIGN_HARDNEG=0

# 2. 学习率设置 (Repair 模式：冻结 Backbone，重训 Ridge)
export MINDEYE_LR_BACKBONE=0
export MINDEYE_LR_PRIOR=0
export MINDEYE_LR_RIDGE=3e-4
export MINDEYE_LR_TEXT=3e-4

# 3. 后台启动命令 (nohup ... &)
# 日志会输出到当前目录的 train_stage0_repair.log
nohup python src/train_textalign_bplan_fixed.py \
  --model_name s1_textalign_stage0_repair_80G \
  --subj 1 \
  --num_sessions 40 \
  --batch_size 32 \
  --num_epochs 30 \
  --no-use_prior \
  --lr_scheduler_type linear \
  --max_lr 3e-4 \
  --ckpt_interval 5 \
  --no-wandb_log \
  --hidden_dim 4096 \
  --multisubject_ckpt "/mnt/work/repos/mindeyev2_ckpts/train_logs/final_multisubject_subj01" \
  --drop_ridge_on_load \
  --textalign_hardneg_path "data/nsd_text/train_coco_captions_hard_negs_clip.pt" > train_stage0_repair.log 2>&1 &


# 训练命令(stage1)
# 1. 再次清理战场 (确保没有僵尸进程)
pkill -f train_textalign

# 2. [关键] 设置续训路径！
# 指向日志里显示的那个保存路径 (不要带 /last.pth，只写到文件夹)
export MINDEYE_RESUME="/mnt/work/repos/TextAlign-mindeye2/train_logs/s1_textalign_stage1_FINAL_BEST_32"

# 3. 环境变量 (重启后必输)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MINDEYE_DTYPE=bf16
unset http_proxy https_proxy all_proxy
export MINDEYE_TEXTALIGN_STAGE=1
export MINDEYE_TEXTALIGN=1
export MINDEYE_TEXTALIGN_HARDNEG=1
export MINDEYE_TEXTALIGN_HARD_SCALE=0.3
export MINDEYE_TEXTALIGN_MARGIN=0.1
export MINDEYE_LR_BACKBONE=1e-5
export MINDEYE_LR_PRIOR=1e-5
export MINDEYE_LR_RIDGE=1e-4
export MINDEYE_LR_TEXT=1e-4

# 4. 续训启动命令
# model_name 必须保持和之前一样，这样日志才会接在后面写，而不是新开一个文件
nohup accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    --main_process_port 29500 \
    src/train_textalign_bplan_fixed.py \
    --model_name s1_textalign_stage1_FINAL_BEST_32 \
    --subj 1 \
    --num_sessions 40 \
    --batch_size 32 \
    --num_epochs 120 \
    --use_prior \
    --lr_scheduler_type linear \
    --max_lr 1e-4 \
    --ckpt_interval 1 \
    --no-wandb_log \
    --hidden_dim 4096 \
    --textalign_hardneg_path "data/nsd_text/train_coco_captions_hard_negs_clip.pt" \
    --multisubject_ckpt "/mnt/work/repos/TextAlign-mindeye2/train_logs/merged_stage0_final" \
    > train_1gpu.log 2>&1 &


  # 推理命令
  python src/recon_inference_run.py \
  --data_path=/mnt/work/repos/TextAlign-mindeye2 \
  --cache_dir=/mnt/work/repos/TextAlign-mindeye2 \
  --output_dir=/mnt/work/repos/TextAlign-mindeye2/evals/s1_textalign_stage1_FINAL_BEST_32 \
  --model_name=s1_textalign_stage1_FINAL_BEST_32 \
  --subj=1 \
  --new_test \
  --seed 42 \
  --export_official_pts \
  --max_save 1000


# 测评命令
python /mnt/work/repos/TextAlign-mindeye2/src/run_debug.py