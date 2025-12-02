#!/bin/bash

# ======================================================================================
# MindEyeV2 - 完整推理流程总控脚本 (run_inference.sh) - v2.4 (subj01+train投影+PEFT版)
#
# 设计哲学:
#   本脚本是一个真正的端到端推理流水线。它接收原始输入，通过一系列模块化
#   步骤处理，最终生成输出。同时，它也支持断点续传以方便调试。
#
# 新增/当前功能:
#   - 使用官方 subj01 40sess 预训练模型做 fMRI→CLIP-image 向量解码
#   - 使用你基于 subj01 训练集 (9000 图像) 训练好的 brain→ViT-H 投影矩阵
#   - 可选 RAG + LLM 融合 Top-K caption，回退为本地简单融合
#   - 统一用 gen_sdxl_with_peft.py 完成 SDXL+IP-Adapter+PEFT 生成
#   - 可选 PEFT 个性化适配器 (train_logs/peft_adapter_subj01)
#   - 显存优化开关 ENABLE_CPU_OFFLOAD
#   - PREP_OFFICIAL_PROJ=1：仅做“官方特征+官方版投影矩阵”准备（输出到 *_official.pt）
# ======================================================================================

# --- 脚本健壮性设置 ---
set -eo pipefail

# --- 推理必须使用 mindeye21 环境 ---
PY_INFER="/home/vipuser/miniconda3/envs/mindeye21/bin/python"
if [[ ! -x "$PY_INFER" ]]; then
    echo "❌ 找不到 mindeye 环境 Python: $PY_INFER" >&2
    exit 1
fi

# 仍然允许加载 .bashrc 以获取 API keys 等
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# ======================================================================================
# §1. 可配置参数
# ======================================================================================

# --- 核心标识符 ---
export EXP_NAME="subj01_inference_run_final"
export SUBJ_ID=1

# --- 显存优化和 PEFT 开关 ---
# 遇到 CUDA OOM 可设为 true
export ENABLE_CPU_OFFLOAD=false

# [重要] PEFT适配器目录（软prompt/LoRA）。如果不想使用，保持为空字符串 ""
# 规则：
#  1) 若外部已通过环境变量 PEFT_ADAPTER_DIR 提供，则严格使用该值；
#  2) 否则，默认尝试使用 ${PROJ_ROOT}/train_logs/peft_adapter_subj%02d（按 SUBJ_ID 补零），若不存在则置空。
export PEFT_ADAPTER_DIR="${PEFT_ADAPTER_DIR:-}"

# --- 路径配置 ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export PROJ_ROOT="$SCRIPT_DIR" # 假设脚本放在 MindEyeV2_Project 目录下

# 如果未显式指定 PEFT_ADAPTER_DIR，则根据 SUBJ_ID 设默认目录
if [ -z "${PEFT_ADAPTER_DIR}" ]; then
    subj_pad=$(printf "%02d" "${SUBJ_ID}")
    candidate="${PROJ_ROOT}/train_logs/peft_adapter_subj${subj_pad}"
    if [ -d "$candidate" ] || [ -f "${candidate}/soft_tokens.pt" ]; then
        export PEFT_ADAPTER_DIR="$candidate"
    else
        export PEFT_ADAPTER_DIR=""
    fi
fi

# NSD 数据路径（如果训练时做了 ds_view，会在后面自动覆盖）
export NSD_DATA_PATH="/home/vipuser/MindEyeV2_Project/src"

# 使用官方 subj01 40sess 完整模型作为 fMRI → CLIP 编码器
export MINDYEYE_MODEL_NAME="${MINDYEYE_MODEL_NAME:-final_subj01_pretrained_40sess_24bs}"

# 允许通过环境变量覆盖输出根目录，否则默认写到项目内 runs/
# 示例：OUTPUT_ROOT=/mnt/mindeye_data/mindeye_runs bash run_inference.sh
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJ_ROOT}/runs}"
export EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"

# 统一缓存路径（支持覆盖 CACHE_ROOT，将缓存指到大盘以避免写满）
# 示例：CACHE_ROOT=/mnt/mindeye_data/mindeye_cache bash run_inference.sh
export CACHE_ROOT="${CACHE_ROOT:-${PROJ_ROOT}/cache}"
export HF_HOME="${CACHE_ROOT}/hf_home"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hub"
export TORCH_HOME="${CACHE_ROOT}/models"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" || true

# 使用镜像加速/回退 Hugging Face 下载（可被环境变量覆盖）
export HF_ENDPOINT="https://hf-mirror.com"

# --- RAG & LLM 配置 ---
# 使用扩建后的 COCO 全量 RAG 知识库作为默认
export TEXT_INDEX_PT="${PROJ_ROOT}/data/coco_full_index.pt"
export ALL_CAPTIONS_PT="${PROJ_ROOT}/data/coco_full_captions.pt"
export TOP_K=16

# --- 图像生成配置 ---
export IP_ADAPTER_DIR="/home/vipuser/models/IP-Adapter" # IP-Adapter 模型根目录

# ★ 当前主线使用：基于 subj01 训练集训练好的 brain→ViT-H 投影矩阵
#   你刚刚跑的是:
#   python tools/train_brain2vith.py \
#       --brain_pt data/proj_subj01_train/all_subjects_brain_vectors.pt \
#       --img_vith_pt data/proj_subj01_train/all_subjects_gt_vith.pt \
#       --out checkpoints/brain2vith_subj01_train.pt
export PROJECTION_MATRIX_PT="${PROJ_ROOT}/checkpoints/brain2vith_subj01_l2_1e3.pt"


export GEN_STEPS=30
export GEN_CFG=5.0
# 可选: 限制生成的前N项。若外部未提供则默认不限制。
export GEN_LIMIT="${GEN_LIMIT:-}"
# 生成分辨率
export GEN_W=${GEN_W:-768}
export GEN_H=${GEN_H:-768}

# --- 评测配置 ---
# 官方提供的 73k 图像特征，用于评测重建效果
export GT_EMBEDS_PT="${PROJ_ROOT}/evals/all_images.pt"

# --- 官方 subj 模型的一键准备模式：提取特征 + 训练“官方版” brain→ViT-H 投影矩阵 ---
# 使用方式：
#   PREP_OFFICIAL_PROJ=1 bash run_inference.sh
# 行为：
#   - 只用官方 MindEye2 → 提取 subj 的 982 图像特征
#   - 训练一个“官方版”投影矩阵，输出到 checkpoints/brain2vith_subjXX_official.pt
#   - 不影响主线用的 checkpoints/brain2vith_subj01_train.pt
export PREP_OFFICIAL_PROJ="${PREP_OFFICIAL_PROJ:-0}"

if [[ "${PREP_OFFICIAL_PROJ}" == "1" ]]; then
    echo "============================================================"
    echo "§OFF.0 官方 MindEye2 subj${SUBJ_ID} 投影矩阵准备 (official 982 图像版)"
    echo "============================================================"

    subj_pad=$(printf "%02d" "${SUBJ_ID}")
    PROJ_DATA_DIR="${PROJ_ROOT}/data/proj_subj${subj_pad}"
    mkdir -p "${PROJ_DATA_DIR}"

    # 官方模型路径
    PROJ_BASE_MODEL_DIR="${PROJ_ROOT}/train_logs/${MINDYEYE_MODEL_NAME}"
    if [[ ! -f "${PROJ_BASE_MODEL_DIR}/last.pth" ]]; then
        echo "❌ 找不到模型权重: ${PROJ_BASE_MODEL_DIR}/last.pth" >&2
        echo "   请确认已下载 Hugging Face 上的 ${MINDYEYE_MODEL_NAME}/last.pth" >&2
        exit 1
    fi

    echo "--- OFF.1 提取 subj${SUBJ_ID} 的 brain / ViT-H 特征 (官方 preset) ---"
    "${PY_INFER}" "${PROJ_ROOT}/tools/extract_all_features.py" \
        --mindeye_model_dir "${PROJ_BASE_MODEL_DIR}" \
        --out_dir "${PROJ_DATA_DIR}" \
        --data_path "${NSD_DATA_PATH}" \
        --device "cuda" \
        --subjects "${SUBJ_ID}"

    BRAIN_PT="${PROJ_DATA_DIR}/all_subjects_brain_vectors.pt"
    IMG_PT="${PROJ_DATA_DIR}/all_subjects_gt_vith.pt"

    if [[ ! -f "${BRAIN_PT}" || ! -f "${IMG_PT}" ]]; then
        echo "❌ 提取特征失败，未找到期望的输出文件：" >&2
        echo "   - ${BRAIN_PT}" >&2
        echo "   - ${IMG_PT}" >&2
        exit 1
    fi

    # 官方版投影矩阵单独输出，避免覆盖 train 版
    OFFICIAL_PROJECTION_MATRIX_PT="${PROJ_ROOT}/checkpoints/brain2vith_subj${subj_pad}_official.pt"

    echo "--- OFF.2 训练 subj${SUBJ_ID} 的官方版 brain→ViT-H 线性投影 ---"
    "${PY_INFER}" "${PROJ_ROOT}/tools/train_brain2vith.py" \
        --brain_pt "${BRAIN_PT}" \
        --img_vith_pt "${IMG_PT}" \
        --out "${OFFICIAL_PROJECTION_MATRIX_PT}" \
        --mode "closed_form" \
        --lambda_l2 1e-3 \
        --device "cuda"

    if [[ -f "${OFFICIAL_PROJECTION_MATRIX_PT}" ]]; then
        echo "✅ subj${SUBJ_ID} 官方版投影矩阵已写入: ${OFFICIAL_PROJECTION_MATRIX_PT}"
        echo "✨ 官方准备模式完成（PREP_OFFICIAL_PROJ=1），不进入完整推理流程。"
        exit 0
    else
        echo "❌ 投影矩阵训练后未找到输出文件: ${OFFICIAL_PROJECTION_MATRIX_PT}" >&2
        exit 1
    fi
fi

# ======================================================================================
# §2. 脚本主逻辑
# ======================================================================================

echo "============================================================"
echo "§0. 初始化实验: ${EXP_NAME}"
echo "============================================================"

TOTAL_START_TIME=$(date +%s)
declare -A STEP_TIMES
declare -a STEP_NAMES

# 解析 --force 参数
FORCE_RUN=false
if [[ "$1" == "--force" ]]; then
    FORCE_RUN=true
    echo "🟡 检测到 --force 标志，将强制重新运行所有步骤。"
    if [ -d "$EXP_DIR" ]; then
        echo "   正在清空旧的实验目录: ${EXP_DIR}"
        rm -rf "${EXP_DIR}"
    fi
fi

# 检查关键环境变量/文件
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "🟡 警告: DEEPSEEK_API_KEY 未设置，将使用本地回退策略生成提示。"
else
    echo "✅ DEEPSEEK_API_KEY 已检测到，将尝试使用 DeepSeek 生成结构化提示。"
fi

if [ ! -f "$TEXT_INDEX_PT" ]; then
    echo "❌ 错误: RAG 文本特征索引未找到: ${TEXT_INDEX_PT}" >&2
    exit 1
fi
if [ ! -f "$PROJECTION_MATRIX_PT" ]; then
    echo "❌ 错误: 投影矩阵未找到: ${PROJECTION_MATRIX_PT}" >&2
    echo "   请确认已训练: checkpoints/brain2vith_subj01_train.pt" >&2
    exit 1
fi

mkdir -p "${EXP_DIR}/decoded_features" "${EXP_DIR}/retrieved_texts" "${EXP_DIR}/llm_prompts" "${EXP_DIR}/generated_images" "${EXP_DIR}/eval_results"
echo "✅ 实验目录已就绪: ${EXP_DIR}"
echo ""

echo "[cfg] ENABLE_CPU_OFFLOAD=${ENABLE_CPU_OFFLOAD}"
echo "[cfg] PEFT_ADAPTER_DIR=${PEFT_ADAPTER_DIR:-<empty>}"

# 辅助函数
run_step() {
    local step_name="$1"
    local output_file="$2"
    shift 2
    local command=("$@")
    
    local step_start_time=$(date +%s)
    local step_key=$(echo "$step_name" | sed 's/.*§\([0-9]\+\.[0-9]\+\|[0-9]\+\)\./\1/')

    echo "============================================================"
    echo "$step_name"
    echo "============================================================"

    local output_exists=false
    if [[ -f "$output_file" ]] || [[ -d "$output_file" && -n "$(ls -A "$output_file" 2>/dev/null)" ]]; then
        output_exists=true
    fi

    if [[ "$FORCE_RUN" == "false" && "$output_exists" == "true" ]]; then
        echo "🟡 输出已存在，跳过此步骤。使用 --force 强制重新运行。"
        echo "   - 路径: ${output_file}"
        STEP_TIMES["$step_key"]=0
        STEP_NAMES+=("$step_name")
    else
        echo "🚀 正在执行命令..."
        eval "${command[@]}"
        
        local output_exists_after=false
        if [[ -f "$output_file" || -d "$output_file" ]]; then
            output_exists_after=true
        fi

        if [ $? -eq 0 ] && [ "$output_exists_after" == "true" ]; then
            echo "✅ 步骤成功完成。"
        else
            echo "❌ 错误: 步骤执行失败或未生成预期的输出。" >&2
            echo "   - 失败的命令: ${command[@]}" >&2
            echo "   - 预期的输出: ${output_file}" >&2
            exit 1
        fi
        
        local step_end_time=$(date +%s)
        local step_duration=$((step_end_time - step_start_time))
        STEP_TIMES["$step_key"]=$step_duration
        STEP_NAMES+=("$step_name")
        
        echo "⏱️  步骤用时: ${step_duration}秒"
    fi
    echo ""
}

# 若训练时做了 subj01 视图划分，则沿用该视图的数据路径
MODEL_DIR="${PROJ_ROOT}/train_logs/${MINDYEYE_MODEL_NAME}"
if [[ -d "${MODEL_DIR}/ds_view/wds/subj01/new_test" ]] && [[ -f "${MODEL_DIR}/ds_view/wds/subj01/new_test/0.tar" ]]; then
    export NSD_DATA_PATH="${MODEL_DIR}/ds_view"
    echo "🟡 使用训练视图数据路径: ${NSD_DATA_PATH}"
fi

# 若需要等待 last.pth，可设 WAIT_FOR_LAST=1
if [[ "${WAIT_FOR_LAST:-0}" == "1" ]]; then
    CKPT_PATH="${MODEL_DIR}/last.pth"
    echo "🕒 WAIT_FOR_LAST=1: 等待模型权重: ${CKPT_PATH}"
    while [[ ! -f "$CKPT_PATH" ]]; do sleep 30; done
    echo "✅ 检测到模型: $CKPT_PATH"
fi

# --- 定义所有中间文件路径 ---
BRAIN_VEC_PT="${EXP_DIR}/decoded_features/brain_clip_vectors.pt"
BRAIN_IDS_JSON="${EXP_DIR}/decoded_features/brain_clip_ids.json"
TOPK_JSONL="${EXP_DIR}/retrieved_texts/topk_texts.jsonl"
LLM_PROMPTS_JSON="${EXP_DIR}/llm_prompts/structured_prompts.json"
GEN_IMAGES_DIR="${EXP_DIR}/generated_images"
RECONS_PT="${EXP_DIR}/eval_results/recons_features.pt"
METRICS_JSON="${EXP_DIR}/eval_results/metrics.json"

# --- §1 fMRI → CLIP-image 向量 ---
CMD_STEP1="${PY_INFER} \"${PROJ_ROOT}/src/extract_clip_vectors.py\" \
    --model_name \"$MINDYEYE_MODEL_NAME\" \
    --data_path \"$NSD_DATA_PATH\" \
    --subj \"$SUBJ_ID\" \
    --clip_out \"$BRAIN_VEC_PT\" \
    --ids_out \"$BRAIN_IDS_JSON\""
run_step "§1. fMRI解码" "$BRAIN_VEC_PT" "$CMD_STEP1"

# --- §2 RAG Top-K 检索 ---
CMD_STEP2="${PY_INFER} \"${PROJ_ROOT}/tools/retrieve_topk.py\" \
    --brain_vec_pt \"$BRAIN_VEC_PT\" \
    --text_index_pt \"$TEXT_INDEX_PT\" \
    --ids_json \"$BRAIN_IDS_JSON\" \
    --captions_pt \"$ALL_CAPTIONS_PT\" \
    --out_jsonl \"$TOPK_JSONL\" \
    --topk \"$TOP_K\""
run_step "§2. RAG检索" "$TOPK_JSONL" "$CMD_STEP2"

# --- §3 LLM 融合成结构化提示（带回退） ---
step3_start_time=$(date +%s)

echo "============================================================"
echo "§3. LLM融合: Top-K 文本 → 结构化提示"
echo "============================================================"
echo "🚀 尝试使用 DeepSeek 生成结构化提示（如果配置了 API Key）..."

CMD_STEP3="${PY_INFER} \"${PROJ_ROOT}/tools/prompts_from_topk_llm.py\" \
    --topk_jsonl \"$TOPK_JSONL\" \
    --out_json \"$LLM_PROMPTS_JSON\" \
    --max_workers 16 \
    --batch_size 100"

set +e
eval "$CMD_STEP3"
ret=$?
set -e

need_fallback=false
if [ ! -f "$LLM_PROMPTS_JSON" ]; then
    need_fallback=true
else
    num_prompts=$("$PY_INFER" - <<PY
import json
try:
    j=json.load(open(r"$LLM_PROMPTS_JSON"))
    print(len(j) if isinstance(j,list) else 0)
except Exception:
    print(0)
PY
)
    if [ "$num_prompts" -eq 0 ]; then
        need_fallback=true
    fi
fi

if [ "$need_fallback" = true ]; then
    echo "🟡 LLM 未生成提示或失败（exit code=${ret}），使用 Top-K 文本回退生成简单提示。"
    "$PY_INFER" - <<PY
import json
topk_path=r"$TOPK_JSONL"
out_path=r"$LLM_PROMPTS_JSON"
prompts=[]
with open(topk_path,'r',encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        rec=json.loads(line)
        topk=rec.get('topk',[])
        if not topk:
            continue
        positive='; '.join(topk[:5])
        prompts.append({
            'id': rec.get('id'),
            'positive': positive,
            'negative': 'blurry, low quality, artifacts, extra limbs, text, watermark'
        })
json.dump(prompts, open(out_path,'w',encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('wrote', out_path, len(prompts))
PY
else
    echo "✅ LLM 已生成结构化提示: $LLM_PROMPTS_JSON"
fi

step3_end_time=$(date +%s)
step3_duration=$((step3_end_time - step3_start_time))
STEP_TIMES["3"]=$step3_duration
STEP_NAMES+=("§3. LLM融合: Top-K 文本 → 结构化提示")
echo "⏱️  步骤用时: ${step3_duration}秒"
echo ""

# --- §4 SDXL + IP-Adapter(+PEFT) 生成图像 ---
CMD_STEP4="${PY_INFER} \"${PROJ_ROOT}/tools/gen_sdxl_with_peft.py\" \
    --adapter_dir \"$IP_ADAPTER_DIR\" \
    --prompts \"$LLM_PROMPTS_JSON\" \
    --brain_vec_pt \"$BRAIN_VEC_PT\" \
    --proj_pt \"$PROJECTION_MATRIX_PT\" \
    --peft_adapter_dir \"$PEFT_ADAPTER_DIR\" \
    --out_dir \"$GEN_IMAGES_DIR\" \
    --steps \"$GEN_STEPS\" \
    --cfg \"$GEN_CFG\" \
    --w \"$GEN_W\" \
    --h \"$GEN_H\" \
    --dtype fp16 \
    --ip_scale 0.8"

if [ "$ENABLE_CPU_OFFLOAD" = true ]; then
    echo "🟡 显存优化已启用，生成速度会变慢。"
    CMD_STEP4+=" --enable_cpu_offload"
fi

if [ -n "$GEN_LIMIT" ]; then
    echo "🟡 使用子集生成模式: 只生成前 $GEN_LIMIT 个样本"
    CMD_STEP4+=" --limit \"$GEN_LIMIT\""
fi

run_step "§4. 图像生成 (统一引擎)" "$GEN_IMAGES_DIR" "$CMD_STEP4"

# --- §5 评测 ---
# 5.1 打包生成结果的特征
if [ ! -d "${EXP_DIR}/images" ]; then
    echo "🟡 创建指向生成图像目录的符号链接: ${EXP_DIR}/images -> ${GEN_IMAGES_DIR}"
    ln -s "${GEN_IMAGES_DIR}" "${EXP_DIR}/images" || true
fi
mkdir -p "${EXP_DIR}/eval_results"

CMD_STEP5_1="bash -c '${PY_INFER} \"${PROJ_ROOT}/tools/pack_recons.py\" --infer_dir \"${EXP_DIR}\" && mv -f \"${EXP_DIR}/recons.pt\" \"${RECONS_PT}\" || true; mkdir -p \"${EXP_DIR}/eval_results\"; if [ -f \"${EXP_DIR}/ids.json\" ]; then mv -f \"${EXP_DIR}/ids.json\" \"${EXP_DIR}/eval_results/recons_ids.json\"; fi'"
run_step "§5.1 评测: 打包生成图像的特征" "$RECONS_PT" "$CMD_STEP5_1"

# 5.2 计算核心指标
CMD_STEP5_2="${PY_INFER} \"${PROJ_ROOT}/tools/eval_recons.py\" \
    --model_dir \"${EXP_DIR}\" \
    --gt_images \"${GT_EMBEDS_PT}\""
run_step "§5.2 评测: 计算核心评测指标" "$METRICS_JSON" "$CMD_STEP5_2"

# --- 最终总结 ---
echo "============================================================"
echo "🎉 推理流程全部完成！"
echo "============================================================"

TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

echo "最终生成的图像位于: ${GEN_IMAGES_DIR}"
echo "最终的评测指标位于: ${METRICS_JSON}"

echo ""
echo "============================================================"
echo "⏱️  时间统计报告"
echo "============================================================"

format_time() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    
    if [ $hours -gt 0 ]; then
        printf "%d小时%d分%d秒" $hours $minutes $seconds
    elif [ $minutes -gt 0 ]; then
        printf "%d分%d秒" $minutes $seconds
    else
        printf "%d秒" $seconds
    fi
}

step_num=0
for step_name in "${STEP_NAMES[@]}"; do
    step_num=$((step_num + 1))
    case $step_num in
        1) step_key="1" ;;
        2) step_key="2" ;;
        3) step_key="3" ;;
        4) step_key="4" ;;
        5) step_key="5.1" ;;
        6) step_key="5.2" ;;
        *) step_key="$step_num" ;;
    esac
    
    duration=${STEP_TIMES[$step_key]:-0}
    formatted_time=$(format_time $duration)
    
    if [ $duration -eq 0 ]; then
        echo "📊 ${step_name}: ${formatted_time} (跳过)"
    else
        echo "📊 ${step_name}: ${formatted_time}"
    fi
done

echo ""
echo "🕐 总执行时间: $(format_time $TOTAL_DURATION)"

if [ $TOTAL_DURATION -gt 0 ]; then
    echo ""
    echo "📈 时间分布:"
    step_idx=0
    for step_name in "${STEP_NAMES[@]}"; do
        step_idx=$((step_idx + 1))
        case $step_idx in
            1) step_key="1" ;;
            2) step_key="2" ;;
            3) step_key="3" ;;
            4) step_key="4" ;;
            5) step_key="5.1" ;;
            6) step_key="5.2" ;;
            *) step_key="$step_idx" ;;
        esac
        
        duration=${STEP_TIMES[$step_key]:-0}
        if [ $duration -gt 0 ]; then
            percentage=$((duration * 100 / TOTAL_DURATION))
            echo "   ${step_name}: ${percentage}%"
        fi
    done
fi

echo ""
echo "您可以查看评测文件以获取量化结果:"
echo "cat ${METRICS_JSON}"
echo "============================================================"
