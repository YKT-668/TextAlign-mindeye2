#!/bin/bash

# ======================================================================================
# MindEyeV2 - 训练总控脚本 (run_training.sh) - "official-subj01" 精简版
#
# 功能:
#   提供统一入口执行当前路线需要的几个训练任务：
#     - mindeye   : 训练自有 MindEye2 主干（如需自己从头训练时用）
#     - projection: 使用已提取的 brain / ViT-H 特征训练 brain→ViT-H 线性投影矩阵
#     - peft      : 训练扩散模型侧的 PEFT 适配器（soft-prompt / LoRA）
#
#   当前主线实验：
#     官方 subj01 40sess MindEye2 模型
#     + subj01 train 9000 样本拟合 brain→ViT-H 线性映射。
#   旧的 multi-subject 预训练 / 微调流水线已删除，以免混淆。
#
# 使用方法:
#   bash run_training.sh [task] [--force] [--fast]
#
#   [task] 可选:
#     - mindeye
#     - projection
#     - peft
#
#   --force : 强制重新训练（删除旧输出）
#   --fast  : 冒烟测试，极小 epoch/steps，只验证流程
# ======================================================================================

set -eo pipefail

# ======================================================================================
# §1. 全局路径 & 环境
# ======================================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
export PROJ_ROOT="$SCRIPT_DIR"
export NSD_DATA_PATH="/home/vipuser/MindEyeV2_Project/src"
export OUTPUT_ROOT="${PROJ_ROOT}/train_logs"

# 临时目录
TMP_DIR_DEFAULT="/home/vipuser/miniconda3/tmp"
if ! mkdir -p "$TMP_DIR_DEFAULT" 2>/dev/null; then
    TMP_DIR_DEFAULT="/home/vipuser/tmp"
    mkdir -p "$TMP_DIR_DEFAULT" 2>/dev/null || true
fi
if [[ ! -d "$TMP_DIR_DEFAULT" ]]; then
    TMP_DIR_DEFAULT="/tmp"
    mkdir -p "$TMP_DIR_DEFAULT" 2>/dev/null || true
fi
export TMPDIR="$TMP_DIR_DEFAULT"
export TEMP="$TMP_DIR_DEFAULT"
export TMP="$TMP_DIR_DEFAULT"

# HF / Torch 缓存集中到项目目录
export HF_HOME="${PROJ_ROOT}/cache/hf_home"
export HUGGINGFACE_HUB_CACHE="${PROJ_ROOT}/cache/hub"
export TRANSFORMERS_CACHE="${PROJ_ROOT}/cache/hub"
export TORCH_HOME="${PROJ_ROOT}/cache/models"
export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" || true

# 通用训练设置
export GLOBAL_SEED=42
export DEVICE="cuda"
export PYTHONUNBUFFERED=1
STDBUF=${STDBUF:-"stdbuf -oL -eL"}

export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"expandable_segments:True,max_split_size_mb:128"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True,max_split_size_mb:128"}

export MINDEYE_DTYPE=${MINDEYE_DTYPE:-bf16}
export CLIP_FP32=${CLIP_FP32:-0}

# 计时工具
declare -a STEP_NAMES=()
declare -a STEP_DURS_NS=()

format_duration_ns() {
    local ns=$1
    if [[ -z "$ns" || "$ns" -le 0 ]]; then
        echo "00:00:00.000"
        return
    fi
    local ms=$((ns/1000000))
    local sec=$((ms/1000))
    local ms_rem=$((ms%1000))
    local h=$((sec/3600))
    local m=$(((sec%3600)/60))
    local s=$((sec%60))
    printf "%02d:%02d:%02d.%03d" "$h" "$m" "$s" "$ms_rem"
}

add_step_timing() {
    local name="$1"
    local dur_ns="$2"
    STEP_NAMES+=("$name")
    STEP_DURS_NS+=("$dur_ns")
}

TOTAL_START_NS=$(date +%s%N)

# Python / Accelerate 解析
PY=${PY:-python}
ACC=${ACC:-accelerate launch}

if [[ -n "$ENV_NAME" ]]; then
    CAND_PY="/home/vipuser/miniconda3/envs/${ENV_NAME}/bin/python"
    CAND_ACC="/home/vipuser/miniconda3/envs/${ENV_NAME}/bin/accelerate"
    if [[ -x "$CAND_PY" ]]; then
        PY="$CAND_PY"
        if [[ -x "$CAND_ACC" ]]; then
            ACC="$CAND_ACC launch"
        else
            ACC="$PY"
        fi
    fi
fi

if [[ -z "$ENV_NAME" && -n "$CONDA_PREFIX" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PY="$CONDA_PREFIX/bin/python"
    if [[ -x "$CONDA_PREFIX/bin/accelerate" ]]; then
        ACC="$CONDA_PREFIX/bin/accelerate launch"
    else
        ACC="$PY"
    fi
fi

echo "🔧 使用 Python 解释器: $PY"

# ======================================================================================
# §2. 各任务配置
# ======================================================================================

# --- 任务1: MindEye2 主干训练 (可选，用于你自己从头训模型时) ---
export MINDYEYE_EXP_NAME="${MINDYEYE_EXP_NAME:-mindeye_v1_subj_all}"
export MINDYEYE_EPOCHS=${MINDYEYE_EPOCHS:-150}
export MINDYEYE_BATCH_SIZE=${MINDYEYE_BATCH_SIZE:-16}
export MINDYEYE_MAX_LR=${MINDYEYE_MAX_LR:-3e-4}
export MINDYEYE_OUT_DIR="${OUTPUT_ROOT}/${MINDYEYE_EXP_NAME}"
export MINDYEYE_MULTI_SUBJECT=false
export MINDYEYE_VALID_SUBJ=1   # 单被试训练默认 subj01 做训练+验证

# --- 任务2: 投影矩阵训练 (brain→ViT-H) ---
# 默认使用我们刚刚跑过的 subj01 train split 9000 样本的特征
export PROJ_BRAIN_VEC_PT="${PROJ_BRAIN_VEC_PT:-${PROJ_ROOT}/data/proj_subj01_train/all_subjects_brain_vectors.pt}"
export PROJ_IMG_VITH_PT="${PROJ_IMG_VITH_PT:-${PROJ_ROOT}/data/proj_subj01_train/all_subjects_gt_vith.pt}"
export PROJ_OUT_PT="${PROJ_OUT_PT:-${PROJ_ROOT}/checkpoints/brain2vith_subj01_train.pt}"
export PROJ_MODE="${PROJ_MODE:-closed_form}"   # closed_form / train
export PROJ_L2=${PROJ_L2:-0.0}                 # 现在我们用 0，与你刚才命令保持一致

# --- 任务3: PEFT 个性化适配器训练 (扩散端) ---
export PEFT_SUBJ_ID=${PEFT_SUBJ_ID:-1}
export PEFT_SUBJ_STR=$(printf "subj%02d" "$PEFT_SUBJ_ID")
export PEFT_EXP_NAME="${PEFT_EXP_NAME:-peft_adapter_${PEFT_SUBJ_STR}}"
export PEFT_OUT_DIR="${PEFT_OUT_DIR:-${OUTPUT_ROOT}/${PEFT_EXP_NAME}}"
export PEFT_CSV_PATH="${PEFT_CSV_PATH:-${PROJ_ROOT}/data/train_pairs_${PEFT_SUBJ_STR}.csv}"
export PEFT_HDF5_PATH="${PEFT_HDF5_PATH:-${NSD_DATA_PATH}/coco_images_224_float16.hdf5}"
export PEFT_EPOCHS=${PEFT_EPOCHS:-5}
export PEFT_STEPS=${PEFT_STEPS:-1000}
export PEFT_LR=${PEFT_LR:-5e-4}
export TRAIN_SOFT_PROMPT=${TRAIN_SOFT_PROMPT:-true}
export TRAIN_LORA=${TRAIN_LORA:-false}

# ======================================================================================
# §3. 主逻辑：各任务的实现
# ======================================================================================

run_mindeye_training() {
    echo "============================================================"
    echo "🚀 开始训练任务: MindEye2 主模型 (自训练分支，可选)"
    echo "============================================================"
    local _t0=$(date +%s%N)

    if [[ "$FORCE_RUN" == "true" && -d "$MINDYEYE_OUT_DIR" ]]; then
        echo "🟡 --force: 正在删除旧的MindEye2输出目录: ${MINDYEYE_OUT_DIR}"
        rm -rf "$MINDYEYE_OUT_DIR"
    fi
    mkdir -p "$MINDYEYE_OUT_DIR"

    LOCAL_DATA_PATH="$NSD_DATA_PATH"

    # ⭐ 可选：subj01 的 train/test 划分视图（仅当你设置了 MINDYEYE_TRAIN_SPLIT 时生效）
    if [[ "${MINDYEYE_MULTI_SUBJECT}" != "true" && "${MINDYEYE_VALID_SUBJ}" == "1" && -n "${MINDYEYE_TRAIN_SPLIT:-}" ]]; then
        echo "🟡 启用 subj01 训练/测试划分视图: 训练占比=${MINDYEYE_TRAIN_SPLIT}"
        DS_VIEW_ROOT="${MINDYEYE_OUT_DIR}/ds_view"
        SRC_TRAIN_DIR="${NSD_DATA_PATH}/wds/subj01/train"
        SRC_TEST_DIR="${NSD_DATA_PATH}/wds/subj01/new_test"
        DEST_TRAIN_DIR="${DS_VIEW_ROOT}/wds/subj01/train"
        DEST_TEST_DIR="${DS_VIEW_ROOT}/wds/subj01/new_test"
        mkdir -p "${DEST_TRAIN_DIR}" "${DEST_TEST_DIR}"

        mapfile -t ALL_SHARDS < <(ls -1 "${SRC_TRAIN_DIR}"/*.tar 2>/dev/null | sort -V)
        if [[ ${#ALL_SHARDS[@]} -lt 2 ]]; then
            echo "❌ subj01 训练分片不足，无法划分。" >&2
            exit 1
        fi
        TOTAL=${#ALL_SHARDS[@]}
        K=$($PY - <<PY
import math
ratio=float("${MINDYEYE_TRAIN_SPLIT}")
total=int("${TOTAL}")
k=max(1, min(total-1, int(round(total*ratio))))
print(k)
PY
)
        for i in $(seq 0 $((K-1))); do
            bn=$(basename "${ALL_SHARDS[$i]}")
            ln -sfn "${ALL_SHARDS[$i]}" "${DEST_TRAIN_DIR}/${bn}"
        done
        printf "" > "${MINDYEYE_OUT_DIR}/test_shards_subj01.txt"
        for i in $(seq ${K} $((TOTAL-1))); do
            echo "${ALL_SHARDS[$i]}" >> "${MINDYEYE_OUT_DIR}/test_shards_subj01.txt"
        done
        FIRST_HOLDOUT=$(head -n1 "${MINDYEYE_OUT_DIR}/test_shards_subj01.txt" || true)
        if [[ -n "$FIRST_HOLDOUT" ]]; then
            ln -sfn "$FIRST_HOLDOUT" "${DEST_TEST_DIR}/0.tar"
        else
            if [[ -f "${SRC_TEST_DIR}/0.tar" ]]; then
                ln -sfn "${SRC_TEST_DIR}/0.tar" "${DEST_TEST_DIR}/0.tar"
            fi
        fi
        cat > "${MINDYEYE_OUT_DIR}/split_meta_subj01.json" <<META
{"total_shards": ${TOTAL}, "k_train": ${K}, "train_ratio": ${MINDYEYE_TRAIN_SPLIT}}
META
        echo "✅ 划分完成: 训练分片=${K}/${TOTAL}; 清单: ${MINDYEYE_OUT_DIR}/test_shards_subj01.txt"
        LOCAL_DATA_PATH="${DS_VIEW_ROOT}"
    fi

    LAUNCHER="$PY"
    CKPT_FLAG="--ckpt_saving"
    if [[ "$FAST_RUN" == "true" ]]; then
        CKPT_FLAG="--no-ckpt_saving"
    fi

    local PRIOR_FLAG=$( [[ "${MINDYEYE_ENABLE_PRIOR:-1}" == "1" ]] && echo "--use_prior" || echo "--no-use_prior" )
    local BG_MODE=${MINDYEYE_BACKGROUND:-0}
    local TS=$(date +%Y%m%d_%H%M%S)
    local LOG_PATH=${MINDYEYE_LOG_PATH:-"${MINDYEYE_OUT_DIR}/mindeye_${TS}.log"}
    local PID_FILE=${MINDYEYE_PID_FILE:-"${MINDYEYE_OUT_DIR}/mindeye.pid"}

    local CMD="${LAUNCHER} \"${PROJ_ROOT}/src/Train.py\" \
        --model_name \"${MINDYEYE_EXP_NAME}\" \
        --data_path \"${LOCAL_DATA_PATH}\" \
        --num_epochs \"${MINDYEYE_EPOCHS}\" \
        $( [[ -n \"${MINDYEYE_NUM_SESSIONS:-}\" ]] && echo --num_sessions \"${MINDYEYE_NUM_SESSIONS}\" ) \
        --batch_size \"${MINDYEYE_BATCH_SIZE}\" \
        --max_lr \"${MINDYEYE_MAX_LR}\" \
        --seed \"${GLOBAL_SEED}\" \
        ${CKPT_FLAG} \
        --no-blurry_recon \
        ${PRIOR_FLAG} \
        $( [ \"${MINDYEYE_MULTI_SUBJECT}\" = true ] && echo \"--multi_subject --subj ${MINDYEYE_VALID_SUBJ}\" || echo \"--subj ${MINDYEYE_VALID_SUBJ}\" ) \
        $( [[ -n \"${MINDYEYE_TRAIN_SPLIT:-}\" ]] && echo --train_split_ratio \"${MINDYEYE_TRAIN_SPLIT}\" )"

    if [[ "${BG_MODE}" == "1" ]]; then
        mkdir -p "${MINDYEYE_OUT_DIR}" || true
        echo "🟡 后台启动 MindEye2，日志: ${LOG_PATH}"
        nohup bash -lc "${CMD}" > "${LOG_PATH}" 2>&1 &
        local MID_PID=$!
        echo ${MID_PID} > "${PID_FILE}"
        echo "📌 MindEye PID: ${MID_PID} (写入 ${PID_FILE})"
        echo "提示: tail -f ${LOG_PATH} 查看进度；kill ${MID_PID} 可终止。"
    else
        TMP_USE=\"$TMPDIR\" \
        TMPDIR=\"$TMP_USE\" TEMP=\"$TMP_USE\" TMP=\"$TMP_USE\" \
        eval ${CMD}
        echo "✅ MindEye2 训练完成。模型保存在: ${MINDYEYE_OUT_DIR}"
        local _t1=$(date +%s%N)
        local _dur=$((_t1-_t0))
        echo "⏱️ MindEye2 用时: $(format_duration_ns $_dur)"
        add_step_timing "mindeye" "$_dur"
    fi
}

run_projection_training() {
    echo "============================================================"
    echo "🚀 训练: brain→ViT-H 线性投影矩阵 (当前主线: subj01 train 9000 样本)"
    echo "============================================================"
    local _t0=$(date +%s%N)

    if [[ "$FORCE_RUN" == "true" && -f "$PROJ_OUT_PT" ]]; then
        echo "🟡 --force: 正在删除旧的投影矩阵: ${PROJ_OUT_PT}"
        rm -f "$PROJ_OUT_PT"
    fi
    if [[ -f "$PROJ_OUT_PT" ]]; then
        echo "🟡 投影矩阵已存在，跳过训练。如需重新拟合，使用 --force。"
        local _t1=$(date +%s%N)
        add_step_timing "projection (skipped)" "$((_t1-_t0))"
        return
    fi

    if [[ ! -f "$PROJ_BRAIN_VEC_PT" || ! -f "$PROJ_IMG_VITH_PT" ]]; then
        echo "❌ 错误: 找不到训练投影矩阵所需的输入文件:" >&2
        echo "   - $PROJ_BRAIN_VEC_PT" >&2
        echo "   - $PROJ_IMG_VITH_PT" >&2
        echo "   你需要先用 tools/extract_all_features.py 把特征提出来。" >&2
        exit 1
    fi

    $PY "${PROJ_ROOT}/tools/train_brain2vith.py" \
        --brain_pt "$PROJ_BRAIN_VEC_PT" \
        --img_vith_pt "$PROJ_IMG_VITH_PT" \
        --out "$PROJ_OUT_PT" \
        --mode "$PROJ_MODE" \
        --lambda_l2 "$PROJ_L2" \
        --device "$DEVICE"

    echo "✅ 投影矩阵训练完成。矩阵保存在: ${PROJ_OUT_PT}"
    local _t1=$(date +%s%N)
    local _dur=$((_t1-_t0))
    echo "⏱️ Projection 用时: $(format_duration_ns $_dur)"
    add_step_timing "projection" "$_dur"
}

run_peft_training() {
    echo "============================================================"
    echo "🚀 训练: PEFT 个性化适配器 (被试 ${PEFT_SUBJ_ID})"
    echo "============================================================"
    local _t0_all=$(date +%s%N)

    echo "--- 1) 准备 CSV 训练数据 ---"
    local _t0_prep=$(date +%s%N)
    if [[ "$FORCE_RUN" == "true" && -f "$PEFT_CSV_PATH" ]]; then
        echo "🟡 --force: 删除旧的 CSV: ${PEFT_CSV_PATH}"
        rm -f "$PEFT_CSV_PATH"
    fi
    if [[ -f "$PEFT_CSV_PATH" ]]; then
        echo "🟡 CSV 已存在，跳过生成。"
        local _t1_prep=$(date +%s%N)
        add_step_timing "peft:data_prep (cached)" "$((_t1_prep-_t0_prep))"
    else
        (cd "${PROJ_ROOT}/src" && python "prepare_data.py")
        if [[ -f "${PROJ_ROOT}/src/train_pairs_subj01.csv" ]]; then
            PEFT_CSV_PATH="${PROJ_ROOT}/src/train_pairs_subj01.csv"
        fi
        echo "✅ CSV 数据准备完成: ${PEFT_CSV_PATH}"
        local _t1_prep=$(date +%s%N)
        add_step_timing "peft:data_prep" "$((_t1_prep-_t0_prep))"
    fi

    echo "--- 2) 训练适配器 ---"
    if [[ "$FORCE_RUN" == "true" && -d "$PEFT_OUT_DIR" ]]; then
        echo "🟡 --force: 删除旧的 PEFT 输出目录: ${PEFT_OUT_DIR}"
        rm -rf "$PEFT_OUT_DIR"
    fi
    mkdir -p "$PEFT_OUT_DIR"

    if ! $PY - <<'PYCHK' >/dev/null 2>&1
import sys
try:
    import diffusers  # noqa: F401
except Exception:
    sys.exit(1)
PYCHK
    then
        if [[ "$FAST_RUN" == "true" ]]; then
            echo "🟡 缺少 diffusers，FAST 模式下跳过 PEFT 训练。"
            local _t1_all=$(date +%s%N)
            add_step_timing "peft:train (skipped)" "$((_t1_all-_t0_all))"
            return
        else
            echo "❌ 错误: 未安装 diffusers，请先安装后再运行 peft。" >&2
            exit 1
        fi
    fi

    PEFT_CMD="$PY \"${PROJ_ROOT}/src/train_peft_adapter.py\" \
        --csv_path \"$PEFT_CSV_PATH\" \
        --subject_id \"$PEFT_SUBJ_STR\" \
        --out_dir \"$PEFT_OUT_DIR\" \
        --images_hdf5_path \"$PEFT_HDF5_PATH\" \
        --epochs \"$PEFT_EPOCHS\" \
        --steps \"$PEFT_STEPS\" \
        --lr \"$PEFT_LR\" \
        --seed \"$GLOBAL_SEED\""

    if [[ "$FAST_RUN" == "true" ]]; then
        PEFT_CMD+=" --model_id stabilityai/sd-turbo"
    fi

    if [[ "$PEFT_CSV_PATH" == *"train_pairs_subj01.csv"* ]]; then
        PEFT_CMD=${PEFT_CMD/--subject_id \"$PEFT_SUBJ_STR\"/--subject_id \"subj01\"}
    fi

    if [[ "$TRAIN_SOFT_PROMPT" == "true" ]]; then
        PEFT_CMD+=" --train_soft"
    fi
    if [[ "$TRAIN_LORA" == "true" ]]; then
        PEFT_CMD+=" --train_lora"
    fi

    unset HF_HUB_OFFLINE || true
    if [[ -z "$HF_ENDPOINT" ]]; then
        export HF_ENDPOINT="https://hf-mirror.com"
    fi

    eval "$PEFT_CMD"

    echo "✅ PEFT 适配器训练完成。适配器保存在: ${PEFT_OUT_DIR}"
    local _t1_all=$(date +%s%N)
    local _dur_all=$((_t1_all-_t0_all))
    echo "⏱️ PEFT-训练总用时: $(format_duration_ns $_dur_all)"
    add_step_timing "peft:train" "$_dur_all"
}

# ======================================================================================
# §4. 入口参数解析 & 调度
# ======================================================================================

TASK=$1
FORCE_RUN=false
FAST_RUN=false
for arg in "$@"; do
    case $arg in
        --force) FORCE_RUN=true ;;
        --fast)  FAST_RUN=true ;;
    esac
done

if [[ "$FAST_RUN" == "true" ]]; then
    echo "🟡 FAST 模式开启：将使用极小 epoch/steps 以便快速验证流程。"
    MINDYEYE_EPOCHS=1
    MINDYEYE_BATCH_SIZE=4
    PEFT_EPOCHS=1
    PEFT_STEPS=50
fi

if [ -z "$TASK" ]; then
    echo "❌ 错误: 未指定训练任务。" >&2
    echo "   用法: bash run_training.sh [mindeye|projection|peft] [--force] [--fast]" >&2
    exit 1
fi

case $TASK in
    mindeye)
        run_mindeye_training
        ;;
    projection)
        run_projection_training
        ;;
    peft)
        run_peft_training
        ;;
    *)
        echo "❌ 错误: 未知的训练任务 '$TASK'。" >&2
        echo "   有效任务: [mindeye|projection|peft]" >&2
        exit 1
        ;;
esac

echo ""
echo "🎉 训练脚本执行完毕。"

TOTAL_END_NS=$(date +%s%N)
TOTAL_ELAPSE_NS=$((TOTAL_END_NS-TOTAL_START_NS))

echo ""
echo "================ 时间统计 (详细) ================"
sum_ns=0
for i in "${!STEP_NAMES[@]}"; do
    name=${STEP_NAMES[$i]}
    dur=${STEP_DURS_NS[$i]}
    sum_ns=$((sum_ns+dur))
    printf " - %-24s %s\n" "$name" "$(format_duration_ns "$dur")"
done
echo "--------------------------------------------------"
echo " 累计(各步骤求和):     $(format_duration_ns "$sum_ns")"
echo " 实际总耗时(墙钟):     $(format_duration_ns "$TOTAL_ELAPSE_NS")"
echo "=================================================="
echo "============================================================"
