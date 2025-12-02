#!/usr/bin/env bash
set -euo pipefail
# 等待训练完成后自动触发 mindeye 环境的推理
# 变量：
#   EXP_NAME: 训练实验名（对应 train_logs/EXP_NAME）
#   INFER_ENV_NAME: 用于推理的 Conda 环境（默认 mindeye21）
#   GEN_LIMIT: 可选，限制生成数量（默认不限制）
#   FORCE: 若为1则 run_inference.sh 使用 --force 重新生成

EXP_NAME=${EXP_NAME:-}
if [[ -z "${EXP_NAME}" ]]; then
  echo "❌ 缺少 EXP_NAME 环境变量（train_logs/<EXP_NAME>）" >&2
  exit 1
fi
INFER_ENV_NAME=${INFER_ENV_NAME:-mindeye21}
FORCE=${FORCE:-1}
GEN_LIMIT=${GEN_LIMIT:-}

PROJ_ROOT=$(cd -- "$(dirname -- "$0")" &>/dev/null && pwd)
OUT_DIR="${PROJ_ROOT}/train_logs/${EXP_NAME}"
CKPT_PATH="${OUT_DIR}/last.pth"

# Ensure output dir exists and create single-instance lock
mkdir -p "${OUT_DIR}"
LOCKFILE="${OUT_DIR}/post_infer.lock"
PIDFILE="${OUT_DIR}/post_infer.pid"
if [[ -f "${LOCKFILE}" ]]; then
  _oldpid=$(cat "${LOCKFILE}" 2>/dev/null || true)
  if [[ -n "${_oldpid}" ]] && ps -p "${_oldpid}" >/dev/null 2>&1; then
    echo "🛑 已有 watcher 运行中: PID ${_oldpid}（${EXP_NAME}）"
    exit 0
  fi
fi
echo "$$" > "${LOCKFILE}"
echo "$$" > "${PIDFILE}"
trap 'rm -f "${LOCKFILE}"' EXIT

echo "🕒 等待训练完成: ${CKPT_PATH} (PID=$$)"
while [[ ! -f "${CKPT_PATH}" ]]; do
  sleep 30
done

echo "✅ 检测到 ckpt: ${CKPT_PATH}"
cd "${PROJ_ROOT}"
ENV_NAME="${INFER_ENV_NAME}" \
MINDYEYE_MODEL_NAME="${EXP_NAME}" \
GEN_LIMIT="${GEN_LIMIT}" \
 bash run_inference.sh $( [[ "${FORCE}" == "1" ]] && echo --force )

echo "🎉 推理完成。"
