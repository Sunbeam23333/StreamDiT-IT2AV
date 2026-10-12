#!/bin/bash
# ============================================================================
# ChatTTS 单模型推理脚本
#
# 默认使用随机 speaker embedding 的固定 seed（可复现）。
# ============================================================================

set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

GPU_ID=${1:-0}
LANGUAGE=${2:-"zh"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
WM_GROUP_BASE=${WM_GROUP_BASE:-${IT2AV_WORKSPACE}}
WM_MODEL_ROOT=${WM_MODEL_ROOT:-${WM_GROUP_BASE}/models/Audio}
WM_RESULT_ROOT=${WM_RESULT_ROOT:-${WM_GROUP_BASE}/results/audio_benchmark}
WM_LOCAL_BASE=${WM_LOCAL_BASE:-${IT2AV_LOCAL_ROOT}}
WM_SYNC_SCRIPT=${WM_SYNC_SCRIPT:-${CODE_ROOT}/fast_sync.py}
WM_ENV_CHATTTS=${WM_ENV_CHATTTS:-chattts}

MODEL_NAME=ChatTTS
MODEL_SRC=${WM_MODEL_ROOT}/${MODEL_NAME}
MODEL_DST=${WM_LOCAL_BASE}/models/Audio/${MODEL_NAME}
if [[ -d "${MODEL_SRC}" && -f "${WM_SYNC_SCRIPT}" ]]; then
  mkdir -p "${MODEL_DST}"
  if [[ ! -f "${MODEL_DST}/.sync_done" ]]; then
    echo ">>> 同步 ${MODEL_NAME} 到本地盘..."
    python "${WM_SYNC_SCRIPT}" "${MODEL_SRC}" "${MODEL_DST}" --workers 32 --skip-existing
    touch "${MODEL_DST}/.sync_done"
  fi
else
  MODEL_DST=${MODEL_SRC}
fi

SAVE_DIR=${WM_RESULT_ROOT}/chattts
mkdir -p "${SAVE_DIR}"
CHAT_SEED=${CHATTTS_SEED:-42}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
run_cmd=(
  python "${CODE_ROOT}/run_tts_benchmark.py"
  --model chattts
  --model_path "${MODEL_DST}"
  --code_dir "${CODE_ROOT}/utils/audio/ChatTTS"
  --prompts_file "${CODE_ROOT}/script/audio/prompts_audio_benchmark.json"
  --output_dir "${SAVE_DIR}"
  --device cuda:0
  --language "${LANGUAGE}"
  --chattts_seed "${CHAT_SEED}"
)

echo "============================================================"
echo "🗣️  ChatTTS 推理"
echo "  模型: ${MODEL_DST}"
echo "  语言: ${LANGUAGE}"
echo "  GPU: ${GPU_ID}"
echo "  输出: ${SAVE_DIR}"
echo "  Seed: ${CHAT_SEED}"
echo "============================================================"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "${WM_ENV_CHATTTS}"; then
    conda activate "${WM_ENV_CHATTTS}"
    "${run_cmd[@]}"
    status=$?
    conda deactivate || true
    exit ${status}
  fi
fi

"${run_cmd[@]}"
