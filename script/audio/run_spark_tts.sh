#!/bin/bash
# ============================================================================
# Spark-TTS 单模型推理脚本
#
# 默认使用 voice creation（gender/pitch/speed）；
# 如需 voice cloning，请在环境变量中提供 SPARK_REF_AUDIO / SPARK_REF_TEXT。
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
WM_ENV_SPARK_TTS=${WM_ENV_SPARK_TTS:-spark_tts}

MODEL_NAME=Spark-TTS-0.5B
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

SAVE_DIR=${WM_RESULT_ROOT}/spark_tts
mkdir -p "${SAVE_DIR}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
run_cmd=(
  python "${CODE_ROOT}/run_tts_benchmark.py"
  --model spark_tts
  --model_path "${MODEL_DST}"
  --code_dir "${CODE_ROOT}/utils/audio/Spark-TTS"
  --prompts_file "${CODE_ROOT}/script/audio/prompts_audio_benchmark.json"
  --output_dir "${SAVE_DIR}"
  --device cuda:0
  --language "${LANGUAGE}"
  --spark_gender "${SPARK_GENDER:-female}"
  --spark_pitch "${SPARK_PITCH:-moderate}"
  --spark_speed "${SPARK_SPEED:-moderate}"
)

if [[ -n "${SPARK_REF_AUDIO:-}" ]]; then
  run_cmd+=(--ref_audio "${SPARK_REF_AUDIO}")
fi
if [[ -n "${SPARK_REF_TEXT:-}" ]]; then
  run_cmd+=(--ref_text "${SPARK_REF_TEXT}")
fi

echo "============================================================"
echo "🗣️  Spark-TTS 推理"
echo "  模型: ${MODEL_DST}"
echo "  语言: ${LANGUAGE}"
echo "  GPU: ${GPU_ID}"
echo "  输出: ${SAVE_DIR}"
echo "  造音色: ${SPARK_GENDER:-female}/${SPARK_PITCH:-moderate}/${SPARK_SPEED:-moderate}"
echo "============================================================"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "${WM_ENV_SPARK_TTS}"; then
    conda activate "${WM_ENV_SPARK_TTS}"
    "${run_cmd[@]}"
    status=$?
    conda deactivate || true
    exit ${status}
  fi
fi

"${run_cmd[@]}"
