#!/bin/bash
# ============================================================================
# CosyVoice 单独运行脚本
#
# 默认：CosyVoice2（更稳定）
# 可选：CosyVoice3（已知存在质量问题，不建议默认使用）
# ============================================================================

set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

GPU_ID=${1:-0}
LANGUAGE=${2:-"zh"}
TTS_ENGINE=${3:-"cosyvoice2"}   # cosyvoice2 | cosyvoice3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
WM_GROUP_BASE=${WM_GROUP_BASE:-${IT2AV_WORKSPACE}}
OUTPUT_DIR=${WM_TTS_OUTPUT_DIR:-${WM_GROUP_BASE}/results/audio_benchmark/${TTS_ENGINE}}
WM_ENV_COSYVOICE=${WM_ENV_COSYVOICE:-}

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

run_cmd=(
  python "${CODE_ROOT}/run_audio_benchmark.py"
  --model cosyvoice
  --prompts_file "${CODE_ROOT}/script/audio/prompts_audio_benchmark.json"
  --output_dir "${OUTPUT_DIR}"
  --device cuda:0
  --language "${LANGUAGE}"
  --tts_engine "${TTS_ENGINE}"
)

echo "============================================================"
echo "🗣️  CosyVoice 推理"
echo "  Engine: ${TTS_ENGINE}"
echo "  语言:   ${LANGUAGE}"
echo "  GPU:    ${GPU_ID}"
echo "  输出:   ${OUTPUT_DIR}"
echo "============================================================"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if [[ -n "${WM_ENV_COSYVOICE}" ]] && conda env list | awk '{print $1}' | grep -qx "${WM_ENV_COSYVOICE}"; then
    conda activate "${WM_ENV_COSYVOICE}"
    "${run_cmd[@]}"
    status=$?
    conda deactivate || true
    exit ${status}
  fi
fi

"${run_cmd[@]}"
