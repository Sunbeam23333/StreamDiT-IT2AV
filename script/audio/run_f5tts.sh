#!/bin/bash
# ============================================================================
# F5-TTS 单模型推理脚本
#
# 默认使用仓库内置参考音频；也支持自行传入参考音频 / 参考文本。
# 用法：
#   bash script/audio/run_f5tts.sh 0 zh
#   F5_REF_AUDIO=/path/ref.wav F5_REF_TEXT="参考文本" bash script/audio/run_f5tts.sh 0 zh
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
WM_ENV_F5TTS=${WM_ENV_F5TTS:-f5tts}

MODEL_NAME=F5-TTS
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

SAVE_DIR=${WM_RESULT_ROOT}/f5tts
mkdir -p "${SAVE_DIR}"

REF_AUDIO=${F5_REF_AUDIO:-}
REF_TEXT=${F5_REF_TEXT:-}
if [[ -z "${REF_AUDIO}" ]]; then
  if [[ "${LANGUAGE}" == "en" ]]; then
    REF_AUDIO="${CODE_ROOT}/utils/audio/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    REF_TEXT="${REF_TEXT:-Some call me nature, others call me mother nature.}"
  else
    REF_AUDIO="${CODE_ROOT}/utils/audio/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_zh.wav"
    REF_TEXT="${REF_TEXT:-对，这就是我，万人敬仰的太乙真人。}"
  fi
fi

export CUDA_VISIBLE_DEVICES=${GPU_ID}
run_cmd=(
  python "${CODE_ROOT}/run_tts_benchmark.py"
  --model f5tts
  --model_path "${MODEL_DST}"
  --code_dir "${CODE_ROOT}/utils/audio/F5-TTS"
  --prompts_file "${CODE_ROOT}/script/audio/prompts_audio_benchmark.json"
  --output_dir "${SAVE_DIR}"
  --device cuda:0
  --language "${LANGUAGE}"
  --ref_audio "${REF_AUDIO}"
  --ref_text "${REF_TEXT}"
)

echo "============================================================"
echo "🗣️  F5-TTS 推理"
echo "  模型: ${MODEL_DST}"
echo "  语言: ${LANGUAGE}"
echo "  GPU: ${GPU_ID}"
echo "  输出: ${SAVE_DIR}"
echo "  参考: ${REF_AUDIO}"
echo "============================================================"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "${WM_ENV_F5TTS}"; then
    conda activate "${WM_ENV_F5TTS}"
    "${run_cmd[@]}"
    status=$?
    conda deactivate || true
    exit ${status}
  fi
fi

"${run_cmd[@]}"
