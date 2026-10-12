#!/bin/bash
# 运行 CosyVoice2 TTS 调试诊断
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"
conda activate ${IT2AV_CONDA_ENV}

# whisper 版本检查 (经验证 v20231117 和最新版 mel_filters 实现相同, 不影响 TTS)
python -c "import whisper; print(f'  whisper={whisper.__version__}')"

LOCAL_BASE=${IT2AV_LOCAL_ROOT}
COSYVOICE_CODE=${LOCAL_BASE}/code/utils/audio/CosyVoice
MATCHA_PATH=${COSYVOICE_CODE}/third_party/Matcha-TTS
if [ ! -f "${MATCHA_PATH}/setup.py" ]; then
    MATCHA_PATH=${LOCAL_BASE}/code/utils/audio/Matcha-TTS
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔍 启动 CosyVoice2 TTS 调试..."
echo "  Script: ${SCRIPT_DIR}/debug_cosyvoice_tts.py"

CUDA_VISIBLE_DEVICES=${GPU_TTS:-5} \
PYTHONPATH="${COSYVOICE_CODE}:${MATCHA_PATH}:$PYTHONPATH" \
python ${SCRIPT_DIR}/debug_cosyvoice_tts.py 2>&1 | tee ${IT2AV_WORKSPACE}/data/audio/digit/debug_tts/debug_log.txt

echo ""
echo "📋 日志已保存到: ${IT2AV_WORKSPACE}/data/audio/digit/debug_tts/debug_log.txt"
