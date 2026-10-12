#!/bin/bash
# ============================================================================
# TangoFlux 文生音效 单独运行脚本
# 模型: declare-lab/TangoFlux (515M params)
# 输出: 44.1kHz WAV, 最长 30 秒
# 显存: ~4GB
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

GPU_ID=${1:-0}

GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
SYNC_SCRIPT=${GROUP_CODE}/fast_sync.py

# 同步模型
MODEL_SRC=${GROUP_BASE}/models/Audio/TangoFlux
MODEL_DST=${LOCAL_BASE}/models/Audio/TangoFlux
mkdir -p ${MODEL_DST}
if [ ! -f "${MODEL_DST}/.sync_done" ]; then
    python ${SYNC_SCRIPT} ${MODEL_SRC} ${MODEL_DST} --workers 32 --skip-existing
    touch ${MODEL_DST}/.sync_done
fi

SAVE_DIR=${GROUP_BASE}/results/audio_benchmark/tangoflux_sfx
mkdir -p ${SAVE_DIR}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH=${GROUP_CODE}/utils/audio/TangoFlux:${GROUP_CODE}:$PYTHONPATH

python ${GROUP_CODE}/run_audio_benchmark.py \
    --model tangoflux \
    --prompts_file ${GROUP_CODE}/script/audio/prompts_audio_benchmark.json \
    --output_dir ${GROUP_BASE}/results/audio_benchmark \
    --device cuda:0
