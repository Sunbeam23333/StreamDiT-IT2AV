#!/bin/bash
# ============================================================================
# ACE-Step 文生BGM/音乐 单独运行脚本
# 模型: ACE-Step v1 3.5B (Apache 2.0)
# 输出: 44.1kHz WAV, 最长 4 分钟
# 显存: ~16GB
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

GPU_ID=${1:-0}

GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
SYNC_SCRIPT=${GROUP_CODE}/fast_sync.py

# 同步模型
MODEL_SRC=${GROUP_BASE}/models/Audio/ACE-Step-v1-3.5B
MODEL_DST=${LOCAL_BASE}/models/Audio/ACE-Step-v1-3.5B
mkdir -p ${MODEL_DST}
if [ ! -f "${MODEL_DST}/.sync_done" ]; then
    python ${SYNC_SCRIPT} ${MODEL_SRC} ${MODEL_DST} --workers 32 --skip-existing
    touch ${MODEL_DST}/.sync_done
fi

SAVE_DIR=${GROUP_BASE}/results/audio_benchmark/ace_step_bgm
mkdir -p ${SAVE_DIR}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH=${GROUP_CODE}/utils/audio/ACE-Step:${GROUP_CODE}:$PYTHONPATH

python ${GROUP_CODE}/run_audio_benchmark.py \
    --model ace_step \
    --prompts_file ${GROUP_CODE}/script/audio/prompts_audio_benchmark.json \
    --output_dir ${GROUP_BASE}/results/audio_benchmark \
    --device_id 0
