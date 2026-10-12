#!/bin/bash
# ============================================================================
# SongGeneration v2 歌曲生成 单独运行脚本
# 模型: SongGeneration-v2-large (4B params)
# 输出: 48kHz FLAC, 最长 4m30s, 人声+伴奏
# 显存: ~32GB (auto), ~16GB (low_mem mode)
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

GPU_ID=${1:-0}
LOW_MEM=${2:-""}   # 传 "--low_mem" 使用低显存模式

GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
SYNC_SCRIPT=${GROUP_CODE}/fast_sync.py

SONGGEN_CODE=${GROUP_CODE}/utils/audio/SongGeneration
MODEL_SRC=${GROUP_BASE}/models/Audio/SongGeneration-v2-large
MODEL_DST=${LOCAL_BASE}/models/Audio/SongGeneration-v2-large
RUNTIME_SRC=${GROUP_BASE}/models/Audio/SongGeneration-Runtime
RUNTIME_DST=${LOCAL_BASE}/models/Audio/SongGeneration-Runtime

# 同步模型
mkdir -p ${MODEL_DST}
if [ ! -f "${MODEL_DST}/.sync_done" ]; then
    python ${SYNC_SCRIPT} ${MODEL_SRC} ${MODEL_DST} --workers 32 --skip-existing
    touch ${MODEL_DST}/.sync_done
fi

# 同步运行时
mkdir -p ${RUNTIME_DST}
if [ ! -f "${RUNTIME_DST}/.sync_done" ]; then
    python ${SYNC_SCRIPT} ${RUNTIME_SRC} ${RUNTIME_DST} --workers 32 --skip-existing
    touch ${RUNTIME_DST}/.sync_done
fi

SAVE_DIR=${GROUP_BASE}/results/audio_benchmark/songgen_song
mkdir -p ${SAVE_DIR}/audios ${SAVE_DIR}/jsonl

# 创建符号链接
for LINK_NAME in ckpt third_party tools; do
    LINK_SRC=${RUNTIME_DST}/${LINK_NAME}
    LINK_DST=${SONGGEN_CODE}/${LINK_NAME}
    if [ -d "$LINK_SRC" ] && [ ! -e "$LINK_DST" ]; then
        ln -s ${LINK_SRC} ${LINK_DST}
        echo "Created symlink: ${LINK_DST} -> ${LINK_SRC}"
    fi
done

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH="${SONGGEN_CODE}/codeclm/tokenizer/:${SONGGEN_CODE}:${SONGGEN_CODE}/codeclm/tokenizer/Flow1dVAE/:$PYTHONPATH"
export TRANSFORMERS_CACHE="${RUNTIME_DST}/hub"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 生成 benchmark JSONL
python ${GROUP_CODE}/run_audio_benchmark.py \
    --model songgen \
    --prompts_file ${GROUP_CODE}/script/audio/prompts_audio_benchmark.json \
    --output_dir ${GROUP_BASE}/results/audio_benchmark \
    --device cuda:0
