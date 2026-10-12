#!/bin/bash
# ============================================================================
# Audio Generation 文生音频 集成推理脚本
#
# 服务器: 8卡 H20 95GB
# 框架:
#   - TangoFlux     (单卡推理, 音效生成, 515M params)
#   - ACE-Step      (单卡推理, BGM/音乐生成, 3.5B params)
#   - SongGeneration(单卡推理, 歌曲生成, 4B params)
#   - CosyVoice3    (单卡推理, 语音合成, 0.5B params)
#
# 场景分类:
#   - 纯音乐类 (Pure Music): 3 个场景
#   - 动漫类   (Anime):      3 个场景
#   - 真人运动 (Live Action): 4 个场景
#
# 用法:
#   # 单模型推理
#   bash run_audio_benchmark.sh tangoflux          # 音效
#   bash run_audio_benchmark.sh ace_step           # BGM
#   bash run_audio_benchmark.sh songgen            # 歌曲
#   bash run_audio_benchmark.sh cosyvoice          # 语音
#
#   # 全部模型
#   bash run_audio_benchmark.sh all
#
#   # 指定 GPU
#   bash run_audio_benchmark.sh tangoflux 0        # GPU 0
#   bash run_audio_benchmark.sh ace_step 1         # GPU 1
#
#   # 多卡并行 (不同模型跑在不同卡上)
#   bash run_audio_benchmark.sh parallel
#
#   # 覆盖 duration (快速测试)
#   bash run_audio_benchmark.sh tangoflux 0 zh 10     # 所有场景只生成10s
#   bash run_audio_benchmark.sh ace_step 1 zh 60       # 只生成60s BGM
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"
conda activate ${IT2AV_CONDA_ENV}
# ===== 参数解析 =====
MODEL=${1:-"all"}        # tangoflux, ace_step, songgen, cosyvoice, all, parallel
GPU_ID=${2:-0}           # GPU ID (单模型模式)
LANGUAGE=${3:-"zh"}      # TTS 语言: zh 或 en
DURATION_OVERRIDE=${4:-""}  # 全局 duration 覆盖 (秒), 空则用 prompts 中的值

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}
GROUP_MODELS=${GROUP_BASE}/models/Audio

LOCAL_BASE=${IT2AV_LOCAL_ROOT}
LOCAL_MODELS=${LOCAL_BASE}/models/Audio
LOCAL_CODE=${LOCAL_BASE}/code

SYNC_SCRIPT=${GROUP_CODE}/fast_sync.py
BENCHMARK_SCRIPT=${GROUP_CODE}/run_audio_benchmark.py
PROMPTS_FILE=${GROUP_CODE}/script/audio/prompts_audio_benchmark.json

# 输出目录
SAVE_DIR=${GROUP_BASE}/results/audio_benchmark

# ===== Conda 环境 =====
# 如果存在 conda，激活音频生成环境
if command -v conda &> /dev/null; then
    echo ">>> Activating conda environment..."
    # conda activate ${IT2AV_CONDA_ENV} || true  # 使用服务器默认环境
fi

echo "============================================"
echo "Audio Generation Benchmark"
echo "  Model:    ${MODEL}"
echo "  GPU:      ${GPU_ID}"
echo "  Language: ${LANGUAGE}"
echo "  Duration: ${DURATION_OVERRIDE:-from_prompts}"
echo "  Output:   ${SAVE_DIR}"
echo "============================================"

# ===== Step 1: 同步模型到本地盘 =====
echo ""
echo ">>> Step 1: Syncing models to local disk..."

sync_model() {
    local MODEL_NAME=$1
    local SRC=${GROUP_MODELS}/${MODEL_NAME}
    local DST=${LOCAL_MODELS}/${MODEL_NAME}

    if [ ! -d "$SRC" ]; then
        echo "[WARN] Model not found at $SRC, skipping sync."
        return 1
    fi

    mkdir -p ${DST}

    if [ ! -f "${DST}/.sync_done" ]; then
        echo ">>> Syncing ${MODEL_NAME}..."
        python ${SYNC_SCRIPT} ${SRC} ${DST} --workers ${SYNC_WORKERS:-32} --skip-existing
        touch ${DST}/.sync_done
        echo ">>> ${MODEL_NAME} sync complete."
    else
        echo ">>> ${MODEL_NAME} already cached, skipping."
    fi
    return 0
}

sync_code_dir() {
    local CODE_NAME=$1
    local SRC=${GROUP_CODE}/utils/audio/${CODE_NAME}
    local DST=${LOCAL_CODE}/utils/audio/${CODE_NAME}

    if [ ! -d "$SRC" ]; then
        echo "[WARN] Code not found at $SRC"
        return 1
    fi

    mkdir -p ${DST}
    echo ">>> Syncing code: ${CODE_NAME}..."
    python ${SYNC_SCRIPT} ${SRC} ${DST} --workers ${SYNC_WORKERS:-32} --skip-existing
    return 0
}

# 根据模型选择同步
case $MODEL in
    tangoflux)
        sync_model "TangoFlux"
        sync_code_dir "TangoFlux"
        ;;
    ace_step)
        sync_model "ACE-Step-v1-3.5B"
        sync_code_dir "ACE-Step"
        ;;
    songgen)
        sync_model "SongGeneration-v2-large"
        sync_model "SongGeneration-Runtime"
        sync_code_dir "SongGeneration"
        ;;
    cosyvoice)
        sync_model "Fun-CosyVoice3-0.5B"
        sync_model "CosyVoice2-0.5B"
        sync_model "CosyVoice-ttsfrd"
        sync_code_dir "CosyVoice"
        ;;
    all)
        sync_model "TangoFlux"
        sync_model "ACE-Step-v1-3.5B"
        sync_model "SongGeneration-v2-large"
        sync_model "SongGeneration-Runtime"
        sync_model "Fun-CosyVoice3-0.5B"
        sync_model "CosyVoice2-0.5B"
        sync_model "CosyVoice-ttsfrd"
        sync_code_dir "TangoFlux"
        sync_code_dir "ACE-Step"
        sync_code_dir "SongGeneration"
        sync_code_dir "CosyVoice"
        ;;
    parallel)
        # 并行同步所有
        sync_model "TangoFlux" &
        sync_model "ACE-Step-v1-3.5B" &
        sync_model "SongGeneration-v2-large" &
        sync_model "SongGeneration-Runtime" &
        sync_model "Fun-CosyVoice3-0.5B" &
        sync_model "CosyVoice2-0.5B" &
        sync_model "CosyVoice-ttsfrd" &
        wait
        sync_code_dir "TangoFlux"
        sync_code_dir "ACE-Step"
        sync_code_dir "SongGeneration"
        sync_code_dir "CosyVoice"
        ;;
    *)
        echo "[ERROR] Unknown model: $MODEL"
        echo "Supported: tangoflux, ace_step, songgen, cosyvoice, all, parallel"
        exit 1
        ;;
esac

echo ""
echo ">>> Step 2: Running audio generation..."

mkdir -p ${SAVE_DIR}

TOTAL_START=$(date +%s)

# ===== 设置 PYTHONPATH =====
export PYTHONPATH=${GROUP_CODE}:$PYTHONPATH

# ===== 构建 duration override 参数 =====
DURATION_ARG=""
if [ -n "$DURATION_OVERRIDE" ]; then
    DURATION_ARG="--duration_override ${DURATION_OVERRIDE}"
fi

if [ "$MODEL" == "parallel" ]; then
    # =========================================================
    # 多卡并行模式: 4 个模型分别跑在不同 GPU 上
    # =========================================================
    echo ">>> Parallel mode: running 4 models on 4 GPUs simultaneously..."

    CUDA_VISIBLE_DEVICES=0 python ${BENCHMARK_SCRIPT} \
        --model tangoflux \
        --prompts_file ${PROMPTS_FILE} \
        --output_dir ${SAVE_DIR} \
        --device cuda:0 \
        --device_id 0 \
        --language ${LANGUAGE} \
        ${DURATION_ARG} \
        > ${SAVE_DIR}/log_tangoflux.txt 2>&1 &
    PID_TANGO=$!
    echo "  [GPU 0] TangoFlux   (PID: $PID_TANGO)"

    CUDA_VISIBLE_DEVICES=1 python ${BENCHMARK_SCRIPT} \
        --model ace_step \
        --prompts_file ${PROMPTS_FILE} \
        --output_dir ${SAVE_DIR} \
        --device cuda:0 \
        --device_id 0 \
        --language ${LANGUAGE} \
        ${DURATION_ARG} \
        > ${SAVE_DIR}/log_ace_step.txt 2>&1 &
    PID_ACE=$!
    echo "  [GPU 1] ACE-Step    (PID: $PID_ACE)"

    CUDA_VISIBLE_DEVICES=2 python ${BENCHMARK_SCRIPT} \
        --model songgen \
        --prompts_file ${PROMPTS_FILE} \
        --output_dir ${SAVE_DIR} \
        --device cuda:0 \
        --device_id 0 \
        --language ${LANGUAGE} \
        ${DURATION_ARG} \
        > ${SAVE_DIR}/log_songgen.txt 2>&1 &
    PID_SONG=$!
    echo "  [GPU 2] SongGen     (PID: $PID_SONG)"

    CUDA_VISIBLE_DEVICES=3 python ${BENCHMARK_SCRIPT} \
        --model cosyvoice \
        --prompts_file ${PROMPTS_FILE} \
        --output_dir ${SAVE_DIR} \
        --device cuda:0 \
        --device_id 0 \
        --language ${LANGUAGE} \
        ${DURATION_ARG} \
        > ${SAVE_DIR}/log_cosyvoice.txt 2>&1 &
    PID_COSY=$!
    echo "  [GPU 3] CosyVoice   (PID: $PID_COSY)"

    echo ""
    echo ">>> Waiting for all models to finish..."

    FAILED=0
    wait $PID_TANGO || { echo "[FAIL] TangoFlux failed"; FAILED=1; }
    wait $PID_ACE   || { echo "[FAIL] ACE-Step failed"; FAILED=1; }
    wait $PID_SONG  || { echo "[FAIL] SongGeneration failed"; FAILED=1; }
    wait $PID_COSY  || { echo "[FAIL] CosyVoice failed"; FAILED=1; }

    if [ $FAILED -eq 1 ]; then
        echo ""
        echo "[WARN] Some models failed. Check logs:"
        echo "  ${SAVE_DIR}/log_tangoflux.txt"
        echo "  ${SAVE_DIR}/log_ace_step.txt"
        echo "  ${SAVE_DIR}/log_songgen.txt"
        echo "  ${SAVE_DIR}/log_cosyvoice.txt"
    fi

elif [ "$MODEL" == "all" ]; then
    # =========================================================
    # 串行运行所有模型 (共享同一个 GPU)
    # =========================================================
    echo ">>> Sequential mode: running all models on GPU ${GPU_ID}..."

    for M in tangoflux ace_step songgen cosyvoice; do
        echo ""
        echo ">>> ===== Running ${M} ====="
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BENCHMARK_SCRIPT} \
            --model ${M} \
            --prompts_file ${PROMPTS_FILE} \
            --output_dir ${SAVE_DIR} \
            --device cuda:0 \
            --device_id 0 \
            --language ${LANGUAGE} \
            ${DURATION_ARG}
    done

else
    # =========================================================
    # 单模型推理
    # =========================================================
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BENCHMARK_SCRIPT} \
        --model ${MODEL} \
        --prompts_file ${PROMPTS_FILE} \
        --output_dir ${SAVE_DIR} \
        --device cuda:0 \
        --device_id 0 \
        --language ${LANGUAGE} \
        ${DURATION_ARG}
fi

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================"
echo "Audio Benchmark completed!"
echo "  Total time: ${TOTAL_TIME}s"
echo "  Output:     ${SAVE_DIR}"
echo ""
echo "Output structure:"
echo "  ${SAVE_DIR}/tangoflux_sfx/   <- 音效 (wav, 44.1kHz, max 30s)"
echo "  ${SAVE_DIR}/ace_step_bgm/    <- BGM音乐 (wav, 44.1kHz, max 240s)"
echo "  ${SAVE_DIR}/songgen_song/    <- 歌曲 (flac, 48kHz, max 4m30s)"
echo "  ${SAVE_DIR}/cosyvoice_tts/   <- 语音旁白 (wav, ~22kHz, 由文本决定)"
echo "  ${SAVE_DIR}/benchmark_report.json  <- 推理时间报告"
echo ""
echo "These audio files can be used with video generation:"
echo "  1. tangoflux_sfx  -> 配合视频画面的环境音效"
echo "  2. ace_step_bgm   -> 配合视频画面的背景音乐"
echo "  3. songgen_song   -> 配合视频画面的歌曲"
echo "  4. cosyvoice_tts  -> 配合视频画面的解说旁白"
echo "============================================"
