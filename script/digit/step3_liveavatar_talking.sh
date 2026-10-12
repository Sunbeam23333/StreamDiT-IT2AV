#!/bin/bash
# ============================================================================
# Step 3: LiveAvatar 数字人说台词视频 (5卡 TPP)
#
# 输入:
#   首帧图片: ${IT2AV_WORKSPACE}/data/pics/digit/20260316/case_{id}.png  (Step1 生成)
#   TTS 音频: ${IT2AV_WORKSPACE}/data/audio/digit/20260316/case_{id}/tts.wav (Step2 生成)
#
# 输出:
#   ${IT2AV_WORKSPACE}/data/video/digit/20260316/case_{id}/talking.mp4
#
# 说明:
#   每个 case 的台词约 10 秒, 生成对镜头说台词的正面视频
#   使用 LiveAvatar 5 卡 TPP 推理 (4 DiT + 1 VAE, FP8, 4 steps)
#   这段视频的最后一帧将作为 Step4 I2V 动作视频的首帧
#
# 参数计算 (10s 台词):
#   fps = 25, infer_frames = 48
#   clip_duration = 48 / 25 = 1.92s
#   num_clips = ceil(10 / 1.92) ≈ 6 (实际由音频时长动态决定)
#
# 用法:
#   bash step3_liveavatar_talking.sh                   # 默认 GPU 0-4
#   GPU_START=0 bash step3_liveavatar_talking.sh       # 指定 GPU 起始
#   CASE_IDS="1 2" bash step3_liveavatar_talking.sh    # 只跑指定 case
#
# 资源: 5×H20, ~20GB VRAM/卡, 每个 case ~2-3 分钟
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"
conda activate ${IT2AV_CONDA_ENV}

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}/utils/video/LiveAvatar

LOCAL_BASE=${IT2AV_LOCAL_ROOT}
LOCAL_CODE=${LOCAL_BASE}/code/LiveAvatar
SYNC_SCRIPT=${IT2AV_CODE_ROOT}/fast_sync.py

LOCAL_MODEL_BASE=${LOCAL_BASE}/models/Wan2.2-S2V-14B
LOCAL_LORA_DIR=${LOCAL_BASE}/models/LiveAvatar
LOCAL_LORA=${LOCAL_LORA_DIR}/liveavatar.safetensors
TRAINING_CONFIG=${LOCAL_CODE}/liveavatar/configs/s2v_causal_sft.yaml

DATE_TAG="20260316"
IMAGE_DIR=${GROUP_BASE}/data/pics/digit/${DATE_TAG}
AUDIO_DIR=${GROUP_BASE}/data/audio/digit/${DATE_TAG}
VIDEO_DIR=${GROUP_BASE}/data/video/digit/${DATE_TAG}

# GPU 配置
GPU_START=${GPU_START:-0}
GPU_END=$((GPU_START + 4))
CUDA_DEVICES=""
for i in $(seq ${GPU_START} ${GPU_END}); do
    [ -n "${CUDA_DEVICES}" ] && CUDA_DEVICES="${CUDA_DEVICES},"
    CUDA_DEVICES="${CUDA_DEVICES}${i}"
done

# 选择 case (默认全部 5 个)
CASE_IDS=${CASE_IDS:-"1 2 3 4 5"}

# ===== 环境设置 =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
export ENABLE_COMPILE=false

echo "============================================================"
echo "🎬 Step 3: LiveAvatar 数字人说台词视频 (5卡 TPP)"
echo "  首帧目录: ${IMAGE_DIR}"
echo "  TTS 目录: ${AUDIO_DIR}"
echo "  输出目录: ${VIDEO_DIR}"
echo "  GPUs: ${CUDA_DEVICES}"
echo "  Cases: ${CASE_IDS}"
echo "============================================================"

# ===== 同步代码和模型 =====
echo ""
echo ">>> 同步代码和模型..."
mkdir -p ${LOCAL_CODE}
python ${SYNC_SCRIPT} ${GROUP_CODE} ${LOCAL_CODE} --workers 32 --skip-existing

for MODEL_DIR in "${LOCAL_MODEL_BASE}" "${LOCAL_LORA_DIR}"; do
    if [ ! -f "${MODEL_DIR}/.sync_done" ]; then
        GROUP_DIR=$(echo ${MODEL_DIR} | sed "s|${LOCAL_BASE}|${GROUP_BASE}|")
        echo "  同步模型: ${GROUP_DIR}"
        mkdir -p ${MODEL_DIR}
        python ${SYNC_SCRIPT} ${GROUP_DIR} ${MODEL_DIR} --workers 32 --skip-existing
        touch "${MODEL_DIR}/.sync_done"
    fi
done

pip install dashscope bitsandbytes -q 2>/dev/null || true

cd ${LOCAL_CODE}

# ===== 动态计算 num_clip 的辅助函数 =====
get_audio_duration_clips() {
    # 用 ffprobe 或 python 获取音频时长, 计算需要的 clip 数
    local AUDIO_FILE=$1
    local DURATION=$(python3 -c "
import subprocess, json
try:
    r = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '$AUDIO_FILE'],
                       capture_output=True, text=True)
    d = json.loads(r.stdout)
    print(f\"{float(d['format']['duration']):.1f}\")
except:
    # fallback: 用 torchaudio
    import torchaudio
    info = torchaudio.info('$AUDIO_FILE')
    print(f'{info.num_frames / info.sample_rate:.1f}')
" 2>/dev/null)
    echo ${DURATION}
}

# ===== 串行推理每个 case =====
TOTAL_START=$(date +%s)

for CASE_ID in ${CASE_IDS}; do
    IMAGE_PATH=${IMAGE_DIR}/case_${CASE_ID}.png
    AUDIO_PATH=${AUDIO_DIR}/case_${CASE_ID}/tts.wav
    SAVE_DIR_CASE=${VIDEO_DIR}/case_${CASE_ID}
    SAVE_FILE=${SAVE_DIR_CASE}/talking.mp4

    mkdir -p ${SAVE_DIR_CASE}

    # 检查输入
    if [ ! -f "${IMAGE_PATH}" ]; then
        echo "  ❌ case_${CASE_ID}: 首帧图片不存在: ${IMAGE_PATH}"
        echo "     请先运行 step1_generate_images.sh"
        continue
    fi
    if [ ! -f "${AUDIO_PATH}" ]; then
        echo "  ❌ case_${CASE_ID}: TTS 音频不存在: ${AUDIO_PATH}"
        echo "     请先运行 step2_generate_audio.sh"
        continue
    fi

    # 跳过已存在的视频
    if [ -f "${SAVE_FILE}" ] && [ $(stat -c%s "${SAVE_FILE}" 2>/dev/null || stat -f%z "${SAVE_FILE}" 2>/dev/null) -gt 10240 ]; then
        echo "  ⏭️  case_${CASE_ID}: 视频已存在: ${SAVE_FILE}"
        continue
    fi

    # 计算音频时长和 clip 数
    AUDIO_DURATION=$(get_audio_duration_clips ${AUDIO_PATH})
    # clip_duration = 48/25 = 1.92s
    NUM_CLIP=$(python3 -c "import math; print(max(1, math.ceil(${AUDIO_DURATION} / 1.92)))")

    PROMPT="A professional Chinese beauty anchor is standing still and speaking directly to the camera, hands naturally at her sides without any gestures, with subtle lip movements and gentle facial expressions. Warm soft lighting, elegant and gentle demeanor, no hand movements."

    echo ""
    echo "============================================================"
    echo "  🎬 case_${CASE_ID}"
    echo "  Image: ${IMAGE_PATH}"
    echo "  Audio: ${AUDIO_PATH} (${AUDIO_DURATION}s)"
    echo "  Clips: ${NUM_CLIP} (1.92s/clip)"
    echo "  Output: ${SAVE_FILE}"
    echo "============================================================"

    CASE_START=$(date +%s)

    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}

    torchrun --nproc_per_node=5 --master_port=29098 \
        minimal_inference/s2v_streaming_interact.py \
        --task s2v-14B \
        --ckpt_dir "${LOCAL_MODEL_BASE}" \
        --size "1280*720" \
        --base_seed 42 \
        --training_config ${TRAINING_CONFIG} \
        --offload_model False \
        --convert_model_dtype \
        --prompt "${PROMPT}" \
        --image "${IMAGE_PATH}" \
        --audio "${AUDIO_PATH}" \
        --infer_frames 48 \
        --load_lora \
        --lora_path_dmd "${LOCAL_LORA}" \
        --sample_steps 4 \
        --sample_guide_scale 0 \
        --num_clip ${NUM_CLIP} \
        --num_gpus_dit 4 \
        --sample_solver euler \
        --enable_vae_parallel \
        --save_dir "${SAVE_DIR_CASE}/" \
        --save_file "${SAVE_FILE}" \
        --fp8

    CASE_END=$(date +%s)
    echo "  ✅ case_${CASE_ID} 完成: $((CASE_END - CASE_START))s"
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

# ===== 提取最后一帧 (供 Step4 I2V 使用) =====
echo ""
echo ">>> 提取每个视频的最后一帧..."
for CASE_ID in ${CASE_IDS}; do
    VIDEO_FILE=${VIDEO_DIR}/case_${CASE_ID}/talking.mp4
    LAST_FRAME=${VIDEO_DIR}/case_${CASE_ID}/last_frame.png

    if [ -f "${VIDEO_FILE}" ] && [ ! -f "${LAST_FRAME}" ]; then
        # 用 ffmpeg 提取最后一帧
        ffmpeg -y -sseof -0.1 -i "${VIDEO_FILE}" -frames:v 1 -q:v 2 "${LAST_FRAME}" 2>/dev/null \
            && echo "  ✅ case_${CASE_ID}: 最后一帧 → ${LAST_FRAME}" \
            || echo "  ⚠️  case_${CASE_ID}: 最后一帧提取失败"
    fi
done

# ===== 汇总 =====
echo ""
echo "============================================================"
echo "🎬 Step 3 完成!"
echo "  耗时: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))min)"
echo ""
echo "  📁 生成的视频:"
for CASE_ID in ${CASE_IDS}; do
    VIDEO_FILE=${VIDEO_DIR}/case_${CASE_ID}/talking.mp4
    LAST_FRAME=${VIDEO_DIR}/case_${CASE_ID}/last_frame.png
    if [ -f "${VIDEO_FILE}" ]; then
        SIZE=$(du -h "${VIDEO_FILE}" | cut -f1)
        echo "    ✅ case_${CASE_ID}/talking.mp4 ($SIZE)"
        [ -f "${LAST_FRAME}" ] && echo "       最后一帧: ${LAST_FRAME}"
    else
        echo "    ❌ case_${CASE_ID}/talking.mp4 (缺失)"
    fi
done
echo ""
echo "  下一步: bash step4_i2v_action.sh"
echo "============================================================"
