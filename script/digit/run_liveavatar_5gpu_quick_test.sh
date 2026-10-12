#!/bin/bash
# ============================================================================
# LiveAvatar 5卡 TPP 快速测试 (使用自带示例, boy.wav 5分52秒完整视频)
# 用于验证 5 卡 TPP 推理环境和模型是否正常工作
#
# 音频: boy.wav (5min52s = 352s)
# 参数计算:
#   fps = 25 (s2v_14B 继承 shared_config.sample_fps)
#   infer_frames = 48 (每 clip 48 帧)
#   clip_duration = 48 / 25 = 1.92s
#   num_clips = ceil(352 / 1.92) = 184
#
# 用法:
#   bash run_liveavatar_5gpu_quick_test.sh              # 默认 GPU 0-4
#   GPU_START=3 bash run_liveavatar_5gpu_quick_test.sh  # GPU 3-7
#
# 资源估算: 5×H20, ~20GB VRAM/卡, ~30-40分钟
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
SAVE_DIR=${GROUP_BASE}/results/liveavatar/5gpu_quick_test

# ===== 环境设置 =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
export ENABLE_COMPILE=false

# GPU 配置: 5 张连续 GPU
GPU_START=${GPU_START:-0}
GPU_END=$((GPU_START + 4))
CUDA_DEVICES=""
for i in $(seq ${GPU_START} ${GPU_END}); do
    if [ -n "${CUDA_DEVICES}" ]; then
        CUDA_DEVICES="${CUDA_DEVICES},"
    fi
    CUDA_DEVICES="${CUDA_DEVICES}${i}"
done

# ===== 同步代码 =====
echo "Syncing code..."
mkdir -p ${LOCAL_CODE}
python ${SYNC_SCRIPT} ${GROUP_CODE} ${LOCAL_CODE} --workers 32 --skip-existing

# ===== 同步模型 (如果还没同步) =====
for MODEL_DIR in "${LOCAL_MODEL_BASE}" "${LOCAL_LORA_DIR}"; do
    if [ ! -f "${MODEL_DIR}/.sync_done" ]; then
        GROUP_DIR=$(echo ${MODEL_DIR} | sed "s|${LOCAL_BASE}|${GROUP_BASE}|")
        echo "Syncing model: ${GROUP_DIR} -> ${MODEL_DIR}"
        mkdir -p ${MODEL_DIR}
        python ${SYNC_SCRIPT} ${GROUP_DIR} ${MODEL_DIR} --workers 32 --skip-existing
        touch "${MODEL_DIR}/.sync_done"
    fi
done

mkdir -p ${SAVE_DIR}
cd ${LOCAL_CODE}

# 安装可能缺失的依赖
pip install dashscope bitsandbytes -q 2>/dev/null || true

# ===== 使用 LiveAvatar 自带示例 =====
EXAMPLE_IMAGE="${LOCAL_CODE}/examples/boy.jpg"
EXAMPLE_AUDIO="${LOCAL_CODE}/examples/boy.wav"
PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."

# 检查示例文件
if [ ! -f "${EXAMPLE_IMAGE}" ]; then
    EXAMPLE_IMAGE="${GROUP_CODE}/examples/boy.jpg"
    EXAMPLE_AUDIO="${GROUP_CODE}/examples/boy.wav"
fi

# boy.wav 参数:
#   时长: 5min52s = 352s
#   fps: 25 (s2v_14B.sample_fps 继承自 shared_config)
#   infer_frames: 48
#   clip_duration: 48/25 = 1.92s
#   num_clips: ceil(352/1.92) = 184
NUM_CLIP=184

echo ""
echo "============================================================"
echo "[5GPU] LiveAvatar 5卡 TPP 快速测试"
echo "  Image: ${EXAMPLE_IMAGE}"
echo "  Audio: ${EXAMPLE_AUDIO}"
echo "  Audio duration: 5min52s (352s)"
echo "  Clips: ${NUM_CLIP} (48 frames × 25fps = 1.92s/clip)"
echo "  Mode: 4 DiT GPU + 1 VAE GPU, FP8, 4 steps"
echo "  Resolution: 720x400"
echo "  GPUs: ${CUDA_DEVICES}"
echo "  Save: ${SAVE_DIR}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}

START_TIME=$(date +%s)

torchrun --nproc_per_node=5 --master_port=29099 \
    minimal_inference/s2v_streaming_interact.py \
    --task s2v-14B \
    --ckpt_dir "${LOCAL_MODEL_BASE}" \
    --size "720*400" \
    --base_seed 420 \
    --training_config ${TRAINING_CONFIG} \
    --offload_model False \
    --convert_model_dtype \
    --prompt "${PROMPT}" \
    --image "${EXAMPLE_IMAGE}" \
    --audio "${EXAMPLE_AUDIO}" \
    --infer_frames 48 \
    --load_lora \
    --lora_path_dmd "${LOCAL_LORA}" \
    --sample_steps 4 \
    --sample_guide_scale 0 \
    --num_clip ${NUM_CLIP} \
    --num_gpus_dit 4 \
    --sample_solver euler \
    --enable_vae_parallel \
    --save_dir "${SAVE_DIR}/" \
    --save_file "${SAVE_DIR}/boy_5gpu_352s.mp4" \
    --fp8

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "✅ 5GPU Quick Test 完成!"
echo "  耗时: ${ELAPSED}s ($((ELAPSED / 60))min)"
echo "  输出: ${SAVE_DIR}/boy_5gpu_352s.mp4"
echo "  音频: 5min52s (352s), ${NUM_CLIP} clips"
echo "  GPU: 5×H20 (${CUDA_DEVICES}), 4 DiT + 1 VAE, TPP, FP8"
echo "============================================================"
