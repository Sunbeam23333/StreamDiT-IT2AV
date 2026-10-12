#!/bin/bash
# ============================================================================
# LiveAvatar 快速推理测试 (单样本, 30秒短视频)
# 用于验证环境和模型是否正常工作
#
# 用法:
#   bash run_liveavatar_quick_test.sh 1gpu    # 单卡快测
#   bash run_liveavatar_quick_test.sh 5gpu    # 5卡快测
#
# 注意: 使用 LiveAvatar 自带的示例图片和音频
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
SAVE_DIR=${GROUP_BASE}/results/liveavatar/quick_test

# ===== 环境设置 =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
export ENABLE_COMPILE=false

# 解析参数
MODE=${1:-1gpu}

# 同步代码
echo "Syncing code..."
mkdir -p ${LOCAL_CODE}
python ${SYNC_SCRIPT} ${GROUP_CODE} ${LOCAL_CODE} --workers 32 --skip-existing

# 同步模型 (如果还没同步)
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

# 安装可能缺失的依赖 (静默安装, 已安装则跳过)
pip install dashscope bitsandbytes -q 2>/dev/null || true

# 使用 LiveAvatar 自带示例
EXAMPLE_IMAGE="${LOCAL_CODE}/examples/boy.jpg"
EXAMPLE_AUDIO="${LOCAL_CODE}/examples/boy.wav"
PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."

# 检查示例文件
if [ ! -f "${EXAMPLE_IMAGE}" ]; then
    # fallback: 检查 group 盘
    EXAMPLE_IMAGE="${GROUP_CODE}/examples/boy.jpg"
    EXAMPLE_AUDIO="${GROUP_CODE}/examples/boy.wav"
fi

echo ""
echo "============================================"
echo "LiveAvatar Quick Test - ${MODE}"
echo "  Image: ${EXAMPLE_IMAGE}"
echo "  Audio: ${EXAMPLE_AUDIO}"
echo "  Save: ${SAVE_DIR}"
echo "============================================"

START_TIME=$(date +%s)

case "${MODE}" in
    1gpu)
        export CUDA_VISIBLE_DEVICES=${CUDA_ID:-0}
        torchrun --nproc_per_node=1 --master_port=29099 \
            minimal_inference/s2v_streaming_interact.py \
            --task s2v-14B \
            --ckpt_dir "${LOCAL_MODEL_BASE}" \
            --size "704*384" \
            --base_seed 420 \
            --training_config ${TRAINING_CONFIG} \
            --offload_model True \
            --convert_model_dtype \
            --prompt "${PROMPT}" \
            --image "${EXAMPLE_IMAGE}" \
            --audio "${EXAMPLE_AUDIO}" \
            --infer_frames 48 \
            --load_lora \
            --lora_path_dmd "${LOCAL_LORA}" \
            --sample_steps 4 \
            --sample_guide_scale 0 \
            --num_clip 15 \
            --num_gpus_dit 1 \
            --sample_solver euler \
            --single_gpu \
            --save_dir "${SAVE_DIR}/" \
            --fp8
        ;;
    5gpu)
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4
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
            --num_clip 15 \
            --num_gpus_dit 4 \
            --sample_solver euler \
            --enable_vae_parallel \
            --save_dir "${SAVE_DIR}/" \
            --fp8
        ;;
    *)
        echo "Unknown mode: ${MODE}. Use 1gpu or 5gpu."
        exit 1
        ;;
esac

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================"
echo "✅ Quick Test 完成!"
echo "  模式: ${MODE}"
echo "  耗时: ${ELAPSED}s ($((ELAPSED / 60))min)"
echo "  输出: ${SAVE_DIR}"
echo "============================================"
