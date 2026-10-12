#!/bin/bash
# ============================================================================
# HunyuanVideo-1.5 自研框架推理 Benchmark
# 服务器: H20 95GB (支持单卡/多卡)
#
# 当前支持的任务:
#   1. T2V 480p    (文生视频, 480p)
#   2. T2V 720p    (文生视频, 720p)
#   3. I2V 480p    (图生视频, 480p)
#   4. I2V 720p    (图生视频, 720p)
#
# 模型: tencent/HunyuanVideo-1.5 (8.3B 参数, DiT + 3D Causal VAE)
#
# 用法:
#   bash run_hyvideo15_native_benchmark.sh t2v_480p        # T2V 480p
#   bash run_hyvideo15_native_benchmark.sh t2v_720p        # T2V 720p
#   bash run_hyvideo15_native_benchmark.sh i2v_480p        # I2V 480p
#   bash run_hyvideo15_native_benchmark.sh i2v_720p        # I2V 720p
#   bash run_hyvideo15_native_benchmark.sh all              # 全部
#
# 环境变量:
#   NUM_GPUS=1     - GPU 数量 (默认 1, 单卡; 设 8 为 8 卡并行)
#   NUM_CASES=5    - 每个任务跑多少个 case (默认 5)
#   CUDA_ID=0      - 单卡模式下指定 GPU (默认 0)
#   ENABLE_SR=false  - 是否启用超分 (默认 false, 节省时间)
#   ENABLE_CACHE=true - 是否启用 DeepCache (默认 true)
#
# 注意:
#   - 单卡 H20 95GB 可以关闭 offloading (--offloading false)
#   - 多卡使用 torchrun sequence parallelism
#   - 121 帧, 50 steps, 24 fps
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

# ===== GPU 配置 =====
NUM_GPUS=${NUM_GPUS:-1}
if [ "${NUM_GPUS}" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_ID:-0}
else
    # 多卡: 使用 0,1,...,N-1
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES=${GPU_LIST}
fi

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}/utils/video/HunyuanVideo-1.5

LOCAL_BASE=${IT2AV_LOCAL_ROOT}
LOCAL_CODE=${LOCAL_BASE}/code/utils/video/HunyuanVideo-1.5
LOCAL_MODEL=${LOCAL_BASE}/models/HunyuanVideo-1.5

SYNC_SCRIPT=${IT2AV_CODE_ROOT}/fast_sync.py

# 测试图片 (I2V 用)
BEAUTY_IMG1=${GROUP_BASE}/data/pics/beauty1.png
BEAUTY_IMG2=${GROUP_BASE}/data/pics/beauty2.png

# 每个任务跑多少个 case
NUM_CASES=${NUM_CASES:-5}

# 功能开关
ENABLE_SR=${ENABLE_SR:-false}
ENABLE_CACHE=${ENABLE_CACHE:-true}
CACHE_TYPE=${CACHE_TYPE:-deepcache}

# ===== 解析命令行参数 =====
RUN_T2V_480P=false
RUN_T2V_720P=false
RUN_I2V_480P=false
RUN_I2V_720P=false

if [ $# -eq 0 ]; then
    echo "Usage: bash $0 {t2v_480p|t2v_720p|i2v_480p|i2v_720p|all} [...]"
    echo ""
    echo "HunyuanVideo-1.5 Tasks (自研框架):"
    echo "  t2v_480p  - Text-to-Video 480p"
    echo "  t2v_720p  - Text-to-Video 720p"
    echo "  i2v_480p  - Image-to-Video 480p"
    echo "  i2v_720p  - Image-to-Video 720p"
    echo "  all       - Run all tasks"
    echo ""
    echo "Env vars:"
    echo "  NUM_GPUS=1        - Number of GPUs (1=single, 8=8-way parallel)"
    echo "  CUDA_ID=0         - GPU index for single-GPU mode (default: 0)"
    echo "  NUM_CASES=5       - Number of cases per task (default: 5)"
    echo "  ENABLE_SR=false   - Enable super resolution (default: false)"
    echo "  ENABLE_CACHE=true - Enable DeepCache (default: true)"
    exit 1
fi

for arg in "$@"; do
    case "${arg}" in
        t2v_480p) RUN_T2V_480P=true ;;
        t2v_720p) RUN_T2V_720P=true ;;
        i2v_480p) RUN_I2V_480P=true ;;
        i2v_720p) RUN_I2V_720P=true ;;
        all)      RUN_T2V_480P=true; RUN_T2V_720P=true; RUN_I2V_480P=true; RUN_I2V_720P=true ;;
        *)        echo "Unknown argument: ${arg}"; exit 1 ;;
    esac
done

# ===== 同步代码和模型 =====
echo "============================================"
echo "Syncing HunyuanVideo-1.5 code and model..."
echo "============================================"
mkdir -p ${LOCAL_CODE}
mkdir -p ${LOCAL_MODEL}

# 同步代码
python ${SYNC_SCRIPT} ${GROUP_CODE} ${LOCAL_CODE} --workers ${SYNC_WORKERS:-32} --skip-existing
echo ">>> Code sync complete."

# 同步模型
GROUP_MODEL=${GROUP_BASE}/models/HunyuanVideo-1.5
if [ ! -f "${LOCAL_MODEL}/.sync_done" ]; then
    echo ">>> Syncing model: HunyuanVideo-1.5 ..."
    python ${SYNC_SCRIPT} ${GROUP_MODEL} ${LOCAL_MODEL} --workers ${SYNC_WORKERS:-32} --skip-existing
    touch ${LOCAL_MODEL}/.sync_done
    echo ">>> Model sync complete."
else
    echo ">>> Model already cached: HunyuanVideo-1.5"
fi

# ===== 环境设置 =====
export PYTHONPATH=${LOCAL_CODE}:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "==============================================================================="
echo "HunyuanVideo-1.5 Native Framework Environment:"
echo "  Code: ${LOCAL_CODE}"
echo "  Model: ${LOCAL_MODEL}"
echo "  GPUs: ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "  SR: ${ENABLE_SR}, Cache: ${ENABLE_CACHE} (${CACHE_TYPE})"
echo "==============================================================================="

# ===== 运行命令封装 =====
# H20 95GB 单卡可以关闭 offloading 获得更快速度
# 多卡时也关闭 offloading (有足够显存)
if [ "${NUM_GPUS}" -gt 1 ]; then
    OFFLOADING=false
    GROUP_OFFLOADING=false
else
    # 单卡 H20 95GB: 8.3B 模型 bf16 约 16.6GB, 加上 text encoder / VAE 约 30-40GB
    # 95GB 足够, 关闭 offloading 更快
    OFFLOADING=false
    GROUP_OFFLOADING=false
fi

run_generate() {
    local resolution=$1
    local task=$2          # t2v or i2v
    local prompt="$3"
    local output_path=$4
    local seed=$5
    local image_path=${6:-none}

    local extra_args=""
    if [ "${task}" = "i2v" ] && [ "${image_path}" != "none" ]; then
        extra_args="--image_path ${image_path}"
    fi

    cd ${LOCAL_CODE}

    torchrun --nproc_per_node=${NUM_GPUS} generate.py \
        --prompt "${prompt}" \
        ${extra_args} \
        --resolution ${resolution} \
        --aspect_ratio 16:9 \
        --seed ${seed} \
        --rewrite false \
        --cfg_distilled false \
        --enable_step_distill false \
        --sparse_attn false \
        --use_sageattn false \
        --offloading ${OFFLOADING} \
        --group_offloading ${GROUP_OFFLOADING} \
        --overlap_group_offloading false \
        --enable_cache ${ENABLE_CACHE} \
        --cache_type ${CACHE_TYPE} \
        --sr ${ENABLE_SR} \
        --output_path "${output_path}" \
        --model_path "${LOCAL_MODEL}"
}

# ===== T2V/I2V Prompts =====
declare -a PROMPTS=(
    "A white cat wearing sunglasses sits on a surfboard at a summer beach, waves gently rolling in the background under a blue sky with white clouds. The cat's fluffy fur sways in the breeze."
    "A young woman in a red dress walks through a field of sunflowers, her hair flowing in the wind as golden light illuminates the scene. Butterflies flutter around her."
    "A futuristic cityscape at sunset with flying cars weaving between gleaming skyscrapers, neon lights reflecting off glass surfaces. The sky transitions from orange to deep purple."
    "An astronaut floating in space with Earth visible in the background, the sun creating a brilliant halo effect around the planet. Stars twinkle in the darkness."
    "A traditional Chinese garden with a koi pond, cherry blossoms falling gently onto the water surface as a light breeze passes through bamboo groves."
    "A majestic eagle soaring over snow-capped mountains, its wings spread wide against a dramatic cloudy sky at golden hour. The bird banks and dives."
    "A cozy coffee shop interior, steam rising from a freshly brewed latte, rain drops sliding down the window pane outside. Warm ambient lighting fills the space."
    "An underwater scene with colorful coral reefs, tropical fish swimming in formation, and rays of sunlight piercing through the ocean surface creating god rays."
    "A vintage steam locomotive rushing through a countryside landscape, smoke billowing from its chimney, autumn leaves swirling in its wake along railroad tracks."
    "A magical forest with bioluminescent mushrooms and fireflies, a deer standing in a misty clearing under moonlight. Particles of light float upward."
    "A professional chef preparing sushi in a modern kitchen, precise knife movements slicing fresh salmon with elegant skill. The blade gleams under spotlights."
    "A group of hot air balloons rising at dawn over Cappadocia, the rocky landscape bathed in warm pink and orange light. Shadows stretch across valleys."
    "An ancient Egyptian temple with hieroglyphics illuminated by torch light, sand particles floating in the beams of light entering through stone columns."
    "A skateboard performing tricks in a neon-lit urban halfpipe, dynamic motion with trailing light effects and street art background. Camera follows the action."
    "A peaceful zen garden with raked sand patterns, a bamboo fountain trickling water, and morning mist gradually clearing to reveal stone arrangements."
    "A powerful waterfall cascading into a turquoise pool, surrounded by lush tropical vegetation and rainbow mist. Birds fly through the spray."
    "A time-lapse of a flower blooming, petals unfurling in slow motion to reveal intricate patterns and vibrant colors. Dewdrops glisten on the petals."
    "A classic car cruising along a coastal highway at sunset, ocean waves crashing against rocks below, seagulls flying overhead in formation."
    "A bustling night market in Southeast Asia, colorful lanterns swaying in the breeze, vendors preparing street food with sizzling woks and aromatic steam."
    "A lone astronaut exploring the surface of Mars, red dust swirling around boots with each step, a rover visible in the distance under an alien sky."
)

declare -a I2V_IMAGES=(
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
)

# ======================================================================
# T2V 480p
# ======================================================================
run_t2v_480p() {
    local save_dir=${GROUP_BASE}/results/hyvideo15_native_t2v_480p_${NUM_GPUS}gpu
    mkdir -p ${save_dir}

    echo ""
    echo "============================================"
    echo "[T2V 480p] Model: HunyuanVideo-1.5 (${NUM_GPUS} GPU)"
    echo "  Resolution: 480p, 121 frames, 50 steps"
    echo "  Offloading: ${OFFLOADING}"
    echo "  Cases: ${NUM_CASES}"
    echo "============================================"

    local START_TIME=$(date +%s)

    for i in $(seq 0 $((NUM_CASES - 1))); do
        local CASE_START=$(date +%s)
        echo ""
        echo ">>> [T2V 480p] Case $((i+1))/${NUM_CASES} - Seed: $((42+i))"
        echo ">>> Prompt: ${PROMPTS[$i]:0:80}..."

        run_generate 480p t2v "${PROMPTS[$i]}" "${save_dir}/case_$((i+1)).mp4" $((42+i))

        local CASE_END=$(date +%s)
        echo ">>> [T2V 480p] Case $((i+1)) done in $((CASE_END - CASE_START))s"
    done

    local END_TIME=$(date +%s)
    echo ""
    echo "[T2V 480p] Total: $((END_TIME - START_TIME))s, Avg: $(( (END_TIME - START_TIME) / NUM_CASES ))s/case"
}

# ======================================================================
# T2V 720p
# ======================================================================
run_t2v_720p() {
    local save_dir=${GROUP_BASE}/results/hyvideo15_native_t2v_720p_${NUM_GPUS}gpu
    mkdir -p ${save_dir}

    echo ""
    echo "============================================"
    echo "[T2V 720p] Model: HunyuanVideo-1.5 (${NUM_GPUS} GPU)"
    echo "  Resolution: 720p, 121 frames, 50 steps"
    echo "  Offloading: ${OFFLOADING}"
    echo "  Cases: ${NUM_CASES}"
    echo "============================================"

    local START_TIME=$(date +%s)

    for i in $(seq 0 $((NUM_CASES - 1))); do
        local CASE_START=$(date +%s)
        echo ""
        echo ">>> [T2V 720p] Case $((i+1))/${NUM_CASES} - Seed: $((42+i))"
        echo ">>> Prompt: ${PROMPTS[$i]:0:80}..."

        run_generate 720p t2v "${PROMPTS[$i]}" "${save_dir}/case_$((i+1)).mp4" $((42+i))

        local CASE_END=$(date +%s)
        echo ">>> [T2V 720p] Case $((i+1)) done in $((CASE_END - CASE_START))s"
    done

    local END_TIME=$(date +%s)
    echo ""
    echo "[T2V 720p] Total: $((END_TIME - START_TIME))s, Avg: $(( (END_TIME - START_TIME) / NUM_CASES ))s/case"
}

# ======================================================================
# I2V 480p
# ======================================================================
run_i2v_480p() {
    local save_dir=${GROUP_BASE}/results/hyvideo15_native_i2v_480p_${NUM_GPUS}gpu
    mkdir -p ${save_dir}

    echo ""
    echo "============================================"
    echo "[I2V 480p] Model: HunyuanVideo-1.5 (${NUM_GPUS} GPU)"
    echo "  Resolution: 480p, 121 frames, 50 steps"
    echo "  Offloading: ${OFFLOADING}"
    echo "  Cases: ${NUM_CASES}"
    echo "============================================"

    local START_TIME=$(date +%s)

    for i in $(seq 0 $((NUM_CASES - 1))); do
        local CASE_START=$(date +%s)
        echo ""
        echo ">>> [I2V 480p] Case $((i+1))/${NUM_CASES} - Seed: $((42+i))"
        echo ">>> Image: ${I2V_IMAGES[$i]}"
        echo ">>> Prompt: ${PROMPTS[$i]:0:80}..."

        run_generate 480p i2v "${PROMPTS[$i]}" "${save_dir}/case_$((i+1)).mp4" $((42+i)) "${I2V_IMAGES[$i]}"

        local CASE_END=$(date +%s)
        echo ">>> [I2V 480p] Case $((i+1)) done in $((CASE_END - CASE_START))s"
    done

    local END_TIME=$(date +%s)
    echo ""
    echo "[I2V 480p] Total: $((END_TIME - START_TIME))s, Avg: $(( (END_TIME - START_TIME) / NUM_CASES ))s/case"
}

# ======================================================================
# I2V 720p
# ======================================================================
run_i2v_720p() {
    local save_dir=${GROUP_BASE}/results/hyvideo15_native_i2v_720p_${NUM_GPUS}gpu
    mkdir -p ${save_dir}

    echo ""
    echo "============================================"
    echo "[I2V 720p] Model: HunyuanVideo-1.5 (${NUM_GPUS} GPU)"
    echo "  Resolution: 720p, 121 frames, 50 steps"
    echo "  Offloading: ${OFFLOADING}"
    echo "  Cases: ${NUM_CASES}"
    echo "============================================"

    local START_TIME=$(date +%s)

    for i in $(seq 0 $((NUM_CASES - 1))); do
        local CASE_START=$(date +%s)
        echo ""
        echo ">>> [I2V 720p] Case $((i+1))/${NUM_CASES} - Seed: $((42+i))"
        echo ">>> Image: ${I2V_IMAGES[$i]}"
        echo ">>> Prompt: ${PROMPTS[$i]:0:80}..."

        run_generate 720p i2v "${PROMPTS[$i]}" "${save_dir}/case_$((i+1)).mp4" $((42+i)) "${I2V_IMAGES[$i]}"

        local CASE_END=$(date +%s)
        echo ">>> [I2V 720p] Case $((i+1)) done in $((CASE_END - CASE_START))s"
    done

    local END_TIME=$(date +%s)
    echo ""
    echo "[I2V 720p] Total: $((END_TIME - START_TIME))s, Avg: $(( (END_TIME - START_TIME) / NUM_CASES ))s/case"
}


# ===== 执行选定的任务 =====
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  HunyuanVideo-1.5 Native Framework Benchmark        ║"
echo "║  GPUs: ${NUM_GPUS}x H20 95GB                                 ║"
echo "║  Cases per task: ${NUM_CASES}                                  ║"
echo "╚══════════════════════════════════════════════════════╝"

GLOBAL_START=$(date +%s)

if [ "${RUN_T2V_480P}" = true ]; then
    run_t2v_480p
fi

if [ "${RUN_T2V_720P}" = true ]; then
    run_t2v_720p
fi

if [ "${RUN_I2V_480P}" = true ]; then
    run_i2v_480p
fi

if [ "${RUN_I2V_720P}" = true ]; then
    run_i2v_720p
fi

GLOBAL_END=$(date +%s)

echo ""
echo "============================================"
echo "All HunyuanVideo-1.5 benchmarks completed!"
echo "Total wall time: $((GLOBAL_END - GLOBAL_START))s"
echo "============================================"
