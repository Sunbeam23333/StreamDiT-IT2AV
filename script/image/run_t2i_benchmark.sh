#!/bin/bash
# ============================================================================
# Text-to-Image 文生图 集成推理脚本
#
# 服务器: 8卡 H20 95GB
# 框架:
#   - Diffusers (单卡推理, 适合调试和小批量)
#   - xDiT (8卡并行推理, 适合大批量生产)
#
# 输出分辨率: 720x1280 (landscape) - 适配 LightX2V Wan2.2 MoE i2v/flf2v
#
# 用法:
#   # 单卡 Diffusers - FLUX.1 schnell (最快)
#   bash run_t2i_benchmark.sh diffusers flux_schnell i2v
#
#   # 单卡 Diffusers - FLUX.1 dev (最高质量)
#   bash run_t2i_benchmark.sh diffusers flux_dev i2v
#
#   # 8卡 xDiT - FLUX.1 schnell
#   bash run_t2i_benchmark.sh xdit flux_schnell i2v
#
#   # 8卡 xDiT - SD3.5 Large
#   bash run_t2i_benchmark.sh xdit sd35 i2v
#
#   # 生成首帧+末帧 (用于 flf2v 视频生成)
#   bash run_t2i_benchmark.sh diffusers flux_schnell flf2v
#   bash run_t2i_benchmark.sh xdit flux_schnell flf2v
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

# ===== 参数解析 =====
FRAMEWORK=${1:-"diffusers"}      # diffusers 或 xdit
MODEL=${2:-"flux_schnell"}       # flux_schnell, flux_dev, sd35, pixart_sigma
MODE=${3:-"i2v"}                 # i2v (单帧) 或 flf2v (首帧+末帧)
NUM_GPUS=${4:-8}                 # GPU 数量 (仅 xdit 模式)

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}
GROUP_MODELS=${GROUP_BASE}/models/Image

LOCAL_BASE=${IT2AV_LOCAL_ROOT}
LOCAL_MODELS=${LOCAL_BASE}/models/Image
LOCAL_CODE=${LOCAL_BASE}/code

SYNC_SCRIPT=${GROUP_CODE}/fast_sync.py
DIFFUSERS_SCRIPT=${GROUP_CODE}/run_t2i_diffusers.py
XDIT_SCRIPT=${GROUP_CODE}/run_t2i_xdit.py

# 输出目录
SAVE_DIR=${GROUP_BASE}/results/t2i_${MODEL}_${MODE}

# ===== 模型映射 =====
declare -A MODEL_DIRS=(
    ["flux_schnell"]="FLUX.1-schnell"
    ["flux_dev"]="FLUX.1-dev"
    ["sd35"]="stable-diffusion-3.5-large"
    ["pixart_sigma"]="PixArt-Sigma-XL-2-1024-MS"
)

declare -A MODEL_STEPS=(
    ["flux_schnell"]="4"
    ["flux_dev"]="28"
    ["sd35"]="28"
    ["pixart_sigma"]="20"
)

declare -A MODEL_GUIDANCE=(
    ["flux_schnell"]="0.0"
    ["flux_dev"]="3.5"
    ["sd35"]="4.5"
    ["pixart_sigma"]="4.5"
)

declare -A MODEL_XDIT_TYPE=(
    ["flux_schnell"]="flux"
    ["flux_dev"]="flux"
    ["sd35"]="sd35"
    ["pixart_sigma"]="pixart_sigma"
)

MODEL_DIR_NAME=${MODEL_DIRS[$MODEL]}
STEPS=${MODEL_STEPS[$MODEL]}
GUIDANCE=${MODEL_GUIDANCE[$MODEL]}
XDIT_TYPE=${MODEL_XDIT_TYPE[$MODEL]}

if [ -z "$MODEL_DIR_NAME" ]; then
    echo "[ERROR] Unknown model: $MODEL"
    echo "Supported models: flux_schnell, flux_dev, sd35, pixart_sigma"
    exit 1
fi

# ===== 分辨率配置 (适配 LightX2V Wan2.2 MoE) =====
HEIGHT=720
WIDTH=1280

# ===== 测试 Prompts =====
declare -a PROMPTS=(
    "A white cat wearing sunglasses sits on a surfboard at a summer beach, waves gently rolling in the background under a blue sky with white clouds. The cat's fluffy fur sways in the breeze."
    "A young woman in a red dress walks through a field of sunflowers, her hair flowing in the wind as golden light illuminates the scene. Butterflies flutter around her."
    "A futuristic cityscape at sunset with flying cars weaving between gleaming skyscrapers, neon lights reflecting off glass surfaces. The sky transitions from orange to deep purple."
    "An astronaut floating in space with Earth visible in the background, the sun creating a brilliant halo effect around the planet. Stars twinkle in the darkness."
    "A traditional Chinese garden with a koi pond, cherry blossoms falling gently onto the water surface as a light breeze passes through bamboo groves."
    "A majestic eagle soaring over snow-capped mountains, its wings spread wide against a dramatic cloudy sky at golden hour."
    "A cozy coffee shop interior, steam rising from a freshly brewed latte, rain drops sliding down the window pane outside. Warm ambient lighting fills the space."
    "An underwater scene with colorful coral reefs, tropical fish swimming in formation, and rays of sunlight piercing through the ocean surface creating god rays."
    "A vintage steam locomotive rushing through a countryside landscape, smoke billowing from its chimney, autumn leaves swirling in its wake along railroad tracks."
    "A magical forest with bioluminescent mushrooms and fireflies, a deer standing in a misty clearing under moonlight. Particles of light float upward."
)

# 末帧 Prompts (用于 flf2v 模式)
declare -a LAST_PROMPTS=(
    "The white cat jumps off the surfboard and splashes into the crystal clear ocean water, creating a dramatic splash of water droplets catching the sunlight."
    "The woman reaches the edge of the sunflower field and turns around with a warm smile, the setting sun painting long golden shadows across the flowers."
    "The flying cars have descended and the city is now bathed in deep blue night with thousands of twinkling lights, a massive holographic advertisement appears between buildings."
    "The astronaut has drifted further away from Earth, now turning to face the vast emptiness of deep space with distant galaxies visible in the background."
    "A sudden gust of wind sends a flurry of cherry blossom petals swirling into the air above the pond, creating a pink vortex of natural beauty."
    "The eagle dives steeply toward a mountain lake, its talons extended, the water surface reflecting the dramatic amber sky."
    "The rain outside has stopped and a rainbow appears through the window, the latte cup is now empty with artistic foam patterns visible at the bottom."
    "A sea turtle glides majestically through the coral reef scene, while a school of silvery fish parts to make way, bioluminescent jellyfish floating in the background."
    "The steam locomotive has crossed a dramatic stone bridge over a deep valley, disappearing into a mountain tunnel with the last wisps of steam trailing behind."
    "The deer leaps gracefully through the magical forest, leaving trails of sparkling light, while the bioluminescent mushrooms pulse brighter in response to its movement."
)

echo "============================================"
echo "Text-to-Image Generation Benchmark"
echo "  Framework:  ${FRAMEWORK}"
echo "  Model:      ${MODEL} (${MODEL_DIR_NAME})"
echo "  Mode:       ${MODE}"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Steps:      ${STEPS}"
echo "  Guidance:   ${GUIDANCE}"
if [ "$FRAMEWORK" == "xdit" ]; then
    echo "  GPUs:       ${NUM_GPUS}"
fi
echo "============================================"

# ===== Step 1: 同步模型到本地盘 =====
echo ""
echo ">>> Step 1: Syncing model to local disk..."

GROUP_MODEL_PATH=${GROUP_MODELS}/${MODEL_DIR_NAME}
LOCAL_MODEL_PATH=${LOCAL_MODELS}/${MODEL_DIR_NAME}

if [ ! -d "$GROUP_MODEL_PATH" ]; then
    echo "[ERROR] Model not found at $GROUP_MODEL_PATH"
    echo "Please run: python download_t2i_models.py --models ${MODEL}"
    exit 1
fi

mkdir -p ${LOCAL_MODEL_PATH}

if [ ! -f "${LOCAL_MODEL_PATH}/.sync_done" ]; then
    echo ">>> Syncing ${MODEL_DIR_NAME} (${SYNC_WORKERS:-32} workers)..."
    python ${SYNC_SCRIPT} ${GROUP_MODEL_PATH} ${LOCAL_MODEL_PATH} \
        --workers ${SYNC_WORKERS:-32} --skip-existing
    touch ${LOCAL_MODEL_PATH}/.sync_done
    echo ">>> Model sync complete."
else
    echo ">>> Model already cached at ${LOCAL_MODEL_PATH}, skipping."
fi

# 同步 xDiT 和 diffusers 代码到本地 (如果走 xDiT)
if [ "$FRAMEWORK" == "xdit" ]; then
    LOCAL_XDIT=${LOCAL_CODE}/xDiT
    LOCAL_DIFFUSERS=${LOCAL_CODE}/diffusers
    mkdir -p ${LOCAL_XDIT} ${LOCAL_DIFFUSERS}

    echo ">>> Syncing xDiT code..."
    python ${SYNC_SCRIPT} ${GROUP_CODE}/xDiT ${LOCAL_XDIT} --workers 32 --skip-existing

    echo ">>> Syncing diffusers code..."
    python ${SYNC_SCRIPT} ${GROUP_CODE}/diffusers ${LOCAL_DIFFUSERS} --workers 32 --skip-existing
fi

echo ""
echo ">>> Step 2: Generating images..."

mkdir -p ${SAVE_DIR}

TOTAL_START=$(date +%s)

if [ "$FRAMEWORK" == "diffusers" ]; then
    # =========================================================
    # Diffusers 单卡推理
    # =========================================================

    # 设置 PYTHONPATH 使用本地 diffusers
    export PYTHONPATH=${GROUP_CODE}/utils/image/diffusers/src:$PYTHONPATH

    if [ "$MODE" == "flf2v" ]; then
        # 逐个生成首帧+末帧对
        for i in $(seq 0 $((${#PROMPTS[@]}-1))); do
            CASE_START=$(date +%s)
            echo ""
            echo ">>> Case $((i+1))/${#PROMPTS[@]} (flf2v mode)"
            echo ">>> First: ${PROMPTS[$i]:0:80}..."
            echo ">>> Last:  ${LAST_PROMPTS[$i]:0:80}..."

            python ${DIFFUSERS_SCRIPT} \
                --model ${MODEL} \
                --mode flf2v \
                --prompt "${PROMPTS[$i]}" \
                --last_prompt "${LAST_PROMPTS[$i]}" \
                --height ${HEIGHT} \
                --width ${WIDTH} \
                --steps ${STEPS} \
                --guidance_scale ${GUIDANCE} \
                --seed $((42+i)) \
                --output_dir "${SAVE_DIR}/case_$((i+1))" \
                --device cuda:0

            CASE_END=$(date +%s)
            echo ">>> Case $((i+1)) done in $((CASE_END - CASE_START))s"
        done
    else
        # 批量生成单帧图像
        # 创建临时 prompts 文件
        PROMPT_FILE=$(mktemp /tmp/t2i_prompts_XXXXX.txt)
        for p in "${PROMPTS[@]}"; do
            echo "$p" >> ${PROMPT_FILE}
        done

        python ${DIFFUSERS_SCRIPT} \
            --model ${MODEL} \
            --mode i2v \
            --prompt_file ${PROMPT_FILE} \
            --height ${HEIGHT} \
            --width ${WIDTH} \
            --steps ${STEPS} \
            --guidance_scale ${GUIDANCE} \
            --seed 42 \
            --output_dir "${SAVE_DIR}" \
            --device cuda:0

        rm -f ${PROMPT_FILE}
    fi

elif [ "$FRAMEWORK" == "xdit" ]; then
    # =========================================================
    # xDiT 多卡并行推理
    # =========================================================

    # 设置 PYTHONPATH
    export PYTHONPATH=${LOCAL_XDIT}:${LOCAL_DIFFUSERS}/src:$PYTHONPATH

    # 并行配置 (8卡 H20)
    # FLUX: pp=2, ulysses=2, ring=2 = 8 GPUs
    # SD3:  pp=2, ulysses=2, ring=2 = 8 GPUs
    # PixArt: pp=2, ulysses=2, ring=2 = 8 GPUs
    PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 2 --ring_degree 2"

    # 根据 GPU 数量调整并行配置
    if [ "$NUM_GPUS" == "1" ]; then
        PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 1 --ring_degree 1"
    elif [ "$NUM_GPUS" == "2" ]; then
        PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 1 --ring_degree 1"
    elif [ "$NUM_GPUS" == "4" ]; then
        PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 2 --ring_degree 1"
    fi

    for i in $(seq 0 $((${#PROMPTS[@]}-1))); do
        CASE_START=$(date +%s)
        echo ""
        echo ">>> Case $((i+1))/${#PROMPTS[@]} - Seed: $((42+i))"
        echo ">>> Prompt: ${PROMPTS[$i]:0:80}..."

        EXTRA_ARGS=""
        if [ "$MODE" == "flf2v" ]; then
            EXTRA_ARGS="--mode flf2v --last_prompt \"${LAST_PROMPTS[$i]}\" --last_seed_offset 10000"
        fi

        torchrun --nproc_per_node=${NUM_GPUS} ${XDIT_SCRIPT} \
            --model ${LOCAL_MODEL_PATH} \
            --model_type ${XDIT_TYPE} \
            ${PARALLEL_ARGS} \
            --height ${HEIGHT} \
            --width ${WIDTH} \
            --no_use_resolution_binning \
            --num_inference_steps ${STEPS} \
            --guidance_scale ${GUIDANCE} \
            --seed $((42+i)) \
            --prompt "${PROMPTS[$i]}" \
            --output_dir "${SAVE_DIR}" \
            --output_prefix "case_$((i+1))" \
            --warmup_steps 0 \
            ${EXTRA_ARGS}

        CASE_END=$(date +%s)
        echo ">>> Case $((i+1)) done in $((CASE_END - CASE_START))s"
    done
fi

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================"
echo "Benchmark completed!"
echo "  Total time:   ${TOTAL_TIME}s"
echo "  Total images: ${#PROMPTS[@]}"
if [ "$MODE" == "flf2v" ]; then
    echo "  Total frames: $((${#PROMPTS[@]} * 2)) (first + last)"
fi
echo "  Average:      $((TOTAL_TIME / ${#PROMPTS[@]}))s/case"
echo "  Output:       ${SAVE_DIR}"
echo "============================================"
echo ""
echo "Generated images can be used with LightX2V for i2v/flf2v video generation."
echo "Example:"
echo "  # i2v: use generated image as first frame"
echo "  torchrun --nproc_per_node=8 -m lightx2v.infer \\"
echo "    --model_cls wan2.2_moe --task i2v \\"
echo "    --image_path ${SAVE_DIR}/case_001_seed42.png ..."
echo ""
if [ "$MODE" == "flf2v" ]; then
    echo "  # flf2v: use first+last frames"
    echo "  torchrun --nproc_per_node=8 -m lightx2v.infer \\"
    echo "    --model_cls wan2.2_moe --task flf2v \\"
    echo "    --image_path ${SAVE_DIR}/case_1/case_001_first_frame.png \\"
    echo "    --last_frame_path ${SAVE_DIR}/case_1/case_001_last_frame.png ..."
fi
