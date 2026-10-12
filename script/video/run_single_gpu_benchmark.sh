#!/bin/bash
# ============================================================================
# 单卡 H20 推理 Benchmark - 支持多模型/多任务
# 服务器: 单卡 H20 95GB (cpu_offload 模式)
#
# 当前支持的模型 (Wan2.2 系列):
#   1. Wan2.2-T2V-A14B        (T2V 文生视频)
#   2. Wan2.2-I2V-A14B        (I2V 图生视频)
#   3. Wan2.2-VACE-Fun-A14B   (VACE FLF2V 首尾帧生视频)
#
# 后续可扩展: Wan2.1, HunyuanVideo 等
#
# 用法:
#   bash run_single_gpu_benchmark.sh t2v          # Wan2.2-T2V-A14B
#   bash run_single_gpu_benchmark.sh i2v          # Wan2.2-I2V-A14B
#   bash run_single_gpu_benchmark.sh vace         # Wan2.2-VACE-Fun-A14B
#   bash run_single_gpu_benchmark.sh all          # 跑全部 Wan2.2 模型
#   bash run_single_gpu_benchmark.sh t2v i2v      # 跑 T2V + I2V
#
# 注意:
#   - 单卡模式使用 cpu_offload, 高/低噪声模型交替加载到GPU
#   - 720p 81帧, 40 steps (与 8 卡配置一致, 只是速度慢)
#   - 使用 python 直接运行, 不需要 torchrun
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

# ===== 选择 GPU =====
export CUDA_VISIBLE_DEVICES=${CUDA_ID:-0}

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}/utils/video/LightX2V-main

LOCAL_BASE=${IT2AV_LOCAL_ROOT}
LOCAL_CODE=${LOCAL_BASE}/code/utils/video/LightX2V-main

lightx2v_path=${LOCAL_CODE}

SYNC_SCRIPT=${IT2AV_CODE_ROOT}/fast_sync.py
PREP_SCRIPT=${IT2AV_CODE_ROOT}/prepare_vace_flf2v.py

# 测试图片
BEAUTY_IMG1=${GROUP_BASE}/data/pics/beauty1.png
BEAUTY_IMG2=${GROUP_BASE}/data/pics/beauty2.png

# 通用负提示词
NEG_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# 每个模型跑多少个 case (默认 5, 可通过环境变量覆盖)
NUM_CASES=${NUM_CASES:-5}

# ===== 解析命令行参数 =====
RUN_T2V=false
RUN_I2V=false
RUN_VACE=false

if [ $# -eq 0 ]; then
    echo "Usage: bash $0 {t2v|i2v|vace|all} [t2v|i2v|vace] ..."
    echo ""
    echo "Wan2.2 Models:"
    echo "  t2v   - Wan2.2-T2V-A14B       (text-to-video)"
    echo "  i2v   - Wan2.2-I2V-A14B       (image-to-video)"
    echo "  vace  - Wan2.2-VACE-Fun-A14B  (first-last-frame-to-video)"
    echo "  all   - Run all Wan2.2 models above"
    echo ""
    echo "Env vars:"
    echo "  CUDA_ID=0       - GPU index (default: 0)"
    echo "  NUM_CASES=5     - Number of cases per model (default: 5)"
    exit 1
fi

for arg in "$@"; do
    case "${arg}" in
        t2v)  RUN_T2V=true ;;
        i2v)  RUN_I2V=true ;;
        vace) RUN_VACE=true ;;
        all)  RUN_T2V=true; RUN_I2V=true; RUN_VACE=true ;;
        *)    echo "Unknown argument: ${arg}"; exit 1 ;;
    esac
done

# ===== 同步代码 (始终需要) =====
echo "============================================"
echo "Syncing code to local disk..."
echo "============================================"
mkdir -p ${LOCAL_CODE}
python ${SYNC_SCRIPT} ${GROUP_CODE} ${LOCAL_CODE} --workers ${SYNC_WORKERS:-32} --skip-existing
echo ">>> Code sync complete."

# ===== 环境设置 (内联 base.sh, 避免全局要求 model_path) =====
# model_path 在每个 run_xxx() 函数内部按具体模型设置
export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DTYPE=${DTYPE:-BF16}
export SENSITIVE_LAYER_DTYPE=${SENSITIVE_LAYER_DTYPE:-None}
export PROFILING_DEBUG_LEVEL=${PROFILING_DEBUG_LEVEL:-2}

echo "==============================================================================="
echo "LightX2V Environment:"
echo "  lightx2v_path: ${lightx2v_path}"
echo "  DTYPE: ${DTYPE}, SENSITIVE_LAYER_DTYPE: ${SENSITIVE_LAYER_DTYPE}"
echo "  PROFILING_DEBUG_LEVEL: ${PROFILING_DEBUG_LEVEL}"
echo "==============================================================================="

# ===== 同步指定模型 =====
sync_model() {
    local model_name=$1
    local group_model=${GROUP_BASE}/models/${model_name}
    local local_model=${LOCAL_BASE}/models/${model_name}

    mkdir -p ${local_model}

    if [ ! -f "${local_model}/.sync_done" ]; then
        echo ">>> Syncing model: ${model_name} ..."
        python ${SYNC_SCRIPT} ${group_model} ${local_model} --workers ${SYNC_WORKERS:-32} --skip-existing
        touch ${local_model}/.sync_done
        echo ">>> Model sync complete: ${model_name}"
    else
        echo ">>> Model already cached: ${model_name}"
    fi
}

# ===== T2V Prompts =====
declare -a T2V_PROMPTS=(
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

# ===== I2V Prompts + Images =====
declare -a I2V_PROMPTS=(
    "A white cat wearing sunglasses sits on a surfboard at a summer beach, waves gently rolling in the background under a blue sky with white clouds."
    "A young woman in a red dress walks through a field of sunflowers, her hair flowing in the wind as golden light illuminates the scene."
    "A futuristic cityscape at sunset with flying cars weaving between gleaming skyscrapers, neon lights reflecting off glass surfaces."
    "An astronaut floating in space, Earth visible in the background with the sun creating a brilliant halo effect around the planet."
    "A traditional Chinese garden with a koi pond, cherry blossoms falling gently onto the water surface as a light breeze passes."
    "A majestic eagle soaring over snow-capped mountains, its wings spread wide against a dramatic cloudy sky at golden hour."
    "A cozy coffee shop interior, steam rising from a freshly brewed latte, rain drops sliding down the window pane outside."
    "An underwater scene with colorful coral reefs, tropical fish swimming in formation, and rays of sunlight piercing through the ocean surface."
    "A vintage steam locomotive rushing through a countryside landscape, smoke billowing from its chimney, autumn leaves swirling in its wake."
    "A magical forest with bioluminescent mushrooms and fireflies, a deer standing in a misty clearing under moonlight."
    "A beautiful woman posing gracefully, her hair gently moving in a light breeze, warm golden lighting creating soft shadows on her face."
    "A gorgeous woman with elegant features shifts her pose, the fabric of her dress flowing with the movement, cinematic lighting."
    "A professional chef preparing sushi in a modern kitchen, precise knife movements slicing fresh salmon with elegant skill."
    "A group of hot air balloons rising at dawn over Cappadocia, the rocky landscape bathed in warm pink and orange light."
    "An ancient Egyptian temple with hieroglyphics illuminated by torch light, sand particles floating in the beams of light."
    "A skateboard performing tricks in a neon-lit urban halfpipe, dynamic motion with trailing light effects and street art background."
    "A peaceful zen garden with raked sand patterns, a bamboo fountain trickling water, and morning mist gradually clearing."
    "A powerful waterfall cascading into a turquoise pool, surrounded by lush tropical vegetation and rainbow mist."
    "A stunning woman smiles softly at the camera, warm ambient light highlighting her features, the background gently blurred."
    "A lone astronaut exploring the surface of Mars, red dust swirling around boots, a rover visible in the distance."
)

declare -a I2V_IMAGES=(
    "${lightx2v_path}/assets/inputs/imgs/img_0.jpg"
    "${lightx2v_path}/assets/inputs/imgs/img_1.jpg"
    "${lightx2v_path}/assets/inputs/imgs/img_2.jpg"
    "${lightx2v_path}/assets/inputs/imgs/girl.png"
    "${lightx2v_path}/assets/inputs/imgs/woman.jpeg"
    "${lightx2v_path}/assets/inputs/imgs/snake.png"
    "${lightx2v_path}/assets/inputs/imgs/img_0.jpg"
    "${lightx2v_path}/assets/inputs/imgs/img_1.jpg"
    "${lightx2v_path}/assets/inputs/imgs/img_2.jpg"
    "${lightx2v_path}/assets/inputs/imgs/girl.png"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
    "${lightx2v_path}/assets/inputs/imgs/woman.jpeg"
    "${lightx2v_path}/assets/inputs/imgs/snake.png"
    "${lightx2v_path}/assets/inputs/imgs/img_0.jpg"
    "${lightx2v_path}/assets/inputs/imgs/img_1.jpg"
    "${lightx2v_path}/assets/inputs/imgs/img_2.jpg"
    "${lightx2v_path}/assets/inputs/imgs/girl.png"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
)

# ===== VACE FLF2V Prompts =====
declare -a VACE_PROMPTS=(
    "A beautiful woman gracefully turns her head, her long hair flowing with the movement, soft lighting accentuating her features."
    "A gorgeous woman slowly smiles, her expression transitioning from serene to radiant, with gentle ambient light."
    "A stunning woman tilts her head slightly, her eyes gazing softly at the camera, hair gently swaying in a light breeze."
    "A beautiful woman raises her hand to brush her hair behind her ear, the motion slow and elegant, warm golden lighting."
    "A woman with delicate features shifts her pose gracefully, the camera capturing the smooth transition of her movement."
    "A gorgeous woman turns from profile to face the camera, her expression warm and inviting, soft bokeh background."
    "A beautiful woman leans forward slightly, her hair cascading over her shoulder, gentle smile forming on her lips."
    "A stunning woman moves her arm gracefully, the fabric of her clothing flowing with the motion, soft diffused lighting."
    "A woman with elegant features slowly turns, the light catching her skin beautifully, creating a dreamy cinematic look."
    "A beautiful woman shifts her weight and changes pose, the transition smooth and natural, with warm color tones."
    "A gorgeous woman reaches up to touch her face gently, the movement slow and deliberate, professional studio lighting."
    "A stunning woman looks down then raises her gaze to the camera, her expression shifting from thoughtful to confident."
    "A beautiful woman sways gently, her body language expressing grace, with soft shadows and warm highlights."
    "A woman with radiant features tilts her chin upward, the light playing across her face, creating elegant shadows."
    "A gorgeous woman turns her body slightly, revealing a different angle, the transition is smooth and cinematic."
    "A beautiful woman adjusts her posture with fluid motion, the scene bathed in warm ambient light, soft focus background."
    "A stunning woman's hair moves gently as she shifts position, the camera capturing every detail in slow motion."
    "A woman with striking features transitions her expression, the mood shifting from mysterious to cheerful, golden hour light."
    "A beautiful woman extends her hand forward, the gesture elegant and inviting, with beautiful depth of field."
    "A gorgeous woman completes a graceful turn, her movement natural and flowing, cinematic lighting throughout."
)


# ======================================================================
# T2V: Wan2.2-T2V-A14B (单卡, cpu_offload)
# ======================================================================
run_t2v() {
    local MODEL_NAME="Wan2.2-T2V-A14B"
    export model_path=${LOCAL_BASE}/models/${MODEL_NAME}
    local save_dir=${GROUP_BASE}/results/wan22_t2v_1gpu
    local CONFIG_JSON=${GROUP_CODE}/configs/wan22/wan_moe_t2v_h20_1gpu.json

    echo ""
    echo "============================================"
    echo "[T2V] Model: ${MODEL_NAME} (单卡推理)"
    echo "  Model path: ${model_path}"
    echo "  Config: 720p, 81 frames, 40 steps"
    echo "  CPU offload: enabled"
    echo "  Cases: ${NUM_CASES}"
    echo "============================================"

    sync_model ${MODEL_NAME}
    mkdir -p ${save_dir}

    local START_TIME=$(date +%s)

    for i in $(seq 0 $((NUM_CASES - 1))); do
        local CASE_START=$(date +%s)
        echo ""
        echo ">>> [T2V] Case $((i+1))/${NUM_CASES} - Seed: $((42+i))"
        echo ">>> Prompt: ${T2V_PROMPTS[$i]:0:80}..."

        python -m lightx2v.infer \
            --model_cls wan2.2_moe \
            --task t2v \
            --model_path ${model_path} \
            --config_json ${CONFIG_JSON} \
            --seed $((42+i)) \
            --prompt "${T2V_PROMPTS[$i]}" \
            --negative_prompt "${NEG_PROMPT}" \
            --save_result_path "${save_dir}/case_$((i+1)).mp4"

        local CASE_END=$(date +%s)
        echo ">>> [T2V] Case $((i+1)) done in $((CASE_END - CASE_START))s"
    done

    local END_TIME=$(date +%s)
    echo ""
    echo "[T2V] Total: $((END_TIME - START_TIME))s, Avg: $(( (END_TIME - START_TIME) / NUM_CASES ))s/case"
}


# ======================================================================
# I2V: Wan2.2-I2V-A14B (单卡, cpu_offload)
# ======================================================================
run_i2v() {
    local MODEL_NAME="Wan2.2-I2V-A14B"
    export model_path=${LOCAL_BASE}/models/${MODEL_NAME}
    local save_dir=${GROUP_BASE}/results/wan22_i2v_1gpu
    local CONFIG_JSON=${GROUP_CODE}/configs/wan22/wan_moe_i2v_h20_1gpu.json

    echo ""
    echo "============================================"
    echo "[I2V] Model: ${MODEL_NAME} (单卡推理)"
    echo "  Model path: ${model_path}"
    echo "  Config: 720p, 81 frames, 40 steps"
    echo "  CPU offload: enabled"
    echo "  Cases: ${NUM_CASES}"
    echo "============================================"

    sync_model ${MODEL_NAME}
    mkdir -p ${save_dir}

    local START_TIME=$(date +%s)

    for i in $(seq 0 $((NUM_CASES - 1))); do
        local CASE_START=$(date +%s)
        echo ""
        echo ">>> [I2V] Case $((i+1))/${NUM_CASES} - Seed: $((42+i))"
        echo ">>> Image: ${I2V_IMAGES[$i]}"
        echo ">>> Prompt: ${I2V_PROMPTS[$i]:0:80}..."

        python -m lightx2v.infer \
            --model_cls wan2.2_moe \
            --task i2v \
            --model_path ${model_path} \
            --config_json ${CONFIG_JSON} \
            --seed $((42+i)) \
            --prompt "${I2V_PROMPTS[$i]}" \
            --negative_prompt "${NEG_PROMPT}" \
            --image_path "${I2V_IMAGES[$i]}" \
            --save_result_path "${save_dir}/case_$((i+1)).mp4"

        local CASE_END=$(date +%s)
        echo ">>> [I2V] Case $((i+1)) done in $((CASE_END - CASE_START))s"
    done

    local END_TIME=$(date +%s)
    echo ""
    echo "[I2V] Total: $((END_TIME - START_TIME))s, Avg: $(( (END_TIME - START_TIME) / NUM_CASES ))s/case"
}


# ======================================================================
# VACE: Wan2.2-VACE-Fun-A14B (单卡, cpu_offload)
# ======================================================================
run_vace() {
    local MODEL_NAME="Wan2.2-VACE-Fun-A14B"
    export model_path=${LOCAL_BASE}/models/${MODEL_NAME}
    local save_dir=${GROUP_BASE}/results/wan22_vace_1gpu
    local CONFIG_JSON=${GROUP_CODE}/configs/wan22_vace/h20_1gpu/wan22_moe_vace_h20_1gpu.json
    local VACE_PREP_DIR=${GROUP_BASE}/data/vace_flf2v_prep

    echo ""
    echo "============================================"
    echo "[VACE] Model: ${MODEL_NAME} (单卡推理, FLF2V)"
    echo "  Model path: ${model_path}"
    echo "  Config: 720p, 81 frames, 40 steps"
    echo "  CPU offload: enabled"
    echo "  First frame: ${BEAUTY_IMG1}"
    echo "  Last frame:  ${BEAUTY_IMG2}"
    echo "  Cases: ${NUM_CASES}"
    echo "============================================"

    sync_model ${MODEL_NAME}
    mkdir -p ${save_dir}

    # Step 1: 预处理首尾帧
    echo ">>> Preparing VACE FLF2V inputs..."
    rm -f "${VACE_PREP_DIR}/src_video.mp4" "${VACE_PREP_DIR}/src_mask.mp4"
    python ${PREP_SCRIPT} \
        --first_frame "${BEAUTY_IMG1}" \
        --last_frame "${BEAUTY_IMG2}" \
        --output_dir "${VACE_PREP_DIR}" \
        --height 720 --width 1280 --num_frames 81 --fps 16
    echo ">>> VACE inputs prepared."

    local SRC_VIDEO=${VACE_PREP_DIR}/src_video.mp4
    local SRC_MASK=${VACE_PREP_DIR}/src_mask.mp4

    local START_TIME=$(date +%s)

    for i in $(seq 0 $((NUM_CASES - 1))); do
        local CASE_START=$(date +%s)
        echo ""
        echo ">>> [VACE] Case $((i+1))/${NUM_CASES} - Seed: $((42+i))"
        echo ">>> Prompt: ${VACE_PROMPTS[$i]:0:80}..."

        python -m lightx2v.infer \
            --model_cls wan2.2_moe_vace \
            --task vace \
            --model_path ${model_path} \
            --config_json ${CONFIG_JSON} \
            --seed $((42+i)) \
            --prompt "${VACE_PROMPTS[$i]}" \
            --negative_prompt "${NEG_PROMPT}" \
            --src_ref_images "${BEAUTY_IMG1}" \
            --src_video "${SRC_VIDEO}" \
            --src_mask "${SRC_MASK}" \
            --save_result_path "${save_dir}/case_$((i+1)).mp4"

        local CASE_END=$(date +%s)
        echo ">>> [VACE] Case $((i+1)) done in $((CASE_END - CASE_START))s"
    done

    local END_TIME=$(date +%s)
    echo ""
    echo "[VACE] Total: $((END_TIME - START_TIME))s, Avg: $(( (END_TIME - START_TIME) / NUM_CASES ))s/case"
}


# ===== 执行选定的任务 =====
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  LightX2V Single GPU Benchmark              ║"
echo "║  GPU: CUDA:${CUDA_VISIBLE_DEVICES}  (H20 95GB)                  ║"
echo "║  Cases per model: ${NUM_CASES}                         ║"
echo "╚══════════════════════════════════════════════╝"

GLOBAL_START=$(date +%s)

if [ "${RUN_T2V}" = true ]; then
    run_t2v
fi

if [ "${RUN_I2V}" = true ]; then
    run_i2v
fi

if [ "${RUN_VACE}" = true ]; then
    run_vace
fi

GLOBAL_END=$(date +%s)

echo ""
echo "============================================"
echo "All benchmarks completed!"
echo "Total wall time: $((GLOBAL_END - GLOBAL_START))s"
echo "============================================"
