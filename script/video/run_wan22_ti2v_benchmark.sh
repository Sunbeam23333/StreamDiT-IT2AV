#!/bin/bash
# ============================================================================
# Wan2.2 TI2V-5B I2V 推理 - 20 cases benchmark (8卡并行)
# 服务器: 8卡 H20 95GB
# 模型: Wan2.2-TI2V-5B (5B Dense, 非MoE, 高压缩VAE stride=[4,16,16])
# 并行: cfg_parallel(2) × seq_parallel_ulysses(4) = 8 GPUs
#
# 注意: TI2V 支持 T2V 和 I2V 两种模式, 这里使用 I2V 模式
#       TI2V 使用高压缩 VAE (stride=[4,16,16], channels=48)
#       分辨率: 704x1280, 帧数: 121, FPS: 24
#       model_cls = wan2.2 (Wan22DenseRunner)
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

# ===== 路径配置 =====
# group 盘源路径（网络存储）
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_MODEL=${GROUP_BASE}/models/Wan2.2-TI2V-5B
GROUP_CODE=${IT2AV_CODE_ROOT}/utils/video/LightX2V-main

# 本地盘缓存路径（高速读取）
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
LOCAL_MODEL=${LOCAL_BASE}/models/Wan2.2-TI2V-5B
LOCAL_CODE=${LOCAL_BASE}/code/utils/video/LightX2V-main

# 实际使用的路径（指向本地盘）
lightx2v_path=${LOCAL_CODE}
model_path=${LOCAL_MODEL}
save_dir=${GROUP_BASE}/results/wan22_ti2v_i2v
input_images_dir=${lightx2v_path}/assets/inputs/imgs

# 额外的美女测试图片
BEAUTY_IMG1=${GROUP_BASE}/data/pics/beauty1.png
BEAUTY_IMG2=${GROUP_BASE}/data/pics/beauty2.png

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# ===== 同步模型和代码到本地盘 =====
echo "============================================"
echo "Syncing files from group to local disk..."
echo "============================================"

mkdir -p ${LOCAL_MODEL}
mkdir -p ${LOCAL_CODE}

SYNC_SCRIPT=${IT2AV_CODE_ROOT}/fast_sync.py

# 同步模型（并行拷贝，跳过已存在且大小一致的文件）
if [ ! -f "${LOCAL_MODEL}/.sync_done" ]; then
    echo ">>> Syncing model: Wan2.2-TI2V-5B (parallel, ${SYNC_WORKERS:-32} workers) ..."
    python ${SYNC_SCRIPT} ${GROUP_MODEL} ${LOCAL_MODEL} --workers ${SYNC_WORKERS:-32} --skip-existing
    touch ${LOCAL_MODEL}/.sync_done
    echo ">>> Model sync complete."
else
    echo ">>> Model already cached at ${LOCAL_MODEL}, skipping sync."
fi

# 同步代码（每次都增量同步，确保最新）
echo ">>> Syncing code: LightX2V-main ..."
python ${SYNC_SCRIPT} ${GROUP_CODE} ${LOCAL_CODE} --workers ${SYNC_WORKERS:-32} --skip-existing
echo ">>> Code sync complete."

echo "============================================"
echo "Local paths:"
echo "  Model: ${LOCAL_MODEL}"
echo "  Code:  ${LOCAL_CODE}"
echo "============================================"

# 环境设置
source ${lightx2v_path}/scripts/base/base.sh

mkdir -p ${save_dir}

# ===== 20 个 I2V 测试 Prompts + 输入图片 =====
declare -a PROMPTS=(
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

# 输入图片: 循环使用LightX2V自带图片 + beauty测试图片
declare -a IMAGES=(
    "${input_images_dir}/img_0.jpg"
    "${input_images_dir}/img_1.jpg"
    "${input_images_dir}/img_2.jpg"
    "${input_images_dir}/girl.png"
    "${input_images_dir}/woman.jpeg"
    "${input_images_dir}/snake.png"
    "${input_images_dir}/img_0.jpg"
    "${input_images_dir}/img_1.jpg"
    "${input_images_dir}/img_2.jpg"
    "${input_images_dir}/girl.png"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
    "${input_images_dir}/woman.jpeg"
    "${input_images_dir}/snake.png"
    "${input_images_dir}/img_0.jpg"
    "${input_images_dir}/img_1.jpg"
    "${input_images_dir}/img_2.jpg"
    "${input_images_dir}/girl.png"
    "${BEAUTY_IMG1}"
    "${BEAUTY_IMG2}"
)

echo "============================================"
echo "Wan2.2 TI2V-5B I2V Benchmark - 20 cases (8 GPU parallel)"
echo "Model: Wan2.2-TI2V-5B (5B Dense, High-Compression VAE)"
echo "Config: 704x1280, 121 frames, 50 steps, FPS=24"
echo "Parallel: cfg_p=2 × seq_p=4 (ulysses) = 8 GPUs"
echo "============================================"

TOTAL_START=$(date +%s)

for i in $(seq 0 19); do
    CASE_START=$(date +%s)
    echo ""
    echo ">>> Case $((i+1))/20 - Seed: $((42+i))"
    echo ">>> Image: ${IMAGES[$i]}"
    echo ">>> Prompt: ${PROMPTS[$i]:0:80}..."
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} -m lightx2v.infer \
        --model_cls wan2.2 \
        --task i2v \
        --model_path ${model_path} \
        --config_json ${lightx2v_path}/configs/wan22/wan_ti2v_i2v_h20_8gpu.json \
        --seed $((42+i)) \
        --prompt "${PROMPTS[$i]}" \
        --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
        --image_path "${IMAGES[$i]}" \
        --save_result_path "${save_dir}/case_$((i+1)).mp4"

    CASE_END=$(date +%s)
    echo ">>> Case $((i+1)) done in $((CASE_END - CASE_START))s"
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================"
echo "Benchmark completed!"
echo "Total time: ${TOTAL_TIME}s"
echo "Average per case: $((TOTAL_TIME / 20))s"
echo "Results saved to: ${save_dir}"
echo "============================================"
