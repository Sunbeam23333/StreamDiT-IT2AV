#!/bin/bash
# ============================================================================
# LingBot Camera Control I2V 推理 - 20 cases benchmark (8卡并行)
# 服务器: 8卡 H20 95GB
# 模型: lingbot-world-base-cam (MoE 双模型, Wan2.1 架构 + Plücker Camera Control)
# 并行: seq_parallel_ulysses(8) = 8 GPUs (无CFG, 不能用cfg_parallel)
#
# 注意: lingbot-world-base-cam 是 I2V 模型, 需要输入图片 + prompt
#       相机控制功能需要提供 action_path (camera pose JSON/NPY)
#       如果不提供 action_path, 则使用默认的静止相机 (等价于普通 I2V)
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

# ===== 路径配置 =====
# group 盘源路径（网络存储）
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_MODEL=${GROUP_BASE}/models/lingbot-world-base-cam
GROUP_CODE=${IT2AV_CODE_ROOT}/utils/video/LightX2V-main

# 本地盘缓存路径（高速读取）
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
LOCAL_MODEL=${LOCAL_BASE}/models/lingbot-world-base-cam
LOCAL_CODE=${LOCAL_BASE}/code/utils/video/LightX2V-main

# 实际使用的路径（指向本地盘）
lightx2v_path=${LOCAL_CODE}
model_path=${LOCAL_MODEL}
save_dir=${GROUP_BASE}/results/lingbot_cam_i2v
input_images_dir=${lightx2v_path}/assets/inputs/imgs

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
    echo ">>> Syncing model: lingbot-world-base-cam (parallel, ${SYNC_WORKERS:-32} workers) ..."
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

# ===== H20 95GB 8卡并行说明 =====
# lingbot-world-base-cam 是 MoE 双模型, 每个模型 ~37GB (bf16)
# enable_cfg=false, 不能用 cfg 并行, 使用 seq_parallel_ulysses(8) 全 8 卡序列并行
# cpu_offload=false, 8 卡显存充足无需 offload

# ===== 20 个 I2V 测试 Case =====
# 每个 case 使用不同 prompt + 不同 seed, 测试多样性和速度
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
    "A professional chef preparing sushi in a modern kitchen, precise knife movements slicing fresh salmon with elegant skill."
    "A group of hot air balloons rising at dawn over Cappadocia, the rocky landscape bathed in warm pink and orange light."
    "An ancient Egyptian temple with hieroglyphics illuminated by torch light, sand particles floating in the beams of light."
    "A skateboard performing tricks in a neon-lit urban halfpipe, dynamic motion with trailing light effects and street art background."
    "A peaceful zen garden with raked sand patterns, a bamboo fountain trickling water, and morning mist gradually clearing."
    "A powerful waterfall cascading into a turquoise pool, surrounded by lush tropical vegetation and rainbow mist."
    "A time-lapse of a flower blooming, petals unfurling in slow motion to reveal intricate patterns and vibrant colors."
    "A classic car cruising along a coastal highway at sunset, ocean waves crashing against rocks, seagulls flying overhead."
    "A bustling night market in Southeast Asia, colorful lanterns swaying, vendors preparing street food with sizzling woks."
    "A lone astronaut exploring the surface of Mars, red dust swirling around boots, a rover visible in the distance."
)

# ===== 选用的输入图片 (循环使用可用图片) =====
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
    "${input_images_dir}/woman.jpeg"
    "${input_images_dir}/snake.png"
    "${input_images_dir}/img_0.jpg"
    "${input_images_dir}/img_1.jpg"
    "${input_images_dir}/img_2.jpg"
    "${input_images_dir}/girl.png"
    "${input_images_dir}/woman.jpeg"
    "${input_images_dir}/snake.png"
    "${input_images_dir}/img_0.jpg"
    "${input_images_dir}/img_1.jpg"
)

echo "============================================"
echo "LingBot Cam I2V Benchmark - 20 cases"
echo "Model: lingbot-world-base-cam (MoE)"
echo "Config: 720p, 81 frames, 40 steps"
echo "Camera: identity (no motion) - 先验证基本推理"
echo "============================================"

TOTAL_START=$(date +%s)

for i in $(seq 0 19); do
    CASE_START=$(date +%s)
    echo ""
    echo ">>> Case $((i+1))/20 - Seed: $((42+i))"
    echo ">>> Prompt: ${PROMPTS[$i]:0:80}..."
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} -m lightx2v.infer \
        --model_cls lingbot_cam_moe \
        --task i2v \
        --model_path ${model_path} \
        --config_json ${lightx2v_path}/configs/lingbot/lingbot_cam_moe_i2v_h20_8gpu.json \
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
