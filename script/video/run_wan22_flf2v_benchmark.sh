#!/bin/bash
# ============================================================================
# Wan2.2 MoE FLF2V 首尾帧生成视频 - 20 cases benchmark (8卡并行)
# 服务器: 8卡 H20 95GB
# 模型: Wan2.2-I2V-A14B (MoE 双模型, 同时支持 I2V 和 FLF2V)
# 并行: seq_parallel_ulysses(8) = 8 GPUs (无CFG, enable_cfg=false)
#
# 注意: FLF2V = First-Last-Frame-to-Video, 需要提供首帧和尾帧图片
#       Wan2.2 MoE I2V 模型也支持 FLF2V 模式 (task=flf2v)
#       boundary=0.900 (MoE 切换阈值)
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

# ===== 路径配置 =====
# group 盘源路径（网络存储）
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_MODEL=${GROUP_BASE}/models/Wan2.2-I2V-A14B
GROUP_CODE=${IT2AV_CODE_ROOT}/utils/video/LightX2V-main

# 本地盘缓存路径（高速读取）
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
LOCAL_MODEL=${LOCAL_BASE}/models/Wan2.2-I2V-A14B
LOCAL_CODE=${LOCAL_BASE}/code/utils/video/LightX2V-main

# 实际使用的路径（指向本地盘）
lightx2v_path=${LOCAL_CODE}
model_path=${LOCAL_MODEL}
save_dir=${GROUP_BASE}/results/wan22_moe_flf2v

# 首尾帧测试图片
FIRST_FRAME=${GROUP_BASE}/data/pics/beauty1.png
LAST_FRAME=${GROUP_BASE}/data/pics/beauty2.png

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
    echo ">>> Syncing model: Wan2.2-I2V-A14B (parallel, ${SYNC_WORKERS:-32} workers) ..."
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

# ===== 20 个 FLF2V 测试 Prompts =====
# 首尾帧: beauty1.png -> beauty2.png (美女擦边类素材)
declare -a PROMPTS=(
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

echo "============================================"
echo "Wan2.2 MoE FLF2V Benchmark - 20 cases (8 GPU parallel)"
echo "Model: Wan2.2-I2V-A14B (MoE, FLF2V mode)"
echo "Config: 720p, 81 frames, 40 steps"
echo "Parallel: seq_p=8 (ulysses) = 8 GPUs (no CFG)"
echo "First frame: ${FIRST_FRAME}"
echo "Last frame:  ${LAST_FRAME}"
echo "============================================"

TOTAL_START=$(date +%s)

for i in $(seq 0 19); do
    CASE_START=$(date +%s)
    echo ""
    echo ">>> Case $((i+1))/20 - Seed: $((42+i))"
    echo ">>> Prompt: ${PROMPTS[$i]:0:80}..."
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} -m lightx2v.infer \
        --model_cls wan2.2_moe \
        --task flf2v \
        --model_path ${model_path} \
        --config_json ${lightx2v_path}/configs/wan22/wan_moe_flf2v_h20_8gpu.json \
        --seed $((42+i)) \
        --prompt "${PROMPTS[$i]}" \
        --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
        --image_path "${FIRST_FRAME}" \
        --last_frame_path "${LAST_FRAME}" \
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
