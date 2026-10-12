#!/bin/bash
# ============================================================================
# Step 4: Wan2.2 I2V 动作续接视频 (8卡并行)
#
# 输入:
#   最后一帧: ${IT2AV_WORKSPACE}/data/video/digit/20260316/case_{id}/last_frame.png (Step3 提取)
#
# 输出:
#   ${IT2AV_WORKSPACE}/data/video/digit/20260316/case_{id}/action.mp4
#
# 说明:
#   用 Step3 LiveAvatar 说台词视频的最后一帧作为 I2V 首帧，
#   通过 Wan2.2-I2V-A14B (MoE) + LightX2V 生成 ~5s 的动作视频
#   (如: 捧着热牛奶微笑、抬头看花、奔跑、过马路等)
#
#   模型: Wan2.2-I2V-A14B (MoE, 双模型 high_noise + low_noise)
#   框架: LightX2V (8卡 ulysses 并行, 无 CFG)
#   参数: 720p, 81 帧, 40 steps → ~3.24s@25fps 视频
#
# 用法:
#   bash step4_i2v_action.sh                          # 默认 8 卡全部
#   CASE_IDS="1 2" bash step4_i2v_action.sh           # 只跑指定 case
#
# 资源: 8×H20, ~80GB VRAM, 每个 case ~3-5 分钟
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"
conda activate ${IT2AV_CONDA_ENV}

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE_LIGHTX2V=${IT2AV_CODE_ROOT}/utils/video/LightX2V-main
GROUP_MODEL=${GROUP_BASE}/models/Wan2.2-I2V-A14B

LOCAL_BASE=${IT2AV_LOCAL_ROOT}
LOCAL_CODE=${LOCAL_BASE}/code/utils/video/LightX2V-main
LOCAL_MODEL=${LOCAL_BASE}/models/Wan2.2-I2V-A14B

SYNC_SCRIPT=${IT2AV_CODE_ROOT}/fast_sync.py

DATE_TAG="20260316"
VIDEO_DIR=${GROUP_BASE}/data/video/digit/${DATE_TAG}

# 选择 case
CASE_IDS=${CASE_IDS:-"1 2 3 4 5"}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

echo "============================================================"
echo "🎬 Step 4: Wan2.2 I2V 动作续接视频 (8卡)"
echo "  模型: Wan2.2-I2V-A14B (MoE)"
echo "  框架: LightX2V"
echo "  并行: seq_parallel_ulysses(8) = 8 GPUs"
echo "  参数: 720p, 81 帧, 40 steps"
echo "  Cases: ${CASE_IDS}"
echo "============================================================"

# ===== 同步模型和代码 =====
echo ""
echo ">>> 同步模型和代码..."

mkdir -p ${LOCAL_MODEL} ${LOCAL_CODE}

if [ ! -f "${LOCAL_MODEL}/.sync_done" ]; then
    echo "  同步模型: Wan2.2-I2V-A14B..."
    python ${SYNC_SCRIPT} ${GROUP_MODEL} ${LOCAL_MODEL} --workers 32 --skip-existing
    touch ${LOCAL_MODEL}/.sync_done
else
    echo "  模型已缓存"
fi

echo "  同步代码: LightX2V-main..."
python ${SYNC_SCRIPT} ${GROUP_CODE_LIGHTX2V} ${LOCAL_CODE} --workers 32 --skip-existing

# 环境变量
lightx2v_path=${LOCAL_CODE}
model_path=${LOCAL_MODEL}
source ${lightx2v_path}/scripts/base/base.sh

# ===== 每个 case 的动作描述 prompt =====
# 从 TSV 的 "镜头" 列提取动作描述
declare -A ACTION_PROMPTS
ACTION_PROMPTS[1]="A beautiful Chinese woman in a bedroom gently picks up a glass of warm milk, holding it with both hands, and looks at the camera with a soft warm smile, cozy bedroom atmosphere, warm lighting, gentle movement"
ACTION_PROMPTS[2]="A beautiful Chinese woman in school uniform on a cherry blossom path at night, she slowly raises her head looking up at falling cherry blossoms, then walks slowly, turning her face to look over her shoulder with a gentle smile, moonlit atmosphere"
ACTION_PROMPTS[3]="A beautiful Chinese woman in a plaid skirt running on a school campus playground, hair flowing in the wind, youthful energy, autumn sunshine, dynamic movement, joyful expression"
ACTION_PROMPTS[4]="A beautiful Chinese woman in high heels and short skirt crossing a city street at a crosswalk, walking gracefully, then slowly turning her face with an elegant side profile look, urban night lights, spring evening atmosphere"
ACTION_PROMPTS[5]="A beautiful Chinese woman in an elegant qipao cheongsam walking gracefully across a small illuminated bridge over flowing water, traditional Chinese garden atmosphere, warm lantern lights, flowing water below, elegant movement"

# ===== 串行推理 =====
TOTAL_START=$(date +%s)

for CASE_ID in ${CASE_IDS}; do
    LAST_FRAME=${VIDEO_DIR}/case_${CASE_ID}/last_frame.png
    SAVE_FILE=${VIDEO_DIR}/case_${CASE_ID}/action.mp4

    # 检查输入
    if [ ! -f "${LAST_FRAME}" ]; then
        echo "  ❌ case_${CASE_ID}: 最后一帧不存在: ${LAST_FRAME}"
        echo "     请先运行 step3_liveavatar_talking.sh"
        continue
    fi

    # 跳过已存在
    if [ -f "${SAVE_FILE}" ] && [ $(stat -c%s "${SAVE_FILE}" 2>/dev/null || stat -f%z "${SAVE_FILE}" 2>/dev/null) -gt 10240 ]; then
        echo "  ⏭️  case_${CASE_ID}: 动作视频已存在: ${SAVE_FILE}"
        continue
    fi

    PROMPT="${ACTION_PROMPTS[${CASE_ID}]}"

    echo ""
    echo "============================================================"
    echo "  🎬 case_${CASE_ID} - I2V 动作视频"
    echo "  Image: ${LAST_FRAME}"
    echo "  Prompt: ${PROMPT:0:80}..."
    echo "  Output: ${SAVE_FILE}"
    echo "============================================================"

    CASE_START=$(date +%s)

    torchrun --nproc_per_node=${NUM_GPUS} -m lightx2v.infer \
        --model_cls wan2.2_moe \
        --task i2v \
        --model_path ${model_path} \
        --config_json ${lightx2v_path}/configs/wan22/wan_moe_i2v_h20_8gpu.json \
        --seed $((42 + CASE_ID)) \
        --prompt "${PROMPT}" \
        --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
        --image_path "${LAST_FRAME}" \
        --save_result_path "${SAVE_FILE}"

    CASE_END=$(date +%s)
    echo "  ✅ case_${CASE_ID} 完成: $((CASE_END - CASE_START))s"
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

# ===== 汇总 =====
echo ""
echo "============================================================"
echo "🎬 Step 4 完成!"
echo "  耗时: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))min)"
echo ""
echo "  📁 生成的动作视频:"
for CASE_ID in ${CASE_IDS}; do
    ACTION_FILE=${VIDEO_DIR}/case_${CASE_ID}/action.mp4
    if [ -f "${ACTION_FILE}" ]; then
        SIZE=$(du -h "${ACTION_FILE}" | cut -f1)
        echo "    ✅ case_${CASE_ID}/action.mp4 ($SIZE)"
    else
        echo "    ❌ case_${CASE_ID}/action.mp4 (缺失)"
    fi
done
echo ""
echo "  📂 完整数据结构:"
echo "    ${GROUP_BASE}/data/"
echo "      pics/digit/${DATE_TAG}/case_{id}.png        ← Step1 首帧"
echo "      audio/digit/${DATE_TAG}/case_{id}/tts.wav    ← Step2 台词"
echo "      audio/digit/${DATE_TAG}/case_{id}/bgm.wav    ← Step2 背景音"
echo "      audio/digit/${DATE_TAG}/case_{id}/sfx.wav    ← Step2 音效"
echo "      video/digit/${DATE_TAG}/case_{id}/talking.mp4 ← Step3 说台词"
echo "      video/digit/${DATE_TAG}/case_{id}/last_frame.png ← Step3 最后一帧"
echo "      video/digit/${DATE_TAG}/case_{id}/action.mp4  ← Step4 动作视频"
echo ""
echo "  💡 后续可用 ffmpeg 拼接 talking.mp4 + action.mp4:"
echo "     ffmpeg -f concat -i list.txt -c copy final.mp4"
echo "============================================================"
