#!/bin/bash
# ============================================================================
# Step 1: 生成首帧人像图 (HunyuanImage T2I API)
#
# 输入: digit_beauty_20260316.tsv 中的 5 个 case
# 输出: ${IT2AV_WORKSPACE}/data/pics/digit/20260316/case_{id}.png
#
# 说明:
#   为每个 case 根据"场景/服装"描述生成一张高质量美女人像首帧
#   这张图将作为:
#     1. LiveAvatar 数字人说台词视频的参考人像
#     2. 后续 I2V 动作视频的首帧
#
# 用法:
#   bash step1_generate_images.sh
#   bash step1_generate_images.sh --skip-existing    # 跳过已存在的图片
#
# 依赖: 内网访问 混元内部 API (HUNYUAN_API_BASE)
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}

DATE_TAG="20260316"
OUTPUT_DIR=${GROUP_BASE}/data/pics/digit/${DATE_TAG}
T2I_SCRIPT=${GROUP_CODE}/utils/image/Hunyuan-Image/test_hy_t2i.py

# 创建 prompts JSON (与 TSV 中的 5 个 case 对应)
PROMPTS_JSON=${OUTPUT_DIR}/t2i_prompts.json

mkdir -p ${OUTPUT_DIR}

echo "============================================================"
echo "🎨 Step 1: 生成首帧人像图 (HunyuanImage T2I)"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  日期标签: ${DATE_TAG}"
echo "============================================================"

# ===== 生成 prompts JSON =====
# 根据 digit_beauty_20260316.tsv 的 5 个 case 构建 prompt
# 每个 case: 场景 + 服装 → 生成美女人像首帧
# 尺寸选 768x1280 (竖屏, 适配 LiveAvatar 和 I2V)

cat > ${PROMPTS_JSON} << 'PROMPTS_EOF'
[
  {
    "id": "case_1",
    "prompt": "A beautiful young Chinese woman in a warm bedroom wearing a comfortable elegant sleeping dress (睡裙), warm ambient lighting, cozy bedroom interior with soft bedding and warm tones, she is standing and looking directly at the camera with a gentle warm smile, full body front view, cinematic photography, high quality, 4K, soft lighting",
    "size": "768x1280"
  },
  {
    "id": "case_2",
    "prompt": "A beautiful young Chinese woman standing on a night cherry blossom tree path (夜晚樱花树小道), wearing a school uniform style short skirt (制服短裙), cherry blossom petals falling around her, soft moonlight and warm street lights, she is looking directly at the camera with a serene smile, full body front view, cinematic photography, high quality, 4K, dreamy atmosphere",
    "size": "768x1280"
  },
  {
    "id": "case_3",
    "prompt": "A beautiful young Chinese woman on a quiet campus pathway (安静的校园小径), wearing a campus style plaid skirt (校园风格子裙), autumn trees and fallen leaves, gentle sunlight filtering through the trees, she is looking directly at the camera with an inspiring smile, full body front view, cinematic photography, high quality, 4K, warm academic atmosphere",
    "size": "768x1280"
  },
  {
    "id": "case_4",
    "prompt": "A beautiful young Chinese woman on a city street with traffic lights and crosswalk (城市街道斑马线), spring evening wind, wearing high heels and a fashionable short skirt (高跟鞋短裙), warm street lights and urban nightscape, she is looking directly at the camera with a confident elegant smile, full body front view, cinematic photography, high quality, 4K, urban chic",
    "size": "768x1280"
  },
  {
    "id": "case_5",
    "prompt": "A beautiful young Chinese woman standing on a small bridge with warm lights (有灯光的小桥上), wearing an elegant traditional Chinese cheongsam/qipao (旗袍), water reflection below, warm lantern lights, she is looking directly at the camera with a graceful gentle smile, full body front view, cinematic photography, high quality, 4K, traditional Chinese elegance",
    "size": "768x1280"
  }
]
PROMPTS_EOF

echo "✅ Prompts JSON 已生成: ${PROMPTS_JSON}"
echo ""

# ===== 调用 HunyuanImage T2I =====
STEP_START=$(date +%s)

python ${T2I_SCRIPT} \
    --prompts_json ${PROMPTS_JSON} \
    --output_dir ${OUTPUT_DIR} \
    --size "768x1280"

STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))

# ===== 检查结果 =====
echo ""
echo "============================================================"
echo "🎨 Step 1 完成!"
echo "  耗时: ${STEP_TIME}s"
echo "  输出目录: ${OUTPUT_DIR}"
echo ""
echo "  📁 生成的图片:"
for f in ${OUTPUT_DIR}/case_*.png; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "     $f ($SIZE)"
    fi
done
echo ""
echo "  下一步: bash step2_generate_audio.sh"
echo "============================================================"
