#!/bin/bash
# ============================================================================
# Pipeline2 - Step 2: HunyuanVideo-1.5 首帧生视频 (I2V → 1080p)
#
# 流程:
#   5 张无水印图片 → HunyuanVideo-1.5 I2V 720p → SR 超分 → 5 段 1080p 视频
#   (121帧, ~5s, 24fps)
#
# 原理:
#   HunyuanVideo-1.5 原生只支持 480p/720p, 1080p 通过内置 SR 网络实现:
#   720p 生成 → 720→1080 SR Step Distilled 超分
#
# 输入:
#   - 5 张图片: data/pics/20260323/no_watermark/*.png
#
# 输出:
#   ${OUTPUT_BASE}/case_{1..5}/raw_video.mp4       (1080p, SR 后)
#   ${OUTPUT_BASE}/case_{1..5}/raw_video_before_sr.mp4 (720p, SR 前, 可选)
#
# 用法:
#   bash step2_generate_video.sh                        # 全部 5 个 case, 8卡
#   bash step2_generate_video.sh --cases 1,3            # 只跑 case 1 和 3
#   bash step2_generate_video.sh --num_gpus 1           # 单卡 (慢, 自动开 offloading)
#   bash step2_generate_video.sh --no_sr                # 不超分, 输出 720p
#   bash step2_generate_video.sh --skip_sync            # 不同步, 直接用 group 盘
#
# GPU 分配:
#   默认: 8 卡 torchrun (H20 95GB × 8)
#
# 依赖: HunyuanVideo-1.5 模型 + Glyph-SDXL-v2 + byt5-small 已下载到 group 盘
# ============================================================================

set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../common/env.sh"

# ===== 激活 conda 环境 =====
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV:-${IT2AV_CONDA_ENV}}"

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
SYNC_SCRIPT=${GROUP_CODE}/fast_sync.py

DATE_TAG="20260323"
OUTPUT_BASE=${GROUP_BASE}/data/digit/${DATE_TAG}

# 图片目录 (group 盘上的无水印图片)
PICS_DIR=${GROUP_BASE}/data/pics/${DATE_TAG}/no_watermark

# GPU 配置 - 默认 8 卡
NUM_GPUS=${NUM_GPUS:-8}

# I2V 参数
# HunyuanVideo-1.5 原生分辨率: 480p 或 720p (不支持直接生成 1080p)
# 1080p 通过 720p + SR 超分实现
RESOLUTION=720p
ENABLE_SR=${ENABLE_SR:-true}         # 默认开启 SR 超分到 1080p
SAVE_PRE_SR=${SAVE_PRE_SR:-true}     # 保存超分前的 720p 视频

# 加速配置 (参考官方 README 推荐)
ENABLE_CACHE=${ENABLE_CACHE:-true}
CACHE_TYPE=${CACHE_TYPE:-deepcache}
CFG_DISTILLED=${CFG_DISTILLED:-true}   # CFG 蒸馏, 约 2x 加速
SAGE_ATTN=${SAGE_ATTN:-false}          # SageAttention (需要额外安装)

# conda 环境
CONDA_ENV=${CONDA_ENV:-${IT2AV_CONDA_ENV}}

FORCE_SYNC=0
SKIP_SYNC=0

# ===== 5 个 Case 的配置 =====
CASE_IMAGES=(
    "1-卧室场景.png"
    "2-校园场景.png"
    "3-樱花道.png"
    "4-街景.png"
    "5-旗袍.png"
)

# I2V 的动作提示词 (描述视频中人物希望做的动作)
CASE_I2V_PROMPTS=(
    "A beautiful young woman in a cozy bedroom, gentle head movements, soft smile, looking at camera warmly, ambient bedroom lighting, night scene, cinematic"
    "A beautiful young woman in school uniform walking along a cherry blossom path at night, hair flowing in gentle breeze, soft moonlight, dreamy atmosphere"
    "A beautiful young woman standing under cherry blossom trees, petals falling gently around her, slight wind blowing her hair, serene expression, spring night"
    "A beautiful young woman crossing a city street at night, confident stride, city lights reflecting, urban night atmosphere, cinematic lighting"
    "A beautiful young woman in traditional qipao dress standing on an illuminated bridge at night, graceful posture, lantern light, elegant movements, Chinese aesthetic"
)

# ===== 命令行参数解析 =====
CASES_CSV="1,2,3,4,5"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cases)        CASES_CSV="$2"; shift 2 ;;
        --num_gpus)     NUM_GPUS="$2"; shift 2 ;;
        --no_sr)        ENABLE_SR=false; shift ;;
        --force_sync)   FORCE_SYNC=1; shift ;;
        --skip_sync)    SKIP_SYNC=1; shift ;;
        --output)       OUTPUT_BASE="$2"; shift 2 ;;
        --no_cfg_distill) CFG_DISTILLED=false; shift ;;
        --sage_attn)    SAGE_ATTN=true; shift ;;
        -h|--help)
            head -33 "$0" | tail -28
            exit 0 ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

IFS=',' read -r -a CASES <<< "${CASES_CSV}"

# ===== GPU 配置 =====
if [[ ${NUM_GPUS} -eq 1 ]]; then
    export CUDA_VISIBLE_DEVICES=${GPU_VIDEO:-0}
else
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES=${GPU_LIST}
fi

# H20 95GB × 8 = 760GB 总显存, 关闭 offloading 更快
# 单卡时也有 95GB, 足以关闭 offloading (8.3B bf16 ≈ 16.6GB + text encoder/VAE ≈ 30-40GB)
OFFLOADING=false
GROUP_OFFLOADING=false

# ===== 初始化 =====
echo "============================================================"
echo "🎥 Pipeline2 Step 2: HunyuanVideo-1.5 I2V → 1080p"
echo "  输出目录:   ${OUTPUT_BASE}"
echo "  Case:       ${CASES_CSV}"
echo "  原生分辨率: ${RESOLUTION} (720p)"
echo "  SR 超分:    ${ENABLE_SR} (720p → 1080p)"
echo "  GPU 数量:   ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "  CFG 蒸馏:   ${CFG_DISTILLED}"
echo "  DeepCache:  ${ENABLE_CACHE} (${CACHE_TYPE})"
echo "  SageAttn:   ${SAGE_ATTN}"
echo "  Offloading: ${OFFLOADING}"
echo "============================================================"

# 创建输出目录
for CASE_ID in "${CASES[@]}"; do
    mkdir -p "${OUTPUT_BASE}/case_${CASE_ID}"
done

# ===== 同步代码和模型 =====
HYVIDEO_CODE_SRC=${GROUP_CODE}/utils/video/HunyuanVideo-1.5
HYVIDEO_CODE_DST=${LOCAL_BASE}/code/utils/video/HunyuanVideo-1.5
HYVIDEO_MODEL_SRC=${GROUP_BASE}/models/HunyuanVideo-1.5
HYVIDEO_MODEL_DST=${LOCAL_BASE}/models/HunyuanVideo-1.5

if [[ ${SKIP_SYNC} -eq 0 ]]; then
    echo ">>> 同步 HunyuanVideo-1.5 代码..."
    mkdir -p "${HYVIDEO_CODE_DST}"
    python "${SYNC_SCRIPT}" "${HYVIDEO_CODE_SRC}" "${HYVIDEO_CODE_DST}" --workers 32 --skip-existing
    echo ">>> 代码同步完成"

    echo ">>> 同步 HunyuanVideo-1.5 模型..."
    mkdir -p "${HYVIDEO_MODEL_DST}"
    if [[ ${FORCE_SYNC} -eq 1 || ! -f "${HYVIDEO_MODEL_DST}/.sync_done" ]]; then
        python "${SYNC_SCRIPT}" "${HYVIDEO_MODEL_SRC}" "${HYVIDEO_MODEL_DST}" --workers 32 --skip-existing
        touch "${HYVIDEO_MODEL_DST}/.sync_done"
        echo ">>> 模型同步完成"
    else
        echo ">>> 模型已缓存"
    fi

    LOCAL_CODE="${HYVIDEO_CODE_DST}"
    LOCAL_MODEL="${HYVIDEO_MODEL_DST}"
else
    echo ">>> 跳过同步, 直接使用 group 盘路径"
    LOCAL_CODE="${HYVIDEO_CODE_SRC}"
    LOCAL_MODEL="${HYVIDEO_MODEL_SRC}"
fi

# ===== 检查并同步 Glyph-SDXL-v2 (byT5 glyph processor) =====
GLYPH_SRC="${HYVIDEO_MODEL_SRC}/text_encoder/Glyph-SDXL-v2"
GLYPH_DST="${LOCAL_MODEL}/text_encoder/Glyph-SDXL-v2"
GLYPH_CKPT_SRC="${GLYPH_SRC}/checkpoints/byt5_model.pt"
GLYPH_CKPT_DST="${GLYPH_DST}/checkpoints/byt5_model.pt"

if [[ ! -f "${GLYPH_CKPT_DST}" ]]; then
    if [[ -f "${GLYPH_CKPT_SRC}" ]]; then
        echo ">>> 同步 Glyph-SDXL-v2 到本地 (group → /tmp)..."
        mkdir -p "${GLYPH_DST}"
        python "${SYNC_SCRIPT}" "${GLYPH_SRC}" "${GLYPH_DST}" --workers 16 --skip-existing
        echo ">>> Glyph-SDXL-v2 同步完成"
    else
        echo ""
        echo "  ❌ 缺少 Glyph-SDXL-v2 模型!"
        echo "     group 盘路径也不存在: ${GLYPH_CKPT_SRC}"
        echo ""
        echo "  请先下载 (在服务器上执行):"
        echo "    python ${GROUP_CODE}/download/download_models.py --models glyph_sdxl"
        echo ""
        exit 1
    fi
else
    echo ">>> Glyph-SDXL-v2 已缓存: ${GLYPH_DST}"
fi

# ===== 环境设置 =====
export PYTHONPATH=${LOCAL_CODE}:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo ""
echo "  Code:  ${LOCAL_CODE}"
echo "  Model: ${LOCAL_MODEL}"

# ===== 逐个 Case 生成视频 =====
STEP_START=$(date +%s)

for CASE_ID in "${CASES[@]}"; do
    IDX=$((CASE_ID - 1))
    IMAGE_NAME="${CASE_IMAGES[$IDX]}"
    IMAGE_PATH="${PICS_DIR}/${IMAGE_NAME}"
    PROMPT="${CASE_I2V_PROMPTS[$IDX]}"
    VIDEO_OUT="${OUTPUT_BASE}/case_${CASE_ID}/raw_video.mp4"

    if [[ -f "${VIDEO_OUT}" && $(stat -c%s "${VIDEO_OUT}" 2>/dev/null || stat -f%z "${VIDEO_OUT}" 2>/dev/null) -gt 10240 ]]; then
        echo "  ⏭️  case_${CASE_ID} 视频已存在: ${VIDEO_OUT}"
        continue
    fi

    if [[ ! -f "${IMAGE_PATH}" ]]; then
        echo "  ⚠️  case_${CASE_ID} 图片不存在: ${IMAGE_PATH}"
        continue
    fi

    echo ""
    echo "  🎥 [case_${CASE_ID}] HunyuanVideo-1.5 I2V (720p → 1080p SR)"
    echo "     图片:   ${IMAGE_PATH}"
    echo "     Prompt: ${PROMPT:0:80}..."
    echo "     输出:   ${VIDEO_OUT}"
    echo "     GPU:    ${NUM_GPUS} 卡, offloading=${OFFLOADING}"
    echo "     加速:   cfg_distilled=${CFG_DISTILLED}, cache=${ENABLE_CACHE}(${CACHE_TYPE}), sage=${SAGE_ATTN}"
    echo ""

    T0=$(date +%s)

    cd "${LOCAL_CODE}"
    torchrun --nproc_per_node=${NUM_GPUS} generate.py \
        --prompt "${PROMPT}" \
        --image_path "${IMAGE_PATH}" \
        --resolution ${RESOLUTION} \
        --aspect_ratio 16:9 \
        --seed $((42 + CASE_ID)) \
        --rewrite false \
        --cfg_distilled ${CFG_DISTILLED} \
        --enable_step_distill false \
        --sparse_attn false \
        --use_sageattn ${SAGE_ATTN} \
        --offloading ${OFFLOADING} \
        --group_offloading ${GROUP_OFFLOADING} \
        --overlap_group_offloading false \
        --enable_cache ${ENABLE_CACHE} \
        --cache_type ${CACHE_TYPE} \
        --sr ${ENABLE_SR} \
        --save_pre_sr_video ${SAVE_PRE_SR} \
        --output_path "${VIDEO_OUT}" \
        --model_path "${LOCAL_MODEL}" \
        2>&1 | while IFS= read -r line; do echo "     ${line}"; done

    T1=$(date +%s)
    ELAPSED=$((T1 - T0))

    if [[ -f "${VIDEO_OUT}" ]]; then
        SIZE=$(du -h "${VIDEO_OUT}" | cut -f1)
        echo "  ✅ case_${CASE_ID} 完成 (${SIZE}, 耗时 ${ELAPSED}s)"
    else
        echo "  ❌ case_${CASE_ID} 生成失败"
    fi
done

STEP_END=$(date +%s)
TOTAL_TIME=$((STEP_END - STEP_START))

# ===== 检查结果 =====
echo ""
echo "============================================================"
echo "🎥 Step 2 完成! (总耗时: ${TOTAL_TIME}s)"
echo "  输出目录: ${OUTPUT_BASE}"
echo "  分辨率:   720p → 1080p (SR=${ENABLE_SR})"
echo ""
for CASE_ID in "${CASES[@]}"; do
    FILE="${OUTPUT_BASE}/case_${CASE_ID}/raw_video.mp4"
    if [[ -f "$FILE" ]]; then
        SIZE=$(du -h "$FILE" | cut -f1)
        echo "  ✅ case_${CASE_ID}/raw_video.mp4 (${SIZE})"
    else
        echo "  ⬜ case_${CASE_ID}/raw_video.mp4 (未生成)"
    fi
done
echo ""
echo "  下一步: bash pipeline2/step3_lipsync_merge.sh"
echo "============================================================"
