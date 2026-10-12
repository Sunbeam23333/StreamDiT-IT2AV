#!/bin/bash
# ============================================================================
# 数字人视频生成完整流水线 (一键运行)
#
# 步骤:
#   Step 1: 生成参考人像 (Banana API 文生图)
#   Step 2: 播客文本 → TTS 音频 (edge-tts)
#   Step 3: LiveAvatar 推理 (数字人视频生成)
#
# 用法:
#   bash run_digit_pipeline.sh 5gpu          # 完整流水线, 5卡推理
#   bash run_digit_pipeline.sh 1gpu          # 完整流水线, 单卡推理
#   bash run_digit_pipeline.sh 5gpu --skip-portrait  # 跳过人像生成
#   bash run_digit_pipeline.sh 5gpu --skip-tts       # 跳过 TTS
#   bash run_digit_pipeline.sh 5gpu --only-infer     # 只跑推理
#
# 数据:
#   输入: ${IT2AV_WORKSPACE}/data/text/digit/digit_test.tsv (3 个播客样本)
#   人像: ${IT2AV_WORKSPACE}/data/pics/digit/ (Banana API 生成)
#   音频: ${IT2AV_WORKSPACE}/data/digit_samples/audio/ (edge-tts 生成)
#   输出: ${IT2AV_WORKSPACE}/results/liveavatar/{1gpu,5gpu}/ (视频)
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
TSV_FILE=${GROUP_BASE}/data/text/digit/digit_test.tsv
PORTRAIT_DIR=${GROUP_BASE}/data/pics/digit
SAMPLE_DIR=${GROUP_BASE}/data/digit_samples

LIVEAVATAR_CODE=${IT2AV_CODE_ROOT}/utils/video/LiveAvatar
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== 解析参数 =====
GPU_MODE=""
SKIP_PORTRAIT=false
SKIP_TTS=false
ONLY_INFER=false

for arg in "$@"; do
    case "${arg}" in
        1gpu|5gpu) GPU_MODE="${arg}" ;;
        --skip-portrait) SKIP_PORTRAIT=true ;;
        --skip-tts) SKIP_TTS=true ;;
        --only-infer) ONLY_INFER=true; SKIP_PORTRAIT=true; SKIP_TTS=true ;;
        *) echo "Unknown argument: ${arg}"; exit 1 ;;
    esac
done

if [ -z "${GPU_MODE}" ]; then
    echo "Usage: bash $0 {1gpu|5gpu} [--skip-portrait] [--skip-tts] [--only-infer]"
    exit 1
fi

PIPELINE_START=$(date +%s)

echo "============================================================"
echo "🎬 数字人视频生成流水线"
echo "============================================================"
echo "  GPU 模式: ${GPU_MODE}"
echo "  输入 TSV: ${TSV_FILE}"
echo "  人像目录: ${PORTRAIT_DIR}"
echo "  样本目录: ${SAMPLE_DIR}"
echo "  跳过人像: ${SKIP_PORTRAIT}"
echo "  跳过 TTS: ${SKIP_TTS}"
echo "============================================================"

# ===== Step 1: 生成参考人像 =====
if ! ${SKIP_PORTRAIT}; then
    echo ""
    echo "============================================================"
    echo "[Step 1/3] 🎨 生成参考人像 (Banana API)"
    echo "============================================================"

    STEP1_START=$(date +%s)
    python ${LIVEAVATAR_CODE}/generate_ref_portraits.py \
        --tsv ${TSV_FILE} \
        --output-dir ${PORTRAIT_DIR} \
        --num-portraits 2
    STEP1_END=$(date +%s)
    echo ">>> Step 1 完成: $((STEP1_END - STEP1_START))s"
else
    echo ""
    echo "[Step 1/3] ⏭️  跳过人像生成"
fi

# ===== Step 2: TTS 生成音频 + 样本列表 =====
if ! ${SKIP_TTS}; then
    echo ""
    echo "============================================================"
    echo "[Step 2/3] 🎙️  TTS 生成音频 + 样本列表"
    echo "============================================================"

    STEP2_START=$(date +%s)
    python ${LIVEAVATAR_CODE}/prepare_digit_samples.py \
        --tsv ${TSV_FILE} \
        --portrait-dir ${PORTRAIT_DIR} \
        --output-dir ${SAMPLE_DIR} \
        --max-duration 600
    STEP2_END=$(date +%s)
    echo ">>> Step 2 完成: $((STEP2_END - STEP2_START))s"
else
    echo ""
    echo "[Step 2/3] ⏭️  跳过 TTS 生成"
fi

# ===== Step 3: LiveAvatar 推理 =====
echo ""
echo "============================================================"
echo "[Step 3/3] 🎬 LiveAvatar 数字人推理 (${GPU_MODE})"
echo "============================================================"

STEP3_START=$(date +%s)
bash ${SCRIPT_DIR}/run_liveavatar_benchmark.sh ${GPU_MODE}
STEP3_END=$(date +%s)
echo ">>> Step 3 完成: $((STEP3_END - STEP3_START))s"

# ===== 汇总 =====
PIPELINE_END=$(date +%s)
PIPELINE_TIME=$((PIPELINE_END - PIPELINE_START))

echo ""
echo "============================================================"
echo "🎉 数字人视频生成流水线完成!"
echo "============================================================"
echo "  总耗时: ${PIPELINE_TIME}s ($((PIPELINE_TIME / 60))min)"
echo ""
echo "  📁 输出文件:"
echo "     人像: ${PORTRAIT_DIR}"
echo "     音频: ${SAMPLE_DIR}/audio/"
echo "     视频: ${GROUP_BASE}/results/liveavatar/${GPU_MODE}/"
echo ""
echo "  📊 10分钟视频资源总结:"
echo "     GPU: ${GPU_MODE}"
if [ "${GPU_MODE}" = "1gpu" ]; then
    echo "     显存: ~80GB (1×H20, offload)"
    echo "     预计推理耗时: ~2.5-3小时"
else
    echo "     显存: ~20GB/卡 (5×H20, TPP)"
    echo "     预计推理耗时: ~25-35分钟"
fi
echo "============================================================"
