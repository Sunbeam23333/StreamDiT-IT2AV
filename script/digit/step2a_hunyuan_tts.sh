#!/bin/bash
# ============================================================================
# Step 2a: 使用混元 TTS API 生成台词语音
#
# 说明:
#   插入在 Step 2 (CosyVoice + BGM + SFX) 和 Step 3 (LiveAvatar) 之间.
#   调用混元 Hunyuan-TTS-2.5 云端 API 生成 TTS, 替代 CosyVoice 本地推理,
#   避免 CosyVoice 本地模型的质量问题 (非人声/乱读等).
#
# 输入:
#   - audio_prompts.json (由 step2_generate_audio.sh 生成)
#
# 输出:
#   ${IT2AV_WORKSPACE}/data/audio/digit/20260316/
#     case_{id}/
#       tts.wav          - 混元 TTS 生成的台词语音 (24kHz, 单声道)
#
# 特点:
#   - 无需 GPU, 纯 API 调用
#   - 高质量中文语音合成
#   - 默认覆盖已有 tts.wav (--no-overwrite 可跳过)
#
# 用法:
#   bash step2a_hunyuan_tts.sh                        # 默认配置
#   bash step2a_hunyuan_tts.sh --voice nana           # 指定音色
#   bash step2a_hunyuan_tts.sh --no-overwrite         # 不覆盖已有文件
#   bash step2a_hunyuan_tts.sh --stream               # 使用流式接口
#
# 依赖: pip install requests
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

# ===== 解析参数 =====
VOICE="wenrouxuemei"
OVERWRITE=true
STREAM="--stream"
SPEED=""
SAMPLE_RATE=24000

while [[ $# -gt 0 ]]; do
    case $1 in
        --voice)
            VOICE="$2"
            shift 2
            ;;
        --no-overwrite)
            OVERWRITE=false
            shift
            ;;
        --stream)
            STREAM="--stream"
            shift
            ;;
        --speed)
            SPEED="--speed $2"
            shift 2
            ;;
        --sample-rate)
            SAMPLE_RATE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}

DATE_TAG="20260316"
OUTPUT_BASE=${GROUP_BASE}/data/audio/digit/${DATE_TAG}
AUDIO_PROMPTS=${OUTPUT_BASE}/audio_prompts.json

# 混元 TTS Python 脚本路径
HUNYUAN_TTS_SCRIPT=${GROUP_CODE}/utils/audio/Hunyuan-tts/hunyuan_tts.py

# ===== API KEY =====
export HUNYUAN_API_KEY="${HUNYUAN_API_KEY:-}"

echo "============================================================"
echo "🎤 Step 2a: 混元 TTS API 语音生成"
echo "  输出目录: ${OUTPUT_BASE}"
echo "  音色: ${VOICE}"
echo "  采样率: ${SAMPLE_RATE}"
echo "  覆盖已有: ${OVERWRITE}"
echo "  流式: ${STREAM:-否}"
echo "============================================================"

# ===== 检查依赖 =====
if [ ! -f "${HUNYUAN_TTS_SCRIPT}" ]; then
    echo "❌ 混元 TTS 脚本不存在: ${HUNYUAN_TTS_SCRIPT}"
    exit 1
fi

if [ ! -f "${AUDIO_PROMPTS}" ]; then
    echo "❌ audio_prompts.json 不存在: ${AUDIO_PROMPTS}"
    echo "   请先运行 step2_generate_audio.sh 生成 audio_prompts.json"
    exit 1
fi

python3 -c "import requests" 2>/dev/null || {
    echo ">>> 安装 requests..."
    pip install -q requests
}

# ===== 如果 overwrite=true, 删除旧的 tts.wav =====
if [ "${OVERWRITE}" = true ]; then
    echo ">>> 清理旧 TTS 文件..."
    for CASE_ID in 1 2 3 4 5; do
        OLD_TTS=${OUTPUT_BASE}/case_${CASE_ID}/tts.wav
        if [ -f "${OLD_TTS}" ]; then
            rm -f "${OLD_TTS}"
            echo "  🗑️  删除: ${OLD_TTS}"
        fi
    done
fi

# ===== 创建输出目录 =====
for CASE_ID in 1 2 3 4 5; do
    mkdir -p ${OUTPUT_BASE}/case_${CASE_ID}
done

# ===== 执行 TTS 生成 =====
STEP_START=$(date +%s)

echo ""
echo ">>> 调用混元 TTS API..."
python3 ${HUNYUAN_TTS_SCRIPT} \
    --prompts_json ${AUDIO_PROMPTS} \
    --output_dir ${OUTPUT_BASE} \
    --voice ${VOICE} \
    --sample_rate ${SAMPLE_RATE} \
    ${STREAM} \
    ${SPEED} \
    2>&1 | tee ${OUTPUT_BASE}/log_hunyuan_tts.txt

STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))

# ===== 检查结果 =====
echo ""
echo "============================================================"
echo "🎤 Step 2a 完成!"
echo "  耗时: ${STEP_TIME}s"
echo "  输出目录: ${OUTPUT_BASE}"
echo ""
echo "  📁 生成的 TTS 音频:"
TOTAL=0
SUCCESS=0
for CASE_ID in 1 2 3 4 5; do
    FILE=${OUTPUT_BASE}/case_${CASE_ID}/tts.wav
    TOTAL=$((TOTAL + 1))
    if [ -f "$FILE" ] && [ $(stat -c%s "$FILE" 2>/dev/null || stat -f%z "$FILE" 2>/dev/null) -gt 1024 ]; then
        SIZE=$(du -h "$FILE" | cut -f1)
        # 用 python 计算时长
        DURATION=$(python3 -c "
import wave
with wave.open('${FILE}', 'rb') as w:
    frames = w.getnframes()
    rate = w.getframerate()
    print(f'{frames/rate:.1f}s')
" 2>/dev/null || echo "?s")
        echo "    ✅ case_${CASE_ID}/tts.wav ($SIZE, ${DURATION})"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "    ❌ case_${CASE_ID}/tts.wav (缺失)"
    fi
done

echo ""
echo "  ✅ ${SUCCESS}/${TOTAL} 成功"

if [ ${SUCCESS} -lt ${TOTAL} ]; then
    echo "  ⚠️  部分 TTS 生成失败, 查看日志:"
    echo "     ${OUTPUT_BASE}/log_hunyuan_tts.txt"
fi

echo ""
echo "  下一步: bash step3_liveavatar_talking.sh"
echo "============================================================"
