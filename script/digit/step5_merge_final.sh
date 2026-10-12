#!/bin/bash
# ============================================================================
# Step 5: 视频 + 音频混合合成
#
# 两种使用模式:
#
# ── 模式 A: 自由指定文件 ──────────────────────────────────────────
#   bash step5_merge_final.sh <video.mp4> [audio1.wav] [audio2.wav] ...
#
#   第一个参数必须是 mp4 视频, 后续参数为要混入的音频文件 (wav/mp3/aac 等)
#   所有音频自动裁剪/循环到视频长度, 结尾淡出
#
#   示例:
#     # 视频 + 一个BGM
#     bash step5_merge_final.sh talking.mp4 bgm.wav
#
#     # 视频 + BGM + SFX
#     bash step5_merge_final.sh talking.mp4 bgm.wav sfx.wav
#
#     # 视频 + BGM, 保留原声, 调音量
#     KEEP_ORIG=1 ORIG_VOL=0.8 AUD_VOL=0.3 bash step5_merge_final.sh talking.mp4 bgm.wav
#
#     # 视频 + BGM, 去除原声 (纯替换)
#     KEEP_ORIG=0 bash step5_merge_final.sh talking.mp4 bgm.wav
#
#   输出: 同目录下 final.mp4 (可通过 OUTPUT=xxx.mp4 自定义)
#
# ── 模式 B: 批量处理 (无位置参数) ────────────────────────────────
#   bash step5_merge_final.sh
#
#   自动遍历 case_1..5, 拼接 talking.mp4 + action.mp4, 混入 tts/bgm/sfx
#
#   CASE_IDS="1 3" bash step5_merge_final.sh
#   ENABLE_BGM=0 bash step5_merge_final.sh
#
# ── 通用参数 ────────────────────────────────────────────────────
#   KEEP_ORIG=1      保留视频原声 (默认1=保留, 0=去除)
#   ORIG_VOL=1.0     视频原声音量
#   AUD_VOL=0.3      额外音频的统一音量 (模式A, 各音频相同)
#   TTS_VOL=1.0      TTS 音量 (模式B)
#   BGM_VOL=0.25     BGM 音量 (模式B)
#   SFX_VOL=0.15     SFX 音量 (模式B)
#   FADE_OUT=2        BGM/SFX 结尾淡出秒数
#   CRF=18            视频编码质量 (越小越好)
#   FORCE=0           是否覆盖已存在的 final.mp4
#   OUTPUT=           自定义输出路径 (模式A)
#
# 资源: 纯 CPU (ffmpeg), 无需 GPU
# ============================================================================

set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"
# 注: ffmpeg 调用处用 || true 防止 set -e 导致跳过错误日志打印

# ===== ffmpeg 路径 (使用静态编译版, 支持 libx264) =====
FFMPEG_DIR=${IT2AV_FFMPEG_DIR:-$(dirname "$(command -v ffmpeg || echo /usr/bin/ffmpeg)")}
FFMPEG=${FFMPEG_DIR}/ffmpeg
FFPROBE=${FFMPEG_DIR}/ffprobe

# ===== 通用参数 =====
KEEP_ORIG=${KEEP_ORIG:-1}       # 保留视频原声
ORIG_VOL=${ORIG_VOL:-1.0}      # 视频原声音量
AUD_VOL=${AUD_VOL:-0.3}        # 模式A: 额外音频统一音量
FADE_OUT=${FADE_OUT:-2}         # 结尾淡出
CRF=${CRF:-18}                  # 编码质量
FORCE=${FORCE:-0}               # 覆盖已有

# ===== 获取时长 =====
get_duration() {
    "${FFPROBE}" -v quiet -print_format json -show_format "$1" 2>/dev/null \
        | python3 -c "import sys,json; print(f\"{float(json.load(sys.stdin)['format']['duration']):.3f}\")" 2>/dev/null
}

# ===== 检测视频是否有音频轨 =====
has_audio_track() {
    local count
    count=$("${FFPROBE}" -v quiet -select_streams a -show_entries stream=index -of csv=p=0 "$1" 2>/dev/null | wc -l)
    [ "$count" -gt 0 ]
}

# =====================================================================
# 模式 A: 自由指定文件
# =====================================================================
if [ $# -ge 1 ]; then
    VIDEO_FILE="$1"
    shift

    if [ ! -f "${VIDEO_FILE}" ]; then
        echo "❌ 视频文件不存在: ${VIDEO_FILE}"
        exit 1
    fi

    # 收集额外音频
    EXTRA_AUDIOS=()
    for f in "$@"; do
        if [ -f "$f" ]; then
            EXTRA_AUDIOS+=("$f")
        else
            echo "⚠️  音频文件不存在, 跳过: $f"
        fi
    done

    # 输出路径
    if [ -n "${OUTPUT:-}" ]; then
        FINAL="${OUTPUT}"
    else
        FINAL="$(dirname "${VIDEO_FILE}")/final.mp4"
    fi

    # 跳过已存在
    if [ -f "${FINAL}" ] && [ ${FORCE} -eq 0 ]; then
        SIZE=$(du -h "${FINAL}" | cut -f1)
        echo "⏭️  已存在: ${FINAL} ($SIZE)  (FORCE=1 覆盖)"
        exit 0
    fi

    VIDEO_DURATION=$(get_duration "${VIDEO_FILE}")
    VIDEO_HAS_AUDIO=false
    has_audio_track "${VIDEO_FILE}" && VIDEO_HAS_AUDIO=true

    echo "============================================================"
    echo "🎬 模式 A: 自由合成"
    echo "  视频: ${VIDEO_FILE} (${VIDEO_DURATION}s)"
    echo "  原声: $(${VIDEO_HAS_AUDIO} && echo "有音轨" || echo "无音轨") → $([ ${KEEP_ORIG} -eq 1 ] && echo "✅ 保留 (vol=${ORIG_VOL})" || echo "❌ 去除")"
    echo "  额外音频: ${#EXTRA_AUDIOS[@]} 个 (vol=${AUD_VOL})"
    for f in "${EXTRA_AUDIOS[@]}"; do
        d=$(get_duration "$f")
        echo "    🔊 $(basename "$f") (${d}s)"
    done
    echo "  输出: ${FINAL}"
    echo "============================================================"

    # 构建 filter_complex (用数组避免路径空格问题)
    INPUTS_ARR=(-i "${VIDEO_FILE}")
    FILTER_PARTS=""
    MIX_LABELS=""
    IDX=1
    NUM_MIX=0

    # 视频原声
    if ${VIDEO_HAS_AUDIO} && [ ${KEEP_ORIG} -eq 1 ]; then
        FILTER_PARTS="${FILTER_PARTS}[0:a]volume=${ORIG_VOL}[orig];"
        MIX_LABELS="${MIX_LABELS}[orig]"
        NUM_MIX=$((NUM_MIX + 1))
    fi

    # 额外音频
    for f in "${EXTRA_AUDIOS[@]}"; do
        INPUTS_ARR+=(-i "${f}")
        FADE_ST=$(python3 -c "print(f'{max(0, ${VIDEO_DURATION} - ${FADE_OUT}):.3f}')")
        FILTER_PARTS="${FILTER_PARTS}[${IDX}:a]aloop=loop=-1:size=2000000000,atrim=0:${VIDEO_DURATION},volume=${AUD_VOL},afade=t=out:st=${FADE_ST}:d=${FADE_OUT}[aud${IDX}];"
        MIX_LABELS="${MIX_LABELS}[aud${IDX}]"
        IDX=$((IDX + 1))
        NUM_MIX=$((NUM_MIX + 1))
    done

    # ffmpeg 日志文件 (排查问题用)
    FFLOG="$(dirname "${VIDEO_FILE}")/_ffmpeg_merge.log"

    echo "  🔧 音轨数: ${NUM_MIX} (原声+额外)"

    if [ ${NUM_MIX} -eq 0 ]; then
        # 无任何音频, 纯视频 copy
        echo "  ⚠️  无音频轨, 输出纯视频"
        "${FFMPEG}" -y -i "${VIDEO_FILE}" -an \
            -c:v copy \
            -movflags +faststart \
            "${FINAL}" 2>"${FFLOG}" || true
    elif [ ${NUM_MIX} -eq 1 ]; then
        FILTER="${FILTER_PARTS}${MIX_LABELS},apad=whole_dur=${VIDEO_DURATION},atrim=0:${VIDEO_DURATION}[aout]"
        echo "  🔧 filter: ${FILTER}"
        "${FFMPEG}" -y "${INPUTS_ARR[@]}" \
            -filter_complex "${FILTER}" \
            -map 0:v -map "[aout]" \
            -c:v libx264 -crf ${CRF} -preset medium \
            -c:a aac -b:a 192k -ar 44100 \
            -shortest -movflags +faststart \
            "${FINAL}" 2>"${FFLOG}" || true
    else
        FILTER="${FILTER_PARTS}${MIX_LABELS}amix=inputs=${NUM_MIX}:duration=longest:dropout_transition=0,atrim=0:${VIDEO_DURATION}[aout]"
        echo "  🔧 filter: ${FILTER}"
        "${FFMPEG}" -y "${INPUTS_ARR[@]}" \
            -filter_complex "${FILTER}" \
            -map 0:v -map "[aout]" \
            -c:v libx264 -crf ${CRF} -preset medium \
            -c:a aac -b:a 192k -ar 44100 \
            -shortest -movflags +faststart \
            "${FINAL}" 2>"${FFLOG}" || true
    fi

    # 如果失败, 打印 ffmpeg 错误日志最后20行
    if [ ! -f "${FINAL}" ] || [ ! -s "${FINAL}" ]; then
        echo "  ❌ ffmpeg 错误日志 (${FFLOG}):"
        tail -20 "${FFLOG}" 2>/dev/null | sed 's/^/    /'
    else
        rm -f "${FFLOG}"  # 成功则删除日志
    fi

    if [ -f "${FINAL}" ]; then
        FINAL_SIZE=$(du -h "${FINAL}" | cut -f1)
        FINAL_DUR=$(get_duration "${FINAL}")
        echo "  ✅ 合成完成: ${FINAL} (${FINAL_SIZE}, ${FINAL_DUR}s)"
    else
        echo "  ❌ 合成失败"
        exit 1
    fi

    exit 0
fi

# =====================================================================
# 模式 B: 批量处理 case
# =====================================================================
GROUP_BASE=${IT2AV_WORKSPACE}
DATE_TAG="${DATE_TAG:-20260316}"
VIDEO_DIR=${GROUP_BASE}/data/video/digit/${DATE_TAG}
AUDIO_DIR=${GROUP_BASE}/data/audio/digit/${DATE_TAG}

CASE_IDS=${CASE_IDS:-"1 2 3 4 5"}

# 模式B 音频开关和音量
ENABLE_TTS=${ENABLE_TTS:-1}
ENABLE_BGM=${ENABLE_BGM:-1}
ENABLE_SFX=${ENABLE_SFX:-1}
TTS_VOL=${TTS_VOL:-1.0}
BGM_VOL=${BGM_VOL:-0.25}
SFX_VOL=${SFX_VOL:-0.15}

echo "============================================================"
echo "🎬 Step 5 (模式 B): 批量合成"
echo "  视频目录: ${VIDEO_DIR}"
echo "  音频目录: ${AUDIO_DIR}"
echo "  Cases: ${CASE_IDS}"
echo ""
echo "  🔊 音频配置:"
echo "    视频原声:  $([ ${KEEP_ORIG} -eq 1 ] && echo '✅ 保留' || echo '❌ 去除')  音量=${ORIG_VOL}"
echo "    TTS 人声:  $([ ${ENABLE_TTS} -eq 1 ] && echo '✅ 启用' || echo '❌ 禁用')  音量=${TTS_VOL}"
echo "    BGM 背景乐: $([ ${ENABLE_BGM} -eq 1 ] && echo '✅ 启用' || echo '❌ 禁用')  音量=${BGM_VOL}"
echo "    SFX 环境音: $([ ${ENABLE_SFX} -eq 1 ] && echo '✅ 启用' || echo '❌ 禁用')  音量=${SFX_VOL}"
echo "    淡出: ${FADE_OUT}s  CRF: ${CRF}  ffmpeg: ${FFMPEG}"
echo "============================================================"

TOTAL_START=$(date +%s)
SUCCESS=0
FAIL=0

for CASE_ID in ${CASE_IDS}; do
    CASE_VIDEO_DIR=${VIDEO_DIR}/case_${CASE_ID}
    CASE_AUDIO_DIR=${AUDIO_DIR}/case_${CASE_ID}

    TALKING=${CASE_VIDEO_DIR}/talking.mp4
    ACTION=${CASE_VIDEO_DIR}/action.mp4
    TTS_WAV=${CASE_AUDIO_DIR}/tts.wav
    BGM_WAV=${CASE_AUDIO_DIR}/bgm.wav
    SFX_WAV=${CASE_AUDIO_DIR}/sfx.wav
    FINAL=${CASE_VIDEO_DIR}/final.mp4

    echo ""
    echo "============================================================"
    echo "  🎬 case_${CASE_ID}"

    # 跳过已存在
    if [ -f "${FINAL}" ] && [ ${FORCE} -eq 0 ]; then
        SIZE=$(du -h "${FINAL}" | cut -f1)
        echo "  ⏭️  已存在: ${FINAL} ($SIZE)  (FORCE=1 覆盖)"
        continue
    fi

    if [ ! -f "${TALKING}" ]; then
        echo "  ❌ talking.mp4 不存在, 跳过"
        FAIL=$((FAIL + 1))
        continue
    fi

    # ── 视频拼接 ──
    CONCAT_VIDEO=${CASE_VIDEO_DIR}/_tmp_concat.mp4
    if [ -f "${ACTION}" ]; then
        echo "  📎 拼接: talking.mp4 + action.mp4"
        CONCAT_LIST=${CASE_VIDEO_DIR}/_tmp_list.txt
        echo "file '${TALKING}'" > "${CONCAT_LIST}"
        echo "file '${ACTION}'" >> "${CONCAT_LIST}"
        "${FFMPEG}" -y -f concat -safe 0 -i "${CONCAT_LIST}" -c copy "${CONCAT_VIDEO}" 2>/dev/null
        rm -f "${CONCAT_LIST}"
    else
        echo "  ⚠️  action.mp4 不存在, 只用 talking.mp4"
        cp "${TALKING}" "${CONCAT_VIDEO}"
    fi

    VIDEO_DURATION=$(get_duration "${CONCAT_VIDEO}")
    VIDEO_HAS_AUDIO=false
    has_audio_track "${CONCAT_VIDEO}" && VIDEO_HAS_AUDIO=true
    echo "  📏 视频: ${VIDEO_DURATION}s (原声: $(${VIDEO_HAS_AUDIO} && echo '有' || echo '无'))"

    # ── 构建滤镜 (用数组避免路径空格问题) ──
    INPUTS_ARR=(-i "${CONCAT_VIDEO}")
    FILTER_PARTS=""
    MIX_LABELS=""
    IDX=1
    NUM_MIX=0

    # 视频原声
    if ${VIDEO_HAS_AUDIO} && [ ${KEEP_ORIG} -eq 1 ]; then
        FILTER_PARTS="${FILTER_PARTS}[0:a]volume=${ORIG_VOL}[orig];"
        MIX_LABELS="${MIX_LABELS}[orig]"
        NUM_MIX=$((NUM_MIX + 1))
    fi

    # TTS
    if [ ${ENABLE_TTS} -eq 1 ] && [ -f "${TTS_WAV}" ]; then
        TTS_DUR=$(get_duration "${TTS_WAV}")
        echo "  🎤 TTS: ${TTS_DUR}s, vol=${TTS_VOL}"
        INPUTS_ARR+=(-i "${TTS_WAV}")
        FILTER_PARTS="${FILTER_PARTS}[${IDX}:a]volume=${TTS_VOL}[tts];"
        MIX_LABELS="${MIX_LABELS}[tts]"
        IDX=$((IDX + 1))
        NUM_MIX=$((NUM_MIX + 1))
    fi

    # BGM
    if [ ${ENABLE_BGM} -eq 1 ] && [ -f "${BGM_WAV}" ]; then
        BGM_DUR=$(get_duration "${BGM_WAV}")
        echo "  🎵 BGM: ${BGM_DUR}s, vol=${BGM_VOL}"
        INPUTS_ARR+=(-i "${BGM_WAV}")
        FADE_ST=$(python3 -c "print(f'{max(0, ${VIDEO_DURATION} - ${FADE_OUT}):.3f}')")
        FILTER_PARTS="${FILTER_PARTS}[${IDX}:a]aloop=loop=-1:size=2000000000,atrim=0:${VIDEO_DURATION},volume=${BGM_VOL},afade=t=out:st=${FADE_ST}:d=${FADE_OUT}[bgm];"
        MIX_LABELS="${MIX_LABELS}[bgm]"
        IDX=$((IDX + 1))
        NUM_MIX=$((NUM_MIX + 1))
    fi

    # SFX
    if [ ${ENABLE_SFX} -eq 1 ] && [ -f "${SFX_WAV}" ]; then
        SFX_DUR=$(get_duration "${SFX_WAV}")
        echo "  🔊 SFX: ${SFX_DUR}s, vol=${SFX_VOL}"
        INPUTS_ARR+=(-i "${SFX_WAV}")
        FADE_ST=$(python3 -c "print(f'{max(0, ${VIDEO_DURATION} - ${FADE_OUT}):.3f}')")
        FILTER_PARTS="${FILTER_PARTS}[${IDX}:a]aloop=loop=-1:size=2000000000,atrim=0:${VIDEO_DURATION},volume=${SFX_VOL},afade=t=out:st=${FADE_ST}:d=${FADE_OUT}[sfx];"
        MIX_LABELS="${MIX_LABELS}[sfx]"
        IDX=$((IDX + 1))
        NUM_MIX=$((NUM_MIX + 1))
    fi

    # ── 合成 ──
    FFLOG=${CASE_VIDEO_DIR}/_ffmpeg_merge.log
    echo "  🔧 音轨数: ${NUM_MIX}"

    if [ ${NUM_MIX} -eq 0 ]; then
        echo "  ⚠️  无音频, 输出纯视频"
        "${FFMPEG}" -y -i "${CONCAT_VIDEO}" -an \
            -c:v copy \
            -movflags +faststart "${FINAL}" 2>"${FFLOG}" || true
    elif [ ${NUM_MIX} -eq 1 ]; then
        FILTER="${FILTER_PARTS}${MIX_LABELS},apad=whole_dur=${VIDEO_DURATION},atrim=0:${VIDEO_DURATION}[aout]"
        echo "  🔧 filter: ${FILTER}"
        "${FFMPEG}" -y "${INPUTS_ARR[@]}" \
            -filter_complex "${FILTER}" \
            -map 0:v -map "[aout]" \
            -c:v libx264 -crf ${CRF} -preset medium \
            -c:a aac -b:a 192k -ar 44100 \
            -shortest -movflags +faststart "${FINAL}" 2>"${FFLOG}" || true
    else
        FILTER="${FILTER_PARTS}${MIX_LABELS}amix=inputs=${NUM_MIX}:duration=longest:dropout_transition=0,atrim=0:${VIDEO_DURATION}[aout]"
        echo "  🔧 filter: ${FILTER}"
        "${FFMPEG}" -y "${INPUTS_ARR[@]}" \
            -filter_complex "${FILTER}" \
            -map 0:v -map "[aout]" \
            -c:v libx264 -crf ${CRF} -preset medium \
            -c:a aac -b:a 192k -ar 44100 \
            -shortest -movflags +faststart "${FINAL}" 2>"${FFLOG}" || true
    fi

    rm -f "${CONCAT_VIDEO}"

    # 如果失败, 打印 ffmpeg 错误日志
    if [ ! -f "${FINAL}" ] || [ ! -s "${FINAL}" ]; then
        echo "  ❌ ffmpeg 错误日志 (${FFLOG}):"
        tail -20 "${FFLOG}" 2>/dev/null | sed 's/^/    /'
    else
        rm -f "${FFLOG}"
    fi

    if [ -f "${FINAL}" ] && [ -s "${FINAL}" ]; then
        FINAL_SIZE=$(du -h "${FINAL}" | cut -f1)
        FINAL_DUR=$(get_duration "${FINAL}")
        echo "  ✅ 完成: ${FINAL} (${FINAL_SIZE}, ${FINAL_DUR}s)"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ❌ 合成失败"
        FAIL=$((FAIL + 1))
    fi
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================================"
echo "🎬 Step 5 完成!  耗时: ${TOTAL_TIME}s  成功: ${SUCCESS}  失败: ${FAIL}"
echo ""
for CASE_ID in ${CASE_IDS}; do
    FINAL=${VIDEO_DIR}/case_${CASE_ID}/final.mp4
    if [ -f "${FINAL}" ]; then
        SIZE=$(du -h "${FINAL}" | cut -f1)
        echo "  ✅ case_${CASE_ID}/final.mp4 ($SIZE)"
    else
        echo "  ❌ case_${CASE_ID}/final.mp4"
    fi
done
echo ""
echo "  💡 示例:"
echo "    # 保留原声 + BGM:"
echo "    KEEP_ORIG=1 ENABLE_BGM=1 ENABLE_SFX=0 ENABLE_TTS=0 bash step5_merge_final.sh"
echo ""
echo "    # 去原声, 只要 TTS + BGM:"
echo "    KEEP_ORIG=0 ENABLE_TTS=1 BGM_VOL=0.2 bash step5_merge_final.sh"
echo ""
echo "    # 自由模式: 指定视频+音频"
echo "    bash step5_merge_final.sh /path/to/talking.mp4 /path/to/bgm.wav"
echo "============================================================"
