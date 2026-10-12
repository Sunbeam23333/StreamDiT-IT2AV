#!/bin/bash
# ============================================================================
# Pipeline2 - Step 3: 视频慢放 + LatentSync 唇形同步 + FFmpeg 合成
#
# 流程:
#   3a. FFmpeg 视频慢放 (raw_video → slowmo_video, 匹配 TTS 时长)
#       HunyuanVideo-1.5 生成 121帧/24fps ≈ 5s, TTS 台词一般 > 5s
#       → 用 setpts 慢放视频, 使时长与 tts.wav 一致
#       → 同时转为 25fps (LatentSync 期望 25fps 输入)
#
#   3b. LatentSync 唇形同步 (slowmo_video + tts.wav → lipsync_video)
#
#   3c. FFmpeg 合成最终视频 (lipsync_video + BGM + SFX → final.mp4)
#       BGM/SFX 截断到与 TTS 相同长度, 不会超出视频时长
#
# 输入 (来自前两步):
#   ${OUTPUT_BASE}/case_{id}/
#     raw_video.mp4  - HunyuanVideo-1.5 生成的原始视频 (Step 2, ~5s)
#     tts.wav        - Spark-TTS 台词语音 (Step 1, 可变长度)
#     bgm.wav        - ACE-Step 背景音乐 (Step 1, ~15s, 可选)
#     sfx.wav        - TangoFlux 环境音效 (Step 1, ~15s, 可选)
#
# 输出:
#   ${OUTPUT_BASE}/case_{id}/
#     slowmo_video.mp4   - 慢放后的视频 (时长 = TTS 时长, 25fps)
#     lipsync_video.mp4  - 唇形同步后的视频
#     final.mp4          - 最终合成视频
#
# 用法:
#   bash step3_lipsync_merge.sh                         # 全部 case
#   bash step3_lipsync_merge.sh --cases 1,3             # 只跑 case 1 和 3
#   bash step3_lipsync_merge.sh --only slowmo           # 只跑视频慢放
#   bash step3_lipsync_merge.sh --only lipsync          # 只跑唇形同步
#   bash step3_lipsync_merge.sh --only merge            # 只跑 FFmpeg 合成
#   bash step3_lipsync_merge.sh --no_bgm                # 最终视频不加 BGM
#   bash step3_lipsync_merge.sh --no_sfx                # 最终视频不加 SFX
#   bash step3_lipsync_merge.sh --no_bgm --no_sfx       # 只有 TTS 语音
#   bash step3_lipsync_merge.sh --skip_sync             # 不同步, 直接用 group 盘
#
# GPU 分配:
#   GPU 4: LatentSync 唇形同步 (~18GB VRAM)
#   慢放和合成步骤不需要 GPU
#
# 依赖:
#   - LatentSync-1.6 模型已下载到 group 盘
#   - FFmpeg 7.0.2 静态编译版 (服务器上已有)
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

# FFmpeg 静态版 (服务器上)
FFMPEG=$(command -v ffmpeg || echo ffmpeg)
FFPROBE=$(command -v ffprobe || echo ffprobe)
DATE_TAG="20260323"
OUTPUT_BASE=${GROUP_BASE}/data/digit/${DATE_TAG}

# GPU 分配
GPU_LIPSYNC=${GPU_LIPSYNC:-4}

# LatentSync 推理参数
LIPSYNC_STEPS=${LIPSYNC_STEPS:-25}
LIPSYNC_GUIDANCE=${LIPSYNC_GUIDANCE:-1.5}

# LatentSync 期望的输入 fps
LIPSYNC_FPS=25

# 混音参数
VOL_TTS=${VOL_TTS:-1.0}
VOL_BGM=${VOL_BGM:-0.15}
VOL_SFX=${VOL_SFX:-0.2}

# BGM/SFX 开关 (默认开启)
USE_BGM=${USE_BGM:-true}
USE_SFX=${USE_SFX:-true}

# conda 环境
CONDA_ENV=${CONDA_ENV:-${IT2AV_CONDA_ENV}}

# ===== 命令行参数解析 =====
CASES_CSV="1,2,3,4,5"
ONLY_CSV="slowmo,lipsync,merge"
FORCE_SYNC=0
SKIP_SYNC=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cases)          CASES_CSV="$2"; shift 2 ;;
        --only)           ONLY_CSV="$2"; shift 2 ;;
        --force_sync)     FORCE_SYNC=1; shift ;;
        --skip_sync)      SKIP_SYNC=1; shift ;;
        --output)         OUTPUT_BASE="$2"; shift 2 ;;
        --lipsync_steps)  LIPSYNC_STEPS="$2"; shift 2 ;;
        --guidance)       LIPSYNC_GUIDANCE="$2"; shift 2 ;;
        --vol_tts)        VOL_TTS="$2"; shift 2 ;;
        --vol_bgm)        VOL_BGM="$2"; shift 2 ;;
        --vol_sfx)        VOL_SFX="$2"; shift 2 ;;
        --no_bgm)         USE_BGM=false; shift ;;
        --no_sfx)         USE_SFX=false; shift ;;
        -h|--help)
            head -50 "$0" | tail -45
            exit 0 ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

IFS=',' read -r -a CASES <<< "${CASES_CSV}"
IFS=',' read -r -a ONLY_PARTS <<< "${ONLY_CSV}"

contains_part() {
    local part=$1
    for p in "${ONLY_PARTS[@]}"; do
        [[ "$p" == "$part" ]] && return 0
    done
    return 1
}

# ===== FFmpeg 检查 =====
if [[ ! -x "${FFMPEG}" ]]; then
    echo "  ⚠️  静态 FFmpeg 不存在: ${FFMPEG}"
    echo "     回退到系统 ffmpeg..."
    FFMPEG=ffmpeg
    FFPROBE=ffprobe
fi

# ===== 辅助函数: 获取媒体时长 (秒, 浮点数) =====
get_duration() {
    local file=$1
    ${FFPROBE} -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "${file}" 2>/dev/null
}

# ===== 辅助函数: 获取视频 fps =====
get_fps() {
    local file=$1
    ${FFPROBE} -v error -select_streams v:0 \
        -show_entries stream=r_frame_rate \
        -of default=noprint_wrappers=1:nokey=1 "${file}" 2>/dev/null
}

# ===== 辅助函数: 获取视频帧数 =====
get_frame_count() {
    local file=$1
    ${FFPROBE} -v error -select_streams v:0 \
        -count_frames -show_entries stream=nb_read_frames \
        -of default=noprint_wrappers=1:nokey=1 "${file}" 2>/dev/null
}

# ===== 初始化 =====
echo "============================================================"
echo "👄 Pipeline2 Step 3: 视频慢放 + 唇形同步 + 合成"
echo "  输出目录:     ${OUTPUT_BASE}"
echo "  Case:         ${CASES_CSV}"
echo "  执行项:       ${ONLY_CSV}"
echo "  GPU:          ${GPU_LIPSYNC}"
echo "  LatentSync:   steps=${LIPSYNC_STEPS}, guidance=${LIPSYNC_GUIDANCE}"
echo "  混音:         TTS=${VOL_TTS}, BGM=${VOL_BGM}, SFX=${VOL_SFX}"
echo "  使用 BGM:     ${USE_BGM}"
echo "  使用 SFX:     ${USE_SFX}"
echo "  FFmpeg:       ${FFMPEG}"
echo "============================================================"

# 创建输出目录
for CASE_ID in "${CASES[@]}"; do
    mkdir -p "${OUTPUT_BASE}/case_${CASE_ID}"
done

# ===== 模型同步函数 =====
sync_model() {
    local MODEL_DIR=$1
    local SRC=${GROUP_BASE}/models/${MODEL_DIR}
    local DST=${LOCAL_BASE}/models/${MODEL_DIR}

    if [[ ! -d "${SRC}" ]]; then
        echo "  ⚠️  模型不存在: ${SRC}" >&2
        echo "${SRC}"
        return 0
    fi

    if [[ ${SKIP_SYNC} -eq 1 ]]; then
        echo "${SRC}"
        return 0
    fi

    mkdir -p "${DST}"
    if [[ ${FORCE_SYNC} -eq 1 || ! -f "${DST}/.sync_done" ]]; then
        echo "  🔄 同步 ${MODEL_DIR} → 本地盘" >&2
        python "${SYNC_SCRIPT}" "${SRC}" "${DST}" --workers 32 --skip-existing >&2
        touch "${DST}/.sync_done"
    else
        echo "  ✅ ${MODEL_DIR} 已缓存" >&2
    fi
    echo "${DST}"
}

sync_code() {
    local CODE_DIR=$1
    local SRC=${GROUP_CODE}/${CODE_DIR}
    local DST=${LOCAL_BASE}/code/${CODE_DIR}

    if [[ ! -d "${SRC}" ]]; then
        echo "${SRC}"
        return 0
    fi

    if [[ ${SKIP_SYNC} -eq 1 ]]; then
        echo "${SRC}"
        return 0
    fi

    mkdir -p "${DST}"
    if [[ ${FORCE_SYNC} -eq 1 || ! -f "${DST}/.sync_done" ]]; then
        echo "  🔄 同步代码 ${CODE_DIR} → 本地盘" >&2
        python "${SYNC_SCRIPT}" "${SRC}" "${DST}" --workers 32 --skip-existing >&2
        touch "${DST}/.sync_done"
    fi
    echo "${DST}"
}


# ################################################################
#  3a. 视频慢放 (raw_video → slowmo_video, 时长匹配 TTS)
# ################################################################
if contains_part "slowmo"; then
    echo ""
    echo "============================================================"
    echo "🐢 Step 3a: FFmpeg 视频慢放 (匹配 TTS 时长)"
    echo "============================================================"

    SLOWMO_START=$(date +%s)

    for CASE_ID in "${CASES[@]}"; do
        CASE_DIR="${OUTPUT_BASE}/case_${CASE_ID}"
        VIDEO_IN="${CASE_DIR}/raw_video.mp4"
        AUDIO_IN="${CASE_DIR}/tts.wav"
        VIDEO_OUT="${CASE_DIR}/slowmo_video.mp4"

        # 跳过已存在的
        if [[ -f "${VIDEO_OUT}" && $(stat -c%s "${VIDEO_OUT}" 2>/dev/null || stat -f%z "${VIDEO_OUT}" 2>/dev/null) -gt 10240 ]]; then
            echo "  ⏭️  case_${CASE_ID} 慢放视频已存在"
            continue
        fi

        if [[ ! -f "${VIDEO_IN}" ]]; then
            echo "  ⚠️  case_${CASE_ID} 原始视频不存在: ${VIDEO_IN}"
            echo "     请先运行 step2_generate_video.sh"
            continue
        fi

        if [[ ! -f "${AUDIO_IN}" ]]; then
            echo "  ⚠️  case_${CASE_ID} TTS 音频不存在: ${AUDIO_IN}"
            echo "     请先运行 step1_generate_audio.sh"
            continue
        fi

        # 获取原始视频时长和 TTS 时长
        VID_DUR=$(get_duration "${VIDEO_IN}")
        TTS_DUR=$(get_duration "${AUDIO_IN}")

        if [[ -z "${VID_DUR}" || -z "${TTS_DUR}" ]]; then
            echo "  ❌ case_${CASE_ID} 无法获取时长: video=${VID_DUR}, tts=${TTS_DUR}"
            continue
        fi

        # 计算慢放倍率: slowdown = TTS_DUR / VID_DUR
        # setpts 的 PTS 倍率 = TTS_DUR / VID_DUR (> 1 表示慢放)
        # 例: video=5s, tts=8s → PTS_FACTOR=1.6 → 每帧时间戳 × 1.6 → 视频变慢
        PTS_FACTOR=$(python3 -c "print(f'{${TTS_DUR} / ${VID_DUR}:.6f}')")

        echo ""
        echo "  🐢 [case_${CASE_ID}] 视频慢放"
        echo "     原始视频: ${VID_DUR}s"
        echo "     TTS 时长: ${TTS_DUR}s"
        echo "     慢放倍率: ${PTS_FACTOR}x (setpts=PTS*${PTS_FACTOR})"
        echo "     输出 fps: ${LIPSYNC_FPS} (LatentSync 期望)"

        T0=$(date +%s)

        # FFmpeg 慢放:
        #   -vf "setpts=PTS*FACTOR" : 慢放视频 (拉长时间戳)
        #   -r 25                   : 输出 25fps (LatentSync 期望)
        #   -an                     : 丢弃原始音频 (原始视频无音频或音频无用)
        #   -c:v libx264 -crf 16    : 高质量编码 (后续还要喂给 LatentSync)
        ${FFMPEG} -y -i "${VIDEO_IN}" \
            -vf "setpts=PTS*${PTS_FACTOR}" \
            -r ${LIPSYNC_FPS} \
            -an \
            -c:v libx264 -crf 16 -preset medium \
            "${VIDEO_OUT}" 2>/dev/null

        T1=$(date +%s)
        ELAPSED=$((T1 - T0))

        if [[ -f "${VIDEO_OUT}" ]]; then
            OUT_DUR=$(get_duration "${VIDEO_OUT}")
            SIZE=$(du -h "${VIDEO_OUT}" | cut -f1)
            echo "  ✅ case_${CASE_ID} 慢放完成: ${VID_DUR}s → ${OUT_DUR}s (${SIZE}, 耗时 ${ELAPSED}s)"
        else
            echo "  ❌ case_${CASE_ID} 慢放失败"
        fi
    done

    SLOWMO_END=$(date +%s)
    SLOWMO_TIME=$((SLOWMO_END - SLOWMO_START))
    echo "✅ Step 3a 完成: 视频慢放 (总耗时: ${SLOWMO_TIME}s)"
fi


# ################################################################
#  3b. LatentSync 唇形同步
# ################################################################
if contains_part "lipsync"; then
    echo ""
    echo "============================================================"
    echo "👄 Step 3b: LatentSync 唇形同步"
    echo "============================================================"

    # 同步 LatentSync 模型和代码
    echo ">>> 同步 LatentSync 模型..."
    LATENTSYNC_MODEL_PATH=$(sync_model "LatentSync-1.6")
    echo ">>> 同步 LatentSync 代码..."
    LATENTSYNC_CODE_PATH=$(sync_code "utils/video/LatentSync")

    # 建立 checkpoints 软链接 (LatentSync 期望 checkpoints/ 在代码目录下)
    CKPT_DIR="${LATENTSYNC_CODE_PATH}/checkpoints"
    if [[ ! -d "${CKPT_DIR}" || ! -f "${CKPT_DIR}/latentsync_unet.pt" ]]; then
        echo "  🔗 链接 checkpoints..."
        mkdir -p "${CKPT_DIR}"
        # latentsync_unet.pt
        if [[ -f "${LATENTSYNC_MODEL_PATH}/latentsync_unet.pt" ]]; then
            ln -sf "${LATENTSYNC_MODEL_PATH}/latentsync_unet.pt" "${CKPT_DIR}/latentsync_unet.pt"
        fi
        # whisper/tiny.pt
        if [[ -d "${LATENTSYNC_MODEL_PATH}/whisper" ]]; then
            ln -sfn "${LATENTSYNC_MODEL_PATH}/whisper" "${CKPT_DIR}/whisper"
        fi
        # auxiliary/
        if [[ -d "${LATENTSYNC_MODEL_PATH}/auxiliary" ]]; then
            ln -sfn "${LATENTSYNC_MODEL_PATH}/auxiliary" "${CKPT_DIR}/auxiliary"
        fi
        echo "  ✅ checkpoints 链接完成"
    else
        echo "  ✅ checkpoints 已就绪"
    fi

    # 将静态 FFmpeg 目录加入 PATH (LatentSync 内部直接调用 "ffmpeg" 命令)
    # 系统自带 ffmpeg 可能没有 libx264, 不认识 -crf 参数
    FFMPEG_DIR=$(dirname "${FFMPEG}")
    if [[ -d "${FFMPEG_DIR}" ]]; then
        export PATH="${FFMPEG_DIR}:${PATH}"
        echo "  🔧 已将 ${FFMPEG_DIR} 加入 PATH"
        echo "     $(which ffmpeg) → $(ffmpeg -version 2>&1 | head -1)"
    fi

    LIPSYNC_START=$(date +%s)

    # 逐个 case 运行唇形同步
    for CASE_ID in "${CASES[@]}"; do
        CASE_DIR="${OUTPUT_BASE}/case_${CASE_ID}"
        # 输入: 使用慢放后的视频 (如果存在), 否则回退到原始视频
        SLOWMO_VIDEO="${CASE_DIR}/slowmo_video.mp4"
        RAW_VIDEO="${CASE_DIR}/raw_video.mp4"
        AUDIO_IN="${CASE_DIR}/tts.wav"
        VIDEO_OUT="${CASE_DIR}/lipsync_video.mp4"

        if [[ -f "${SLOWMO_VIDEO}" ]]; then
            VIDEO_IN="${SLOWMO_VIDEO}"
        else
            echo "  ⚠️  case_${CASE_ID} 慢放视频不存在, 使用原始视频"
            VIDEO_IN="${RAW_VIDEO}"
        fi

        if [[ -f "${VIDEO_OUT}" && $(stat -c%s "${VIDEO_OUT}" 2>/dev/null || stat -f%z "${VIDEO_OUT}" 2>/dev/null) -gt 10240 ]]; then
            echo "  ⏭️  case_${CASE_ID} 唇形同步已存在"
            continue
        fi

        if [[ ! -f "${VIDEO_IN}" ]]; then
            echo "  ⚠️  case_${CASE_ID} 视频不存在: ${VIDEO_IN}"
            echo "     请先运行 step2 和 step3a"
            continue
        fi

        if [[ ! -f "${AUDIO_IN}" ]]; then
            echo "  ⚠️  case_${CASE_ID} TTS 音频不存在: ${AUDIO_IN}"
            echo "     请先运行 step1_generate_audio.sh"
            continue
        fi

        VID_DUR=$(get_duration "${VIDEO_IN}")
        TTS_DUR=$(get_duration "${AUDIO_IN}")

        echo ""
        echo "  👄 [case_${CASE_ID}] LatentSync 唇形同步"
        echo "     视频: ${VIDEO_IN} (${VID_DUR}s)"
        echo "     音频: ${AUDIO_IN} (${TTS_DUR}s)"
        echo "     输出: ${VIDEO_OUT}"

        T0=$(date +%s)

        cd "${LATENTSYNC_CODE_PATH}"
        CUDA_VISIBLE_DEVICES=${GPU_LIPSYNC} \
        python -m scripts.inference \
            --unet_config_path "${LATENTSYNC_CODE_PATH}/configs/unet/stage2_512.yaml" \
            --inference_ckpt_path "${CKPT_DIR}/latentsync_unet.pt" \
            --inference_steps ${LIPSYNC_STEPS} \
            --guidance_scale ${LIPSYNC_GUIDANCE} \
            --enable_deepcache \
            --video_path "${VIDEO_IN}" \
            --audio_path "${AUDIO_IN}" \
            --video_out_path "${VIDEO_OUT}" \
            --temp_dir "${CASE_DIR}/temp_lipsync" \
            2>&1 | while IFS= read -r line; do echo "     ${line}"; done

        T1=$(date +%s)
        ELAPSED=$((T1 - T0))

        if [[ -f "${VIDEO_OUT}" ]]; then
            SIZE=$(du -h "${VIDEO_OUT}" | cut -f1)
            OUT_DUR=$(get_duration "${VIDEO_OUT}")
            echo "  ✅ case_${CASE_ID} 唇形同步完成 (${OUT_DUR}s, ${SIZE}, 耗时 ${ELAPSED}s)"
        else
            echo "  ❌ case_${CASE_ID} 唇形同步失败"
        fi
    done

    LIPSYNC_END=$(date +%s)
    LIPSYNC_TIME=$((LIPSYNC_END - LIPSYNC_START))
    echo "✅ Step 3b 完成: LatentSync 唇形同步 (总耗时: ${LIPSYNC_TIME}s)"
fi


# ################################################################
#  3c. FFmpeg 合成最终视频
#      BGM/SFX 截断到与视频 (≈TTS) 相同长度
# ################################################################
if contains_part "merge"; then
    echo ""
    echo "============================================================"
    echo "🎬 Step 3c: FFmpeg 合成最终视频"
    echo "  混音:    TTS=${VOL_TTS}, BGM=${VOL_BGM}, SFX=${VOL_SFX}"
    echo "  使用BGM: ${USE_BGM}"
    echo "  使用SFX: ${USE_SFX}"
    echo "============================================================"

    for CASE_ID in "${CASES[@]}"; do
        CASE_DIR="${OUTPUT_BASE}/case_${CASE_ID}"
        LIPSYNC_VIDEO="${CASE_DIR}/lipsync_video.mp4"
        TTS_AUDIO="${CASE_DIR}/tts.wav"
        BGM="${CASE_DIR}/bgm.wav"
        SFX="${CASE_DIR}/sfx.wav"
        FINAL="${CASE_DIR}/final.mp4"

        if [[ ! -f "${LIPSYNC_VIDEO}" ]]; then
            echo "  ⚠️  case_${CASE_ID}: 缺少唇形同步视频, 跳过"
            continue
        fi

        # 获取 TTS 时长作为最终视频时长的基准
        if [[ -f "${TTS_AUDIO}" ]]; then
            FINAL_DUR=$(get_duration "${TTS_AUDIO}")
        else
            FINAL_DUR=$(get_duration "${LIPSYNC_VIDEO}")
        fi

        echo ""
        echo "  🎬 [case_${CASE_ID}] 合成最终视频 (目标时长: ${FINAL_DUR}s)"

        # 判断是否有额外音频需要混入
        HAS_BGM=false
        HAS_SFX=false
        [[ "${USE_BGM}" == "true" && -f "${BGM}" ]] && HAS_BGM=true
        [[ "${USE_SFX}" == "true" && -f "${SFX}" ]] && HAS_SFX=true

        if [[ "${HAS_BGM}" == "false" && "${HAS_SFX}" == "false" ]]; then
            # ---- 只有视频自带的 TTS 音频, 直接截断到目标时长 ----
            echo "     模式: 仅 TTS (无 BGM/SFX)"
            ${FFMPEG} -y -i "${LIPSYNC_VIDEO}" \
                -t "${FINAL_DUR}" \
                -c:v libx264 -crf 18 -preset medium \
                -c:a aac -b:a 192k \
                "${FINAL}" 2>/dev/null
        else
            # ---- 多音频混音: TTS + BGM + SFX ----
            # 构建 FFmpeg 输入和 filter_complex
            #   输入流:
            #     [0] = lipsync_video (含 TTS 音轨)
            #     [1] = BGM (如果有)
            #     [2] = SFX (如果有, 编号取决于 BGM 是否存在)
            FFMPEG_INPUTS="-i ${LIPSYNC_VIDEO}"
            STREAM_IDX=1

            if [[ "${HAS_BGM}" == "true" ]]; then
                FFMPEG_INPUTS="${FFMPEG_INPUTS} -i ${BGM}"
                BGM_IDX=${STREAM_IDX}
                STREAM_IDX=$((STREAM_IDX + 1))
            fi
            if [[ "${HAS_SFX}" == "true" ]]; then
                FFMPEG_INPUTS="${FFMPEG_INPUTS} -i ${SFX}"
                SFX_IDX=${STREAM_IDX}
                STREAM_IDX=$((STREAM_IDX + 1))
            fi

            AUDIO_COUNT=${STREAM_IDX}  # 总音频流数 (含 TTS)

            # filter_complex:
            #   1. TTS 音频: 从 lipsync_video 提取, 截断到目标时长
            #   2. BGM: 截断到目标时长, 调音量
            #   3. SFX: 截断到目标时长, 调音量
            #   4. amix 混合所有音频
            FILTER="[0:a]atrim=0:${FINAL_DUR},asetpts=PTS-STARTPTS,volume=${VOL_TTS}[tts];"
            MIX_INPUTS="[tts]"

            if [[ "${HAS_BGM}" == "true" ]]; then
                FILTER="${FILTER}[${BGM_IDX}:a]atrim=0:${FINAL_DUR},asetpts=PTS-STARTPTS,volume=${VOL_BGM}[bgm];"
                MIX_INPUTS="${MIX_INPUTS}[bgm]"
                echo "     BGM: ${BGM} (截断到 ${FINAL_DUR}s, vol=${VOL_BGM})"
            fi

            if [[ "${HAS_SFX}" == "true" ]]; then
                FILTER="${FILTER}[${SFX_IDX}:a]atrim=0:${FINAL_DUR},asetpts=PTS-STARTPTS,volume=${VOL_SFX}[sfx];"
                MIX_INPUTS="${MIX_INPUTS}[sfx]"
                echo "     SFX: ${SFX} (截断到 ${FINAL_DUR}s, vol=${VOL_SFX})"
            fi

            FILTER="${FILTER}${MIX_INPUTS}amix=inputs=${AUDIO_COUNT}:duration=first:dropout_transition=0[aout]"

            echo "     混音流数: ${AUDIO_COUNT}"

            ${FFMPEG} -y ${FFMPEG_INPUTS} \
                -t "${FINAL_DUR}" \
                -filter_complex "${FILTER}" \
                -map 0:v -map "[aout]" \
                -c:v libx264 -crf 18 -preset medium \
                -c:a aac -b:a 192k \
                "${FINAL}" 2>/dev/null
        fi

        if [[ -f "${FINAL}" ]]; then
            SIZE=$(du -h "${FINAL}" | cut -f1)
            OUT_DUR=$(get_duration "${FINAL}")
            echo "  ✅ case_${CASE_ID}: final.mp4 (${OUT_DUR}s, ${SIZE})"
        else
            echo "  ❌ case_${CASE_ID}: 合成失败"
        fi
    done
    echo "✅ Step 3c 完成: 最终视频合成"
fi

# ===== 总结 =====
echo ""
echo "============================================================"
echo "🎉 Pipeline2 Step 3 完成!"
echo "  输出目录: ${OUTPUT_BASE}"
echo ""
for CASE_ID in "${CASES[@]}"; do
    CASE_DIR="${OUTPUT_BASE}/case_${CASE_ID}"
    echo "  📁 case_${CASE_ID}:"
    for FILE in tts.wav bgm.wav sfx.wav raw_video.mp4 slowmo_video.mp4 lipsync_video.mp4 final.mp4; do
        FPATH="${CASE_DIR}/${FILE}"
        if [[ -f "${FPATH}" ]]; then
            SIZE=$(du -h "${FPATH}" | cut -f1)
            DUR=$(get_duration "${FPATH}" 2>/dev/null || echo "?")
            echo "    ✅ ${FILE} (${SIZE}, ${DUR}s)"
        else
            echo "    ⬜ ${FILE}"
        fi
    done
done
echo "============================================================"
