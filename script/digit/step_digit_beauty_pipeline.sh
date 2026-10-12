#!/bin/bash
# ============================================================================
# 深夜治愈IP 数字人美女 视频制作全 Pipeline
#
# 流程:
#   Step 1: Spark-TTS 生成台词语音 (TTS)
#   Step 2: ACE-Step BGM + TangoFlux SFX (音频, 复用 step2_generate_audio.sh 逻辑)
#   Step 3: HunyuanVideo-1.5 首帧生视频 (图片 → 视频)
#   Step 4: LatentSync 唇形同步 (视频 + TTS → 对口型视频)
#   Step 5: FFmpeg 合成最终视频 (对口型视频 + BGM + SFX → 最终输出)
#
# 输入:
#   - 5 张无水印图片: data/pics/20260323/no_watermark/*.png
#   - 5 条台词: data/text/digit/digit_beauty_20260316.tsv
#
# 输出:
#   - data/digit/20260323/case_{id}/final.mp4
#
# 用法:
#   bash step_digit_beauty_pipeline.sh                           # 全部步骤
#   bash step_digit_beauty_pipeline.sh --steps tts               # 只跑 TTS
#   bash step_digit_beauty_pipeline.sh --steps tts,lipsync       # 只跑 TTS + 唇形同步
#   bash step_digit_beauty_pipeline.sh --cases 1,2               # 只跑 case 1 和 2
#
# 多卡分配 (默认):
#   GPU 0: Spark-TTS
#   GPU 1: ACE-Step BGM
#   GPU 2: TangoFlux SFX
#   GPU 3: HunyuanVideo-1.5
#   GPU 4: LatentSync 唇形同步
#
# 依赖: 模型已下载到 group 盘, fast_sync.py 可用
# ============================================================================

set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
SYNC_SCRIPT=${GROUP_CODE}/fast_sync.py

DATE_TAG="20260323"
OUTPUT_BASE=${GROUP_BASE}/data/digit/${DATE_TAG}

# 图片目录 (group 盘上的无水印图片)
PICS_DIR=${GROUP_BASE}/data/pics/${DATE_TAG}/no_watermark

# GPU 分配
GPU_TTS=${GPU_TTS:-0}
GPU_BGM=${GPU_BGM:-1}
GPU_SFX=${GPU_SFX:-2}
GPU_VIDEO=${GPU_VIDEO:-3}
GPU_LIPSYNC=${GPU_LIPSYNC:-4}

# conda 环境
CONDA_ENV=${CONDA_ENV:-${IT2AV_CONDA_ENV}}

# ===== 5 个 Case 的配置 =====
# 图片按文件名排序对应 TSV 中的 5 条
CASE_IMAGES=(
    "1-卧室场景.png"
    "2-校园场景.png"
    "3-樱花道.png"
    "4-街景.png"
    "5-旗袍.png"
)

CASE_TTS_TEXTS=(
    "晚上好，生活再忙再累，也别忘了好好爱自己。"
    "晚上好，黑夜无论怎样悠长，白昼总会到来，一切都会慢慢变好。"
    "晚上好，世上只有一种英雄主义，就是在认清生活真相后依然热爱生活。"
    "晚上好，不必行色匆匆，不必光芒四射，只需做自己，早点休息吧。"
    "晚上好，万物皆有裂痕，那是光照进来的地方，愿你今夜好梦。"
)

CASE_BGM_PROMPTS=(
    "warm cozy ambient, soft piano and strings, gentle, relaxing, bedroom atmosphere, intimate, lo-fi, 70 bpm, instrumental"
    "dreamy night walk, cherry blossom, japanese ambient, soft piano and acoustic guitar, melancholic yet hopeful, 75 bpm, instrumental, peaceful"
    "campus youth, acoustic guitar, warm, inspirational, folk style, gentle drums, 80 bpm, instrumental, youthful energy"
    "urban night chill, lo-fi beats, city pop, smooth bass, soft synth pad, 85 bpm, instrumental, sophisticated night walk"
    "traditional Chinese, guzheng and bamboo flute, elegant, flowing water, peaceful bridge, 65 bpm, instrumental, ancient beauty"
)

CASE_SFX_PROMPTS=(
    "Quiet bedroom ambience at night, soft wind outside window, gentle fabric rustling sounds, warm cozy indoor atmosphere, peaceful nighttime"
    "Night cherry blossom garden ambience, gentle wind through trees, flower petals falling softly, distant crickets, quiet nighttime nature sounds"
    "Quiet campus pathway ambience, gentle breeze through trees, leaves rustling, distant birds singing, footsteps on stone path, peaceful school atmosphere"
    "City street night ambience, traffic light clicking, distant car sounds, heels clicking on crosswalk, gentle spring evening wind, urban nightlife atmosphere"
    "Small bridge over stream at night, flowing water sounds, lantern lights crackling softly, gentle wind, peaceful Chinese garden atmosphere, distant temple bells"
)

# ===== 命令行参数解析 =====
STEPS_CSV="tts,bgm,sfx,video,lipsync,merge"
CASES_CSV="1,2,3,4,5"
FORCE_SYNC=0
SKIP_SYNC=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)    STEPS_CSV="$2"; shift 2 ;;
        --cases)    CASES_CSV="$2"; shift 2 ;;
        --force_sync) FORCE_SYNC=1; shift ;;
        --skip_sync)  SKIP_SYNC=1; shift ;;
        --output)   OUTPUT_BASE="$2"; shift 2 ;;
        -h|--help)
            head -35 "$0" | tail -30
            exit 0 ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

IFS=',' read -r -a STEPS <<< "${STEPS_CSV}"
IFS=',' read -r -a CASES <<< "${CASES_CSV}"

# ===== 初始化 =====
echo "============================================================"
echo "🎬 深夜治愈IP 数字人美女 全 Pipeline"
echo "  输出目录:   ${OUTPUT_BASE}"
echo "  步骤:       ${STEPS_CSV}"
echo "  Case:       ${CASES_CSV}"
echo "  GPU 分配:   TTS=${GPU_TTS} BGM=${GPU_BGM} SFX=${GPU_SFX} Video=${GPU_VIDEO} LipSync=${GPU_LIPSYNC}"
echo "============================================================"

# 创建输出目录
for CASE_ID in "${CASES[@]}"; do
    mkdir -p "${OUTPUT_BASE}/case_${CASE_ID}"
done

# ===== 模型同步函数 =====
sync_model() {
    local MODEL_DIR=$1    # group 盘下的相对路径
    local SRC=${GROUP_BASE}/models/${MODEL_DIR}
    local DST=${LOCAL_BASE}/models/${MODEL_DIR}

    if [[ ! -d "${SRC}" ]]; then
        echo "  ⚠️  模型不存在: ${SRC}"
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

contains_step() {
    local step=$1
    for s in "${STEPS[@]}"; do
        [[ "$s" == "$step" ]] && return 0
    done
    return 1
}

# ################################################################
#  Step 1: Spark-TTS 台词语音生成
# ################################################################
if contains_step "tts"; then
    echo ""
    echo "============================================================"
    echo "🎤 Step 1: Spark-TTS 生成台词语音"
    echo "============================================================"

    # 同步 Spark-TTS 模型和代码
    echo ">>> 同步 Spark-TTS 模型..."
    SPARK_MODEL_PATH=$(sync_model "Audio/Spark-TTS-0.5B")
    echo ">>> 同步 Spark-TTS 代码..."
    SPARK_CODE_PATH=$(sync_code "utils/audio/Spark-TTS")

    echo ">>> 开始 TTS 推理..."
    for CASE_ID in "${CASES[@]}"; do
        IDX=$((CASE_ID - 1))
        TEXT="${CASE_TTS_TEXTS[$IDX]}"
        SAVE_PATH="${OUTPUT_BASE}/case_${CASE_ID}/tts.wav"

        if [[ -f "${SAVE_PATH}" && $(stat -c%s "${SAVE_PATH}" 2>/dev/null || stat -f%z "${SAVE_PATH}" 2>/dev/null) -gt 1024 ]]; then
            echo "  ⏭️  case_${CASE_ID} TTS 已存在"
            continue
        fi

        echo "  🎤 [case_${CASE_ID}] ${TEXT}"
        CUDA_VISIBLE_DEVICES=${GPU_TTS} \
        python -c "
import sys, os, torch, soundfile as sf
sys.path.insert(0, '${SPARK_CODE_PATH}')
from cli.SparkTTS import SparkTTS

device = torch.device('cuda:0')
model = SparkTTS('${SPARK_MODEL_PATH}', device)

with torch.no_grad():
    wav = model.inference(
        '''${TEXT}''',
        None,
        gender='female',
        pitch='moderate',
        speed='moderate',
    )
os.makedirs(os.path.dirname('${SAVE_PATH}'), exist_ok=True)
sf.write('${SAVE_PATH}', wav, samplerate=16000)
print(f'  ✅ 保存: ${SAVE_PATH} ({len(wav)/16000:.1f}s)')
del model
torch.cuda.empty_cache()
" 2>&1
    done
    echo "✅ Step 1 完成: TTS 语音生成"
fi

# ################################################################
#  Step 2: BGM + SFX (复用 step2_generate_audio.sh 逻辑)
# ################################################################
if contains_step "bgm" || contains_step "sfx"; then
    echo ""
    echo "============================================================"
    echo "🎵 Step 2: BGM + SFX 音频生成"
    echo "============================================================"

    # 生成 audio_prompts.json
    AUDIO_PROMPTS=${OUTPUT_BASE}/audio_prompts.json
    python3 -c "
import json

cases = []
tts_texts = $(python3 -c "import json; print(json.dumps([$(printf '"""%s""",' "${CASE_TTS_TEXTS[@]}" | sed 's/,$//')]))")
bgm_prompts = $(python3 -c "import json; print(json.dumps([$(printf '"""%s""",' "${CASE_BGM_PROMPTS[@]}" | sed 's/,$//')]))")
sfx_prompts = $(python3 -c "import json; print(json.dumps([$(printf '"""%s""",' "${CASE_SFX_PROMPTS[@]}" | sed 's/,$//')]))")

for i in range(5):
    cases.append({
        'id': f'case_{i+1}',
        'tts_text': tts_texts[i],
        'bgm_prompt': bgm_prompts[i],
        'bgm_lyrics': '[inst-long]',
        'bgm_duration': 15,
        'sfx_prompt': sfx_prompts[i],
        'sfx_duration': 15,
    })

with open('${AUDIO_PROMPTS}', 'w', encoding='utf-8') as f:
    json.dump({'cases': cases}, f, ensure_ascii=False, indent=2)
print(f'✅ Audio prompts: ${AUDIO_PROMPTS}')
" 2>&1

    echo "  ℹ️  BGM/SFX 生成请运行: bash step2_generate_audio.sh"
    echo "  或者在此脚本中 --steps 中只选择需要的步骤"
fi

# ################################################################
#  Step 3: HunyuanVideo-1.5 首帧生视频 (图片→视频)
# ################################################################
if contains_step "video"; then
    echo ""
    echo "============================================================"
    echo "🎥 Step 3: HunyuanVideo-1.5 首帧生视频"
    echo "============================================================"
    echo "  ℹ️  此步骤需要单独运行 HunyuanVideo-1.5 I2V 推理"
    echo "  输入: ${PICS_DIR}/*.png"
    echo "  输出: ${OUTPUT_BASE}/case_{id}/raw_video.mp4"
    echo ""
    echo "  请参考已有的 HunyuanVideo 推理脚本生成视频后再进行下一步"
fi

# ################################################################
#  Step 4: LatentSync 唇形同步
# ################################################################
if contains_step "lipsync"; then
    echo ""
    echo "============================================================"
    echo "👄 Step 4: LatentSync 唇形同步"
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

    # 逐个 case 运行唇形同步
    for CASE_ID in "${CASES[@]}"; do
        CASE_DIR="${OUTPUT_BASE}/case_${CASE_ID}"
        VIDEO_IN="${CASE_DIR}/raw_video.mp4"
        AUDIO_IN="${CASE_DIR}/tts.wav"
        VIDEO_OUT="${CASE_DIR}/lipsync_video.mp4"

        if [[ -f "${VIDEO_OUT}" && $(stat -c%s "${VIDEO_OUT}" 2>/dev/null || stat -f%z "${VIDEO_OUT}" 2>/dev/null) -gt 10240 ]]; then
            echo "  ⏭️  case_${CASE_ID} 唇形同步已存在"
            continue
        fi

        if [[ ! -f "${VIDEO_IN}" ]]; then
            echo "  ⚠️  case_${CASE_ID} 原始视频不存在: ${VIDEO_IN}"
            echo "     请先运行 Step 3 生成视频"
            continue
        fi

        if [[ ! -f "${AUDIO_IN}" ]]; then
            echo "  ⚠️  case_${CASE_ID} TTS 音频不存在: ${AUDIO_IN}"
            echo "     请先运行 Step 1 生成 TTS"
            continue
        fi

        echo ""
        echo "  👄 [case_${CASE_ID}] LatentSync 唇形同步..."
        echo "     视频: ${VIDEO_IN}"
        echo "     音频: ${AUDIO_IN}"
        echo "     输出: ${VIDEO_OUT}"

        CUDA_VISIBLE_DEVICES=${GPU_LIPSYNC} \
        python -m scripts.inference \
            --unet_config_path "${LATENTSYNC_CODE_PATH}/configs/unet/stage2_512.yaml" \
            --inference_ckpt_path "${CKPT_DIR}/latentsync_unet.pt" \
            --inference_steps 25 \
            --guidance_scale 1.5 \
            --enable_deepcache \
            --video_path "${VIDEO_IN}" \
            --audio_path "${AUDIO_IN}" \
            --video_out_path "${VIDEO_OUT}" \
            --temp_dir "${CASE_DIR}/temp_lipsync" \
            2>&1 | while IFS= read -r line; do echo "     ${line}"; done

        if [[ -f "${VIDEO_OUT}" ]]; then
            echo "  ✅ case_${CASE_ID} 唇形同步完成"
        else
            echo "  ❌ case_${CASE_ID} 唇形同步失败"
        fi
    done
    echo "✅ Step 4 完成: LatentSync 唇形同步"
fi

# ################################################################
#  Step 5: FFmpeg 合成最终视频
# ################################################################
if contains_step "merge"; then
    echo ""
    echo "============================================================"
    echo "🎬 Step 5: 合成最终视频 (对口型视频 + BGM + SFX)"
    echo "============================================================"

    for CASE_ID in "${CASES[@]}"; do
        CASE_DIR="${OUTPUT_BASE}/case_${CASE_ID}"
        LIPSYNC_VIDEO="${CASE_DIR}/lipsync_video.mp4"
        BGM="${CASE_DIR}/bgm.wav"
        SFX="${CASE_DIR}/sfx.wav"
        FINAL="${CASE_DIR}/final.mp4"

        if [[ ! -f "${LIPSYNC_VIDEO}" ]]; then
            echo "  ⚠️  case_${CASE_ID}: 缺少唇形同步视频, 跳过"
            continue
        fi

        echo "  🎬 [case_${CASE_ID}] 合成最终视频..."

        # 构建 FFmpeg 命令: 视频(含TTS音频) + BGM + SFX 混音
        FFMPEG_INPUTS="-i ${LIPSYNC_VIDEO}"
        FILTER_PARTS=""
        AUDIO_STREAMS=1    # [0:a] = lipsync video 自带的 TTS 音频

        if [[ -f "${BGM}" ]]; then
            FFMPEG_INPUTS="${FFMPEG_INPUTS} -i ${BGM}"
            AUDIO_STREAMS=$((AUDIO_STREAMS + 1))
        fi

        if [[ -f "${SFX}" ]]; then
            FFMPEG_INPUTS="${FFMPEG_INPUTS} -i ${SFX}"
            AUDIO_STREAMS=$((AUDIO_STREAMS + 1))
        fi

        if [[ ${AUDIO_STREAMS} -eq 1 ]]; then
            # 只有视频自带音频, 直接复制
            ffmpeg -y -i "${LIPSYNC_VIDEO}" -c copy "${FINAL}" 2>/dev/null
        else
            # 多音频混音: TTS 音量 1.0, BGM 音量 0.15, SFX 音量 0.2
            FILTER="[0:a]volume=1.0[tts];"
            MIX_INPUTS="[tts]"
            STREAM_IDX=1

            if [[ -f "${BGM}" ]]; then
                FILTER="${FILTER}[${STREAM_IDX}:a]volume=0.15[bgm];"
                MIX_INPUTS="${MIX_INPUTS}[bgm]"
                STREAM_IDX=$((STREAM_IDX + 1))
            fi

            if [[ -f "${SFX}" ]]; then
                FILTER="${FILTER}[${STREAM_IDX}:a]volume=0.2[sfx];"
                MIX_INPUTS="${MIX_INPUTS}[sfx]"
                STREAM_IDX=$((STREAM_IDX + 1))
            fi

            FILTER="${FILTER}${MIX_INPUTS}amix=inputs=${AUDIO_STREAMS}:duration=shortest[aout]"

            ffmpeg -y ${FFMPEG_INPUTS} \
                -filter_complex "${FILTER}" \
                -map 0:v -map "[aout]" \
                -c:v libx264 -crf 18 -preset medium \
                -c:a aac -b:a 192k \
                -shortest \
                "${FINAL}" 2>/dev/null

        fi

        if [[ -f "${FINAL}" ]]; then
            SIZE=$(du -h "${FINAL}" | cut -f1)
            echo "  ✅ case_${CASE_ID}: ${FINAL} (${SIZE})"
        else
            echo "  ❌ case_${CASE_ID}: 合成失败"
        fi
    done
    echo "✅ Step 5 完成: 最终视频合成"
fi

# ===== 总结 =====
echo ""
echo "============================================================"
echo "🎉 Pipeline 完成!"
echo "  输出目录: ${OUTPUT_BASE}"
echo ""
for CASE_ID in "${CASES[@]}"; do
    CASE_DIR="${OUTPUT_BASE}/case_${CASE_ID}"
    echo "  📁 case_${CASE_ID}:"
    for FILE in tts.wav bgm.wav sfx.wav raw_video.mp4 lipsync_video.mp4 final.mp4; do
        FPATH="${CASE_DIR}/${FILE}"
        if [[ -f "${FPATH}" ]]; then
            SIZE=$(du -h "${FPATH}" | cut -f1)
            echo "    ✅ ${FILE} (${SIZE})"
        else
            echo "    ⬜ ${FILE} (未生成)"
        fi
    done
done
echo "============================================================"
