#!/bin/bash
# ============================================================================
# Pipeline2 - Step 1: 生成音频 (TTS + BGM + SFX)
#
# 流程:
#   1a. Spark-TTS 生成台词语音 (GPU 0)
#   1b. ACE-Step 生成 BGM (GPU 1)
#   1c. TangoFlux 生成 SFX (GPU 2)
#   三者并行跑在不同 GPU 上
#
# 输入:
#   - 5 条台词 (硬编码在脚本中, 源自 digit_beauty_20260316.tsv)
#   - BGM/SFX prompts
#
# 输出:
#   ${OUTPUT_BASE}/case_{1..5}/
#     tts.wav    - Spark-TTS 台词语音
#     bgm.wav    - ACE-Step 背景音乐 (15s)
#     sfx.wav    - TangoFlux 环境音效 (15s)
#
# 用法:
#   bash step1_generate_audio.sh                        # 全部 case, TTS+BGM+SFX
#   bash step1_generate_audio.sh --cases 1,3            # 只跑 case 1 和 3
#   bash step1_generate_audio.sh --only tts             # 只跑 TTS
#   bash step1_generate_audio.sh --only bgm,sfx         # 只跑 BGM + SFX
#   bash step1_generate_audio.sh --skip_sync            # 不同步, 直接用 group 盘
#
# GPU 分配:
#   GPU 0: Spark-TTS
#   GPU 1: ACE-Step BGM
#   GPU 2: TangoFlux SFX
#
# 依赖: Spark-TTS / ACE-Step / TangoFlux 模型已下载到 group 盘
# ============================================================================

set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../common/env.sh"

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
SYNC_SCRIPT=${GROUP_CODE}/fast_sync.py

DATE_TAG="20260323"
OUTPUT_BASE=${GROUP_BASE}/data/digit/${DATE_TAG}

# GPU 分配
GPU_TTS=${GPU_TTS:-0}
GPU_BGM=${GPU_BGM:-1}
GPU_SFX=${GPU_SFX:-2}

# conda 环境
CONDA_ENV=${CONDA_ENV:-${IT2AV_CONDA_ENV}}

# ===== 5 个 Case 的配置 =====
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
CASES_CSV="1,2,3,4,5"
ONLY_CSV="tts,bgm,sfx"
FORCE_SYNC=0
SKIP_SYNC=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cases)      CASES_CSV="$2"; shift 2 ;;
        --only)       ONLY_CSV="$2"; shift 2 ;;
        --force_sync) FORCE_SYNC=1; shift ;;
        --skip_sync)  SKIP_SYNC=1; shift ;;
        --output)     OUTPUT_BASE="$2"; shift 2 ;;
        -h|--help)
            head -38 "$0" | tail -33
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

# ===== 初始化 =====
echo "============================================================"
echo "🎵 Pipeline2 Step 1: 生成音频 (TTS + BGM + SFX)"
echo "  输出目录:   ${OUTPUT_BASE}"
echo "  Case:       ${CASES_CSV}"
echo "  生成项:     ${ONLY_CSV}"
echo "  GPU 分配:   TTS=${GPU_TTS} BGM=${GPU_BGM} SFX=${GPU_SFX}"
echo "============================================================"

# 创建输出目录
for CASE_ID in "${CASES[@]}"; do
    mkdir -p "${OUTPUT_BASE}/case_${CASE_ID}"
done

# ===== 模型同步函数 =====
sync_model() {
    local MODEL_DIR=$1    # group 盘 models/ 下的相对路径
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
#  1a. Spark-TTS 台词语音生成
# ################################################################
run_tts() {
    echo ""
    echo "============================================================"
    echo "🎤 Step 1a: Spark-TTS 生成台词语音"
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
    echo "✅ TTS 语音生成完成"
}

# ################################################################
#  1b. ACE-Step BGM 背景音乐生成
# ################################################################
run_bgm() {
    echo ""
    echo "============================================================"
    echo "🎵 Step 1b: ACE-Step 生成 BGM"
    echo "============================================================"

    echo ">>> 同步 ACE-Step 模型..."
    ACE_MODEL_PATH=$(sync_model "Audio/ACE-Step-v1-3.5B")
    echo ">>> 同步 ACE-Step 代码..."
    ACE_CODE_PATH=$(sync_code "utils/audio/ACE-Step")

    echo ">>> 开始 BGM 生成..."
    for CASE_ID in "${CASES[@]}"; do
        IDX=$((CASE_ID - 1))
        PROMPT="${CASE_BGM_PROMPTS[$IDX]}"
        SAVE_PATH="${OUTPUT_BASE}/case_${CASE_ID}/bgm.wav"

        if [[ -f "${SAVE_PATH}" && $(stat -c%s "${SAVE_PATH}" 2>/dev/null || stat -f%z "${SAVE_PATH}" 2>/dev/null) -gt 1024 ]]; then
            echo "  ⏭️  case_${CASE_ID} BGM 已存在"
            continue
        fi

        echo "  🎵 [case_${CASE_ID}] ${PROMPT:0:60}..."
        CUDA_VISIBLE_DEVICES=${GPU_BGM} \
        PYTHONPATH="${ACE_CODE_PATH}:${GROUP_CODE}:${PYTHONPATH}" \
        python -c "
import sys, os, gc, time, torch
sys.path.insert(0, '${ACE_CODE_PATH}')
from acestep.pipeline_ace_step import ACEStepPipeline

pipeline = ACEStepPipeline(
    checkpoint_dir='${ACE_MODEL_PATH}',
    dtype='bfloat16',
    torch_compile=False,
    cpu_offload=False,
    overlapped_decode=False,
)

os.makedirs(os.path.dirname('${SAVE_PATH}'), exist_ok=True)
t0 = time.time()
pipeline(
    audio_duration=15,
    prompt='''${PROMPT}''',
    lyrics='[inst-long]',
    infer_step=100,
    guidance_scale=15.0,
    scheduler_type='euler',
    cfg_type='apg',
    omega_scale=10.0,
    manual_seeds='42',
    guidance_interval=0.5,
    guidance_interval_decay=0.0,
    min_guidance_scale=3.0,
    use_erg_tag=True,
    use_erg_lyric=False,
    use_erg_diffusion=True,
    oss_steps='',
    guidance_scale_text=0.0,
    guidance_scale_lyric=0.0,
    save_path='${SAVE_PATH}',
)
elapsed = time.time() - t0
print(f'  ✅ 保存: ${SAVE_PATH} (15s, 耗时 {elapsed:.1f}s)')
del pipeline
gc.collect()
torch.cuda.empty_cache()
" 2>&1
    done
    echo "✅ BGM 生成完成"
}

# ################################################################
#  1c. TangoFlux SFX 环境音效生成
# ################################################################
run_sfx() {
    echo ""
    echo "============================================================"
    echo "🔊 Step 1c: TangoFlux 生成 SFX"
    echo "============================================================"

    echo ">>> 同步 TangoFlux 模型..."
    TANGO_MODEL_PATH=$(sync_model "Audio/TangoFlux")
    FLANT5_MODEL_PATH=$(sync_model "Audio/flan-t5-large")
    echo ">>> 同步 TangoFlux 代码..."
    TANGO_CODE_PATH=$(sync_code "utils/audio/TangoFlux")

    echo ">>> 开始 SFX 生成..."
    for CASE_ID in "${CASES[@]}"; do
        IDX=$((CASE_ID - 1))
        PROMPT="${CASE_SFX_PROMPTS[$IDX]}"
        SAVE_PATH="${OUTPUT_BASE}/case_${CASE_ID}/sfx.wav"

        if [[ -f "${SAVE_PATH}" && $(stat -c%s "${SAVE_PATH}" 2>/dev/null || stat -f%z "${SAVE_PATH}" 2>/dev/null) -gt 1024 ]]; then
            echo "  ⏭️  case_${CASE_ID} SFX 已存在"
            continue
        fi

        echo "  🔊 [case_${CASE_ID}] ${PROMPT:0:60}..."
        CUDA_VISIBLE_DEVICES=${GPU_SFX} \
        PYTHONPATH="${TANGO_CODE_PATH}:${GROUP_CODE}:${PYTHONPATH}" \
        python -c "
import sys, os, gc, json, time, torch, torchaudio
sys.path.insert(0, '${TANGO_CODE_PATH}')
from diffusers import AutoencoderOobleck
from safetensors.torch import load_file
from tangoflux.model import TangoFlux as TangoFluxModel

device = 'cuda:0'
model_path = '${TANGO_MODEL_PATH}'
flant5_path = '${FLANT5_MODEL_PATH}'

vae = AutoencoderOobleck()
vae_weights = load_file(os.path.join(model_path, 'vae.safetensors'))
vae.load_state_dict(vae_weights)

with open(os.path.join(model_path, 'config.json'), 'r') as f:
    config = json.load(f)

if os.path.isdir(flant5_path):
    model = TangoFluxModel(config, text_encoder_dir=flant5_path)
else:
    model = TangoFluxModel(config)
weights = load_file(os.path.join(model_path, 'tangoflux.safetensors'))
model.load_state_dict(weights, strict=False)

vae = vae.to(device)
model = model.to(device)

os.makedirs(os.path.dirname('${SAVE_PATH}'), exist_ok=True)
t0 = time.time()
with torch.no_grad():
    latents = model.inference_flow(
        '''${PROMPT}''', duration=15, num_inference_steps=50, guidance_scale=4.5,
    )
    wave = vae.decode(latents.transpose(2, 1)).sample.cpu()[0]
waveform_end = int(15 * vae.config.sampling_rate)
audio = wave[:, :waveform_end]
torchaudio.save('${SAVE_PATH}', audio, sample_rate=44100)
elapsed = time.time() - t0
print(f'  ✅ 保存: ${SAVE_PATH} (15s, 耗时 {elapsed:.1f}s)')

del model, vae
gc.collect()
torch.cuda.empty_cache()
" 2>&1
    done
    echo "✅ SFX 生成完成"
}

# ################################################################
#  执行: 三者可并行
# ################################################################
STEP_START=$(date +%s)
PIDS=()
PID_NAMES=()

if contains_part "tts"; then
    run_tts > "${OUTPUT_BASE}/log_tts.txt" 2>&1 &
    PIDS+=($!)
    PID_NAMES+=("TTS")
    echo "  🚀 TTS 已启动 (PID: ${PIDS[-1]}, GPU ${GPU_TTS})"
fi

if contains_part "bgm"; then
    run_bgm > "${OUTPUT_BASE}/log_bgm.txt" 2>&1 &
    PIDS+=($!)
    PID_NAMES+=("BGM")
    echo "  🚀 BGM 已启动 (PID: ${PIDS[-1]}, GPU ${GPU_BGM})"
fi

if contains_part "sfx"; then
    run_sfx > "${OUTPUT_BASE}/log_sfx.txt" 2>&1 &
    PIDS+=($!)
    PID_NAMES+=("SFX")
    echo "  🚀 SFX 已启动 (PID: ${PIDS[-1]}, GPU ${GPU_SFX})"
fi

echo ""
echo ">>> 等待所有音频生成完成..."

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  ✅ ${PID_NAMES[$i]} 完成"
    else
        echo "  ❌ ${PID_NAMES[$i]} 失败! 查看 ${OUTPUT_BASE}/log_${PID_NAMES[$i],,}.txt"
        FAILED=1
    fi
done

STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))

# ===== 检查结果 =====
echo ""
echo "============================================================"
echo "🎵 Step 1 完成! (耗时: ${STEP_TIME}s)"
echo "  输出目录: ${OUTPUT_BASE}"
echo ""
for CASE_ID in "${CASES[@]}"; do
    echo "  📁 case_${CASE_ID}:"
    for TYPE in tts bgm sfx; do
        FILE="${OUTPUT_BASE}/case_${CASE_ID}/${TYPE}.wav"
        if [[ -f "$FILE" ]]; then
            SIZE=$(du -h "$FILE" | cut -f1)
            echo "    ✅ ${TYPE}.wav (${SIZE})"
        else
            echo "    ⬜ ${TYPE}.wav (未生成)"
        fi
    done
done

if [[ ${FAILED} -ne 0 ]]; then
    echo ""
    echo "  ⚠️  部分音频生成失败, 请检查日志:"
    echo "     ${OUTPUT_BASE}/log_tts.txt"
    echo "     ${OUTPUT_BASE}/log_bgm.txt"
    echo "     ${OUTPUT_BASE}/log_sfx.txt"
fi

echo ""
echo "  下一步: bash pipeline2/step2_generate_video.sh"
echo "============================================================"
