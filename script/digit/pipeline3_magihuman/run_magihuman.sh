#!/bin/bash
# ============================================================================
# Pipeline3 - MagiHuman: 首帧+文本 → 音频+视频 一键生成
#
# daVinci-MagiHuman 是一个 single-stream 音视频联合生成模型 (15B),
# 可以直接从 首帧图片 + Enhanced Prompt 同时生成音频和视频.
# 无需分步编排(TTS→Video→LipSync), 一步到位.
#
# 流程:
#   1. fast_sync 同步模型到本地盘 (/tmp)
#   2. torchrun 启动 MagiHuman 推理 (支持多卡)
#   3. 输出 mp4 (含音频+视频)
#
# 模式:
#   distill  - 蒸馏模型, 8步, 无CFG, 256p (最快, ~2s/5s视频 on H100)
#   sr540p   - base模型32步 + 540p超分 (推荐, 画质均衡)
#   sr1080p  - base模型32步 + 1080p超分 (最高画质, 需要更多显存)
#
# 输入:
#   ${INPUT_DIR}/case_{N}/
#     image.png       - 首帧图片
#     prompt.txt      - Enhanced Prompt (遵循 MagiHuman 格式规范)
#
# 输出:
#   ${OUTPUT_DIR}/case_{N}/
#     magihuman_video.mp4        - 音视频合并文件
#     magihuman_video_video.mp4  - 纯视频
#     magihuman_video_audio.wav  - 纯音频
#
# 用法:
#   bash run_magihuman.sh                                # 全部 case, distill 模式
#   bash run_magihuman.sh --mode sr540p                  # 540p 超分模式
#   bash run_magihuman.sh --mode sr1080p                 # 1080p 超分模式
#   bash run_magihuman.sh --cases 1,3                    # 只跑 case 1 和 3
#   bash run_magihuman.sh --seconds 10                   # 生成 10 秒视频
#   bash run_magihuman.sh --num_gpus 8                   # 8 卡并行 (cp_size=8)
#   bash run_magihuman.sh --skip_sync                    # 不同步, 直接用 group 盘
#
# GPU: 默认单卡 (distill 模式), H20 95GB 足够跑 256p/540p
# ============================================================================

set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../common/env.sh"

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
SYNC_SCRIPT=${GROUP_CODE}/fast_sync.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATE_TAG="20260324"

# ===== 默认参数 =====
MODE="sr1080p"               # distill | sr540p | sr1080p
CASES_CSV="1,2,3,4,5"
NUM_GPUS=${NUM_GPUS:-8}      # 8卡 H20 95GB
SECONDS_GEN=${SECONDS_GEN:-5}
SEED=42
FORCE_SYNC=0
SKIP_SYNC=0
INPUT_DIR=""                 # 留空则用默认路径
OUTPUT_DIR=""                # 留空则用默认路径

# conda
CONDA_ENV=${CONDA_ENV:-${IT2AV_CONDA_ENV}}

# ===== 命令行参数解析 =====
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)        MODE="$2"; shift 2 ;;
        --cases)       CASES_CSV="$2"; shift 2 ;;
        --num_gpus)    NUM_GPUS="$2"; shift 2 ;;
        --seconds)     SECONDS_GEN="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --input_dir)   INPUT_DIR="$2"; shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --force_sync)  FORCE_SYNC=1; shift ;;
        --skip_sync)   SKIP_SYNC=1; shift ;;
        --date_tag)    DATE_TAG="$2"; shift 2 ;;
        -h|--help)
            head -45 "$0" | tail -40
            exit 0 ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

# 设置输入/输出路径
INPUT_DIR=${INPUT_DIR:-${GROUP_BASE}/data/digit/${DATE_TAG}}
OUTPUT_DIR=${OUTPUT_DIR:-${GROUP_BASE}/data/digit/${DATE_TAG}}

IFS=',' read -r -a CASES <<< "${CASES_CSV}"

# 根据模式选择配置文件
case "${MODE}" in
    distill)
        CONFIG_JSON="${SCRIPT_DIR}/config_distill.json"
        BR_WIDTH=448; BR_HEIGHT=256
        SR_ARGS=""
        LABEL="distill (256p, 8步)"
        ;;
    sr540p)
        CONFIG_JSON="${SCRIPT_DIR}/config_sr540p.json"
        BR_WIDTH=448; BR_HEIGHT=256
        SR_ARGS="--sr_width 896 --sr_height 512"
        LABEL="base+SR (540p, 32+5步)"
        ;;
    sr1080p)
        CONFIG_JSON="${SCRIPT_DIR}/config_sr1080p.json"
        BR_WIDTH=448; BR_HEIGHT=256
        SR_ARGS="--sr_width 1920 --sr_height 1088"
        LABEL="base+SR (1080p, 32+5步)"
        export SR2_1080=true
        export CPU_OFFLOAD=true
        ;;
    *)
        echo "❌ 未知模式: ${MODE} (可选: distill, sr540p, sr1080p)"
        exit 1
        ;;
esac

# sr540p 也需要 CPU_OFFLOAD (官方 example/sr_540p/run.sh 默认开启)
if [[ "${MODE}" == "sr540p" ]]; then
    export CPU_OFFLOAD=true
fi

# ===== 初始化 =====
echo "============================================================"
echo "🎬 Pipeline3 - MagiHuman: 音视频联合生成"
echo "  模式:       ${MODE} - ${LABEL}"
echo "  GPU 数量:   ${NUM_GPUS}"
echo "  视频时长:   ${SECONDS_GEN} 秒"
echo "  种子:       ${SEED}"
echo "  输入目录:   ${INPUT_DIR}"
echo "  输出目录:   ${OUTPUT_DIR}"
echo "  Case:       ${CASES_CSV}"
echo "============================================================"

# ===== conda 环境激活 =====
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}
echo "✅ Conda 环境: $(conda info --envs | grep '*' | awk '{print $1}')"

# ===== Patch: diffusers torchao_quantizer.py 缺少 logger 定义 =====
# diffusers bug: torchao_quantizer.py 顶层 _update_torch_safe_globals() 的 except 分支
# 调用 logger.warning() 但该模块未定义 logger.
# 策略: 尝试 sed/cp 直接修改文件; 如果无权限则 chmod 后重试.
_CONDA_PFX="${CONDA_PREFIX:-${CONDA_ROOT}/envs/${CONDA_ENV}}"
TORCHAO_QF="${_CONDA_PFX}/lib/python3.12/site-packages/diffusers/quantizers/torchao/torchao_quantizer.py"
echo ">>> Patching diffusers torchao_quantizer.py logger bug..."
echo "    Target: ${TORCHAO_QF}"
echo "    File exists: $(test -f "${TORCHAO_QF}" && echo YES || echo NO)"
echo "    Writable: $(test -w "${TORCHAO_QF}" && echo YES || echo NO)"

if [[ -f "${TORCHAO_QF}" ]]; then
    # 检查是否已经 patch 过
    if head -3 "${TORCHAO_QF}" | grep -q '_logging'; then
        echo "    ✅ Already patched"
    else
        echo "    >>> File head before patch:"
        head -3 "${TORCHAO_QF}"

        # 确保可写
        chmod u+w "${TORCHAO_QF}" 2>/dev/null || true

        # 用 cp + cat 方式 (比 sed -i 更可靠)
        _TMP="${IT2AV_LOCAL_ROOT}/_tqf_patch_$$.py"
        mkdir -p "$(dirname "${_TMP}")"
        {
            echo 'import logging as _logging'
            echo 'logger = _logging.getLogger(__name__)'
            cat "${TORCHAO_QF}"
        } > "${_TMP}"

        if cp -f "${_TMP}" "${TORCHAO_QF}"; then
            echo "    ✅ Patched successfully"
        else
            echo "    ❌ Cannot write to ${TORCHAO_QF}"
            echo "    ❌ Please fix permissions: chmod a+w '${TORCHAO_QF}'"
            echo "    ❌ Or run: cp '${_TMP}' '${TORCHAO_QF}'"
            # 不要因为 patch 失败而退出整个脚本
        fi
        rm -f "${_TMP}"

        echo "    >>> File head after patch:"
        head -3 "${TORCHAO_QF}"
    fi
else
    echo "    ⚠️  File not found!"
    echo "    Listing parent dir:"
    ls -la "$(dirname "${TORCHAO_QF}")" 2>/dev/null || echo "    Parent dir does not exist"
fi

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
    # 代码目录总是强制重新同步(可能有新增文件如 magi_compiler shim)
    echo "  🔄 同步代码 ${CODE_DIR} → 本地盘" >&2
    python "${SYNC_SCRIPT}" "${SRC}" "${DST}" --workers 32 >&2
    touch "${DST}/.sync_done"
    echo "${DST}"
}

# ################################################################
#  Step 1: 同步模型
# ################################################################
echo ""
echo "============================================================"
echo "📦 Step 1: 同步模型到本地盘"
echo "============================================================"

echo ">>> 同步 daVinci-MagiHuman 模型..."
MAGIHUMAN_MODEL_PATH=$(sync_model "daVinci-MagiHuman")

echo ">>> 同步 stable-audio-open-1.0 (Audio VAE)..."
SA_AUDIO_PATH=$(sync_model "stable-audio-open-1.0")

echo ">>> 同步 t5gemma-9b-9b-ul2 (Text Encoder)..."
T5GEMMA_PATH=$(sync_model "t5gemma-9b-9b-ul2")

echo ">>> 同步 Wan2.2-TI2V-5B (Video VAE)..."
WAN_VAE_PATH=$(sync_model "Wan2.2-TI2V-5B")

echo ">>> 同步 daVinci-MagiHuman 代码..."
MAGIHUMAN_CODE_PATH=$(sync_code "utils/video/daVinci-MagiHuman")

echo ""
echo "  模型路径:"
echo "    MagiHuman:   ${MAGIHUMAN_MODEL_PATH}"
echo "    Audio VAE:   ${SA_AUDIO_PATH}"
echo "    Text Enc:    ${T5GEMMA_PATH}"
echo "    Video VAE:   ${WAN_VAE_PATH}"
echo "    代码:        ${MAGIHUMAN_CODE_PATH}"

# ################################################################
#  Step 2: 生成 config.json (动态替换路径)
# ################################################################
echo ""
echo "============================================================"
echo "⚙️  Step 2: 生成运行时 config"
echo "============================================================"

# 生成一个替换了实际路径的 config.json
RUNTIME_CONFIG="${OUTPUT_DIR}/_runtime_config_${MODE}.json"
mkdir -p "$(dirname "${RUNTIME_CONFIG}")"

python -c "
import json, sys

with open('${CONFIG_JSON}', 'r') as f:
    config = json.load(f)

# 替换模型路径
ec = config.get('engine_config', {})
ev = config.get('evaluation_config', {})

# MagiHuman 主模型
if ec.get('load'):
    # distill -> /path/models/daVinci-MagiHuman/distill
    # base    -> /path/models/daVinci-MagiHuman/base
    suffix = ec['load'].split('/')[-1]  # 'distill' or 'base'
    ec['load'] = '${MAGIHUMAN_MODEL_PATH}/' + suffix

# Audio VAE
if ev.get('audio_model_path'):
    ev['audio_model_path'] = '${SA_AUDIO_PATH}'

# Text Encoder
if ev.get('txt_model_path'):
    ev['txt_model_path'] = '${T5GEMMA_PATH}'

# Video VAE
if ev.get('vae_model_path'):
    ev['vae_model_path'] = '${WAN_VAE_PATH}'

# SR model
if ev.get('sr_model_path'):
    suffix = ev['sr_model_path'].split('/')[-1]  # '540p_sr' or '1080p_sr'
    ev['sr_model_path'] = '${MAGIHUMAN_MODEL_PATH}/' + suffix

# Turbo VAE
if ev.get('student_config_path'):
    ev['student_config_path'] = '${MAGIHUMAN_MODEL_PATH}/turbo_vae/TurboV3-Wan22-TinyShallow_7_7.json'
if ev.get('student_ckpt_path'):
    ev['student_ckpt_path'] = '${MAGIHUMAN_MODEL_PATH}/turbo_vae/checkpoint-340000.ckpt'

# 更新 cp_size 以匹配 NUM_GPUS
ec['cp_size'] = ${NUM_GPUS}

with open('${RUNTIME_CONFIG}', 'w') as f:
    json.dump(config, f, indent=2)

print(f'✅ Runtime config: ${RUNTIME_CONFIG}')
"

# ################################################################
#  Step 3: 逐 Case 推理
# ################################################################
echo ""
echo "============================================================"
echo "🎬 Step 3: MagiHuman 推理"
echo "============================================================"

STEP_START=$(date +%s)
TOTAL_CASES=${#CASES[@]}
CASE_IDX=0

for CASE_ID in "${CASES[@]}"; do
    CASE_IDX=$((CASE_IDX + 1))
    CASE_DIR="${OUTPUT_DIR}/case_${CASE_ID}"
    INPUT_CASE_DIR="${INPUT_DIR}/case_${CASE_ID}"
    mkdir -p "${CASE_DIR}"

    # 检查输入文件
    IMAGE_PATH="${INPUT_CASE_DIR}/image.png"
    PROMPT_PATH="${INPUT_CASE_DIR}/prompt.txt"

    # 也支持 jpg
    if [[ ! -f "${IMAGE_PATH}" ]]; then
        IMAGE_PATH="${INPUT_CASE_DIR}/image.jpg"
    fi

    if [[ ! -f "${IMAGE_PATH}" ]]; then
        echo "  ⚠️  [case_${CASE_ID}] 首帧图片不存在: ${INPUT_CASE_DIR}/image.{png,jpg}, 跳过"
        continue
    fi

    if [[ ! -f "${PROMPT_PATH}" ]]; then
        echo "  ⚠️  [case_${CASE_ID}] prompt.txt 不存在: ${PROMPT_PATH}, 跳过"
        continue
    fi

    # ---- 根据输入图片方向自适应分辨率 ----
    # 竖屏图片 (h > w) 时交换 width/height, 保持正确的宽高比
    IMG_DIMS=$(python -c "
from PIL import Image
im = Image.open('${IMAGE_PATH}')
w, h = im.size
print(f'{w} {h}')
")
    IMG_W=$(echo "${IMG_DIMS}" | awk '{print $1}')
    IMG_H=$(echo "${IMG_DIMS}" | awk '{print $2}')

    CASE_BR_W=${BR_WIDTH}
    CASE_BR_H=${BR_HEIGHT}
    CASE_SR_ARGS="${SR_ARGS}"

    if [[ ${IMG_H} -gt ${IMG_W} ]]; then
        # 竖屏: 交换 width 和 height
        CASE_BR_W=${BR_HEIGHT}
        CASE_BR_H=${BR_WIDTH}
        if [[ -n "${SR_ARGS}" ]]; then
            SR_W=$(echo "${SR_ARGS}" | grep -oP '(?<=--sr_width )\d+')
            SR_H=$(echo "${SR_ARGS}" | grep -oP '(?<=--sr_height )\d+')
            CASE_SR_ARGS="--sr_width ${SR_H} --sr_height ${SR_W}"
        fi
        echo "     📐 竖屏图片 (${IMG_W}x${IMG_H}): 分辨率交换为 ${CASE_BR_W}x${CASE_BR_H}"
    else
        echo "     📐 横屏图片 (${IMG_W}x${IMG_H}): 保持分辨率 ${CASE_BR_W}x${CASE_BR_H}"
    fi

    OUTPUT_PREFIX="${CASE_DIR}/magihuman_video"

    # pipeline.py 实际生成的文件名带有分辨率后缀:
    #   有SR: {prefix}_{seconds}s_{br_w}x{br_h}_{sr_w}x{sr_h}.mp4
    #   无SR: {prefix}_{seconds}s_{br_w}x{br_h}.mp4
    if [[ -n "${CASE_SR_ARGS}" ]]; then
        CASE_SR_W=$(echo "${CASE_SR_ARGS}" | grep -oP '(?<=--sr_width )\d+')
        CASE_SR_H=$(echo "${CASE_SR_ARGS}" | grep -oP '(?<=--sr_height )\d+')
        ACTUAL_OUTPUT="${OUTPUT_PREFIX}_${SECONDS_GEN}s_${CASE_BR_W}x${CASE_BR_H}_${CASE_SR_W}x${CASE_SR_H}.mp4"
    else
        ACTUAL_OUTPUT="${OUTPUT_PREFIX}_${SECONDS_GEN}s_${CASE_BR_W}x${CASE_BR_H}.mp4"
    fi

    # 检查是否已存在输出 (幂等)
    if [[ -f "${ACTUAL_OUTPUT}" ]]; then
        FILESIZE=$(stat -c%s "${ACTUAL_OUTPUT}" 2>/dev/null || stat -f%z "${ACTUAL_OUTPUT}" 2>/dev/null)
        if [[ ${FILESIZE} -gt 10240 ]]; then
            echo "  ⏭️  [${CASE_IDX}/${TOTAL_CASES}] case_${CASE_ID} 已存在 (${FILESIZE} bytes)"
            continue
        fi
    fi

    echo ""
    echo "  🎬 [${CASE_IDX}/${TOTAL_CASES}] case_${CASE_ID}: ${MODE} 模式, ${SECONDS_GEN}s"
    echo "     图片: ${IMAGE_PATH}"
    echo "     Prompt: $(head -c 80 "${PROMPT_PATH}")..."
    echo ""

    # 环境变量
    export MASTER_ADDR="${MASTER_ADDR:-localhost}"
    export MASTER_PORT="${MASTER_PORT:-6020}"
    export NNODES="${NNODES:-1}"
    export NODE_RANK="${NODE_RANK:-0}"
    export GPUS_PER_NODE="${NUM_GPUS}"
    export WORLD_SIZE="$((GPUS_PER_NODE * NNODES))"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
    export PYTHONPATH="${MAGIHUMAN_CODE_PATH}:${PYTHONPATH:-}"

    export TORCHDYNAMO_DISABLE=1

    DISTRIBUTED_ARGS="--nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${GPUS_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}"

    LOG_FILE="${CASE_DIR}/log_magihuman_${MODE}.txt"

    CASE_START=$(date +%s)

    torchrun ${DISTRIBUTED_ARGS} \
        "${MAGIHUMAN_CODE_PATH}/inference/pipeline/entry.py" \
        --config-load-path "${RUNTIME_CONFIG}" \
        --prompt "$(cat "${PROMPT_PATH}")" \
        --image_path "${IMAGE_PATH}" \
        --seconds "${SECONDS_GEN}" \
        --seed "${SEED}" \
        --br_width "${CASE_BR_W}" \
        --br_height "${CASE_BR_H}" \
        ${CASE_SR_ARGS} \
        --output_path "${OUTPUT_PREFIX}" \
        2>&1 | tee "${LOG_FILE}"

    CASE_END=$(date +%s)
    CASE_TIME=$((CASE_END - CASE_START))

    if [[ -f "${ACTUAL_OUTPUT}" ]]; then
        echo "  ✅ case_${CASE_ID} 完成 (${CASE_TIME}s): $(basename "${ACTUAL_OUTPUT}")"
    else
        echo "  ❌ case_${CASE_ID} 失败 (${CASE_TIME}s), 查看日志: ${LOG_FILE}"
    fi
done

STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))

# ===== 汇总结果 =====
echo ""
echo "============================================================"
echo "🎬 Pipeline3 MagiHuman 完成! (总耗时: ${STEP_TIME}s)"
echo "  模式:     ${MODE} - ${LABEL}"
echo "  输出目录: ${OUTPUT_DIR}"
echo ""
for CASE_ID in "${CASES[@]}"; do
    CASE_DIR="${OUTPUT_DIR}/case_${CASE_ID}"
    echo "  📁 case_${CASE_ID}:"
    # pipeline.py 生成的文件名带分辨率后缀, 用通配符匹配
    shopt -s nullglob
    FOUND_FILES=("${CASE_DIR}"/magihuman_video_*.mp4 "${CASE_DIR}"/magihuman_video_*_video.mp4 "${CASE_DIR}"/magihuman_video_*_audio.wav)
    shopt -u nullglob
    if [[ ${#FOUND_FILES[@]} -gt 0 ]]; then
        for FILE in "${FOUND_FILES[@]}"; do
            SIZE=$(du -h "${FILE}" | cut -f1)
            echo "    ✅ $(basename "${FILE}") (${SIZE})"
        done
    else
        echo "    ⬜ 未找到输出文件"
    fi
done
echo "============================================================"
