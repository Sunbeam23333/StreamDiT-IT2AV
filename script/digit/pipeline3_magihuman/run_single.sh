#!/bin/bash
# ============================================================================
# MagiHuman 单次推理脚本 (精简版)
#
# 只跑一个 case: 指定一张图片 + prompt, 生成音视频.
# 自动检测输入图片方向(竖屏/横屏)并适配分辨率.
#
# 用法:
#   bash run_single.sh --image /path/to/image.png --prompt "your prompt"
#   bash run_single.sh --image /path/to/image.png --prompt_file /path/to/prompt.txt
#   bash run_single.sh --image /path/to/image.png --prompt "..." --seconds 10
#   bash run_single.sh --image /path/to/image.png --prompt "..." --mode sr540p
#   bash run_single.sh --image /path/to/image.png --prompt "..." --output_dir /path/to/output
#   bash run_single.sh --image /path/to/image.png --prompt "..." --skip_sync
#
# 模式:
#   distill  - 蒸馏模型, 8步, 无CFG, 256p (最快)
#   sr540p   - base模型32步 + 540p超分
#   sr1080p  - base模型32步 + 1080p超分 (最高画质)
#
# 输出:
#   {output_dir}/magihuman_video_{seconds}s_{w}x{h}[_{sr_w}x{sr_h}].mp4
#   {output_dir}/magihuman_video_{seconds}s_..._video.mp4
#   {output_dir}/magihuman_video_{seconds}s_..._audio.wav
#   {output_dir}/log_magihuman_{mode}.txt
# ============================================================================

set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../common/env.sh"

# ===== 路径配置 (默认值, 可通过参数覆盖) =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
SYNC_SCRIPT=${GROUP_CODE}/fast_sync.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== 默认参数 =====
MODE="sr1080p"
SECONDS_GEN=${SECONDS_GEN:-5}
SEED=42
NUM_GPUS=${NUM_GPUS:-8}
SKIP_SYNC=0
FORCE_SYNC=0
IMAGE_PATH=""
PROMPT_TEXT=""
PROMPT_FILE=""
OUTPUT_DIR=""
LOG_DIR=""
CONDA_ENV=${CONDA_ENV:-${IT2AV_CONDA_ENV}}

# ===== 命令行参数解析 =====
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)        IMAGE_PATH="$2"; shift 2 ;;
        --prompt)       PROMPT_TEXT="$2"; shift 2 ;;
        --prompt_file)  PROMPT_FILE="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --seconds)      SECONDS_GEN="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --num_gpus)     NUM_GPUS="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --log_dir)      LOG_DIR="$2"; shift 2 ;;
        --skip_sync)    SKIP_SYNC=1; shift ;;
        --force_sync)   FORCE_SYNC=1; shift ;;
        -h|--help)
            head -30 "$0" | tail -25
            exit 0 ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

# ===== 参数校验 =====
if [[ -z "${IMAGE_PATH}" ]]; then
    echo "❌ 必须指定 --image 参数"
    echo "   用法: bash run_single.sh --image /path/to/image.png --prompt \"your prompt\""
    exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "❌ 图片不存在: ${IMAGE_PATH}"
    exit 1
fi

# prompt: 优先用 --prompt_file, 其次 --prompt
if [[ -n "${PROMPT_FILE}" ]]; then
    if [[ ! -f "${PROMPT_FILE}" ]]; then
        echo "❌ prompt 文件不存在: ${PROMPT_FILE}"
        exit 1
    fi
    PROMPT_TEXT="$(cat "${PROMPT_FILE}")"
fi

if [[ -z "${PROMPT_TEXT}" ]]; then
    echo "❌ 必须指定 --prompt 或 --prompt_file"
    exit 1
fi

# 默认输出目录: 图片所在目录
OUTPUT_DIR=${OUTPUT_DIR:-$(dirname "${IMAGE_PATH}")}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}}
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ===== 根据模式选择配置 =====
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

# ===== 根据输入图片方向自适应分辨率 =====
IMG_DIMS=$(python -c "
from PIL import Image
im = Image.open('${IMAGE_PATH}')
w, h = im.size
print(f'{w} {h}')
")
IMG_W=$(echo "${IMG_DIMS}" | awk '{print $1}')
IMG_H=$(echo "${IMG_DIMS}" | awk '{print $2}')

if [[ ${IMG_H} -gt ${IMG_W} ]]; then
    # 竖屏: 交换 width 和 height
    ACTUAL_BR_W=${BR_HEIGHT}
    ACTUAL_BR_H=${BR_WIDTH}
    if [[ -n "${SR_ARGS}" ]]; then
        SR_W=$(echo "${SR_ARGS}" | grep -oP '(?<=--sr_width )\d+')
        SR_H=$(echo "${SR_ARGS}" | grep -oP '(?<=--sr_height )\d+')
        SR_ARGS="--sr_width ${SR_H} --sr_height ${SR_W}"
    fi
    ORIENTATION="竖屏"
else
    ACTUAL_BR_W=${BR_WIDTH}
    ACTUAL_BR_H=${BR_HEIGHT}
    ORIENTATION="横屏"
fi

# ===== 显示配置 =====
echo "============================================================"
echo "🎬 MagiHuman 单次推理"
echo "  模式:       ${MODE} - ${LABEL}"
echo "  GPU 数量:   ${NUM_GPUS}"
echo "  视频时长:   ${SECONDS_GEN} 秒"
echo "  种子:       ${SEED}"
echo "  图片:       ${IMAGE_PATH} (${IMG_W}x${IMG_H}, ${ORIENTATION})"
echo "  Base分辨率: ${ACTUAL_BR_W}x${ACTUAL_BR_H}"
if [[ -n "${SR_ARGS}" ]]; then
    echo "  SR分辨率:   ${SR_ARGS}"
fi
echo "  Prompt:     $(echo "${PROMPT_TEXT}" | head -c 100)..."
echo "  输出目录:   ${OUTPUT_DIR}"
echo "  日志目录:   ${LOG_DIR}"
echo "============================================================"

# ===== conda 环境激活 =====
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}
echo "✅ Conda 环境: $(conda info --envs | grep '*' | awk '{print $1}')"

# ===== Patch: diffusers torchao_quantizer.py 缺少 logger 定义 =====
_CONDA_PFX="${CONDA_PREFIX:-${CONDA_ROOT}/envs/${CONDA_ENV}}"
TORCHAO_QF="${_CONDA_PFX}/lib/python3.12/site-packages/diffusers/quantizers/torchao/torchao_quantizer.py"
if [[ -f "${TORCHAO_QF}" ]]; then
    if ! head -3 "${TORCHAO_QF}" | grep -q '_logging'; then
        echo ">>> Patching diffusers torchao_quantizer.py logger bug..."
        chmod u+w "${TORCHAO_QF}" 2>/dev/null || true
        _TMP="${IT2AV_LOCAL_ROOT}/_tqf_patch_$$.py"
        mkdir -p "$(dirname "${_TMP}")"
        { echo 'import logging as _logging'; echo 'logger = _logging.getLogger(__name__)'; cat "${TORCHAO_QF}"; } > "${_TMP}"
        cp -f "${_TMP}" "${TORCHAO_QF}" 2>/dev/null && echo "  ✅ Patched" || echo "  ⚠️  Patch failed (non-critical)"
        rm -f "${_TMP}"
    fi
fi

# ===== 模型同步 =====
sync_model() {
    local MODEL_DIR=$1
    local SRC=${GROUP_BASE}/models/${MODEL_DIR}
    local DST=${LOCAL_BASE}/models/${MODEL_DIR}

    if [[ ! -d "${SRC}" ]]; then
        echo "  ⚠️  模型不存在: ${SRC}" >&2
        echo "${SRC}"; return 0
    fi
    if [[ ${SKIP_SYNC} -eq 1 ]]; then
        echo "${SRC}"; return 0
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
        echo "${SRC}"; return 0
    fi
    if [[ ${SKIP_SYNC} -eq 1 ]]; then
        echo "${SRC}"; return 0
    fi
    mkdir -p "${DST}"
    echo "  🔄 同步代码 ${CODE_DIR} → 本地盘" >&2
    python "${SYNC_SCRIPT}" "${SRC}" "${DST}" --workers 32 >&2
    touch "${DST}/.sync_done"
    echo "${DST}"
}

echo ""
echo "📦 同步模型到本地盘..."
MAGIHUMAN_MODEL_PATH=$(sync_model "daVinci-MagiHuman")
SA_AUDIO_PATH=$(sync_model "stable-audio-open-1.0")
T5GEMMA_PATH=$(sync_model "t5gemma-9b-9b-ul2")
WAN_VAE_PATH=$(sync_model "Wan2.2-TI2V-5B")
MAGIHUMAN_CODE_PATH=$(sync_code "utils/video/daVinci-MagiHuman")

echo ""
echo "  模型路径:"
echo "    MagiHuman:   ${MAGIHUMAN_MODEL_PATH}"
echo "    Audio VAE:   ${SA_AUDIO_PATH}"
echo "    Text Enc:    ${T5GEMMA_PATH}"
echo "    Video VAE:   ${WAN_VAE_PATH}"
echo "    代码:        ${MAGIHUMAN_CODE_PATH}"

# ===== 生成 config.json =====
echo ""
echo "⚙️  生成运行时 config..."
RUNTIME_CONFIG="${OUTPUT_DIR}/_runtime_config_${MODE}.json"

python -c "
import json

with open('${CONFIG_JSON}', 'r') as f:
    config = json.load(f)

ec = config.get('engine_config', {})
ev = config.get('evaluation_config', {})

if ec.get('load'):
    suffix = ec['load'].split('/')[-1]
    ec['load'] = '${MAGIHUMAN_MODEL_PATH}/' + suffix

if ev.get('audio_model_path'):
    ev['audio_model_path'] = '${SA_AUDIO_PATH}'
if ev.get('txt_model_path'):
    ev['txt_model_path'] = '${T5GEMMA_PATH}'
if ev.get('vae_model_path'):
    ev['vae_model_path'] = '${WAN_VAE_PATH}'
if ev.get('sr_model_path'):
    suffix = ev['sr_model_path'].split('/')[-1]
    ev['sr_model_path'] = '${MAGIHUMAN_MODEL_PATH}/' + suffix
if ev.get('student_config_path'):
    ev['student_config_path'] = '${MAGIHUMAN_MODEL_PATH}/turbo_vae/TurboV3-Wan22-TinyShallow_7_7.json'
if ev.get('student_ckpt_path'):
    ev['student_ckpt_path'] = '${MAGIHUMAN_MODEL_PATH}/turbo_vae/checkpoint-340000.ckpt'

ec['cp_size'] = ${NUM_GPUS}

with open('${RUNTIME_CONFIG}', 'w') as f:
    json.dump(config, f, indent=2)

print(f'✅ Runtime config: ${RUNTIME_CONFIG}')
"

# ===== 推理 =====
echo ""
echo "============================================================"
echo "🎬 开始推理..."
echo "============================================================"

OUTPUT_PREFIX="${OUTPUT_DIR}/magihuman_video"

# 计算预期输出文件名
if [[ -n "${SR_ARGS}" ]]; then
    _SR_W=$(echo "${SR_ARGS}" | grep -oP '(?<=--sr_width )\d+')
    _SR_H=$(echo "${SR_ARGS}" | grep -oP '(?<=--sr_height )\d+')
    ACTUAL_OUTPUT="${OUTPUT_PREFIX}_${SECONDS_GEN}s_${ACTUAL_BR_W}x${ACTUAL_BR_H}_${_SR_W}x${_SR_H}.mp4"
else
    ACTUAL_OUTPUT="${OUTPUT_PREFIX}_${SECONDS_GEN}s_${ACTUAL_BR_W}x${ACTUAL_BR_H}.mp4"
fi

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

LOG_FILE="${LOG_DIR}/log_magihuman_${MODE}.txt"

INFERENCE_START=$(date +%s)

torchrun ${DISTRIBUTED_ARGS} \
    "${MAGIHUMAN_CODE_PATH}/inference/pipeline/entry.py" \
    --config-load-path "${RUNTIME_CONFIG}" \
    --prompt "${PROMPT_TEXT}" \
    --image_path "${IMAGE_PATH}" \
    --seconds "${SECONDS_GEN}" \
    --seed "${SEED}" \
    --br_width "${ACTUAL_BR_W}" \
    --br_height "${ACTUAL_BR_H}" \
    ${SR_ARGS} \
    --output_path "${OUTPUT_PREFIX}" \
    2>&1 | tee "${LOG_FILE}"

INFERENCE_END=$(date +%s)
INFERENCE_TIME=$((INFERENCE_END - INFERENCE_START))

# ===== 结果 =====
echo ""
echo "============================================================"
if [[ -f "${ACTUAL_OUTPUT}" ]]; then
    FILESIZE=$(du -h "${ACTUAL_OUTPUT}" | cut -f1)
    echo "✅ 推理完成! (耗时: ${INFERENCE_TIME}s)"
    echo "  输出: ${ACTUAL_OUTPUT} (${FILESIZE})"
    # 列出所有生成的文件
    echo ""
    echo "  生成文件:"
    shopt -s nullglob
    for f in "${OUTPUT_DIR}"/magihuman_video_*; do
        SIZE=$(du -h "${f}" | cut -f1)
        echo "    $(basename "${f}") (${SIZE})"
    done
    shopt -u nullglob
else
    echo "❌ 推理失败! (耗时: ${INFERENCE_TIME}s)"
    echo "  请查看日志: ${LOG_FILE}"
fi
echo "============================================================"
