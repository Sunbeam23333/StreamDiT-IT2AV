#!/bin/bash
# ============================================================================
# LiveAvatar 5卡 TPP 数字人视频生成 Benchmark
# 服务器: 8×H20 95GB GPU (使用其中 5 卡: 4 DiT + 1 VAE)
# 模型: Wan2.2-S2V-14B (14B 基座) + LiveAvatar LoRA (DMD 4步蒸馏)
#
# 用法:
#   bash run_liveavatar_5gpu.sh                    # 默认跑 3 个样本
#   NUM_CASES=1 bash run_liveavatar_5gpu.sh        # 只跑 1 个样本
#   GPU_START=3 bash run_liveavatar_5gpu.sh        # 从 GPU3 开始用 (3,4,5,6,7)
#
# 数据准备 (自动执行，首次会自动生成):
#   1. generate_ref_portraits.py  # 调 API 生成参考人像 (需内网)
#   2. prepare_digit_samples.py   # edge-tts 生成音频 + JSON 样本列表
#   只需确保 digit_test.tsv 已同步到 group 盘即可
#
# 资源估算 (10分钟视频, ~300 clips × 48 frames):
#   5卡 H20: ~25-35分钟 | ~20GB VRAM/卡 | TPP 实时模式, FP8
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"
conda activate ${IT2AV_CONDA_ENV}

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}/utils/video/LiveAvatar

LOCAL_BASE=${IT2AV_LOCAL_ROOT}
LOCAL_CODE=${LOCAL_BASE}/code/LiveAvatar

SYNC_SCRIPT=${IT2AV_CODE_ROOT}/fast_sync.py

# 模型路径 (group 盘)
GROUP_MODEL_BASE=${GROUP_BASE}/models/Wan2.2-S2V-14B
GROUP_LORA=${GROUP_BASE}/models/LiveAvatar

# 模型路径 (本地高速盘)
LOCAL_MODEL_BASE=${LOCAL_BASE}/models/Wan2.2-S2V-14B
LOCAL_LORA_DIR=${LOCAL_BASE}/models/LiveAvatar
LOCAL_LORA=${LOCAL_LORA_DIR}/liveavatar.safetensors

# 数据路径
SAMPLE_LIST=${GROUP_BASE}/data/digit_samples/digit_sample_list.json
PORTRAIT_DIR=${GROUP_BASE}/data/pics/digit
AUDIO_DIR=${GROUP_BASE}/data/digit_samples/audio
TSV_FILE=${GROUP_BASE}/data/text/digit/digit_test.tsv

# 数据准备脚本 (在 LiveAvatar 代码目录下)
PREPARE_PORTRAITS=${GROUP_CODE}/generate_ref_portraits.py
PREPARE_SAMPLES=${GROUP_CODE}/prepare_digit_samples.py

# 输出路径 (保存到 group 盘)
RESULT_BASE=${GROUP_BASE}/results/liveavatar/5gpu

# 配置
TRAINING_CONFIG=${LOCAL_CODE}/liveavatar/configs/s2v_causal_sft.yaml
NUM_CASES=${NUM_CASES:-3}  # 默认跑 3 个样本

# GPU 配置: 支持从指定 GPU 开始连续选 5 张
GPU_START=${GPU_START:-0}
GPU_END=$((GPU_START + 4))
CUDA_DEVICES=""
for i in $(seq ${GPU_START} ${GPU_END}); do
    if [ -n "${CUDA_DEVICES}" ]; then
        CUDA_DEVICES="${CUDA_DEVICES},"
    fi
    CUDA_DEVICES="${CUDA_DEVICES}${i}"
done

# ===== 环境设置 =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
export ENABLE_COMPILE=false

# ===== 前置检查 & 数据准备 =====
echo ""
echo "============================================"
echo "前置检查 & 数据准备..."
echo "============================================"

# 1. 检查 TSV 源数据是否存在
if [ ! -f "${TSV_FILE}" ]; then
    echo "❌ TSV 源数据不存在: ${TSV_FILE}"
    echo "   请确保 digit_test.tsv 已同步到 group 盘"
    exit 1
fi
echo "✅ TSV 源数据: ${TSV_FILE}"

# 2. 生成参考人像 (如果人像目录为空或不存在)
PORTRAIT_COUNT=$(find ${PORTRAIT_DIR} -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
if [ "${PORTRAIT_COUNT}" -lt 1 ]; then
    echo ""
    echo "🎨 参考人像不存在，正在生成..."
    mkdir -p ${PORTRAIT_DIR}
    python ${PREPARE_PORTRAITS} \
        --tsv "${TSV_FILE}" \
        --output-dir "${PORTRAIT_DIR}" \
        --num-portraits 1
    echo ">>> 人像生成完成."
else
    echo "✅ 参考人像已存在: ${PORTRAIT_DIR} (${PORTRAIT_COUNT} 张)"
fi

# 3. 生成音频 + JSON 样本列表 (如果不存在)
#    使用 CosyVoice3 TTS (本地离线, 无需外网)
#    TTS 使用剩余 GPU (LiveAvatar 用 GPU_START~GPU_START+4, TTS 用其他卡)
if [ ! -f "${SAMPLE_LIST}" ]; then
    echo ""
    echo "🎙️  样本列表不存在，正在生成 (CosyVoice3 TTS + JSON)..."

    # 同步 CosyVoice 模型到本地盘
    COSYVOICE_CODE=${IT2AV_CODE_ROOT}/utils/audio/CosyVoice
    for MODEL_NAME in Fun-CosyVoice3-0.5B CosyVoice2-0.5B CosyVoice-ttsfrd; do
        MODEL_SRC=${GROUP_BASE}/models/Audio/${MODEL_NAME}
        MODEL_DST=${LOCAL_BASE}/models/Audio/${MODEL_NAME}
        if [ -d "${MODEL_SRC}" ] && [ ! -f "${MODEL_DST}/.sync_done" ]; then
            echo "  同步 ${MODEL_NAME}..."
            mkdir -p ${MODEL_DST}
            python ${SYNC_SCRIPT} ${MODEL_SRC} ${MODEL_DST} --workers 32 --skip-existing
            touch ${MODEL_DST}/.sync_done
        fi
    done

    # 安装 CosyVoice 依赖
    pip install pyworld torchaudio -q 2>/dev/null || true

    # 选择 TTS 用的 GPU: 用 LiveAvatar 5 卡之外的一张
    TTS_GPU=$((GPU_END + 1))
    if [ ${TTS_GPU} -gt 7 ]; then
        # 如果没有剩余 GPU, 用最后一张 (GPU7)
        TTS_GPU=7
    fi

    # CosyVoice 需要 Matcha-TTS 在 path 中
    export PYTHONPATH="${COSYVOICE_CODE}:${COSYVOICE_CODE}/third_party/Matcha-TTS:${IT2AV_CODE_ROOT}:$PYTHONPATH"

    mkdir -p $(dirname ${SAMPLE_LIST})
    CUDA_VISIBLE_DEVICES=${TTS_GPU} python ${PREPARE_SAMPLES} \
        --tsv "${TSV_FILE}" \
        --portrait-dir "${PORTRAIT_DIR}" \
        --output-dir "$(dirname ${SAMPLE_LIST})" \
        --max-duration 600 \
        --cosyvoice-model "${LOCAL_BASE}/models/Audio/Fun-CosyVoice3-0.5B" \
        --cosyvoice-code "${COSYVOICE_CODE}"
    echo ">>> 样本准备完成."

    if [ ! -f "${SAMPLE_LIST}" ]; then
        echo "❌ 样本列表生成失败: ${SAMPLE_LIST}"
        exit 1
    fi
else
    echo "✅ 样本列表已存在: ${SAMPLE_LIST}"
fi

# ===== 同步代码到本地盘 =====
echo "============================================"
echo "Syncing LiveAvatar code to local disk..."
echo "============================================"
mkdir -p ${LOCAL_CODE}
python ${SYNC_SCRIPT} ${GROUP_CODE} ${LOCAL_CODE} --workers ${SYNC_WORKERS:-32} --skip-existing
echo ">>> Code sync complete."

# ===== 同步模型到本地盘 =====
sync_models() {
    local MODEL_SRC=$1
    local MODEL_DST=$2
    local MODEL_NAME=$3
    local DONE_FLAG="${MODEL_DST}/.sync_done"

    if [ -f "${DONE_FLAG}" ]; then
        echo ">>> ${MODEL_NAME} 已同步到本地盘, 跳过."
        return
    fi

    echo "============================================"
    echo "Syncing ${MODEL_NAME} to local disk..."
    echo "  ${MODEL_SRC} -> ${MODEL_DST}"
    echo "============================================"
    mkdir -p ${MODEL_DST}
    python ${SYNC_SCRIPT} ${MODEL_SRC} ${MODEL_DST} --workers ${SYNC_WORKERS:-32} --skip-existing
    touch "${DONE_FLAG}"
    echo ">>> ${MODEL_NAME} sync complete."
}

sync_models "${GROUP_MODEL_BASE}" "${LOCAL_MODEL_BASE}" "Wan2.2-S2V-14B"
sync_models "${GROUP_LORA}" "${LOCAL_LORA_DIR}" "LiveAvatar-LoRA"

# ===== 安装可能缺失的依赖 =====
pip install dashscope bitsandbytes -q 2>/dev/null || true

# ===== 加载样本列表 =====
load_samples() {
    python3 -c "
import json, sys
with open('${SAMPLE_LIST}', 'r') as f:
    samples = json.load(f)
names = list(samples.keys())[:${NUM_CASES}]
for name in names:
    s = samples[name]
    print(f\"{name}|{s['image']}|{s['audio']}|{s['num_clip']}|{s['prompt']}|{s.get('duration_sec', 0)}\")
"
}

# ===== 5卡 TPP 推理 =====
mkdir -p ${RESULT_BASE}

echo ""
echo "============================================================"
echo "[5GPU] LiveAvatar 5卡 TPP 实时推理"
echo "  Model: Wan2.2-S2V-14B + LiveAvatar LoRA"
echo "  Model path: ${LOCAL_MODEL_BASE}"
echo "  LoRA path: ${LOCAL_LORA}"
echo "  Mode: 4 DiT GPU + 1 VAE GPU, FP8, 4 steps"
echo "  Resolution: 720x400"
echo "  GPUs: ${CUDA_DEVICES}"
echo "  Cases: ${NUM_CASES}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}

OVERALL_START=$(date +%s)
CASE_IDX=0

while IFS='|' read -r SAMPLE_NAME IMAGE_PATH AUDIO_PATH NUM_CLIP PROMPT DURATION; do
    CASE_IDX=$((CASE_IDX + 1))
    echo ""
    echo "------------------------------------------------------------"
    echo "[5GPU] Case ${CASE_IDX}/${NUM_CASES}: ${SAMPLE_NAME}"
    echo "  Image: ${IMAGE_PATH}"
    echo "  Audio: ${AUDIO_PATH}"
    echo "  Clips: ${NUM_CLIP}"
    echo "  Duration: ~${DURATION}s"
    echo "------------------------------------------------------------"

    CASE_START=$(date +%s)

    cd ${LOCAL_CODE}
    torchrun --nproc_per_node=5 --master_port=$((29200 + CASE_IDX)) \
        minimal_inference/s2v_streaming_interact.py \
        --task s2v-14B \
        --ckpt_dir "${LOCAL_MODEL_BASE}" \
        --size "720*400" \
        --base_seed 420 \
        --training_config ${TRAINING_CONFIG} \
        --offload_model False \
        --convert_model_dtype \
        --prompt "${PROMPT}" \
        --image "${IMAGE_PATH}" \
        --audio "${AUDIO_PATH}" \
        --infer_frames 48 \
        --load_lora \
        --lora_path_dmd "${LOCAL_LORA}" \
        --sample_steps 4 \
        --sample_guide_scale 0 \
        --num_clip ${NUM_CLIP} \
        --num_gpus_dit 4 \
        --sample_solver euler \
        --enable_vae_parallel \
        --save_dir "${RESULT_BASE}/" \
        --save_file "${RESULT_BASE}/${SAMPLE_NAME}.mp4" \
        --fp8

    CASE_END=$(date +%s)
    CASE_TIME=$((CASE_END - CASE_START))
    echo ">>> [5GPU] Case ${CASE_IDX} 完成: ${CASE_TIME}s (${SAMPLE_NAME})"

done < <(load_samples)

OVERALL_END=$(date +%s)
OVERALL_TIME=$((OVERALL_END - OVERALL_START))

echo ""
echo "============================================================"
echo "[5GPU] Benchmark 完成!"
echo "  总耗时: ${OVERALL_TIME}s ($((OVERALL_TIME / 60))min)"
echo "  平均每样本: $((OVERALL_TIME / CASE_IDX))s"
echo "  输出目录: ${RESULT_BASE}"
echo "  GPU: 5×H20 (${CUDA_DEVICES}), 4 DiT + 1 VAE, TPP mode, FP8"
echo "============================================================"
