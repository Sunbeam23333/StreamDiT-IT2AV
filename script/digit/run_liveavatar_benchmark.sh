#!/bin/bash
# ============================================================================
# LiveAvatar 数字人视频生成 Benchmark
# 服务器: 8×H20 95GB GPU
# 模型: Wan2.2-S2V-14B (14B 基座) + LiveAvatar LoRA (DMD 4步蒸馏)
#
# 支持模式:
#   1gpu  - 单卡 H20 95GB (offload 模式, 适合调试)
#   5gpu  - 5卡 TPP 实时推理 (4 DiT + 1 VAE, 推荐生产)
#   8gpu  - 8卡 (5卡推理, 3卡空闲, 方便资源利用)
#
# 用法:
#   bash run_liveavatar_benchmark.sh 1gpu          # 单卡推理
#   bash run_liveavatar_benchmark.sh 5gpu          # 5卡 TPP
#   bash run_liveavatar_benchmark.sh all           # 两种模式都跑
#   NUM_CASES=1 bash run_liveavatar_benchmark.sh 1gpu  # 只跑第1个样本
#
# 数据准备 (需先执行):
#   1. python generate_ref_portraits.py   # 生成参考人像
#   2. python prepare_digit_samples.py    # TTS 生成音频 + JSON 样本列表
#
# 10分钟视频资源估算:
#   - 单卡 H20: ~2.5~3小时 (offload 模式, FP8)
#   - 5卡 H20: ~25~35分钟 (TPP 实时模式, FP8, 约45 FPS)
#   - 显存: 单卡 ~80GB, 5卡 ~20GB/卡 (FP8)
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

# 输出路径 (保存到 group 盘)
RESULT_BASE=${GROUP_BASE}/results/liveavatar

# 配置
TRAINING_CONFIG=${LOCAL_CODE}/liveavatar/configs/s2v_causal_sft.yaml
NUM_CASES=${NUM_CASES:-3}  # 默认跑 3 个样本 (TSV 中有 3 个)

# ===== 解析命令行参数 =====
RUN_1GPU=false
RUN_5GPU=false

if [ $# -eq 0 ]; then
    echo "============================================================"
    echo "LiveAvatar 数字人视频生成 Benchmark"
    echo "============================================================"
    echo ""
    echo "Usage: bash $0 {1gpu|5gpu|all}"
    echo ""
    echo "Modes:"
    echo "  1gpu  - 单卡 H20 95GB (offload, 调试用)"
    echo "  5gpu  - 5卡 TPP 实时推理 (推荐)"
    echo "  all   - 两种模式都跑"
    echo ""
    echo "Env vars:"
    echo "  NUM_CASES=3     - 跑几个样本 (默认 3)"
    echo "  CUDA_ID=0       - 单卡模式 GPU index"
    echo ""
    echo "10分钟视频资源估算:"
    echo "  单卡 H20:  ~2.5-3小时 | ~80GB VRAM"
    echo "  5卡 H20:   ~25-35分钟 | ~20GB VRAM/卡"
    exit 1
fi

for arg in "$@"; do
    case "${arg}" in
        1gpu) RUN_1GPU=true ;;
        5gpu) RUN_5GPU=true ;;
        all)  RUN_1GPU=true; RUN_5GPU=true ;;
        *)    echo "Unknown argument: ${arg}"; exit 1 ;;
    esac
done

# ===== 环境设置 =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
# H20 不支持 torch.compile 的某些优化，关闭
export ENABLE_COMPILE=false

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
    # 用 python 解析 JSON, 提取每个样本的信息
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

# ===== 单卡推理 =====
run_1gpu() {
    local save_dir=${RESULT_BASE}/1gpu
    mkdir -p ${save_dir}

    echo ""
    echo "============================================================"
    echo "[1GPU] LiveAvatar 单卡 H20 推理"
    echo "  Model: Wan2.2-S2V-14B + LiveAvatar LoRA"
    echo "  Model path: ${LOCAL_MODEL_BASE}"
    echo "  LoRA path: ${LOCAL_LORA}"
    echo "  Mode: offload, FP8, 4 steps"
    echo "  Resolution: 704x384"
    echo "  Cases: ${NUM_CASES}"
    echo "============================================================"

    export CUDA_VISIBLE_DEVICES=${CUDA_ID:-0}

    local OVERALL_START=$(date +%s)
    local CASE_IDX=0

    while IFS='|' read -r SAMPLE_NAME IMAGE_PATH AUDIO_PATH NUM_CLIP PROMPT DURATION; do
        CASE_IDX=$((CASE_IDX + 1))
        echo ""
        echo "------------------------------------------------------------"
        echo "[1GPU] Case ${CASE_IDX}/${NUM_CASES}: ${SAMPLE_NAME}"
        echo "  Image: ${IMAGE_PATH}"
        echo "  Audio: ${AUDIO_PATH}"
        echo "  Clips: ${NUM_CLIP}"
        echo "  Duration: ~${DURATION}s"
        echo "------------------------------------------------------------"

        local CASE_START=$(date +%s)

        cd ${LOCAL_CODE}
        torchrun --nproc_per_node=1 --master_port=$((29100 + CASE_IDX)) \
            minimal_inference/s2v_streaming_interact.py \
            --task s2v-14B \
            --ckpt_dir "${LOCAL_MODEL_BASE}" \
            --size "704*384" \
            --base_seed 420 \
            --training_config ${TRAINING_CONFIG} \
            --offload_model True \
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
            --num_gpus_dit 1 \
            --sample_solver euler \
            --single_gpu \
            --save_dir "${save_dir}/" \
            --save_file "${save_dir}/${SAMPLE_NAME}.mp4" \
            --fp8

        local CASE_END=$(date +%s)
        local CASE_TIME=$((CASE_END - CASE_START))
        echo ">>> [1GPU] Case ${CASE_IDX} 完成: ${CASE_TIME}s (${SAMPLE_NAME})"

    done < <(load_samples)

    local OVERALL_END=$(date +%s)
    local OVERALL_TIME=$((OVERALL_END - OVERALL_START))

    echo ""
    echo "============================================================"
    echo "[1GPU] Benchmark 完成!"
    echo "  总耗时: ${OVERALL_TIME}s ($((OVERALL_TIME / 60))min)"
    echo "  平均每样本: $((OVERALL_TIME / CASE_IDX))s"
    echo "  输出目录: ${save_dir}"
    echo "  GPU: 1×H20 95GB, offload mode, FP8"
    echo "============================================================"
}

# ===== 5卡 TPP 推理 =====
run_5gpu() {
    local save_dir=${RESULT_BASE}/5gpu
    mkdir -p ${save_dir}

    echo ""
    echo "============================================================"
    echo "[5GPU] LiveAvatar 5卡 TPP 实时推理"
    echo "  Model: Wan2.2-S2V-14B + LiveAvatar LoRA"
    echo "  Model path: ${LOCAL_MODEL_BASE}"
    echo "  LoRA path: ${LOCAL_LORA}"
    echo "  Mode: 4 DiT GPU + 1 VAE GPU, FP8, 4 steps"
    echo "  Resolution: 720x400"
    echo "  Cases: ${NUM_CASES}"
    echo "============================================================"

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4

    local OVERALL_START=$(date +%s)
    local CASE_IDX=0

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

        local CASE_START=$(date +%s)

        cd ${LOCAL_CODE}
        torchrun --nproc_per_node=5 --master_port=$((29200 + CASE_IDX)) \
            minimal_inference/s2v_streaming_interact.py \
            --task s2v-14B \
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
            --save_dir "${save_dir}/" \
            --save_file "${save_dir}/${SAMPLE_NAME}.mp4" \
            --fp8

        local CASE_END=$(date +%s)
        local CASE_TIME=$((CASE_END - CASE_START))
        echo ">>> [5GPU] Case ${CASE_IDX} 完成: ${CASE_TIME}s (${SAMPLE_NAME})"

    done < <(load_samples)

    local OVERALL_END=$(date +%s)
    local OVERALL_TIME=$((OVERALL_END - OVERALL_START))

    echo ""
    echo "============================================================"
    echo "[5GPU] Benchmark 完成!"
    echo "  总耗时: ${OVERALL_TIME}s ($((OVERALL_TIME / 60))min)"
    echo "  平均每样本: $((OVERALL_TIME / CASE_IDX))s"
    echo "  输出目录: ${save_dir}"
    echo "  GPU: 5×H20 (4 DiT + 1 VAE), TPP mode, FP8"
    echo "============================================================"
}

# ===== 前置检查 =====
echo ""
echo "============================================"
echo "前置检查..."
echo "============================================"

# 检查样本列表
if [ ! -f "${SAMPLE_LIST}" ]; then
    echo "❌ 样本列表不存在: ${SAMPLE_LIST}"
    echo "   请先运行: python prepare_digit_samples.py"
    exit 1
fi

# 检查模型
if [ ! -d "${LOCAL_MODEL_BASE}" ] || [ -z "$(ls -A ${LOCAL_MODEL_BASE} 2>/dev/null)" ]; then
    echo "⚠️  基座模型未同步到本地盘"
    echo "   确保已运行: python download_models.py --models liveavatar"
fi

# 打印资源估算
echo ""
echo "📊 10分钟视频 (~300 clips × 48 frames) 资源估算:"
echo "┌──────────┬───────────┬──────────────┬──────────────┐"
echo "│  模式    │ GPU       │ 显存/卡      │ 预计耗时     │"
echo "├──────────┼───────────┼──────────────┼──────────────┤"
echo "│  1gpu    │ 1×H20     │ ~80GB        │ ~2.5-3小时   │"
echo "│  5gpu    │ 5×H20     │ ~20GB/卡     │ ~25-35分钟   │"
echo "└──────────┴───────────┴──────────────┴──────────────┘"
echo ""

# ===== 执行 =====
if ${RUN_1GPU}; then
    run_1gpu
fi

if ${RUN_5GPU}; then
    run_5gpu
fi

echo ""
echo "============================================"
echo "🎉 LiveAvatar Benchmark 全部完成!"
echo "   结果目录: ${RESULT_BASE}"
echo "============================================"
