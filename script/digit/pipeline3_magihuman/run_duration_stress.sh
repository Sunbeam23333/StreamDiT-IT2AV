#!/bin/bash
# ============================================================================
# MagiHuman 视频时长压测脚本
#
# 目标：测试 20s / 30s / 60s 长视频生成是否 OOM，记录显存占用和耗时。
#
# 核心结论（参考代码分析）：
#   - decode chunk 内部计算显存与时长无关（每次处理固定 7 latent 帧）
#   - 但 decode 最终拼接完整视频 tensor (out.to(device)) 与时长线性相关！
#     60s 1080p 完整 tensor 约 19 GB (bf16)，已修复为留在 CPU 不搬回 GPU
#   - DiT 去噪阶段 token 数也随时长线性增长，是另一个 OOM 风险点
#
# 用法：
#   bash run_duration_stress.sh                      # 测试所有时长
#   bash run_duration_stress.sh --durations "5,10,20" # 只测 5/10/20 秒
#   bash run_duration_stress.sh --mode sr540p          # 用 540p 模式压测
#   bash run_duration_stress.sh --image /path/img.png  # 自定义图片
#   bash run_duration_stress.sh --dry_run              # 仅打印计划，不执行
# ============================================================================

set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../common/env.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== 压测默认配置 =====
DURATIONS_CSV="60,30"    # 要压测的时长（秒）
MODE="sr1080p"                    # 默认用最高画质压测
IMAGE_PATH=""                     # 为空时使用内置 base64 测试图片
OUTPUT_BASE=""                    # 为空时自动生成到 /tmp
DRY_RUN=0
# 60s 1080p 竖屏 SR token~769K:
#   cp=1 → 每卡 769K token → OOM (激活 ~108 GB)
#   cp=2 → 每卡 384K token → OOM (激活 ~54 GB + 权重 28.5 GB + all2all > 95 GB)
#   cp=4 → 每卡 192K token → 激活 ~27 GB + 权重 28.5 GB ≈ 56 GB ✅
#   cp=8 → 每卡  96K token → 激活 ~14 GB + 权重 28.5 GB ≈ 42 GB ✅✅
NUM_GPUS=${NUM_GPUS:-4}
SEED=42

# ===== 命令行解析 =====
while [[ $# -gt 0 ]]; do
    case "$1" in
        --durations)   DURATIONS_CSV="$2"; shift 2 ;;
        --mode)        MODE="$2"; shift 2 ;;
        --image)       IMAGE_PATH="$2"; shift 2 ;;
        --output_base) OUTPUT_BASE="$2"; shift 2 ;;
        --num_gpus)    NUM_GPUS="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --dry_run)     DRY_RUN=1; shift ;;
        -h|--help)
            head -25 "$0" | tail -20
            exit 0 ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

IFS=',' read -r -a DURATIONS <<< "${DURATIONS_CSV}"

# ===== 输出目录 =====
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_BASE=${OUTPUT_BASE:-${IT2AV_LOCAL_ROOT}/duration_stress_${TIMESTAMP}}
mkdir -p "${OUTPUT_BASE}"

RESULT_CSV="${OUTPUT_BASE}/results.csv"
echo "seconds,mode,status,inference_time_s,file_size_mb,peak_vram_mb,note" > "${RESULT_CSV}"

# ===== 理论 token 数预计算（用于参考） =====
calc_tokens() {
    local secs=$1
    local fps=25
    local t_comp=4
    local lat_h=34; local lat_w=60   # 256p base
    local p_t=1; local p_h=2; local p_w=2  # patch size

    local num_frames=$(( secs * fps + 1 ))
    local T_lat=$(( (num_frames - 1) / t_comp + 1 ))
    local video_tokens=$(( T_lat * lat_h / p_h * lat_w / p_w ))
    local audio_tokens=${num_frames}
    local total=$(( video_tokens + audio_tokens + 640 ))  # +640 text tokens

    local T_lat_sr
    # sr1080p latent: 68×120
    local sr_h=68; local sr_w=120
    local sr_video=$(( T_lat * sr_h / p_h * sr_w / p_w ))
    local sr_total=$(( sr_video + audio_tokens + 640 ))

    echo "T_lat=${T_lat} base_tokens=${total} sr_tokens=${sr_total}"
}

calc_chunk_count() {
    local secs=$1
    local fps=25
    local t_comp=4
    local first_chunk=3
    local step=5

    local num_frames=$(( secs * fps + 1 ))
    local T_lat=$(( (num_frames - 1) / t_comp + 1 ))
    local chunks=1
    if [[ $T_lat -gt $first_chunk ]]; then
        local remaining=$(( T_lat - first_chunk ))
        chunks=$(( 1 + (remaining + step - 1) / step ))
    fi
    echo "${chunks}"
}

# ===== 显示压测计划 =====
echo "============================================================"
echo "🔬 MagiHuman 视频时长压测"
echo "  模式:     ${MODE}"
echo "  GPU 数量: ${NUM_GPUS}"
echo "  种子:     ${SEED}"
echo "  输出目录: ${OUTPUT_BASE}"
echo "  测试时长: ${DURATIONS_CSV} 秒"
echo ""
echo "  理论分析（decode 与时长无关，DiT token 数线性增长）："
echo "  ┌──────┬──────────┬──────────────┬────────────┬────────────────┐"
echo "  │ 时长 │ latent T │ decode chunk │ base token │  SR token      │"
echo "  ├──────┼──────────┼──────────────┼────────────┼────────────────┤"
for secs in "${DURATIONS[@]}"; do
    info=$(calc_tokens "${secs}")
    T_lat=$(echo "$info" | grep -oP '(?<=T_lat=)\d+')
    base_tok=$(echo "$info" | grep -oP '(?<=base_tokens=)\d+')
    sr_tok=$(echo "$info" | grep -oP '(?<=sr_tokens=)\d+')
    chunks=$(calc_chunk_count "${secs}")
    printf "  │ %4ds │ %8d │ %12d │ %10d │ %14d │\n" "${secs}" "${T_lat}" "${chunks}" "${base_tok}" "${sr_tok}"
done
echo "  └──────┴──────────┴──────────────┴────────────┴────────────────┘"
echo ""
  echo "  注: decode chunk 内部固定 7 latent 帧，但拼接完整 tensor 与时长线性相关"
  echo "      已修复: output_offload 模式下不再把完整 tensor 搬回 GPU"
  echo "      OOM 风险: DiT 去噪阶段 SR token 数随时长线性增长"
echo "============================================================"

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "🔍 [dry_run] 仅打印计划，不实际执行"
    exit 0
fi

# ===== 创建测试 prompt 文件 =====
PROMPT_FILE="${OUTPUT_BASE}/test_prompt.txt"
cat > "${PROMPT_FILE}" << 'PROMPT_EOF'
Medium close-up portrait of a young East Asian woman standing beneath a canopy of blooming cherry blossom trees at night. She wears a loose, off-shoulder cream-white knit sweater and holds a small brown hardcover book against her chest with her left hand. Her long, wavy dark brown hair cascades over her right shoulder, adorned with a delicate white lace bow clip. With her right hand, she holds a single pink cherry blossom petal up to her cheek, smiling warmly at the camera.

She begins speaking softly in Mandarin Chinese, her voice gentle and melodic, as if narrating a passage from the book she holds. Her lips move with natural precision, forming each syllable clearly. She lowers the cherry blossom from her cheek and gestures lightly with her right hand as she describes the beauty of the spring night around her. Occasional cherry blossom petals drift slowly past her face, caught in a gentle breeze.

She pauses briefly, glancing down at the open book with a thoughtful expression, then looks back up at the camera with a widening smile. She continues speaking, now sharing a personal reflection about the fleeting nature of cherry blossoms and how they remind her of precious moments. Her eyebrows rise slightly with emotion, and she tilts her head gently to the side.

A soft gust of wind causes several petals to swirl around her. She laughs lightly, brushing a petal from her hair with her free hand. She then reads aloud a short poetic passage from the book, her voice becoming slower and more deliberate, each word carefully enunciated. Her facial expression shifts to one of serene contemplation, eyes softening.

She closes the book gently, holds it against her chest with both hands, and delivers a final reflection to the camera — a warm, sincere message about cherishing the present moment. She ends with a bright, genuine smile and a small nod.

Throughout the entire sequence, the background features abundant pink cherry blossom branches in full bloom, illuminated by warm golden light from a street lamp behind her right shoulder. The night sky is dark, creating beautiful contrast with the soft pink blossoms. The lighting is warm and atmospheric with gentle bokeh from the blossoms and lamp. The camera remains steady in a vertical portrait composition, mid-chest framing with shallow depth of field. Natural blinking, subtle head movements, and authentic micro-expressions maintain lifelike presence throughout the extended performance.
PROMPT_EOF

echo "✅ 测试 prompt 已写入: ${PROMPT_FILE}"
echo ""

# ===== 测试图片检查 =====
if [[ -n "${IMAGE_PATH}" && ! -f "${IMAGE_PATH}" ]]; then
    echo "❌ 图片不存在: ${IMAGE_PATH}"
    exit 1
fi

if [[ -z "${IMAGE_PATH}" ]]; then
    echo "⚠️  未指定 --image，尝试使用示例图片..."
    # 尝试找到任意可用的测试图片
    SAMPLE_IMG=$(find ${IT2AV_WORKSPACE}/data -name "image.png" -o -name "image.jpg" 2>/dev/null | head -1)
    if [[ -z "${SAMPLE_IMG}" ]]; then
        SAMPLE_IMG=$(find ${IT2AV_LOCAL_ROOT} -name "*.png" -o -name "*.jpg" 2>/dev/null | head -1)
    fi
    if [[ -z "${SAMPLE_IMG}" ]]; then
        echo "❌ 未找到任何测试图片，请通过 --image 指定"
        echo "   用法: bash run_duration_stress.sh --image /path/to/face.png"
        exit 1
    fi
    IMAGE_PATH="${SAMPLE_IMG}"
    echo "   使用: ${IMAGE_PATH}"
fi

echo ""
echo "============================================================"
echo "🚀 开始压测..."
echo "============================================================"

# ===== 逐时长压测 =====
PASS_COUNT=0
FAIL_COUNT=0
OOM_COUNT=0

for SECS in "${DURATIONS[@]}"; do
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "⏱️  测试 ${SECS} 秒视频 (${MODE})"
    info=$(calc_tokens "${SECS}")
    T_lat=$(echo "$info" | grep -oP '(?<=T_lat=)\d+')
    sr_tok=$(echo "$info" | grep -oP '(?<=sr_tokens=)\d+')
    chunks=$(calc_chunk_count "${SECS}")
    echo "   latent T=${T_lat}, SR token≈${sr_tok}, decode chunk数=${chunks}"
    echo "────────────────────────────────────────────────────────────"

    CASE_DIR="${OUTPUT_BASE}/duration_${SECS}s"
    mkdir -p "${CASE_DIR}"

    # 获取运行前 GPU 显存（卡0）
    VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 || echo "N/A")

    START_T=$(date +%s)

    # 调用 run_single.sh
    bash "${SCRIPT_DIR}/run_single.sh" \
        --image "${IMAGE_PATH}" \
        --prompt_file "${PROMPT_FILE}" \
        --mode "${MODE}" \
        --seconds "${SECS}" \
        --seed "${SEED}" \
        --num_gpus "${NUM_GPUS}" \
        --output_dir "${CASE_DIR}" \
        --log_dir "${CASE_DIR}" \
        2>&1 | tee "${CASE_DIR}/run_output.txt"
    STATUS=$?

    END_T=$(date +%s)
    ELAPSED=$(( END_T - START_T ))

    # 获取运行后 GPU 显存
    VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 || echo "N/A")

    # 检查是否 OOM
    IS_OOM=0
    if grep -q "CUDA out of memory\|RuntimeError.*OOM\|out of memory" "${CASE_DIR}/run_output.txt" 2>/dev/null; then
        IS_OOM=1
    fi

    # 找到输出文件
    OUTPUT_FILE=$(find "${CASE_DIR}" -name "magihuman_video_${SECS}s*.mp4" ! -name "*_video.mp4" ! -name "*_audio*" 2>/dev/null | head -1)
    FILE_SIZE_MB="N/A"
    if [[ -n "${OUTPUT_FILE}" && -f "${OUTPUT_FILE}" ]]; then
        FILE_SIZE_MB=$(du -m "${OUTPUT_FILE}" | cut -f1)
    fi

    # 从日志中提取显存峰值（如果有打印）
    PEAK_VRAM=$(grep -oP '(?<=memory_allocated: )\d+\.\d+' "${CASE_DIR}/run_output.txt" 2>/dev/null | tail -1 || echo "N/A")

    # 记录结果
    if [[ ${IS_OOM} -eq 1 ]]; then
        STATUS_STR="OOM"
        OOM_COUNT=$((OOM_COUNT + 1))
        echo "  ❌ OOM! 时长 ${SECS}s 显存不足"
    elif [[ -n "${OUTPUT_FILE}" && -f "${OUTPUT_FILE}" ]]; then
        STATUS_STR="OK"
        PASS_COUNT=$((PASS_COUNT + 1))
        echo "  ✅ 成功! 耗时 ${ELAPSED}s, 文件大小 ${FILE_SIZE_MB}MB"
    else
        STATUS_STR="FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "  ❌ 失败! 耗时 ${ELAPSED}s (非 OOM 错误)"
    fi

    echo "${SECS},${MODE},${STATUS_STR},${ELAPSED},${FILE_SIZE_MB},${PEAK_VRAM},vram_before=${VRAM_BEFORE}MB_after=${VRAM_AFTER}MB" >> "${RESULT_CSV}"
done

# ===== nvidia-smi 显存快照（所有卡） =====
echo ""
echo "============================================================"
echo "📊 测试完成后 GPU 显存状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
    --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi 不可用)"
echo "============================================================"

# ===== 汇总 =====
echo ""
echo "============================================================"
echo "📋 压测汇总"
echo "  模式:     ${MODE}"
echo "  通过:     ${PASS_COUNT}"
echo "  失败:     ${FAIL_COUNT}"
echo "  OOM:      ${OOM_COUNT}"
echo ""
echo "  详细结果: ${RESULT_CSV}"
echo ""
echo "  各时长结果:"
echo "  ┌──────┬──────────┬────────────┬────────────┐"
echo "  │ 时长 │ 状态     │ 耗时(s)    │ 文件大小   │"
echo "  ├──────┼──────────┼────────────┼────────────┤"
while IFS=',' read -r secs mode status elapsed fsize pvram note; do
    [[ "${secs}" == "seconds" ]] && continue
    printf "  │ %4ds │ %-8s │ %10s │ %8sMB   │\n" "${secs}" "${status}" "${elapsed}" "${fsize}"
done < "${RESULT_CSV}"
echo "  └──────┴──────────┴────────────┴────────────┘"
echo ""

if [[ ${OOM_COUNT} -gt 0 ]]; then
    echo "  ⚠️  存在 OOM 案例，建议排查方向："
    echo "     1. SR DiT 推理 OOM（token 数太大）→ 增加 num_gpus，用 CP 分割 token:"
    echo "        bash $0 --num_gpus 8  (60s 1080p 必须 8 卡)"
    echo "     2. decode 阶段 OOM（完整 tensor 搬 GPU）→ 已修复为 CPU 处理，确认代码已同步"
    echo "     3. 确认 expandable_segments 生效（碎片问题）:"
    echo "        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    echo "     4. 降低分辨率: --mode sr540p"
fi

echo "============================================================"
echo "📁 输出目录: ${OUTPUT_BASE}"
