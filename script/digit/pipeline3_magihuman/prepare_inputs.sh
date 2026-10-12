#!/bin/bash
# ============================================================================
# Pipeline3 - MagiHuman: 准备输入数据 (复用 Pipeline2 的 5 个 Case)
#
# 直接从 Pipeline2 的图片目录复制首帧图片，并根据台词自动生成
# MagiHuman Enhanced Prompt (遵循 prompts/enhanced_prompt_design.md 规范)。
#
# 5 个 Case (与 Pipeline2 完全一致):
#   case_1: 卧室场景 — "晚上好，生活再忙再累，也别忘了好好爱自己。"
#   case_2: 校园场景 — "晚上好，黑夜无论怎样悠长，白昼总会到来..."
#   case_3: 樱花道   — "晚上好，世上只有一种英雄主义..."
#   case_4: 城市街景 — "晚上好，不必行色匆匆，不必光芒四射..."
#   case_5: 旗袍小桥 — "晚上好，万物皆有裂痕，那是光照进来的地方..."
#
# 输出 (每个 case 目录):
#   case_{N}/image.png   - 首帧图片 (从 pipeline2 图片目录复制/软链接)
#   case_{N}/prompt.txt  - Enhanced Prompt (MagiHuman 格式, CJK逐字加空格)
#
# 用法:
#   bash prepare_inputs.sh
#   bash prepare_inputs.sh --date_tag 20260324
#   bash prepare_inputs.sh --cases 1,3
#   bash prepare_inputs.sh --link            # 用软链接代替复制图片
# ============================================================================

set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../common/env.sh"

GROUP_BASE=${IT2AV_WORKSPACE}
DATE_TAG="20260324"
PIPELINE2_DATE_TAG="20260323"   # Pipeline2 使用的日期
CASES_CSV="1,2,3,4,5"
USE_LINK=0                       # 0=复制, 1=软链接

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cases)               CASES_CSV="$2"; shift 2 ;;
        --date_tag)            DATE_TAG="$2"; shift 2 ;;
        --pipeline2_date_tag)  PIPELINE2_DATE_TAG="$2"; shift 2 ;;
        --link)                USE_LINK=1; shift ;;
        -h|--help)             head -30 "$0" | tail -25; exit 0 ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

IFS=',' read -r -a CASES <<< "${CASES_CSV}"
OUTPUT_DIR=${GROUP_BASE}/data/digit/${DATE_TAG}
PICS_DIR=${GROUP_BASE}/data/pics/${PIPELINE2_DATE_TAG}/no_watermark

# ===== 5 个 Case 的配置 (与 Pipeline2 完全一致) =====
CASE_IMAGES=(
    "1-卧室场景.png"
    "2-校园场景.png"
    "3-樱花道.png"
    "4-街景.png"
    "5-旗袍.png"
)

# MagiHuman Enhanced Prompt — 遵循 enhanced_prompt_design.md 规范:
#   第一段: 场景/外貌/表情/动作描述 (英文, 150-200 words, 静止角色)
#   Dialogue 段: <角色, 语言>: "CJK逐字加空格的台词"
#   Background Sound 段: <背景音描述>
#
# 台词内容与 Pipeline2 step1_generate_audio.sh 完全一致

CASE_PROMPTS=()

# ── Case 1: 卧室场景 ──
read -r -d '' PROMPT_1 << 'PROMPT_EOF' || true
Low-angle medium close-up shot of a young woman sitting still in a warm, dimly lit bedroom at night. She wears a comfortable sleepwear, her hair falling softly around her shoulders. Her overall disposition is warm and nurturing. She faces the camera directly with a gentle, inviting gaze. Her eyes crinkle slightly at the outer corners, the orbicularis oculi contracting to form a soft, genuine smile. Her lips part slowly as she begins to speak in a calm, unhurried, and tender tone, "晚 上 好 ， 生 活 再 忙 再 累 ， 也 别 忘 了 好 好 爱 自 己 。" The lip muscles articulate each character with precise, gentle movements — the jaw opens slightly for each syllable while the lip corners maintain a subtle upward curve throughout. As she finishes, her smile deepens slightly, her cheeks lifting. The lighting is warm amber from a bedside lamp, casting a golden glow on the left side of her face with soft shadows on the right. The depth of field is shallow, keeping her face in sharp focus against the blurred cozy bedroom interior. Her torso remains completely stationary throughout.

Dialogue:
<Young woman, Mandarin>: "晚 上 好 ， 生 活 再 忙 再 累 ， 也 别 忘 了 好 好 爱 自 己 。"

Background Sound:
<Quiet bedroom ambience at night with soft wind outside the window>
PROMPT_EOF
CASE_PROMPTS+=("${PROMPT_1}")

# ── Case 2: 校园场景 ──
read -r -d '' PROMPT_2 << 'PROMPT_EOF' || true
Static medium shot of a young woman in a school uniform with a short pleated skirt, standing motionless on a cherry blossom path at night. Soft moonlight illuminates the scene from above, casting pale blue-white highlights on her hair and shoulders. Her disposition is reflective and gently hopeful. She gazes directly at the camera with a composed expression — her brows relaxed, her eyes soft and slightly glistening. She begins to speak in a measured, melodic, and quietly encouraging tone, "晚 上 好 ， 黑 夜 无 论 怎 样 悠 长 ， 白 昼 总 会 到 来 ， 一 切 都 会 慢 慢 变 好 。" Her lips move with careful articulation, the lower lip dropping gently for each syllable, her jaw shifting subtly. As she reaches the final phrase, a faint, hopeful smile forms — the zygomaticus major pulling the lip corners upward while her eyes narrow slightly with warmth. Her head tilts a fraction to the right at the end. The background shows static cherry blossom branches with petals frozen in the air. The camera remains fixed, with shallow depth of field focusing sharply on her face.

Dialogue:
<Young woman, Mandarin>: "晚 上 好 ， 黑 夜 无 论 怎 样 悠 长 ， 白 昼 总 会 到 来 ， 一 切 都 会 慢 慢 变 好 。"

Background Sound:
<Gentle wind through cherry blossom trees at night with distant crickets>
PROMPT_EOF
CASE_PROMPTS+=("${PROMPT_2}")

# ── Case 3: 樱花道 ──
read -r -d '' PROMPT_3 << 'PROMPT_EOF' || true
Static medium close-up shot of a young woman standing beneath cherry blossom trees on a quiet campus path at spring night. She wears a campus-style plaid skirt and stands perfectly still. Her overall mood is earnest and inspiring. She looks directly at the camera with steady, determined eyes, her brow smooth and composed. She speaks in a clear, warm, and slightly impassioned tone, "晚 上 好 ， 世 上 只 有 一 种 英 雄 主 义 ， 就 是 在 认 清 生 活 真 相 后 依 然 热 爱 生 活 。" Her lip movements are distinct and expressive — the mentalis and orbicularis oris muscles working in coordination as she enunciates each character. At the word "英 雄," her chin lifts slightly, a micro-expression of conviction crossing her face, her eyebrows rising subtly. By the final phrase "热 爱 生 活," a genuine, broad smile breaks across her face, the zygomaticus major contracting fully. Her eyes crinkle at the corners with emotional warmth. The lighting is a mix of cool moonlight from above and warm ambient light from nearby path lamps. Depth of field is shallow; the background cherry blossoms are softly blurred. Her torso and posture remain fixed throughout.

Dialogue:
<Young woman, Mandarin>: "晚 上 好 ， 世 上 只 有 一 种 英 雄 主 义 ， 就 是 在 认 清 生 活 真 相 后 依 然 热 爱 生 活 。"

Background Sound:
<Quiet campus pathway ambience with gentle breeze through trees and leaves rustling>
PROMPT_EOF
CASE_PROMPTS+=("${PROMPT_3}")

# ── Case 4: 城市街景 ──
read -r -d '' PROMPT_4 << 'PROMPT_EOF' || true
Static medium shot of a young woman standing still on a city street at night, near a crosswalk under warm streetlight glow. She wears a fashionable short skirt and high heels. Her posture is upright and poised, conveying quiet confidence. Her face carries a relaxed, reassuring expression. She looks directly at the camera with calm, steady eyes. She speaks in a soft, unhurried, and soothing tone, "晚 上 好 ， 不 必 行 色 匆 匆 ， 不 必 光 芒 四 射 ， 只 需 做 自 己 ， 早 点 休 息 吧 。" Her lips articulate each syllable with relaxed precision, the jaw dropping minimally, the lip corners maintaining a neutral-to-warm position. At "只 需 做 自 己," her expression softens further, the frontalis muscle relaxing completely, her eyelids lowering slightly in a gentle, knowing look. At "早 点 休 息 吧," she offers a tender, closing smile — the orbicularis oculi engaging to create genuine warmth around her eyes. City lights create colorful bokeh in the blurred background. Warm tungsten streetlight illuminates her face from the upper left. Her body remains completely motionless throughout the performance.

Dialogue:
<Young woman, Mandarin>: "晚 上 好 ， 不 必 行 色 匆 匆 ， 不 必 光 芒 四 射 ， 只 需 做 自 己 ， 早 点 休 息 吧 。"

Background Sound:
<City street night ambience with distant traffic and gentle spring evening wind>
PROMPT_EOF
CASE_PROMPTS+=("${PROMPT_4}")

# ── Case 5: 旗袍小桥 ──
read -r -d '' PROMPT_5 << 'PROMPT_EOF' || true
Static medium close-up shot of a young woman in an elegant traditional qipao dress, standing motionless on a small illuminated bridge at night. Warm lantern light casts a golden-amber glow across her face and the silk fabric of her dress. Her disposition is serene and poetic. She faces the camera with a tranquil gaze, her expression composed and graceful. She speaks in a gentle, lilting, and slightly wistful tone, "晚 上 好 ， 万 物 皆 有 裂 痕 ， 那 是 光 照 进 来 的 地 方 ， 愿 你 今 夜 好 梦 。" Her lips move with delicate precision, each CJK character articulated through subtle jaw and lip coordination. At "裂 痕," a micro-expression of contemplation flickers across her face — her brows drawing together almost imperceptibly. At "光 照 进 来," her expression brightens, the corrugator supercilii releasing as her eyes widen fractionally. She concludes with "好 梦," her eyelids lowering halfway in a peaceful, blessing-like expression, a soft smile forming. The background shows the static bridge railings and soft reflections of lantern light on still water below. The camera is fixed with shallow depth of field. Her torso and global position remain stationary throughout.

Dialogue:
<Young woman, Mandarin>: "晚 上 好 ， 万 物 皆 有 裂 痕 ， 那 是 光 照 进 来 的 地 方 ， 愿 你 今 夜 好 梦 。"

Background Sound:
<Small bridge over stream at night with flowing water sounds and gentle wind>
PROMPT_EOF
CASE_PROMPTS+=("${PROMPT_5}")

# ===== 执行 =====
echo "============================================================"
echo "📋 Pipeline3 - 准备 MagiHuman 输入 (复用 Pipeline2 的 5 个 Case)"
echo "  输出目录:    ${OUTPUT_DIR}"
echo "  图片来源:    ${PICS_DIR}"
echo "  Cases:       ${CASES_CSV}"
echo "  图片模式:    $([ ${USE_LINK} -eq 1 ] && echo '软链接' || echo '复制')"
echo "============================================================"

SUCCESS_COUNT=0
FAIL_COUNT=0

for CASE_ID in "${CASES[@]}"; do
    IDX=$((CASE_ID - 1))
    CASE_DIR="${OUTPUT_DIR}/case_${CASE_ID}"
    mkdir -p "${CASE_DIR}"

    IMAGE_NAME="${CASE_IMAGES[$IDX]}"
    IMAGE_SRC="${PICS_DIR}/${IMAGE_NAME}"
    IMAGE_DST="${CASE_DIR}/image.png"
    PROMPT_DST="${CASE_DIR}/prompt.txt"

    echo ""
    echo "  📁 case_${CASE_ID}: ${IMAGE_NAME}"

    # ── 复制/链接图片 ──
    if [[ -f "${IMAGE_DST}" ]]; then
        echo "    ✅ image.png 已存在"
    elif [[ -f "${IMAGE_SRC}" ]]; then
        if [[ ${USE_LINK} -eq 1 ]]; then
            ln -sf "${IMAGE_SRC}" "${IMAGE_DST}"
            echo "    ✅ image.png 已链接 ← ${IMAGE_NAME}"
        else
            cp "${IMAGE_SRC}" "${IMAGE_DST}"
            echo "    ✅ image.png 已复制 ← ${IMAGE_NAME}"
        fi
    else
        echo "    ❌ 图片不存在: ${IMAGE_SRC}"
        echo "       请确认 Pipeline2 的图片已上传到 group 盘"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # ── 写入 Enhanced Prompt ──
    if [[ -f "${PROMPT_DST}" ]]; then
        echo "    ✅ prompt.txt 已存在 (跳过覆盖)"
    else
        echo "${CASE_PROMPTS[$IDX]}" > "${PROMPT_DST}"
        WORD_COUNT=$(wc -w < "${PROMPT_DST}")
        echo "    ✅ prompt.txt 已生成 (${WORD_COUNT} words)"
    fi

    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
done

echo ""
echo "============================================================"
echo "📋 准备完成! (成功: ${SUCCESS_COUNT}, 失败: ${FAIL_COUNT})"
echo ""
echo "  输出目录结构:"
echo "    ${OUTPUT_DIR}/"
for CASE_ID in "${CASES[@]}"; do
    echo "    ├── case_${CASE_ID}/"
    echo "    │   ├── image.png"
    echo "    │   └── prompt.txt"
done
echo ""
echo "  下一步:"
echo "    bash pipeline3_magihuman/run_magihuman.sh --mode distill"
echo "    bash pipeline3_magihuman/run_magihuman.sh --mode sr540p"
echo "============================================================"
