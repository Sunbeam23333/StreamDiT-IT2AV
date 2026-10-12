#!/bin/bash
# ============================================================================
# 统一 TTS Benchmark 脚本
#
# 默认比较：CosyVoice2 / Spark-TTS / ChatTTS / F5-TTS
# - CosyVoice3 可选，但默认不推荐
# - Fish-Speech 当前标记为 experimental（S2 repo 与旧 runner 不兼容）
#
# 用法：
#   bash script/audio/run_tts_benchmark.sh
#   bash script/audio/run_tts_benchmark.sh --models cosyvoice2,spark_tts,chattts,f5tts --gpu 0 --language zh
#   bash script/audio/run_tts_benchmark.sh --models f5tts --ref_audio /path/ref.wav --ref_text "参考音频文本"
#   bash script/audio/run_tts_benchmark.sh --scene_ids scene_01_pure_music scene_04_anime
#
# 可用环境变量：
#   WM_GROUP_BASE      远端 group 根目录（默认 ${IT2AV_WORKSPACE}）
#   WM_MODEL_ROOT      模型目录（默认 ${WM_GROUP_BASE}/models/Audio）
#   WM_RESULT_ROOT     输出根目录（默认 ${WM_GROUP_BASE}/results/tts_benchmark）
#   WM_LOCAL_BASE      本地高速盘同步目录（默认 ${IT2AV_LOCAL_ROOT}）
#   WM_CONDA_ENV       统一 conda 环境名（默认 ${IT2AV_CONDA_ENV}）
# ============================================================================

set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

WM_GROUP_BASE=${WM_GROUP_BASE:-${IT2AV_WORKSPACE}}
WM_MODEL_ROOT=${WM_MODEL_ROOT:-${WM_GROUP_BASE}/models/Audio}
WM_RESULT_ROOT=${WM_RESULT_ROOT:-${WM_GROUP_BASE}/results/tts_benchmark}
WM_LOCAL_BASE=${WM_LOCAL_BASE:-${IT2AV_LOCAL_ROOT}}
WM_SYNC_SCRIPT=${WM_SYNC_SCRIPT:-${CODE_ROOT}/fast_sync.py}
WM_TTS_PROMPTS=${WM_TTS_PROMPTS:-${CODE_ROOT}/script/audio/prompts_audio_benchmark.json}

WM_CONDA_ENV=${WM_CONDA_ENV:-${IT2AV_CONDA_ENV}}

GPU_ID=0
LANGUAGE=zh
MODELS_CSV="cosyvoice2,spark_tts,chattts,f5tts"
FORCE_SYNC=0
SKIP_SYNC=0
RUN_TAG=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="${WM_RESULT_ROOT}/${RUN_TAG}"
F5_REF_AUDIO=""
F5_REF_TEXT=""
SPARK_GENDER="female"
SPARK_PITCH="moderate"
SPARK_SPEED="moderate"
CHATTTS_SEED=42
SCENE_IDS=()

usage() {
  cat <<'EOF'
统一 TTS Benchmark

选项：
  --models <csv>            模型列表，默认 cosyvoice2,spark_tts,chattts,f5tts
                            可选: cosyvoice2,cosyvoice3,spark_tts,chattts,f5tts,fish_speech
  --gpu <id>                GPU 编号，默认 0
  --language <zh|en>        语言，默认 zh
  --prompts_file <json>     指定 prompts 文件，默认 script/audio/prompts_audio_benchmark.json
  --scene_ids <id...>       只跑指定场景
  --output_root <dir>       输出根目录
  --skip_sync               不同步到 /tmp，直接使用 group 盘模型
  --force_sync              强制重新同步模型

  --ref_audio <wav>         F5-TTS 参考音频
  --ref_text <text>         F5-TTS 参考文本

  --spark_gender <v>        Spark-TTS 造音色 gender，默认 female
  --spark_pitch <v>         Spark-TTS 造音色 pitch，默认 moderate
  --spark_speed <v>         Spark-TTS 造音色 speed，默认 moderate
  --chattts_seed <int>      ChatTTS seed，默认 42

  --cosyvoice_speaker <v>   CosyVoice 内置 speaker，例如 中文女 / 中文男
  --cosyvoice_ref_audio     CosyVoice zero-shot 参考音频
  --cosyvoice_ref_text      CosyVoice zero-shot 参考文本
  --cosyvoice_speed <v>     CosyVoice SFT 语速，默认 1.0

  -h, --help                显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODELS_CSV="$2"; shift 2 ;;
    --gpu)
      GPU_ID="$2"; shift 2 ;;
    --language)
      LANGUAGE="$2"; shift 2 ;;
    --scene_ids)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SCENE_IDS+=("$1")
        shift
      done ;;
    --output_root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --skip_sync)
      SKIP_SYNC=1; shift ;;
    --force_sync)
      FORCE_SYNC=1; shift ;;
    --ref_audio)
      F5_REF_AUDIO="$2"; shift 2 ;;
    --ref_text)
      F5_REF_TEXT="$2"; shift 2 ;;
    --spark_gender)
      SPARK_GENDER="$2"; shift 2 ;;
    --spark_pitch)
      SPARK_PITCH="$2"; shift 2 ;;
    --spark_speed)
      SPARK_SPEED="$2"; shift 2 ;;
    --chattts_seed)
      CHATTTS_SEED="$2"; shift 2 ;;
    --cosyvoice_speaker)
      COSYVOICE_SPEAKER="$2"; shift 2 ;;
    --cosyvoice_ref_audio)
      COSYVOICE_REF_AUDIO="$2"; shift 2 ;;
    --cosyvoice_ref_text)
      COSYVOICE_REF_TEXT="$2"; shift 2 ;;
    --cosyvoice_speed)
      COSYVOICE_SPEED="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "❌ 未知参数: $1"
      usage
      exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"

if [[ ! -f "${WM_TTS_PROMPTS}" ]]; then
  echo "❌ prompts 文件不存在: ${WM_TTS_PROMPTS}"
  exit 1
fi

run_in_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "❌ 未找到 conda 命令，无法执行: conda activate ${WM_CONDA_ENV}"
    return 1
  fi

  eval "$(conda shell.bash hook)"
  echo ">>> conda activate ${WM_CONDA_ENV}"
  conda activate "${WM_CONDA_ENV}"
  "$@"
  local status=$?
  conda deactivate || true
  return ${status}
}

sync_model_dir() {
  local model_dir_name="$1"
  local src="${WM_MODEL_ROOT}/${model_dir_name}"
  local dst="${WM_LOCAL_BASE}/models/Audio/${model_dir_name}"

  if [[ ! -d "${src}" ]]; then
    echo "${src}"
    return 0
  fi

  if [[ ${SKIP_SYNC} -eq 1 || ! -f "${WM_SYNC_SCRIPT}" ]]; then
    echo "${src}"
    return 0
  fi

  mkdir -p "${dst}"
  if [[ ${FORCE_SYNC} -eq 1 || ! -f "${dst}/.sync_done" ]]; then
    echo ">>> 同步模型 ${model_dir_name} 到本地盘"
    python "${WM_SYNC_SCRIPT}" "${src}" "${dst}" --workers 32 --skip-existing
    touch "${dst}/.sync_done"
  fi

  echo "${dst}"
}

sync_code_dir() {
  local src="$1"
  local rel_path
  local dst

  if [[ ! -d "${src}" ]]; then
    echo "${src}"
    return 0
  fi

  if [[ "${src}" == "${CODE_ROOT}/"* ]]; then
    rel_path="${src#${CODE_ROOT}/}"
  else
    rel_path="$(basename "${src}")"
  fi
  dst="${WM_LOCAL_BASE}/code/${rel_path}"

  if [[ ${SKIP_SYNC} -eq 1 || ! -f "${WM_SYNC_SCRIPT}" ]]; then
    echo "${src}"
    return 0
  fi

  mkdir -p "${dst}"
  if [[ ${FORCE_SYNC} -eq 1 || ! -f "${dst}/.sync_done" ]]; then
    echo ">>> 同步代码 $(basename "${src}") 到本地盘"
    python "${WM_SYNC_SCRIPT}" "${src}" "${dst}" --workers 32 --skip-existing
    touch "${dst}/.sync_done"
  fi

  echo "${dst}"
}

append_scene_args() {
  if [[ ${#SCENE_IDS[@]} -gt 0 ]]; then
    printf '%s\n' "--scene_ids" "${SCENE_IDS[@]}"
  fi
}

get_f5_default_ref_audio() {
  local base_dir="${1:-${CODE_ROOT}/utils/audio/F5-TTS}"
  if [[ "${LANGUAGE}" == "en" ]]; then
    echo "${base_dir}/src/f5_tts/infer/examples/basic/basic_ref_en.wav"
  else
    echo "${base_dir}/src/f5_tts/infer/examples/basic/basic_ref_zh.wav"
  fi
}

get_f5_default_ref_text() {
  if [[ "${LANGUAGE}" == "en" ]]; then
    printf '%s' "Some call me nature, others call me mother nature."
  else
    printf '%s' "对，这就是我，万人敬仰的太乙真人。"
  fi
}

run_cosyvoice() {
  local engine="$1"
  local out_dir="${OUTPUT_ROOT}/${engine}"
  mkdir -p "${out_dir}"

  local cmd=(
    python "${CODE_ROOT}/run_audio_benchmark.py"
    --model cosyvoice
    --prompts_file "${WM_TTS_PROMPTS}"
    --output_dir "${out_dir}"
    --device cuda:0
    --language "${LANGUAGE}"
    --tts_engine "${engine}"
  )

  if [[ ${#SCENE_IDS[@]} -gt 0 ]]; then
    cmd+=(--scene_ids "${SCENE_IDS[@]}")
  fi

  run_in_conda_env "${cmd[@]}"
}

run_unified_tts() {
  local model_key="$1"
  local model_dir_name="$2"
  local code_dir="$3"
  local out_dir="${OUTPUT_ROOT}/${model_key}"
  local model_path
  local actual_code_dir

  model_path=$(sync_model_dir "${model_dir_name}")
  actual_code_dir=$(sync_code_dir "${code_dir}")
  mkdir -p "${out_dir}"

  local cmd=(
    python "${CODE_ROOT}/run_tts_benchmark.py"
    --model "${model_key}"
    --model_path "${model_path}"
    --code_dir "${actual_code_dir}"
    --prompts_file "${WM_TTS_PROMPTS}"
    --output_dir "${out_dir}"
    --device cuda:0
    --language "${LANGUAGE}"
  )

  if [[ ${#SCENE_IDS[@]} -gt 0 ]]; then
    cmd+=(--scene_ids "${SCENE_IDS[@]}")
  fi

  case "${model_key}" in
    f5tts)
      local ref_audio="${F5_REF_AUDIO}"
      local ref_text="${F5_REF_TEXT}"
      if [[ -z "${ref_audio}" ]]; then
        ref_audio=$(get_f5_default_ref_audio "${actual_code_dir}")
      fi
      if [[ -z "${ref_text}" ]]; then
        ref_text=$(get_f5_default_ref_text)
      fi
      cmd+=(--ref_audio "${ref_audio}" --ref_text "${ref_text}")
      ;;
    spark_tts)
      cmd+=(
        --spark_gender "${SPARK_GENDER}"
        --spark_pitch "${SPARK_PITCH}"
        --spark_speed "${SPARK_SPEED}"
      )
      ;;
    chattts)
      cmd+=(--chattts_seed "${CHATTTS_SEED}")
      ;;
  esac

  run_in_conda_env "${cmd[@]}"
}

echo "============================================================"
echo "🎤 TTS Benchmark"
echo "  Code root:   ${CODE_ROOT}"
echo "  Model root:  ${WM_MODEL_ROOT}"
echo "  Output root: ${OUTPUT_ROOT}"
echo "  GPU:         ${GPU_ID}"
echo "  Language:    ${LANGUAGE}"
echo "  Models:      ${MODELS_CSV}"
echo "============================================================"

action_failed=0
for model in "${MODELS[@]}"; do
  echo
  echo ">>> Running ${model}"
  case "${model}" in
    cosyvoice2|cosyvoice3)
      CUDA_VISIBLE_DEVICES=${GPU_ID} run_cosyvoice "${model}" || action_failed=1
      ;;
    spark_tts)
      CUDA_VISIBLE_DEVICES=${GPU_ID} run_unified_tts "spark_tts" "Spark-TTS-0.5B" "${CODE_ROOT}/utils/audio/Spark-TTS" || action_failed=1
      ;;
    chattts)
      CUDA_VISIBLE_DEVICES=${GPU_ID} run_unified_tts "chattts" "ChatTTS" "${CODE_ROOT}/utils/audio/ChatTTS" || action_failed=1
      ;;
    f5tts)
      CUDA_VISIBLE_DEVICES=${GPU_ID} run_unified_tts "f5tts" "F5-TTS" "${CODE_ROOT}/utils/audio/F5-TTS" || action_failed=1
      ;;
    fish_speech)
      echo "⚠️  Fish-Speech 当前未纳入统一 benchmark：vendored repo 为 S2，需走官方 API Server 路线。"
      echo "    参考: python ${CODE_ROOT}/utils/audio/fish-speech/tools/api_server.py --listen 0.0.0.0:8080"
      ;;
    *)
      echo "❌ 不支持的模型: ${model}"
      action_failed=1
      ;;
  esac
done

python - "${OUTPUT_ROOT}" <<'PY'
import json
import os
import sys
import wave
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None

output_root = Path(sys.argv[1])
summary_rows = []


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return 0.0 if rate == 0 else frames / float(rate)
    except wave.Error:
        if sf is None:
            raise
        info = sf.info(str(path))
        rate = info.samplerate
        return 0.0 if rate == 0 else info.frames / float(rate)


for subdir in sorted(p for p in output_root.iterdir() if p.is_dir()):
    summary = None
    summary_files = list(subdir.glob("summary_*.json"))
    if summary_files:
        summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
        rows = [r for r in summary.get("results", []) if "error" not in r and "rtf" in r]
        avg_rtf = round(sum(r["rtf"] for r in rows) / len(rows), 3) if rows else None
        summary_rows.append({
            "model": subdir.name,
            "successful": summary.get("successful"),
            "failed": summary.get("failed"),
            "total_audio_duration": summary.get("total_audio_duration"),
            "total_time_s": round(sum(r.get("gen_time", 0.0) for r in summary.get("results", [])), 2),
            "avg_rtf": avg_rtf,
            "report": str(summary_files[0]),
        })
        continue

    report_path = subdir / "benchmark_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        wav_dir = subdir / "cosyvoice_tts"
        wavs = sorted(wav_dir.glob("*.wav")) if wav_dir.exists() else []
        total_audio_duration = round(sum(wav_duration(w) for w in wavs), 2) if wavs else None
        summary_rows.append({
            "model": subdir.name,
            "successful": len(wavs),
            "failed": None,
            "total_audio_duration": total_audio_duration,
            "total_time_s": report.get("results", {}).get("cosyvoice", {}).get("total_time_s", report.get("total_time_s")),
            "avg_rtf": None,
            "report": str(report_path),
        })

summary_json = output_root / "comparison_summary.json"
summary_tsv = output_root / "comparison_summary.tsv"
summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
with summary_tsv.open("w", encoding="utf-8") as f:
    f.write("model\tsuccessful\tfailed\ttotal_audio_duration\ttotal_time_s\tavg_rtf\treport\n")
    for row in summary_rows:
        f.write(
            f"{row['model']}\t{row['successful']}\t{row['failed']}\t{row['total_audio_duration']}\t{row['total_time_s']}\t{row['avg_rtf']}\t{row['report']}\n"
        )

print("\n============================================================")
print("📊 Comparison summary generated")
print(f"  JSON: {summary_json}")
print(f"  TSV:  {summary_tsv}")
print("============================================================")
PY

if [[ ${action_failed} -ne 0 ]]; then
  echo "⚠️  部分模型执行失败，请检查各子目录日志与 summary 文件。"
  exit 1
fi

echo "✅ 全部 benchmark 执行完成: ${OUTPUT_ROOT}"
