#!/bin/bash
# ============================================================================
# 安装 TTS 备选方案的代码仓库和 conda 环境
#
# 支持的 TTS 框架:
#   1. F5-TTS      : Flow Matching + DiT, 零样本3秒克隆
#   2. Fish-Speech  : 极速推理, 15万小时训练
#   3. Spark-TTS    : LLM-based, 音高/语速可控
#   4. ChatTTS      : 对话场景专用, 韵律自然
#
# GitHub:
#   - https://github.com/SWivid/F5-TTS
#   - https://github.com/fishaudio/fish-speech
#   - https://github.com/SparkAudio/Spark-TTS
#   - https://github.com/2noise/ChatTTS
#
# 用法:
#   bash setup_tts_alternatives.sh               # 安装全部
#   bash setup_tts_alternatives.sh f5tts          # 只安装 F5-TTS
#   bash setup_tts_alternatives.sh fish_speech    # 只安装 Fish-Speech
#   bash setup_tts_alternatives.sh spark_tts      # 只安装 Spark-TTS
#   bash setup_tts_alternatives.sh chattts        # 只安装 ChatTTS
#
# 代理配置 (访问 GitHub / PyPI 受限时):
#   在仓库根目录 .env 中设置 IT2AV_PROXY_URL=http://user:pass@host:port
#   (或直接 export HTTPS_PROXY), 账号密码不要写进脚本
# ============================================================================

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
UTILS_AUDIO=${CODE_ROOT}/utils/audio

# 代理 / .env (IT2AV_PROXY_URL 会在 env.sh 中导出为 HTTP(S)_PROXY)
source "${SCRIPT_DIR}/../common/env.sh"

setup_proxy() {
    if [[ -n "${HTTPS_PROXY:-}${https_proxy:-}" ]]; then
        echo "✅ 使用代理: $(echo "${HTTPS_PROXY:-${https_proxy}}" | sed -E 's#//[^@/]+@#//***@#')"
    else
        echo "ℹ️  未配置代理, 直连 GitHub / PyPI"
    fi
}

# ============================================================
# F5-TTS
# GitHub: https://github.com/SWivid/F5-TTS
# HuggingFace 模型: SWivid/F5-TTS
# ============================================================
install_f5tts() {
    echo ""
    echo "============================================================"
    echo "🔧 安装 F5-TTS"
    echo "   GitHub: https://github.com/SWivid/F5-TTS"
    echo "============================================================"

    local CODE_DIR=${UTILS_AUDIO}/F5-TTS

    if [ -d "${CODE_DIR}" ] && [ -f "${CODE_DIR}/src/f5_tts/infer/infer_cli.py" ]; then
        echo "  ⏭️  代码已存在: ${CODE_DIR}"
        cd "${CODE_DIR}" && git pull 2>/dev/null || true
    else
        echo "  >>> 克隆代码..."
        git clone https://github.com/SWivid/F5-TTS.git "${CODE_DIR}"
    fi

    # 创建 conda 环境
    echo "  >>> 创建/更新 conda 环境: f5tts"
    if ! conda env list | grep -q "f5tts"; then
        conda create -n f5tts python=3.11 -y
    fi

    # 安装依赖
    echo "  >>> 安装依赖..."
    eval "$(conda shell.bash hook)"
    conda activate f5tts
    cd "${CODE_DIR}"
    pip install -e . 2>&1 | tail -5
    # F5-TTS 需要 vocos, transformers 等
    pip install soundfile cached_path 2>&1 | tail -3
    conda deactivate

    echo "  ✅ F5-TTS 安装完成"
    echo "     代码: ${CODE_DIR}"
    echo "     环境: conda activate f5tts"
    echo "     模型: 需运行 download_audio_models.py --models f5tts"
}

# ============================================================
# Fish-Speech
# GitHub: https://github.com/fishaudio/fish-speech
# HuggingFace 模型: fishaudio/fish-speech-1.5
# ============================================================
install_fish_speech() {
    echo ""
    echo "============================================================"
    echo "🔧 安装 Fish-Speech"
    echo "   GitHub: https://github.com/fishaudio/fish-speech"
    echo "============================================================"

    local CODE_DIR=${UTILS_AUDIO}/fish-speech

    if [ -d "${CODE_DIR}" ] && [ -f "${CODE_DIR}/tools/api_server.py" ]; then
        echo "  ⏭️  代码已存在: ${CODE_DIR}"
        cd "${CODE_DIR}" && git pull 2>/dev/null || true
    else
        echo "  >>> 克隆代码..."
        git clone https://github.com/fishaudio/fish-speech.git "${CODE_DIR}"
    fi

    # 创建 conda 环境
    echo "  >>> 创建/更新 conda 环境: fish_speech"
    if ! conda env list | grep -q "fish_speech"; then
        conda create -n fish_speech python=3.11 -y
    fi

    echo "  >>> 安装依赖..."
    eval "$(conda shell.bash hook)"
    conda activate fish_speech
    cd "${CODE_DIR}"
    pip install -e . 2>&1 | tail -5
    conda deactivate

    echo "  ✅ Fish-Speech 安装完成"
    echo "     代码: ${CODE_DIR}"
    echo "     环境: conda activate fish_speech"
    echo "     模型: 需运行 download_audio_models.py --models fish_speech"
}

# ============================================================
# Spark-TTS
# GitHub: https://github.com/SparkAudio/Spark-TTS
# HuggingFace 模型: SparkAudio/Spark-TTS-0.5B
# ============================================================
install_spark_tts() {
    echo ""
    echo "============================================================"
    echo "🔧 安装 Spark-TTS"
    echo "   GitHub: https://github.com/SparkAudio/Spark-TTS"
    echo "============================================================"

    local CODE_DIR=${UTILS_AUDIO}/Spark-TTS

    if [ -d "${CODE_DIR}" ] && [ -f "${CODE_DIR}/cli/inference.py" ]; then
        echo "  ⏭️  代码已存在: ${CODE_DIR}"
        cd "${CODE_DIR}" && git pull 2>/dev/null || true
    else
        echo "  >>> 克隆代码..."
        git clone https://github.com/SparkAudio/Spark-TTS.git "${CODE_DIR}"
    fi

    # 创建 conda 环境
    echo "  >>> 创建/更新 conda 环境: spark_tts"
    if ! conda env list | grep -q "spark_tts"; then
        conda create -n spark_tts python=3.12 -y
    fi

    echo "  >>> 安装依赖..."
    eval "$(conda shell.bash hook)"
    conda activate spark_tts
    cd "${CODE_DIR}"
    pip install -r requirements.txt 2>&1 | tail -5
    conda deactivate

    echo "  ✅ Spark-TTS 安装完成"
    echo "     代码: ${CODE_DIR}"
    echo "     环境: conda activate spark_tts"
    echo "     模型: 需运行 download_audio_models.py --models spark_tts"
}

# ============================================================
# ChatTTS
# GitHub: https://github.com/2noise/ChatTTS
# HuggingFace 模型: 2Noise/ChatTTS
# ============================================================
install_chattts() {
    echo ""
    echo "============================================================"
    echo "🔧 安装 ChatTTS"
    echo "   GitHub: https://github.com/2noise/ChatTTS"
    echo "============================================================"

    local CODE_DIR=${UTILS_AUDIO}/ChatTTS

    if [ -d "${CODE_DIR}" ] && [ -f "${CODE_DIR}/ChatTTS/core.py" ]; then
        echo "  ⏭️  代码已存在: ${CODE_DIR}"
        cd "${CODE_DIR}" && git pull 2>/dev/null || true
    else
        echo "  >>> 克隆代码..."
        git clone https://github.com/2noise/ChatTTS.git "${CODE_DIR}"
    fi

    # 创建 conda 环境
    echo "  >>> 创建/更新 conda 环境: chattts"
    if ! conda env list | grep -q "chattts"; then
        conda create -n chattts python=3.11 -y
    fi

    echo "  >>> 安装依赖..."
    eval "$(conda shell.bash hook)"
    conda activate chattts
    cd "${CODE_DIR}"
    pip install --upgrade -r requirements.txt 2>&1 | tail -5
    conda deactivate

    echo "  ✅ ChatTTS 安装完成"
    echo "     代码: ${CODE_DIR}"
    echo "     环境: conda activate chattts"
    echo "     模型: 需运行 download_audio_models.py --models chattts"
}

# ============================================================
# Main
# ============================================================
setup_proxy

TARGET=${1:-"all"}

case ${TARGET} in
    f5tts)
        install_f5tts
        ;;
    fish_speech)
        install_fish_speech
        ;;
    spark_tts)
        install_spark_tts
        ;;
    chattts)
        install_chattts
        ;;
    all)
        install_f5tts
        install_fish_speech
        install_spark_tts
        install_chattts
        ;;
    *)
        echo "❌ 未知目标: ${TARGET}"
        echo "用法: bash setup_tts_alternatives.sh [f5tts|fish_speech|spark_tts|chattts|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "🎉 TTS 备选方案安装完成!"
echo ""
echo "📦 模型下载 (HuggingFace):"
echo "  python download_audio_models.py --models tts_alt"
echo ""
echo "🚀 各框架推理脚本:"
echo "  bash script/audio/run_f5tts.sh"
echo "  bash script/audio/run_fish_speech.sh"
echo "  bash script/audio/run_spark_tts.sh"
echo "  bash script/audio/run_chattts.sh"
echo "============================================================"
