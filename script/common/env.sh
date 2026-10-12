#!/usr/bin/env bash
# ============================================================================
# StreamDiT-IT2AV · 公共环境 / 路径约定 (所有 script/*.sh 都会 source 本文件)
#
#   IT2AV_CODE_ROOT    本仓库根目录 (自动推导, 一般不用设置)
#   IT2AV_WORKSPACE    共享工作区: models/ data/ results/ output/ 都在这里
#                      (原始实验中是一块 NFS 共享盘), 默认 ~/it2av_workspace
#   IT2AV_LOCAL_ROOT   本地 SSD 缓存: fast_sync.py 把模型从共享盘拷到这里再加载
#                      默认 /tmp/it2av
#   IT2AV_CONDA_ENV    主 conda 环境名 (视频 / 数字人 / CosyVoice 等), 默认 it2av
#   IT2AV_FFMPEG_DIR   (可选) 带 libx264 的静态 ffmpeg 所在目录
#
# 私密配置 (HF_TOKEN / 代理 / API Key) 请写进仓库根目录的 .env
# (参考 .env.example, .env 已被 .gitignore 忽略, 不会被提交)
# ============================================================================

_IT2AV_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IT2AV_CODE_ROOT="${IT2AV_CODE_ROOT:-$(cd "${_IT2AV_ENV_DIR}/../.." && pwd)}"

if [[ -f "${IT2AV_CODE_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${IT2AV_CODE_ROOT}/.env"
    set +a
fi

export IT2AV_WORKSPACE="${IT2AV_WORKSPACE:-${HOME}/it2av_workspace}"
export IT2AV_LOCAL_ROOT="${IT2AV_LOCAL_ROOT:-/tmp/it2av}"
export IT2AV_CONDA_ENV="${IT2AV_CONDA_ENV:-it2av}"
export CONDA_ROOT="${CONDA_ROOT:-${CONDA_EXE:+$(dirname "$(dirname "${CONDA_EXE}")")}}"
export CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"

# 需要代理访问 HuggingFace / GitHub / PyPI 时, 在 .env 中设置 IT2AV_PROXY_URL
# (形如 http://user:pass@host:port), 这里只在环境里还没有代理时才导出。
if [[ -n "${IT2AV_PROXY_URL:-}" && -z "${HTTPS_PROXY:-}${https_proxy:-}" ]]; then
    export HTTP_PROXY="${IT2AV_PROXY_URL}" HTTPS_PROXY="${IT2AV_PROXY_URL}"
    export http_proxy="${IT2AV_PROXY_URL}" https_proxy="${IT2AV_PROXY_URL}"
    export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}" no_proxy="${no_proxy:-localhost,127.0.0.1}"
fi

if [[ -n "${IT2AV_FFMPEG_DIR:-}" && -x "${IT2AV_FFMPEG_DIR}/ffmpeg" ]]; then
    export PATH="${IT2AV_FFMPEG_DIR}:${PATH}"
fi
