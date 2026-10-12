#!/bin/bash
# ============================================================================
# 主环境安装脚本 (视频 / 数字人 / 统一 TTS 共用的 ${IT2AV_CONDA_ENV})
# 实验服务器: 8 x H20 95GB, Python 3.12, CUDA 12.x
#
# 音频模型 (TangoFlux / ACE-Step / SongGeneration / CosyVoice) 的依赖冲突较多,
# 见 script/audio/setup_audio_envs.sh 与 docs/environment_pitfalls.md
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/script/common/env.sh"

echo "============================================"
echo "Step 1: 创建 conda 环境 (${IT2AV_CONDA_ENV})"
echo "============================================"
eval "$(conda shell.bash hook)"
conda create -n "${IT2AV_CONDA_ENV}" python=3.12 -y
conda activate "${IT2AV_CONDA_ENV}"

echo "============================================"
echo "Step 2: 安装 PyTorch (CUDA 12.1)"
echo "============================================"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "============================================"
echo "Step 3: 安装 Flash Attention"
echo "============================================"
pip install flash-attn --no-build-isolation

echo "============================================"
echo "Step 4: 安装 LightX2V 依赖"
echo "============================================"
cd "${IT2AV_CODE_ROOT}/utils/video/LightX2V-main"
pip install -e .

echo "============================================"
echo "Step 5: 安装额外依赖 (lingbot-world + common)"
echo "============================================"
pip install "diffusers>=0.31.0" "transformers>=4.49.0" "tokenizers>=0.20.3" "accelerate>=1.1.1"
pip install safetensors einops scipy "opencv-python>=4.9.0.80" "imageio[ffmpeg]" imageio-ffmpeg
pip install loguru tqdm numpy easydict ftfy sgl-kernel huggingface_hub

echo "============================================"
echo "Step 5b: 安装 CosyVoice 额外依赖"
echo "============================================"
pip install pyworld hyperpyyaml conformer lightning
# 必须是 GPU 版且 >=1.19 (numpy 2.x ABI); CPU 版会让 speech tokenizer 静默跑在 CPU 上
pip install onnxruntime-gpu==1.20.1 --no-deps

echo "============================================"
echo "Step 5c: 安装 MagiHuman 额外依赖"
echo "============================================"
pip install pydantic pydantic-settings sentencepiece unfoldNd torchao
pip install timm==1.0.20 ffmpeg-python==0.2.0 av==15.1.0 mosaicml_streaming==0.8.0
pip install openai-whisper --no-deps
pip install tiktoken rich

echo "============================================"
echo "Step 6: 下载模型"
echo "============================================"
cd "${IT2AV_CODE_ROOT}"
python download/download_models.py

echo "============================================"
echo "安装完成！"
echo "============================================"
echo ""
echo "模型目录: ${IT2AV_WORKSPACE}/models/"
echo "代码目录: ${IT2AV_CODE_ROOT}/utils/video/LightX2V-main/"
echo ""
echo "激活环境: conda activate ${IT2AV_CONDA_ENV}"
