#!/bin/bash
# ============================================================================
# LiveAvatar 环境安装脚本
# 服务器: 8×H20 95GB GPU
#
# 用法:
#   bash setup_liveavatar_env.sh
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"
conda activate ${IT2AV_CONDA_ENV}
echo "============================================"
echo "LiveAvatar 环境安装"
echo "============================================"

# ===== 路径 =====
GROUP_BASE=${IT2AV_WORKSPACE}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== Step 1: Conda 环境 =====
echo ""
echo "[Step 1] 创建 conda 环境..."
if conda info --envs | grep -q "liveavatar"; then
    echo "  环境 liveavatar 已存在, 跳过创建"
else
    conda create -n liveavatar python=3.10 -y
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate liveavatar

# ===== Step 2: PyTorch =====
echo ""
echo "[Step 2] 安装 PyTorch..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio --index-url https://download.pytorch.org/whl/cu128

# ===== Step 3: Flash Attention 2 (H20 GPU) =====
echo ""
echo "[Step 3] 安装 Flash Attention 2 (H20 GPU 不支持 FA3)..."
pip install flash-attn==2.8.3

# ===== Step 4: 依赖包 =====
echo ""
echo "[Step 4] 安装 Python 依赖..."
pip install -r ${SCRIPT_DIR}/requirements_liveavatar.txt

# ===== Step 5: mmlab packages (for face detection) =====
echo ""
echo "[Step 5] 安装 mmlab 相关包..."
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"

# ===== Step 6: ffmpeg =====
echo ""
echo "[Step 6] 检查 ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "  ffmpeg 已安装: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "  ⚠️  ffmpeg 未安装, 请手动安装:"
    echo "    apt-get install -y ffmpeg"
    echo "  或下载静态版本:"
    echo "    wget https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.4-amd64-static.tar.xz"
fi

# ===== Step 7: 下载模型 =====
echo ""
echo "[Step 7] 下载 LiveAvatar 模型..."
echo "  使用 download_models.py 下载:"
echo "    python ${IT2AV_CODE_ROOT}/download/download_models.py --models liveavatar"
echo ""
echo "  这将下载:"
echo "    1. Wan2.2-S2V-14B (~28GB)  -> ${GROUP_BASE}/models/Wan2.2-S2V-14B/"
echo "    2. LiveAvatar LoRA (~200MB) -> ${GROUP_BASE}/models/LiveAvatar/"

echo ""
echo "============================================"
echo "✅ LiveAvatar 环境安装完成!"
echo ""
echo "后续步骤:"
echo "  1. 下载模型: python ${IT2AV_CODE_ROOT}/download/download_models.py --models liveavatar"
echo "  2. 生成人像: python ${IT2AV_CODE_ROOT}/utils/video/LiveAvatar/generate_ref_portraits.py"
echo "  3. 准备样本: python ${IT2AV_CODE_ROOT}/utils/video/LiveAvatar/prepare_digit_samples.py"
echo "  4. 运行推理: bash ${SCRIPT_DIR}/run_liveavatar_benchmark.sh"
echo "============================================"
