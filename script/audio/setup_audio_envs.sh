#!/bin/bash
# ============================================================================
# 音频生成环境安装脚本
#
# 由于各模型的 diffusers/transformers 版本有冲突，建议分环境安装:
#
# | 模型              | diffusers  | transformers | 最低显存 |
# |-------------------|------------|--------------|----------|
# | TangoFlux         | ==0.30.0   | ==4.44.0     | ~4GB     |
# | ACE-Step          | >=0.33.0   | ==4.50.0     | ~16GB    |
# | SongGeneration v2 | ==0.27.2   | ==4.37.2     | ~32GB    |
# | CosyVoice3        | ==0.29.0   | ==4.51.3     | ~4GB     |
#
# 用法:
#   bash setup_audio_envs.sh all          # 安装全部环境
#   bash setup_audio_envs.sh tangoflux    # 只安装 TangoFlux
#   bash setup_audio_envs.sh ace_step     # 只安装 ACE-Step
#   bash setup_audio_envs.sh songgen      # 只安装 SongGeneration
#   bash setup_audio_envs.sh cosyvoice    # 只安装 CosyVoice
# ============================================================================

set -e

TARGET=${1:-"all"}
CUDA_VERSION=${2:-"cu121"}  # cu118, cu121, cu124

echo "============================================"
echo "Audio Generation Environment Setup"
echo "  Target:       ${TARGET}"
echo "  CUDA:         ${CUDA_VERSION}"
echo "============================================"

# ===== TangoFlux 环境 =====
setup_tangoflux() {
    echo ""
    echo ">>> Setting up TangoFlux environment..."
    conda create -n audio_tangoflux python=3.10 -y 2>/dev/null || true
    eval "$(conda shell.bash hook)"
    conda activate audio_tangoflux

    pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

    pip install \
        diffusers==0.30.0 \
        transformers==4.44.0 \
        accelerate==0.34.2 \
        datasets==2.21.0 \
        huggingface_hub \
        safetensors \
        torchlibrosa==0.1.0 \
        librosa \
        tqdm \
        soundfile

    echo ">>> TangoFlux environment ready: conda activate audio_tangoflux"
}

# ===== ACE-Step 环境 =====
setup_ace_step() {
    echo ""
    echo ">>> Setting up ACE-Step environment..."
    conda create -n audio_ace_step python=3.10 -y 2>/dev/null || true
    eval "$(conda shell.bash hook)"
    conda activate audio_ace_step

    pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

    pip install \
        diffusers>=0.33.0 \
        transformers==4.50.0 \
        accelerate>=1.0.0 \
        datasets==3.4.1 \
        huggingface_hub \
        safetensors \
        librosa==0.11.0 \
        soundfile==0.13.1 \
        loguru==0.7.3 \
        pypinyin==0.53.0 \
        pytorch_lightning==2.5.1 \
        py3langid==0.3.0 \
        hangul-romanize==0.1.0 \
        num2words==0.5.14 \
        "spacy>=3.8.0" \
        cutlet \
        "fugashi[unidic-lite]" \
        click \
        peft \
        einops \
        tqdm \
        tensorboard \
        tensorboardX \
        matplotlib

    echo ">>> ACE-Step environment ready: conda activate audio_ace_step"
}

# ===== SongGeneration 环境 =====
setup_songgen() {
    echo ""
    echo ">>> Setting up SongGeneration environment..."
    conda create -n audio_songgen python=3.10 -y 2>/dev/null || true
    eval "$(conda shell.bash hook)"
    conda activate audio_songgen

    pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

    pip install \
        diffusers==0.27.2 \
        transformers==4.37.2 \
        huggingface_hub==0.25.2 \
        safetensors \
        omegaconf==2.3.0 \
        alias-free-torch \
        descript-audio-codec \
        einops \
        einops-exts \
        flashy \
        julius \
        k-diffusion==0.1.1 \
        kaldiio \
        lameenc \
        librosa \
        "lightning>=2.2.4" \
        ninja \
        nnAudio \
        openunmix \
        peft==0.10.0 \
        "vector-quantize-pytorch>=1.22.17" \
        "x-transformers>=2.3.25" \
        tqdm \
        soundfile

    # Flash Attention (可选, 但推荐)
    pip install flash-attn --no-build-isolation 2>/dev/null || \
        echo "[WARN] flash-attn install failed, SongGeneration will work without it (slower)"

    echo ">>> SongGeneration environment ready: conda activate audio_songgen"
}

# ===== CosyVoice 环境 =====
setup_cosyvoice() {
    echo ""
    echo ">>> Setting up CosyVoice environment..."
    conda create -n audio_cosyvoice python=3.10 -y 2>/dev/null || true
    eval "$(conda shell.bash hook)"
    conda activate audio_cosyvoice

    pip install torch==2.3.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

    pip install \
        diffusers==0.29.0 \
        transformers==4.51.3 \
        huggingface_hub \
        safetensors \
        conformer==0.3.2 \
        hydra-core==1.3.2 \
        HyperPyYAML==1.2.3 \
        inflect==7.3.1 \
        modelscope==1.20.0 \
        networkx==3.1 \
        numpy==1.26.4 \
        omegaconf==2.3.0 \
        protobuf==4.25 \
        pyarrow==18.1.0 \
        pydantic==2.7.0 \
        pyworld==0.3.4 \
        rich==13.7.1 \
        soundfile==0.12.1 \
        tensorboard==2.14.0 \
        librosa==0.10.2 \
        x-transformers==2.11.24 \
        wetext==0.0.4 \
        wget==3.2 \
        tqdm

    # Linux 专用依赖
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        pip install deepspeed==0.15.1 2>/dev/null || echo "[WARN] deepspeed install failed"
        pip install onnxruntime-gpu==1.18.0 \
            --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
            2>/dev/null || echo "[WARN] onnxruntime-gpu install failed, using CPU onnxruntime"
    else
        pip install onnxruntime==1.18.0 2>/dev/null || true
    fi

    echo ">>> CosyVoice environment ready: conda activate audio_cosyvoice"
}

# ===== 执行 =====
case $TARGET in
    tangoflux)  setup_tangoflux ;;
    ace_step)   setup_ace_step ;;
    songgen)    setup_songgen ;;
    cosyvoice)  setup_cosyvoice ;;
    all)
        setup_tangoflux
        setup_ace_step
        setup_songgen
        setup_cosyvoice
        ;;
    *)
        echo "[ERROR] Unknown target: $TARGET"
        echo "Supported: tangoflux, ace_step, songgen, cosyvoice, all"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Available environments:"
echo "  conda activate audio_tangoflux   # TangoFlux (音效, ~4GB VRAM)"
echo "  conda activate audio_ace_step    # ACE-Step (BGM, ~16GB VRAM)"
echo "  conda activate audio_songgen     # SongGeneration (歌曲, ~32GB VRAM)"
echo "  conda activate audio_cosyvoice   # CosyVoice3 (语音, ~4GB VRAM)"
echo ""
echo "Quick start:"
echo "  conda activate audio_tangoflux"
echo "  bash script/audio/run_audio_benchmark.sh tangoflux"
echo "============================================"
