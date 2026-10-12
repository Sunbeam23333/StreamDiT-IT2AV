#!/usr/bin/env python3
"""
Download Audio Generation models from HuggingFace for world model audio pipeline.

支持的模型:
  === 文生音效 (Text-to-Audio) ===
  - TangoFlux          : 515M参数, Flow Matching + CRPO, 30秒44.1kHz, ICLR 2026
  - flan-t5-large      : TangoFlux text encoder (google/flan-t5-large, 780M参数)
  
  === 文生音乐 (Text-to-Music / Song Generation) ===
  - ACE-Step v1 3.5B   : 3.5B参数, Apache 2.0, 19语言, 4分钟/20秒生成
  - SongGeneration-v2-large : 4B参数, 商用级歌曲生成, 4m30s, 人声+伴奏
  - SongGeneration Runtime  : 运行时依赖 (codec, third_party)
  
  === 人声合成 (Text-to-Speech) ===
  - CosyVoice3 (Fun-CosyVoice3-0.5B) : 0.5B参数, 9语言+18方言, SOTA TTS
  - CosyVoice2-0.5B    : 流式推理, 5语言, 零样本声音克隆
  - CosyVoice-ttsfrd   : 文本前端资源包
  - F5-TTS             : ~330M参数, Flow Matching + DiT, 零样本3秒克隆, 中英双语
  - Fish-Speech 1.5    : ~500M参数, 15万小时训练, 极速推理, 多语言
  - Spark-TTS 0.5B     : 0.5B参数, LLM-based单流解耦, 音高/语速可控, 中英双语
  - ChatTTS            : ~400M参数, 对话场景专用, 韵律自然, 10万+小时训练

目标:
  为视频/图文 prompt 匹配对应的音效/BGM/人声

用法:
    python download_audio_models.py --models all
    python download_audio_models.py --models tangoflux ace_step
    python download_audio_models.py --models cosyvoice3 cosyvoice2
    python download_audio_models.py --models tts         # 所有 TTS 模型
    python download_audio_models.py --models f5tts fish_speech spark_tts chattts
    python download_audio_models.py --models songgen_v2_large songgen_runtime
    python download_audio_models.py --target-root /custom/path
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import login, snapshot_download

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ============================================================================
# 代理配置区域 - Proxy Configuration
# ============================================================================
USERNAME = os.environ.get('IT2AV_PROXY_USER', '')
PASSWORD = os.environ.get('IT2AV_PROXY_PASS', '')

PROXY_HOST = os.environ.get('IT2AV_PROXY_HOST', '')
PROXY_PORT = os.environ.get('IT2AV_PROXY_PORT', '')
NO_PROXY_DOMAINS = os.environ.get('NO_PROXY', 'localhost,127.0.0.1')

# ============================================================================
# HuggingFace Token
# ============================================================================
HF_TOKEN = os.environ.get('HF_TOKEN', '')

# ============================================================================
# 模型配置
# ============================================================================
AUDIO_MODEL_REPOS = {
    # === Text-to-Audio (音效) ===
    "tangoflux": {
        "repo_id": "declare-lab/TangoFlux",
        "local_dir": "TangoFlux",
        "description": "TangoFlux - 515M params, Flow Matching + CRPO, 30s 44.1kHz audio, ICLR 2026",
    },
    "flan_t5_large": {
        "repo_id": "google/flan-t5-large",
        "local_dir": "flan-t5-large",
        "description": "Flan-T5 Large - TangoFlux text encoder (780M params, Google T5)",
    },

    # === Text-to-Music (音乐/歌曲生成) ===
    "ace_step": {
        "repo_id": "ACE-Step/ACE-Step-v1-3.5B",
        "local_dir": "ACE-Step-v1-3.5B",
        "description": "ACE-Step v1 - 3.5B params, 19 languages, 4min in 20s, Apache 2.0",
    },
    "songgen_v2_large": {
        "repo_id": "lglg666/SongGeneration-v2-large",
        "local_dir": "SongGeneration-v2-large",
        "description": "SongGeneration v2 Large - 4B params, commercial-grade song gen, vocals+accompaniment",
    },
    "songgen_runtime": {
        "repo_id": "lglg666/SongGeneration-Runtime",
        "local_dir": "SongGeneration-Runtime",
        "description": "SongGeneration Runtime - codecs, tokenizers, third_party dependencies",
    },

    # === Text-to-Speech (人声) ===
    "cosyvoice3": {
        "repo_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "local_dir": "Fun-CosyVoice3-0.5B",
        "description": "Fun-CosyVoice3 - 0.5B params, 9 languages + 18 dialects, SOTA zero-shot TTS",
    },
    "cosyvoice2": {
        "repo_id": "FunAudioLLM/CosyVoice2-0.5B",
        "local_dir": "CosyVoice2-0.5B",
        "description": "CosyVoice2 - 0.5B params, 5 languages, streaming, zero-shot voice cloning",
    },
    "cosyvoice_ttsfrd": {
        "repo_id": "FunAudioLLM/CosyVoice-ttsfrd",
        "local_dir": "CosyVoice-ttsfrd",
        "description": "CosyVoice ttsfrd - text frontend resource for better normalization",
    },

    # === Text-to-Speech (备选方案) ===
    "f5tts": {
        "repo_id": "SWivid/F5-TTS",
        "local_dir": "F5-TTS",
        "description": "F5-TTS - ~330M params, Flow Matching + DiT, zero-shot 3s voice cloning, CN+EN",
        "github": "https://github.com/SWivid/F5-TTS",
    },
    "fish_speech": {
        "repo_id": "fishaudio/fish-speech-1.5",
        "local_dir": "Fish-Speech-1.5",
        "description": "Fish-Speech 1.5 - ~500M params, 150K hours trained, ultra-fast, multi-lingual",
        "github": "https://github.com/fishaudio/fish-speech",
    },
    "spark_tts": {
        "repo_id": "SparkAudio/Spark-TTS-0.5B",
        "local_dir": "Spark-TTS-0.5B",
        "description": "Spark-TTS - 0.5B params, LLM-based single-stream, pitch/speed control, CN+EN",
        "github": "https://github.com/SparkAudio/Spark-TTS",
    },
    "chattts": {
        "repo_id": "2Noise/ChatTTS",
        "local_dir": "ChatTTS",
        "description": "ChatTTS - ~400M params, dialogue-optimized, natural prosody, 100K+ hours trained",
        "github": "https://github.com/2noise/ChatTTS",
    },
}

DEFAULT_TARGET = os.path.join(os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace")), "models/Audio")


def setup_proxy() -> None:
    """配置代理以访问外网."""
    if os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy'):
        logging.info("Proxy already configured in environment")
        return

    if not PROXY_HOST:

        logging.info("No proxy configured (IT2AV_PROXY_HOST unset), using direct connection")
        return

    proxy_url = (f'http://{USERNAME}:{PASSWORD}@{PROXY_HOST}:{PROXY_PORT}' if USERNAME

                 else f'http://{PROXY_HOST}:{PROXY_PORT}')

    os.environ['NO_PROXY'] = NO_PROXY_DOMAINS
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['no_proxy'] = NO_PROXY_DOMAINS
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url

    logging.info("Proxy configured: %s:%s (user: %s)", PROXY_HOST, PROXY_PORT, USERNAME or "-")


def setup_hf_token(token: str | None = None) -> None:
    """登录 HuggingFace."""
    token = token or HF_TOKEN
    if not token or token == 'YOUR_HF_TOKEN':
        logging.warning("HuggingFace token not configured!")
        return

    try:
        login(token=token, add_to_git_credential=False)
        logging.info("HuggingFace login successful (token: %s...%s)", token[:5], token[-4:])
    except Exception as e:
        logging.warning("HuggingFace login failed: %s", e)
        os.environ['HF_TOKEN'] = token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = token
        logging.info("Set HF_TOKEN in environment as fallback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Audio Generation models from HuggingFace")
    parser.add_argument("--target-root", default=DEFAULT_TARGET, help="Destination directory")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--revision", default=None, help="Specific revision to pin")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (overrides built-in)")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(AUDIO_MODEL_REPOS.keys()) + ["all", "audio", "music", "tts", "tts_alt"],
        default=["all"],
        help=(
            "Which models to download. Groups: "
            "all (everything), "
            "audio (TangoFlux), "
            "music (ACE-Step + SongGeneration), "
            "tts (CosyVoice3 + CosyVoice2), "
            "tts_alt (F5-TTS + Fish-Speech + Spark-TTS + ChatTTS)"
        ),
    )
    return parser.parse_args()


# 模型组映射
MODEL_GROUPS = {
    "audio": ["tangoflux", "flan_t5_large"],
    "music": ["ace_step", "songgen_v2_large", "songgen_runtime"],
    "tts": ["cosyvoice3", "cosyvoice2", "cosyvoice_ttsfrd", "f5tts", "fish_speech", "spark_tts", "chattts"],
    "tts_alt": ["f5tts", "fish_speech", "spark_tts", "chattts"],
}


def get_selected_repos(models_arg: list[str]) -> list[dict]:
    """Get repos based on user selection, supporting group aliases."""
    selected_keys: set[str] = set()

    for m in models_arg:
        if m == "all":
            selected_keys = set(AUDIO_MODEL_REPOS.keys())
            break
        elif m in MODEL_GROUPS:
            selected_keys.update(MODEL_GROUPS[m])
        elif m in AUDIO_MODEL_REPOS:
            selected_keys.add(m)

    # 保持定义顺序
    selected = []
    for key in AUDIO_MODEL_REPOS:
        if key in selected_keys:
            selected.append(AUDIO_MODEL_REPOS[key])
    return selected


def download_models(
    target_root: Path,
    repos: list[dict],
    max_workers: int,
    revision: str | None,
    token: str | None = None,
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    token = token or HF_TOKEN

    total = len(repos)
    for idx, repo_info in enumerate(repos, 1):
        repo_id = repo_info["repo_id"]
        local_dir = target_root / repo_info["local_dir"]

        logging.info("[%d/%d] %s", idx, total, repo_info["description"])
        logging.info("  Repo: %s -> %s", repo_id, local_dir)

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token,
                **({"revision": revision} if revision else {}),
            )
            logging.info("  ✅ Successfully downloaded %s (%d/%d)", repo_id, idx, total)
        except Exception as e:
            logging.error("  ❌ Failed to download %s: %s", repo_id, e)
            raise


def setup_songgen_runtime(target_root: Path) -> None:
    """Setup SongGeneration runtime: move ckpt and third_party to proper locations."""
    runtime_dir = target_root / "SongGeneration-Runtime"
    if not runtime_dir.exists():
        return

    logging.info("Setting up SongGeneration runtime structure...")

    # 创建符号链接或提示信息
    ckpt_src = runtime_dir / "ckpt"
    tp_src = runtime_dir / "third_party"

    if ckpt_src.exists():
        logging.info("  SongGeneration ckpt available at: %s", ckpt_src)
    if tp_src.exists():
        logging.info("  SongGeneration third_party available at: %s", tp_src)

    logging.info(
        "  NOTE: When running SongGeneration, set working directory or symlink:\n"
        "    ln -s %s/ckpt <SongGeneration_code>/ckpt\n"
        "    ln -s %s/third_party <SongGeneration_code>/third_party",
        runtime_dir, runtime_dir,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("Audio Generation Model Downloader")
    logging.info("=" * 60)
    logging.info("")
    logging.info("Target categories:")
    logging.info("  🎵 Text-to-Audio (音效): TangoFlux")
    logging.info("  🎶 Text-to-Music (BGM/歌曲): ACE-Step, SongGeneration")
    logging.info("  🗣️  Text-to-Speech (人声): CosyVoice3, CosyVoice2")
    logging.info("  🗣️  Text-to-Speech (备选): F5-TTS, Fish-Speech, Spark-TTS, ChatTTS")
    logging.info("")

    logging.info("Step 1: Configuring Network Proxy")
    setup_proxy()
    print()

    logging.info("Step 2: Logging in to HuggingFace")
    setup_hf_token(args.hf_token)
    print()

    repos = get_selected_repos(args.models)
    logging.info("Models to download (%d):", len(repos))
    for r in repos:
        logging.info("  - [%s] %s", r["local_dir"], r["description"])
        logging.info("    -> %s/%s", args.target_root, r["local_dir"])
    print()

    logging.info("Step 3: Downloading Models to %s", args.target_root)
    try:
        download_models(Path(args.target_root), repos, args.max_workers, args.revision, args.hf_token)

        # 后处理: SongGeneration runtime 设置
        setup_songgen_runtime(Path(args.target_root))

        logging.info("\n" + "=" * 60)
        logging.info("All audio models downloaded successfully!")
        logging.info("  Target: %s", args.target_root)
        logging.info("")
        logging.info("Model layout:")
        logging.info("  %s/TangoFlux              <- 音效生成", args.target_root)
        logging.info("  %s/flan-t5-large          <- TangoFlux text encoder", args.target_root)
        logging.info("  %s/ACE-Step-v1-3.5B       <- 音乐/歌曲生成", args.target_root)
        logging.info("  %s/SongGeneration-v2-large <- 商用级歌曲生成", args.target_root)
        logging.info("  %s/SongGeneration-Runtime  <- SongGen 运行时", args.target_root)
        logging.info("  %s/Fun-CosyVoice3-0.5B    <- TTS (人声)", args.target_root)
        logging.info("  %s/CosyVoice2-0.5B        <- TTS (人声)", args.target_root)
        logging.info("  %s/CosyVoice-ttsfrd       <- TTS 前端资源", args.target_root)
        logging.info("  %s/F5-TTS                 <- TTS 备选: F5-TTS", args.target_root)
        logging.info("  %s/Fish-Speech-1.5        <- TTS 备选: Fish-Speech", args.target_root)
        logging.info("  %s/Spark-TTS-0.5B         <- TTS 备选: Spark-TTS", args.target_root)
        logging.info("  %s/ChatTTS                <- TTS 备选: ChatTTS", args.target_root)
        logging.info("=" * 60)
    except KeyboardInterrupt:
        logging.warning("\nDownload interrupted. Re-run to resume (auto-resume supported).")
        sys.exit(1)
    except Exception as e:
        logging.error("\nDownload failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
