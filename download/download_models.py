#!/usr/bin/env python3
"""Download World Model models from HuggingFace/ModelScope (lingbot, Wan2.1, Wan2.2, VACE, HunyuanVideo, LiveAvatar, MagiHuman)."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import login, snapshot_download
from tqdm import tqdm

# ============================================================================
# 代理配置区域 - Proxy Configuration
# ============================================================================
USERNAME = os.environ.get('IT2AV_PROXY_USER', '')
PASSWORD = os.environ.get('IT2AV_PROXY_PASS', '')

PROXY_HOST = os.environ.get('IT2AV_PROXY_HOST', '')
PROXY_PORT = os.environ.get('IT2AV_PROXY_PORT', '')
NO_PROXY_DOMAINS = os.environ.get('NO_PROXY', 'localhost,127.0.0.1')

# ============================================================================
# HuggingFace Token - 用于访问 gated repo (Gemma, FLUX.1 等)
# ============================================================================
HF_TOKEN = os.environ.get('HF_TOKEN', '')

# ============================================================================
# 模型配置
# ============================================================================
MODEL_REPOS = [
    "robbyant/lingbot-world-base-cam",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "Wan-AI/Wan2.1-FLF2V-14B-720P",
    "Wan-AI/Wan2.2-I2V-A14B",
    "Wan-AI/Wan2.2-TI2V-5B",
    "Wan-AI/Wan2.1-VACE-14B",
    "Alibaba-PAI/Wan2.2-VACE-Fun-A14B",
    # HunyuanVideo-1.5 主模型 (含 transformer + VAE)
    "tencent/HunyuanVideo-1.5",
    # HunyuanVideo-1.5 Text Encoders
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "google/byt5-small",
    # HunyuanVideo-1.5 Vision Encoder (I2V 需要)
    # 注意: FLUX.1-Redux-dev 需要 HuggingFace 申请访问权限, 请先在 HF 网页上 request access
    "black-forest-labs/FLUX.1-Redux-dev",
    # LiveAvatar 数字人 (基于 Wan2.2-S2V-14B + LoRA)
    "Wan-AI/Wan2.2-S2V-14B",
    "Quark-Vision/Live-Avatar",
    # LatentSync 1.6 唇形同步 (UNet + Whisper + auxiliary)
    "ByteDance/LatentSync-1.6",
    # daVinci-MagiHuman 音视频联合生成 (base, distill, SR, turbo_vae)
    "GAIR/daVinci-MagiHuman",
    # MagiHuman 依赖: T5-Gemma Text Encoder
    "google/t5gemma-9b-9b-ul2",
    # MagiHuman 依赖: Stable Audio VAE
    "stabilityai/stable-audio-open-1.0",
]

DEFAULT_TARGET = os.path.join(os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace")), "models")


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
    """登录 HuggingFace 以访问 gated repo (如 Gemma, FLUX.1)."""
    token = token or HF_TOKEN
    if not token or token == 'YOUR_HF_TOKEN':
        logging.warning("HuggingFace token not configured! Gated repos (Gemma, FLUX.1) will fail.")
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
    parser = argparse.ArgumentParser(description="Download World Model models from HuggingFace")
    parser.add_argument("--target-root", default=DEFAULT_TARGET, help="Destination directory")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--revision", default=None, help="Specific revision to pin")
    parser.add_argument("--resume", action="store_true", help="(deprecated, downloads always resume)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (overrides built-in)")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=[
            "lingbot", "wan22", "wan21_flf2v", "wan22_i2v", "wan22_ti2v",
            "wan21_vace", "wan22_vace_fun",
            "hyvideo15", "hyvideo15_text_enc", "hyvideo15_vision_enc",
            "glyph_sdxl",
            "liveavatar", "wan22_s2v",
            "latentsync",
            "magihuman",
            "all",
        ],
        default=["all"],
        help="Which models to download (default: all)",
    )
    return parser.parse_args()


def get_local_dir_name(repo_id: str) -> str:
    """Map repo_id to local directory name."""
    mapping = {
        "robbyant/lingbot-world-base-cam": "lingbot-world-base-cam",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "Wan2.2-T2V-A14B-Diffusers",
        "Wan-AI/Wan2.1-FLF2V-14B-720P": "Wan2.1-FLF2V-14B-720P",
        "Wan-AI/Wan2.2-I2V-A14B": "Wan2.2-I2V-A14B",
        "Wan-AI/Wan2.2-TI2V-5B": "Wan2.2-TI2V-5B",
        "Wan-AI/Wan2.1-VACE-14B": "Wan2.1-VACE-14B",
        "Alibaba-PAI/Wan2.2-VACE-Fun-A14B": "Wan2.2-VACE-Fun-A14B",
        # HunyuanVideo-1.5: 按官方 ckpts 目录结构放置
        "tencent/HunyuanVideo-1.5": "HunyuanVideo-1.5",
        "Qwen/Qwen2.5-VL-7B-Instruct": "HunyuanVideo-1.5/text_encoder/llm",
        "google/byt5-small": "HunyuanVideo-1.5/text_encoder/byt5-small",
        "black-forest-labs/FLUX.1-Redux-dev": "HunyuanVideo-1.5/vision_encoder/siglip",
        # LiveAvatar
        "Wan-AI/Wan2.2-S2V-14B": "Wan2.2-S2V-14B",
        "Quark-Vision/Live-Avatar": "LiveAvatar",
        # LatentSync 唇形同步
        "ByteDance/LatentSync-1.6": "LatentSync-1.6",
        # daVinci-MagiHuman 音视频联合生成
        "GAIR/daVinci-MagiHuman": "daVinci-MagiHuman",
        # MagiHuman 依赖模型
        "google/t5gemma-9b-9b-ul2": "t5gemma-9b-9b-ul2",
        "stabilityai/stable-audio-open-1.0": "stable-audio-open-1.0",
    }
    return mapping.get(repo_id, repo_id.replace("/", "__"))


def get_selected_repos(models_arg: list[str]) -> list[str]:
    """Get repos based on user selection."""
    if "all" in models_arg:
        return MODEL_REPOS

    selected = []
    for m in models_arg:
        if m == "lingbot":
            selected.append("robbyant/lingbot-world-base-cam")
        elif m == "wan22":
            selected.append("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        elif m == "wan21_flf2v":
            selected.append("Wan-AI/Wan2.1-FLF2V-14B-720P")
        elif m == "wan22_i2v":
            selected.append("Wan-AI/Wan2.2-I2V-A14B")
        elif m == "wan22_ti2v":
            selected.append("Wan-AI/Wan2.2-TI2V-5B")
        elif m == "wan21_vace":
            selected.append("Wan-AI/Wan2.1-VACE-14B")
        elif m == "wan22_vace_fun":
            selected.append("Alibaba-PAI/Wan2.2-VACE-Fun-A14B")
        elif m == "hyvideo15":
            selected.append("tencent/HunyuanVideo-1.5")
        elif m == "hyvideo15_text_enc":
            selected.append("Qwen/Qwen2.5-VL-7B-Instruct")
            selected.append("google/byt5-small")
        elif m == "glyph_sdxl":
            # Glyph-SDXL-v2 从 ModelScope 下载, 标记为特殊处理
            selected.append("__modelscope__/AI-ModelScope/Glyph-SDXL-v2")
        elif m == "hyvideo15_vision_enc":
            selected.append("black-forest-labs/FLUX.1-Redux-dev")
        elif m == "liveavatar":
            # LiveAvatar 需要基座模型 + LoRA
            selected.append("Wan-AI/Wan2.2-S2V-14B")
            selected.append("Quark-Vision/Live-Avatar")
        elif m == "wan22_s2v":
            selected.append("Wan-AI/Wan2.2-S2V-14B")
        elif m == "latentsync":
            selected.append("ByteDance/LatentSync-1.6")
        elif m == "magihuman":
            # MagiHuman 主模型 + 依赖 (Text Encoder + Audio VAE)
            # Video VAE (Wan2.2-TI2V-5B) 已在 wan22_ti2v 中, 但自动包含
            selected.append("GAIR/daVinci-MagiHuman")
            selected.append("google/t5gemma-9b-9b-ul2")
            selected.append("stabilityai/stable-audio-open-1.0")
            if "Wan-AI/Wan2.2-TI2V-5B" not in selected:
                selected.append("Wan-AI/Wan2.2-TI2V-5B")
    return selected


def download_glyph_sdxl_v2(target_root: Path) -> None:
    """Download Glyph-SDXL-v2 from ModelScope (required by HunyuanVideo-1.5 byT5 text encoder)."""
    glyph_target = target_root / "HunyuanVideo-1.5" / "text_encoder" / "Glyph-SDXL-v2"
    marker_file = glyph_target / "checkpoints" / "byt5_model.pt"

    if marker_file.exists():
        logging.info("  ⏭️  Glyph-SDXL-v2 already exists at: %s", glyph_target)
        return

    logging.info("  Downloading AI-ModelScope/Glyph-SDXL-v2 from ModelScope...")
    try:
        from modelscope import snapshot_download as ms_snapshot_download
    except ImportError:
        logging.error("  ❌ modelscope package not installed! Run: pip install modelscope")
        raise

    glyph_cache = ms_snapshot_download(
        "AI-ModelScope/Glyph-SDXL-v2",
        cache_dir="/tmp/glyph_cache",
    )

    glyph_target.mkdir(parents=True, exist_ok=True)
    for item in os.listdir(glyph_cache):
        src = os.path.join(glyph_cache, item)
        dst = str(glyph_target / item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    logging.info("  ✅ Glyph-SDXL-v2 downloaded to: %s", glyph_target)


def download_models(
    target_root: Path,
    repos: list[str],
    max_workers: int,
    revision: str | None,
    token: str | None = None,
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    token = token or HF_TOKEN

    # 分离 ModelScope 下载 和 HuggingFace 下载
    hf_repos = [r for r in repos if not r.startswith("__modelscope__/")]
    ms_repos = [r for r in repos if r.startswith("__modelscope__/")]

    total = len(hf_repos) + len(ms_repos)
    idx = 0

    # HuggingFace 下载
    for repo_id in hf_repos:
        idx += 1
        local_dir = target_root / get_local_dir_name(repo_id)

        logging.info("[%d/%d] Downloading %s -> %s", idx, total, repo_id, local_dir)

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

    # ModelScope 下载 (Glyph-SDXL-v2 等)
    for repo_tag in ms_repos:
        idx += 1
        repo_name = repo_tag.replace("__modelscope__/", "")
        logging.info("[%d/%d] Downloading %s (ModelScope) -> %s", idx, total, repo_name, target_root)

        if "Glyph-SDXL-v2" in repo_name:
            download_glyph_sdxl_v2(target_root)
        else:
            logging.warning("  ⚠️ Unknown ModelScope repo: %s", repo_name)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("Step 1: Configuring Network Proxy")
    logging.info("=" * 60)
    setup_proxy()
    print()

    logging.info("=" * 60)
    logging.info("Step 2: Logging in to HuggingFace (for gated repos)")
    logging.info("=" * 60)
    setup_hf_token(args.hf_token)
    print()

    repos = get_selected_repos(args.models)

    # 如果下载 all 或 hyvideo15_text_enc，自动包含 Glyph-SDXL-v2
    glyph_tag = "__modelscope__/AI-ModelScope/Glyph-SDXL-v2"
    if "all" in args.models or "hyvideo15_text_enc" in args.models or "glyph_sdxl" in args.models:
        if glyph_tag not in repos:
            repos.append(glyph_tag)

    logging.info("Models to download:")
    for r in repos:
        if r.startswith("__modelscope__/"):
            logging.info("  - %s (ModelScope)", r.replace("__modelscope__/", ""))
        else:
            logging.info("  - %s -> %s/%s", r, args.target_root, get_local_dir_name(r))
    print()

    logging.info("=" * 60)
    logging.info("Step 3: Downloading Models")
    logging.info("=" * 60)
    try:
        download_models(Path(args.target_root), repos, args.max_workers, args.revision, args.hf_token)
        logging.info("\n" + "=" * 60)
        logging.info("All models downloaded successfully to %s", args.target_root)
        logging.info("=" * 60)
    except KeyboardInterrupt:
        logging.warning("\nDownload interrupted. Use --resume to continue later.")
        sys.exit(1)
    except Exception as e:
        logging.error("\nDownload failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
