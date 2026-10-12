#!/usr/bin/env python3
"""
Download stabilityai/stable-audio-open-1.0 for MagiHuman Audio VAE.

MagiHuman 的 SAAudioFeatureExtractor 需要这个模型:
  - model_config.json  (模型结构配置)
  - model.safetensors  (权重文件)

用法:
    python download_stable_audio.py
    python download_stable_audio.py --target-root /custom/path/models
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import login, snapshot_download

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
REPO_ID = "stabilityai/stable-audio-open-1.0"
LOCAL_DIR_NAME = "stable-audio-open-1.0"
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
    """登录 HuggingFace."""
    token = token or HF_TOKEN
    if not token or token == 'YOUR_HF_TOKEN':
        logging.warning("HuggingFace token not configured!")
        return

    try:
        login(token=token, add_to_git_credential=False)
        logging.info("HuggingFace login successful")
    except Exception as e:
        logging.warning("HuggingFace login failed: %s", e)
        os.environ['HF_TOKEN'] = token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = token
        logging.info("Set HF_TOKEN in environment as fallback")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download stable-audio-open-1.0 for MagiHuman")
    parser.add_argument("--target-root", default=DEFAULT_TARGET, help="Parent directory for models")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("Downloading: %s", REPO_ID)
    logging.info("  -> %s/%s", args.target_root, LOCAL_DIR_NAME)
    logging.info("=" * 60)

    setup_proxy()
    setup_hf_token(args.hf_token)

    target_dir = Path(args.target_root) / LOCAL_DIR_NAME
    target_dir.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or HF_TOKEN

    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(target_dir),
            token=token,
        )
        logging.info("✅ Successfully downloaded %s", REPO_ID)
        logging.info("   Location: %s", target_dir)

        # 验证关键文件
        for required_file in ["model_config.json", "model.safetensors"]:
            fpath = target_dir / required_file
            if fpath.exists():
                size_mb = fpath.stat().st_size / (1024 * 1024)
                logging.info("   ✅ %s (%.1f MB)", required_file, size_mb)
            else:
                logging.warning("   ⚠️  %s NOT FOUND!", required_file)

    except Exception as e:
        logging.error("❌ Download failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
