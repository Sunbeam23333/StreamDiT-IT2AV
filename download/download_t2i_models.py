#!/usr/bin/env python3
"""
Download Text-to-Image models from HuggingFace for LightX2V first/last frame generation.

支持的模型:
  - FLUX.1 [schnell]  : 12B参数, Apache 2.0, 4步出图, 开源最强T2I
  - FLUX.1 [dev]      : 12B参数, 非商用, 28步出图, 质量更高
  - SD3.5 Large       : 8B参数, 开源, MMDiT架构
  - SD3.5 Medium      : 2.5B参数, 轻量高效, 下载量最高
  - PixArt-Sigma      : 0.6B参数, 支持4K, 训练高效 (备选轻量方案)

目标分辨率: 720x1280 / 1280x720 (服务于 LightX2V Wan2.2 MoE i2v/flf2v)

用法:
    python download_t2i_models.py --models all
    python download_t2i_models.py --models flux_schnell flux_dev
    python download_t2i_models.py --models sd35 --target-root /custom/path
"""
from __future__ import annotations

import argparse
import logging
import os
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
# HuggingFace Token - 用于访问 gated repo (FLUX.1, SD3.5 等)
# ============================================================================
HF_TOKEN = os.environ.get('HF_TOKEN', '')

# ============================================================================
# 模型配置
# ============================================================================
T2I_MODEL_REPOS = {
    "flux_schnell": {
        "repo_id": "black-forest-labs/FLUX.1-schnell",
        "local_dir": "FLUX.1-schnell",
        "description": "FLUX.1 schnell - 12B params, Apache 2.0, 4-step, fastest quality",
    },
    "flux_dev": {
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "local_dir": "FLUX.1-dev",
        "description": "FLUX.1 dev - 12B params, non-commercial, 28-step, highest quality",
    },
    "sd35_large": {
        "repo_id": "stabilityai/stable-diffusion-3.5-large",
        "local_dir": "stable-diffusion-3.5-large",
        "description": "SD3.5 Large - 8B params, MMDiT + Rectified Flow",
    },
    "sd35_medium": {
        "repo_id": "stabilityai/stable-diffusion-3.5-medium",
        "local_dir": "stable-diffusion-3.5-medium",
        "description": "SD3.5 Medium - 2.5B params, lightweight & popular, MMDiT",
    },
    "pixart_sigma": {
        "repo_id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "local_dir": "PixArt-Sigma-XL-2-1024-MS",
        "description": "PixArt-Sigma - 0.6B params, DiT, training-efficient (lightweight option)",
    },
}

DEFAULT_TARGET = os.path.join(os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace")), "models/Image")


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
    """登录 HuggingFace 以访问 gated repo (如 FLUX.1, SD3.5)."""
    token = token or HF_TOKEN
    if not token or token == 'YOUR_HF_TOKEN':
        logging.warning("HuggingFace token not configured! Gated repos will fail.")
        return

    try:
        login(token=token, add_to_git_credential=False)
        logging.info("HuggingFace login successful (token: %s...%s)", token[:5], token[-4:])
    except Exception as e:
        logging.warning("HuggingFace login failed: %s", e)
        # 即使 login 失败，也尝试通过环境变量传递 token
        os.environ['HF_TOKEN'] = token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = token
        logging.info("Set HF_TOKEN in environment as fallback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download T2I models from HuggingFace")
    parser.add_argument("--target-root", default=DEFAULT_TARGET, help="Destination directory")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--revision", default=None, help="Specific revision to pin")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (overrides built-in)")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(T2I_MODEL_REPOS.keys()) + ["all"],
        default=["all"],
        help="Which models to download (default: all)",
    )
    return parser.parse_args()


def get_selected_repos(models_arg: list[str]) -> list[dict]:
    """Get repos based on user selection."""
    if "all" in models_arg:
        return list(T2I_MODEL_REPOS.values())

    selected = []
    for m in models_arg:
        if m in T2I_MODEL_REPOS:
            selected.append(T2I_MODEL_REPOS[m])
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("Text-to-Image Model Downloader")
    logging.info("=" * 60)
    logging.info("")

    logging.info("Step 1: Configuring Network Proxy")
    setup_proxy()
    print()

    logging.info("Step 2: Logging in to HuggingFace (for gated repos)")
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
        logging.info("\n" + "=" * 60)
        logging.info("All T2I models downloaded successfully!")
        logging.info("  Target: %s", args.target_root)
        logging.info("=" * 60)
    except KeyboardInterrupt:
        logging.warning("\nDownload interrupted. Re-run to resume (auto-resume supported).")
        sys.exit(1)
    except Exception as e:
        logging.error("\nDownload failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
