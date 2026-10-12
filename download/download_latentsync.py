#!/usr/bin/env python3
"""
Download LatentSync-1.6 模型 from HuggingFace to group 盘.

模型文件:
  - latentsync_unet.pt    (5.07 GB)  主 UNet 模型
  - whisper/tiny.pt        (75.6 MB)  Whisper 音频编码器
  - auxiliary/             (~2.88 GB) 人脸检测/VGG16/SyncNet 等辅助模型
  - stable_syncnet.pt      (1.61 GB)  SyncNet (训练用, 推理可选)

总计约 ~9.6 GB

用法:
  python download_latentsync.py                       # 下载全部
  python download_latentsync.py --inference-only       # 只下载推理必需文件
  python download_latentsync.py --link-checkpoints     # 下载后自动链接到代码目录
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

# ============================================================================
# 代理配置
# ============================================================================
USERNAME = os.environ.get('IT2AV_PROXY_USER', '')
PASSWORD = os.environ.get('IT2AV_PROXY_PASS', '')
PROXY_HOST = os.environ.get('IT2AV_PROXY_HOST', '')
PROXY_PORT = os.environ.get('IT2AV_PROXY_PORT', '')
NO_PROXY_DOMAINS = os.environ.get('NO_PROXY', 'localhost,127.0.0.1')

# ============================================================================
# 路径配置
# ============================================================================
REPO_ID = "ByteDance/LatentSync-1.6"
DEFAULT_TARGET = os.path.join(os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace")), "models/LatentSync-1.6")
LATENTSYNC_CODE_DIR = os.path.join(
    os.environ.get("IT2AV_CODE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "utils", "video", "LatentSync",
)

# 推理必需文件 (不含 stable_syncnet.pt 和部分 auxiliary)
INFERENCE_FILES = [
    "latentsync_unet.pt",
    "whisper/tiny.pt",
    "auxiliary/sfd_face.pth",       # 人脸检测
    "auxiliary/vgg16-397923af.pth",  # 感知损失 (pipeline 中会加载)
]

# 推理可选但推荐的文件
OPTIONAL_FILES = [
    "auxiliary/syncnet_v2.model",
    "auxiliary/koniq_pretrained.pkl",
    "auxiliary/i3d_torchscript.pt",
    "auxiliary/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
    "stable_syncnet.pt",
]


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


def download_full(target_dir: str) -> None:
    """下载全部文件 (snapshot_download)."""
    logging.info("📦 下载 LatentSync-1.6 全部文件 → %s", target_dir)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=target_dir,
        ignore_patterns=["*.md", ".git*"],
    )
    logging.info("✅ 全部文件下载完成: %s", target_dir)


def download_inference_only(target_dir: str) -> None:
    """只下载推理必需的文件."""
    logging.info("📦 下载 LatentSync-1.6 推理文件 → %s", target_dir)
    for filename in INFERENCE_FILES:
        logging.info("  下载: %s", filename)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=target_dir,
        )
    logging.info("✅ 推理文件下载完成: %s", target_dir)


def link_checkpoints(model_dir: str, code_dir: str) -> None:
    """在 LatentSync 代码目录下创建 checkpoints 软链接."""
    ckpt_dir = Path(code_dir) / "checkpoints"
    model_path = Path(model_dir)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    links = {
        "latentsync_unet.pt": model_path / "latentsync_unet.pt",
        "whisper": model_path / "whisper",
        "auxiliary": model_path / "auxiliary",
    }

    for name, src in links.items():
        dst = ckpt_dir / name
        if not src.exists():
            logging.warning("  ⚠️  源不存在, 跳过: %s", src)
            continue
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            logging.info("  ⏭️  已存在 (非软链接): %s", dst)
            continue

        dst.symlink_to(src)
        logging.info("  🔗 %s → %s", dst, src)

    # stable_syncnet.pt (可选)
    syncnet_src = model_path / "stable_syncnet.pt"
    syncnet_dst = ckpt_dir / "stable_syncnet.pt"
    if syncnet_src.exists() and not syncnet_dst.exists():
        syncnet_dst.symlink_to(syncnet_src)
        logging.info("  🔗 %s → %s", syncnet_dst, syncnet_src)

    logging.info("✅ Checkpoints 链接完成: %s", ckpt_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LatentSync-1.6 from HuggingFace")
    parser.add_argument("--target-dir", default=DEFAULT_TARGET, help="下载目标目录")
    parser.add_argument("--inference-only", action="store_true", help="只下载推理必需文件 (~5.7GB)")
    parser.add_argument("--link-checkpoints", action="store_true", help="下载后自动链接到代码目录")
    parser.add_argument("--code-dir", default=LATENTSYNC_CODE_DIR, help="LatentSync 代码目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("LatentSync-1.6 模型下载")
    logging.info("  Repo:   %s", REPO_ID)
    logging.info("  Target: %s", args.target_dir)
    logging.info("  Mode:   %s", "inference-only" if args.inference_only else "full")
    logging.info("=" * 60)

    setup_proxy()

    try:
        if args.inference_only:
            download_inference_only(args.target_dir)
        else:
            download_full(args.target_dir)

        if args.link_checkpoints:
            link_checkpoints(args.target_dir, args.code_dir)

        logging.info("\n" + "=" * 60)
        logging.info("✅ LatentSync-1.6 下载完成!")
        logging.info("=" * 60)

    except KeyboardInterrupt:
        logging.warning("\n下载中断")
        sys.exit(1)
    except Exception as e:
        logging.error("\n下载失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
