#!/usr/bin/env python3
"""
VACE FLF2V 首尾帧预处理工具

将两张图片（首帧 + 尾帧）构造为 VACE 所需的 src_video.mp4 和 src_mask.mp4。

VACE FLF2V 原理:
  - src_video: 首帧 + 空白帧(全黑) + 尾帧 => 81帧视频
  - src_mask:  首帧mask=0(保留) + 中间mask=255(生成) + 尾帧mask=0(保留)
  - VACE 模型根据首尾帧的视觉信息 + prompt 生成中间过渡视频

注意: 输出使用 ffmpeg H.264 编码以确保 decord 能正确读取。

用法:
  python prepare_vace_flf2v.py \
      --first_frame /path/to/first.png \
      --last_frame /path/to/last.png \
      --output_dir /path/to/output \
      --height 720 --width 1280 --num_frames 81 --fps 16
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np


def create_flf2v_videos(
    first_frame_path: str,
    last_frame_path: str,
    output_dir: str,
    height: int = 720,
    width: int = 1280,
    num_frames: int = 81,
    fps: int = 16,
) -> tuple[str, str]:
    """构造 VACE FLF2V 所需的 src_video 和 src_mask 视频文件。

    使用 ffmpeg H.264 编码输出，确保 decord 能正确读取。

    Args:
        first_frame_path: 首帧图片路径
        last_frame_path: 尾帧图片路径
        output_dir: 输出目录
        height: 视频高度
        width: 视频宽度
        num_frames: 总帧数 (应满足 (num_frames-1) % 4 == 0)
        fps: 帧率

    Returns:
        (src_video_path, src_mask_path) 元组
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取并 resize 首尾帧
    first_frame = cv2.imread(first_frame_path)
    last_frame = cv2.imread(last_frame_path)

    if first_frame is None:
        raise FileNotFoundError(f"Cannot read first frame: {first_frame_path}")
    if last_frame is None:
        raise FileNotFoundError(f"Cannot read last frame: {last_frame_path}")

    first_frame = resize_center_crop(first_frame, height, width)
    last_frame = resize_center_crop(last_frame, height, width)

    src_video_path = os.path.join(output_dir, "src_video.mp4")
    src_mask_path = os.path.join(output_dir, "src_mask.mp4")

    # --- 构造 src_video: 首帧 + (num_frames-2)黑帧 + 尾帧 ---
    _write_video_ffmpeg(
        output_path=src_video_path,
        frames=_gen_src_video_frames(first_frame, last_frame, height, width, num_frames),
        height=height,
        width=width,
        fps=fps,
    )

    # --- 构造 src_mask: 首帧=0(保留) + 中间=255(生成) + 尾帧=0(保留) ---
    _write_video_ffmpeg(
        output_path=src_mask_path,
        frames=_gen_src_mask_frames(height, width, num_frames),
        height=height,
        width=width,
        fps=fps,
    )

    print(f"[OK] src_video: {src_video_path} ({num_frames} frames, {width}x{height}, {fps}fps)")
    print(f"[OK] src_mask:  {src_mask_path} ({num_frames} frames)")

    return src_video_path, src_mask_path


def _gen_src_video_frames(first_frame, last_frame, height, width, num_frames):
    """生成 src_video 的帧序列 (BGR)."""
    yield first_frame
    black = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(num_frames - 2):
        yield black
    yield last_frame


def _gen_src_mask_frames(height, width, num_frames):
    """生成 src_mask 的帧序列 (BGR, 0=保留 255=生成)."""
    keep = np.zeros((height, width, 3), dtype=np.uint8)
    gen = np.full((height, width, 3), 255, dtype=np.uint8)
    yield keep
    for _ in range(num_frames - 2):
        yield gen
    yield keep


def _write_video_ffmpeg(output_path: str, frames, height: int, width: int, fps: int):
    """用 ffmpeg 将帧序列写成 H.264 编码的 mp4 文件。

    使用 rawvideo pipe 输入 + libx264 编码，确保最大兼容性。
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "18",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}):\n{stderr.decode()}")


def resize_center_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize + center crop 到目标尺寸，保持内容不变形。"""
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # center crop
    y_start = (new_h - target_h) // 2
    x_start = (new_w - target_w) // 2
    img = img[y_start : y_start + target_h, x_start : x_start + target_w]
    return img


def main():
    parser = argparse.ArgumentParser(description="VACE FLF2V 首尾帧预处理")
    parser.add_argument("--first_frame", type=str, required=True, help="首帧图片路径")
    parser.add_argument("--last_frame", type=str, required=True, help="尾帧图片路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--height", type=int, default=720, help="视频高度 (default: 720)")
    parser.add_argument("--width", type=int, default=1280, help="视频宽度 (default: 1280)")
    parser.add_argument("--num_frames", type=int, default=81, help="总帧数 (default: 81)")
    parser.add_argument("--fps", type=int, default=16, help="帧率 (default: 16)")
    args = parser.parse_args()

    create_flf2v_videos(
        first_frame_path=args.first_frame,
        last_frame_path=args.last_frame,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
