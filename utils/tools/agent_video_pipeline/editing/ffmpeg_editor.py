"""
FFmpeg-based video editing primitives.

Provides CLI and Python API for:
  - concat        : concatenate multiple video clips
  - merge-audio  : overlay multiple audio tracks onto a video
  - extract-last-frame : pull the final frame from a video as a PNG
  - add-subtitles: burn SRT subtitles into a video
  - apply-transitions: crossfade between adjacent clips
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


def _run(cmd: List[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _probe_video_size(video_path: str) -> tuple[int, int]:
    """Probe video width/height with ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = _run(cmd)
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"无法读取视频分辨率: {video_path}")
    width = int(streams[0].get("width") or 0)
    height = int(streams[0].get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"视频分辨率无效: {video_path}")
    return width, height


def concat_videos(video_paths: List[str], output: str, keep_orig: bool = False) -> str:
    """Concatenate multiple videos robustly.

    使用 filter concat + 统一分辨率/FPS 重新编码，避免因 timebase、fps、分辨率不一致导致的丢段。
    """
    if not video_paths:
        raise ValueError("video_paths 不能为空")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_paths: List[str] = []
    for vp in video_paths:
        src = Path(vp)
        if keep_orig:
            fixed = src.parent / f"_orig_{src.stem}{src.suffix}"
            shutil.copy(src, fixed)
            normalized_paths.append(str(fixed))
        else:
            normalized_paths.append(str(src))

    target_w, target_h = _probe_video_size(normalized_paths[0])

    input_args: List[str] = []
    filter_parts: List[str] = []
    concat_inputs: List[str] = []
    for i, vp in enumerate(normalized_paths):
        input_args += ["-i", vp]
        filter_parts.append(
            (
                f"[{i}:v]"
                f"fps=24,"
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,"
                "setsar=1,format=yuv420p,setpts=PTS-STARTPTS"
                f"[v{i}]"
            )
        )
        concat_inputs.append(f"[v{i}]")

    filter_complex = ";".join(filter_parts + [f"{''.join(concat_inputs)}concat=n={len(normalized_paths)}:v=1:a=0[vout]"])
    cmd = [
        "ffmpeg",
        *input_args,
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-y",
        str(out_path),
    ]
    _run(cmd)
    return str(out_path)


def merge_audio(
    video: str,
    audio_specs: List[Tuple[str, float]],
    output: str,
    keep_orig: bool = False,
) -> str:
    """Overlay multiple audio tracks onto a video with per-track volume.

    Args:
        video: Input video path (video stream is copied, audio replaced).
        audio_specs: List of (audio_path, volume) tuples, e.g. [("tts.wav", 1.0), ("bgm.wav", 0.2)].
        output: Output video path.
        keep_orig: If True, keep original video as well.

    Returns:
        The output path.
    """
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not audio_specs:
        shutil.copy(video, str(out_path))
        return str(out_path)

    input_args: List[str] = ["-i", str(video)]
    filter_parts: List[str] = []
    for i, (audio_path, vol) in enumerate(audio_specs):
        input_args += ["-i", str(audio_path)]
        filter_parts.append(f"[{i + 1}]volume={vol}[a{i}]")

    mix_inputs = "".join(f"[a{i}]" for i in range(len(audio_specs)))
    filter_complex = (
        f"{';'.join(filter_parts)};"
        f"{mix_inputs}amix=inputs={len(audio_specs)}:duration=longest:dropout_transition=2[aout]"
    )

    cmd = [
        "ffmpeg", "-i", str(video),
        *input_args[2:],  # strip leading "-i video" already added
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-y", str(out_path),
    ]
    _run(cmd)
    return str(out_path)


def extract_frame(video: str, output: str, time_s: float = 0.0) -> str:
    """Extract a frame near a target timestamp as a PNG."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seek_s = max(float(time_s), 0.0)
    cmd = [
        "ffmpeg",
        "-ss", f"{seek_s:.3f}",
        "-i", str(video),
        "-frames:v", "1",
        "-q:v", "2",
        "-y", str(out_path),
    ]
    _run(cmd)
    return str(out_path)


def extract_last_frame(video: str, output: str) -> str:
    """Extract the final frame of a video as a PNG.

    Args:
        video: Input video path.
        output: Output PNG path.

    Returns:
        The output path.
    """
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-sseof", "-0.1",
        "-i", str(video),
        "-frames:v", "1",
        "-q:v", "2",
        "-y", str(out_path),
    ]
    _run(cmd, check=False)
    if not out_path.exists():
        extract_frame(video=video, output=output, time_s=0.0)
    return str(out_path)


def trim_video(video: str, output: str, duration_s: float) -> str:
    """Trim a video to the target duration, re-encoding for reliable short clips."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", str(video),
        "-t", f"{max(duration_s, 0.2):.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-an",
        "-y", str(out_path),
    ]
    _run(cmd)
    return str(out_path)


def add_subtitles(video: str, srt_path: str, output: str, style: str | None = None) -> str:
    """Burn SRT subtitles into a video.

    Args:
        video: Input video path.
        srt_path: Path to .srt subtitle file.
        output: Output video path.
        style: Optional FFmpeg subtitles filter style string.

    Returns:
        The output path.
    """
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    default_style = (
        "FontName=Arial,FontSize=24,"
        "PrimaryColour=&HFFFFFF,"
        "OutlineColour=&H000000,"
        "Outline=2,Shadow=1"
    )
    style_str = style or default_style
    subtitle_filter = f"subtitles='{Path(srt_path).as_posix()}':force_style='{style_str}'"

    cmd = [
        "ffmpeg", "-i", str(video),
        "-vf", subtitle_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        "-y", str(out_path),
    ]
    try:
        _run(cmd)
    except subprocess.CalledProcessError:
        # Fallback without custom style
        cmd2 = [
            "ffmpeg", "-i", str(video),
            "-vf", f"subtitles='{Path(srt_path).as_posix()}'",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            "-y", str(out_path),
        ]
        _run(cmd2)
    return str(out_path)


def make_motion_video_from_image(
    image: str,
    output: str,
    duration_s: float,
    size: str = "768x1280",
    fps: int = 12,
    motion: str = "zoom_in",
) -> str:
    """Create a short motion video from a still image using FFmpeg zoompan."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = (int(part) for part in size.lower().split("x", 1))
    frames = max(int(duration_s * fps), fps)
    if motion == "zoom_out":
        zoom_expr = "if(lte(on,1),1.06,max(1.0,zoom-0.0008))"
    else:
        zoom_expr = "min(zoom+0.0008,1.06)"
    vf = (
        f"scale={width}:{height},"
        f"zoompan=z='{zoom_expr}':d={frames}:s={width}x{height}:fps={fps},"
        "format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-loop", "1",
        "-i", str(image),
        "-vf", vf,
        "-t", str(duration_s),
        "-pix_fmt", "yuv420p",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-y", str(out_path),
    ]
    _run(cmd)
    return str(out_path)


def make_story_motion_video_from_frames(
    first_image: str,
    output: str,
    duration_s: float,
    size: str = "768x1280",
    fps: int = 15,
    last_image: str | None = None,
    middle_image: str | None = None,
) -> str:
    """Create a more narrative fallback clip by transitioning between start/middle/end keyframes."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = (int(part) for part in size.lower().split("x", 1))

    first_path = Path(first_image)
    last_path = Path(last_image) if last_image else None
    middle_path = Path(middle_image) if middle_image else None

    if not last_path or last_path.resolve() == first_path.resolve():
        return make_motion_video_from_image(
            image=first_image,
            output=output,
            duration_s=duration_s,
            size=size,
            fps=fps,
            motion="zoom_in",
        )

    fade_duration = min(max(duration_s * 0.18, 0.35), max(duration_s - 0.3, 0.35))
    first_zoom = "min(zoom+0.0006,1.04)"
    mid_zoom = "if(lte(on,1),1.02,min(zoom+0.0003,1.03))"
    last_zoom = "if(lte(on,1),1.04,max(1.0,zoom-0.0006))"

    if middle_path and middle_path.exists() and middle_path.resolve() not in {first_path.resolve(), last_path.resolve()}:
        first_offset = max(duration_s * 0.28, 0.1)
        second_offset = max(duration_s * 0.62, first_offset + 0.1)
        filter_complex = (
            f"[0:v]scale={width}:{height},zoompan=z='{first_zoom}':d=1:s={width}x{height}:fps={fps},trim=duration={duration_s},setpts=PTS-STARTPTS[v0];"
            f"[1:v]scale={width}:{height},zoompan=z='{mid_zoom}':d=1:s={width}x{height}:fps={fps},trim=duration={duration_s},setpts=PTS-STARTPTS[v1];"
            f"[2:v]scale={width}:{height},zoompan=z='{last_zoom}':d=1:s={width}x{height}:fps={fps},trim=duration={duration_s},setpts=PTS-STARTPTS[v2];"
            f"[v0][v1]xfade=transition=fade:duration={fade_duration:.3f}:offset={first_offset:.3f}[vm];"
            f"[vm][v2]xfade=transition=fade:duration={fade_duration:.3f}:offset={second_offset:.3f},format=yuv420p[vout]"
        )
        cmd = [
            "ffmpeg",
            "-loop", "1",
            "-t", str(duration_s),
            "-i", str(first_image),
            "-loop", "1",
            "-t", str(duration_s),
            "-i", str(middle_path),
            "-loop", "1",
            "-t", str(duration_s),
            "-i", str(last_path),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-pix_fmt", "yuv420p",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-t", str(duration_s),
            "-y", str(out_path),
        ]
        _run(cmd)
        return str(out_path)

    offset = max(duration_s - fade_duration, 0.1)
    filter_complex = (
        f"[0:v]scale={width}:{height},zoompan=z='{first_zoom}':d=1:s={width}x{height}:fps={fps},trim=duration={duration_s},setpts=PTS-STARTPTS[v0];"
        f"[1:v]scale={width}:{height},zoompan=z='{last_zoom}':d=1:s={width}x{height}:fps={fps},trim=duration={duration_s},setpts=PTS-STARTPTS[v1];"
        f"[v0][v1]xfade=transition=fade:duration={fade_duration:.3f}:offset={offset:.3f},format=yuv420p[vout]"
    )
    cmd = [
        "ffmpeg",
        "-loop", "1",
        "-t", str(duration_s),
        "-i", str(first_image),
        "-loop", "1",
        "-t", str(duration_s),
        "-i", str(last_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-pix_fmt", "yuv420p",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-t", str(duration_s),
        "-y", str(out_path),
    ]
    _run(cmd)
    return str(out_path)


def apply_crossfade(file_list: List[str], transition_s: float, output: str) -> str:
    """Concatenate adjacent video clips with a crossfade transition.

    Args:
        file_list: Ordered list of video paths.
        transition_s: Duration of the crossfade in seconds.
        output: Output video path.

    Returns:
        The output path.
    """
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(file_list) < 2:
        if file_list:
            shutil.copy(file_list[0], str(out_path))
        return str(out_path)

    filter_complex_parts = []
    input_args: List[str] = []
    for i, fp in enumerate(file_list):
        input_args += ["-i", str(fp)]
        filter_complex_parts.append(f"[{i}:v]")

    n = len(file_list)
    fade_len = transition_s

    # Build crossfade chain: [0:v][1:v] crossfade [1:v][2:v] ...
    cur = "[0:v]"
    clips = [f"[{i}:v]" for i in range(n)]
    chain: List[str] = []
    for i in range(1, n):
        chain.append(f"{clips[i - 1]}{clips[i]}crossfade=duration={fade_len}:d={i - 1}[v{i}]")
        clips[i] = f"[v{i}]"

    fc = ";".join(chain + [f"[v{n - 1}]null[outv]"])
    cmd = [
        "ffmpeg",
        *input_args,
        "-filter_complex", fc,
        "-map", "[outv]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        "-y", str(out_path),
    ]
    _run(cmd)
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(prog="ffmpeg_editor", description="FFmpeg video editing primitives")
    sub = parser.add_subparsers(dest="command", required=True)

    p_concat = sub.add_parser("concat", help="Concatenate multiple videos")
    p_concat.add_argument("--video", action="append", required=True, help="Video file (repeatable)")
    p_concat.add_argument("--output", required=True, help="Output path")
    p_concat.add_argument("--keep-orig", action="store_true", help="Keep intermediate files")

    p_merge = sub.add_parser("merge-audio", help="Overlay audio tracks onto video")
    p_merge.add_argument("--video", required=True, help="Base video")
    p_merge.add_argument("--audio", action="append", required=True, help="audio_path:volume (repeatable)")
    p_merge.add_argument("--output", required=True, help="Output video")
    p_merge.add_argument("--keep-orig", action="store_true")

    p_extract = sub.add_parser("extract-last-frame", help="Extract final frame as PNG")
    p_extract.add_argument("--video", required=True)
    p_extract.add_argument("--output", required=True)

    p_sub = sub.add_parser("add-subtitles", help="Burn subtitles into video")
    p_sub.add_argument("--video", required=True)
    p_sub.add_argument("--srt", required=True)
    p_sub.add_argument("--output", required=True)

    p_xfade = sub.add_parser("apply-transitions", help="Crossfade between clips")
    p_xfade.add_argument("--video", action="append", required=True)
    p_xfade.add_argument("--transition-s", type=float, default=0.5)
    p_xfade.add_argument("--output", required=True)

    args = parser.parse_args()

    def _parse_audio(spec: str) -> Tuple[str, float]:
        parts = spec.rsplit(":", 1)
        return parts[0], float(parts[1]) if len(parts) == 2 else 1.0

    if args.command == "concat":
        concat_videos(args.video, args.output, keep_orig=args.keep_orig)
    elif args.command == "merge-audio":
        specs = [_parse_audio(s) for s in args.audio]
        merge_audio(args.video, specs, args.output, keep_orig=args.keep_orig)
    elif args.command == "extract-last-frame":
        extract_last_frame(args.video, args.output)
    elif args.command == "add-subtitles":
        add_subtitles(args.video, args.srt, args.output)
    elif args.command == "apply-transitions":
        apply_crossfade(args.video, args.transition_s, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
