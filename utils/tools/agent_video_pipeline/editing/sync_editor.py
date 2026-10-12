"""Audio-Video synchronization utilities.

Provides tools for aligning video and audio durations through:
- Video speed adjustment (time stretching)
- Audio stretching/compression
- Duration matching with intelligent strategy selection
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Tuple


def get_media_duration(path: str) -> float:
    """Get the duration of a video or audio file in seconds.

    Args:
        path: Path to media file.

    Returns:
        Duration in seconds.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def adjust_video_speed(
    video: str,
    speed_factor: float,
    output: str,
    audio_pitch_correction: bool = True,
) -> str:
    """Adjust video playback speed.

    Args:
        video: Input video path.
        speed_factor: Speed multiplier (>1 faster, <1 slower).
        output: Output video path.
        audio_pitch_correction: If True, preserve audio pitch when speeding up/down.

    Returns:
        Output video path.
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate setpts factor (inverse of speed)
    setpts_factor = 1.0 / speed_factor
    atempo = speed_factor

    # Build filter complex
    video_filter = f"setpts={setpts_factor}*PTS"

    if audio_pitch_correction and atempo > 0.5 and atempo < 2.0:
        # atempo only works in range [0.5, 2.0], chain multiple if needed
        atempo_filters = []
        remaining = atempo
        while remaining > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            atempo_filters.append("atempo=0.5")
            remaining /= 0.5
        atempo_filters.append(f"atempo={remaining:.4f}")
        audio_filter = ",".join(atempo_filters)
        filter_complex = f"[0:v]{video_filter}[v];[0:a]{audio_filter}[a]"
        cmd = [
            "ffmpeg",
            "-i", str(video),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-y", str(output_path),
        ]
    else:
        # Video only, no audio pitch correction
        cmd = [
            "ffmpeg",
            "-i", str(video),
            "-vf", video_filter,
            "-af", f"atempo={min(max(atempo, 0.5), 2.0)}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-y", str(output_path),
        ]

    subprocess.run(cmd, check=True, capture_output=True)
    return str(output_path)


def stretch_audio(
    audio: str,
    target_duration: float,
    output: str,
    method: Literal["atempo", "rubberband"] = "atempo",
) -> str:
    """Stretch or compress audio to match target duration.

    Args:
        audio: Input audio path.
        target_duration: Desired duration in seconds.
        output: Output audio path.
        method: Stretching method (atempo for simple, rubberband for quality).

    Returns:
        Output audio path.
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    current_duration = get_media_duration(audio)
    ratio = target_duration / current_duration

    if abs(ratio - 1.0) < 0.01:
        # No significant change needed
        import shutil
        shutil.copy(audio, output)
        return output

    if method == "atempo":
        # Chain atempo filters for ratios outside [0.5, 2.0]
        filters = []
        remaining = ratio
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.4f}")
        filter_str = ",".join(filters)

        cmd = [
            "ffmpeg",
            "-i", str(audio),
            "-af", filter_str,
            "-c:a", "aac", "-b:a", "192k",
            "-y", str(output_path),
        ]
    else:
        # rubberband - higher quality but requires external library
        cmd = [
            "ffmpeg",
            "-i", str(audio),
            "-af", f"rubberband=tempo={ratio:.4f}",
            "-c:a", "aac", "-b:a", "192k",
            "-y", str(output_path),
        ]

    subprocess.run(cmd, check=True, capture_output=True)
    return str(output_path)


def sync_video_audio(
    video: str,
    audio: str,
    output: str,
    strategy: Literal["stretch_video", "stretch_audio", "auto"] = "auto",
    max_speed_change: float = 0.2,
) -> Tuple[str, dict]:
    """Synchronize video and audio by matching their durations.

    Args:
        video: Input video path.
        audio: Input audio path.
        output: Output synchronized video path.
        strategy: Synchronization strategy:
            - "stretch_video": Adjust video speed to match audio
            - "stretch_audio": Adjust audio speed to match video
            - "auto": Choose based on content type (default)
        max_speed_change: Maximum allowed speed change ratio (0.2 = 20%).

    Returns:
        Tuple of (output path, sync info dict).
    """
    video_duration = get_media_duration(video)
    audio_duration = get_media_duration(audio)

    info = {
        "original_video_duration": video_duration,
        "original_audio_duration": audio_duration,
        "strategy": strategy,
        "speed_factor": 1.0,
    }

    if abs(video_duration - audio_duration) < 0.1:
        # Already in sync
        import shutil
        shutil.copy(video, output)
        info["action"] = "no_change"
        return output, info

    # Determine strategy
    if strategy == "auto":
        # For lip-sync videos, prefer stretching audio
        # For action videos, prefer stretching video
        ratio = audio_duration / video_duration
        if 1 - max_speed_change < ratio < 1 + max_speed_change:
            strategy = "stretch_video"
        else:
            strategy = "stretch_audio"

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if strategy == "stretch_video":
        speed_factor = audio_duration / video_duration
        info["speed_factor"] = speed_factor
        info["action"] = "video_speed_adjusted"

        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tmp:
            tmp_audio = tmp.name

        # Extract audio
        subprocess.run([
            "ffmpeg", "-i", audio,
            "-c:a", "aac", "-b:a", "192k",
            "-y", tmp_audio,
        ], check=True, capture_output=True)

        # Adjust video speed
        temp_video = output_path.parent / f"{output_path.stem}_temp{output_path.suffix}"
        adjust_video_speed(video, speed_factor, str(temp_video), audio_pitch_correction=False)

        # Merge adjusted video with original audio
        subprocess.run([
            "ffmpeg",
            "-i", str(temp_video),
            "-i", tmp_audio,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-y", str(output_path),
        ], check=True, capture_output=True)

        # Cleanup
        Path(tmp_audio).unlink(missing_ok=True)
        temp_video.unlink(missing_ok=True)

    else:  # stretch_audio
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tmp:
            tmp_audio = tmp.name

        # Stretch audio to match video
        stretch_audio(audio, video_duration, tmp_audio, method="atempo")

        # Merge video with stretched audio
        subprocess.run([
            "ffmpeg",
            "-i", video,
            "-i", tmp_audio,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-y", str(output_path),
        ], check=True, capture_output=True)

        Path(tmp_audio).unlink(missing_ok=True)
        info["action"] = "audio_stretched"

    return str(output_path), info


def align_for_lipsync(
    video: str,
    audio: str,
    output: str,
    tolerance: float = 0.05,
) -> Tuple[str, dict]:
    """Specialized alignment for lip-sync videos.

    Prioritizes preserving video frame rate and motion smoothness,
    stretching audio when possible.

    Args:
        video: Input video path.
        audio: Input audio (TTS output).
        output: Output aligned video path.
        tolerance: Acceptable duration difference ratio.

    Returns:
        Tuple of (output path, alignment info).
    """
    video_duration = get_media_duration(video)
    audio_duration = get_media_duration(audio)

    ratio = abs(audio_duration - video_duration) / video_duration

    if ratio <= tolerance:
        # Within tolerance, just concatenate
        import shutil
        from .ffmpeg_editor import merge_audio

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        merge_audio(video, [(audio, 1.0)], output)
        return output, {
            "strategy": "direct_merge",
            "video_duration": video_duration,
            "audio_duration": audio_duration,
        }

    # Otherwise, stretch audio to match video
    return sync_video_audio(
        video, audio, output,
        strategy="stretch_audio",
        max_speed_change=0.3,
    )
