"""Video editing toolkit built on FFmpeg."""
from agent_video_pipeline.editing.ffmpeg_editor import (
    add_subtitles,
    apply_crossfade,
    concat_videos,
    extract_last_frame,
    merge_audio,
)
from agent_video_pipeline.editing.sync_editor import (
    adjust_video_speed,
    align_for_lipsync,
    get_media_duration,
    stretch_audio,
    sync_video_audio,
)

__all__ = [
    # FFmpeg primitives
    "add_subtitles",
    "apply_crossfade",
    "concat_videos",
    "extract_last_frame",
    "merge_audio",
    # Sync utilities
    "adjust_video_speed",
    "align_for_lipsync",
    "get_media_duration",
    "stretch_audio",
    "sync_video_audio",
]
