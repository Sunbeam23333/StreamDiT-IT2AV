"""Agent video pipeline toolkit.

A multi-agent framework for video generation with multiple backend support.

Example:
    from agent_video_pipeline import run_pipeline, PipelineConfig
    from agent_video_pipeline.configs import ModelSelector, VideoBackend
    
    # Simple usage
    result = run_pipeline(
        user_prompt="一个主播在讲解最新的手机产品",
        duration_s=5.0,
        video_backend="wan22_i2v",  # or "hyvideo", "magihuman"
    )
    
    # With explicit backend selection
    result = run_pipeline(
        user_prompt="A music video with ambient atmosphere",
        duration_s=8.0,
        preferred_route="magihuman_av",
    )
"""

__version__ = "0.3.0"

# Core components (Mora-inspired)
from agent_video_pipeline.core import (
    Action,
    Memory,
    Message,
    MessageQueue,
    Role,
    RoleReactMode,
    Workflow,
    WorkflowState,
)
from agent_video_pipeline.configs import (
    LLMConfig,
    ModelSelector,
    PipelineConfig,
    VideoBackend,
    ImageBackend,
    AudioBackend,
    create_selector_for_env,
)
from agent_video_pipeline.adapters import (
    HunyuanImageStage,
    BananaImageStage,
    HyVideoStage,
    WanLightX2VStage,
    MagiHumanStage,
    LatentSyncStage,
    HunyuanTTSStage,
    BGMStage,
    SFXStage,
)
from agent_video_pipeline.editing import (
    concat_videos,
    merge_audio,
    extract_last_frame,
    add_subtitles,
    apply_crossfade,
    sync_video_audio,
    align_for_lipsync,
    adjust_video_speed,
)
from agent_video_pipeline.orchestrator import run_pipeline
from agent_video_pipeline.schemas import ProjectPlan, ShotSpec, StageCommand

__all__ = [
    # Core framework
    "Action",
    "Memory",
    "Message",
    "MessageQueue",
    "Role",
    "RoleReactMode",
    "Workflow",
    "WorkflowState",
    # Configuration
    "LLMConfig",
    "PipelineConfig",
    "ModelSelector",
    "VideoBackend",
    "ImageBackend",
    "AudioBackend",
    "create_selector_for_env",
    # Adapters
    "HunyuanImageStage",
    "BananaImageStage",
    "HyVideoStage",
    "WanLightX2VStage",
    "MagiHumanStage",
    "LatentSyncStage",
    "HunyuanTTSStage",
    "BGMStage",
    "SFXStage",
    # Editing
    "concat_videos",
    "merge_audio",
    "extract_last_frame",
    "add_subtitles",
    "apply_crossfade",
    "sync_video_audio",
    "align_for_lipsync",
    "adjust_video_speed",
    # High-level API
    "run_pipeline",
    "run_cpu_pipeline",
    "available_scenarios",
    "detect_scenario",
    "ProjectPlan",
    "StoryboardBeat",
    "ShotSpec",
    "StageCommand",
    "AgentRoleSpec",
    "ChainStep",
    "ReviewSpec",
]
