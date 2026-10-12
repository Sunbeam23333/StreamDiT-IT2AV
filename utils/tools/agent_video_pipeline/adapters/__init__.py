"""Adapters for existing generation tools."""
from agent_video_pipeline.adapters.audio_stage import BananaImageStage, BGMStage, SFXStage
from agent_video_pipeline.adapters.image_stage import HunyuanImageStage
from agent_video_pipeline.adapters.lipsync_stage import LatentSyncStage
from agent_video_pipeline.adapters.tts_stage import HunyuanTTSStage
from agent_video_pipeline.adapters.video_stage import HyVideoStage, MagiHumanStage
from agent_video_pipeline.adapters.wan_lightx2v_stage import WanLightX2VStage

__all__ = [
    "BananaImageStage",
    "BGMStage",
    "HunyuanImageStage",
    "HunyuanTTSStage",
    "HyVideoStage",
    "LatentSyncStage",
    "MagiHumanStage",
    "SFXStage",
    "WanLightX2VStage",
]
