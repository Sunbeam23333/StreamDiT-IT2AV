"""Built-in actions for video generation pipeline.

Actions are reusable components that can be composed into Roles.
"""
from .image_actions import GenerateImageAction
from .video_actions import GenerateVideoAction
from .audio_actions import GenerateTTSAction
from .edit_actions import LipSyncAction, MergeAudioAction

__all__ = [
    "GenerateImageAction",
    "GenerateVideoAction",
    "GenerateTTSAction",
    "LipSyncAction",
    "MergeAudioAction",
]
