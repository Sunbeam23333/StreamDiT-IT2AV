"""Video editing actions."""
from __future__ import annotations

from pathlib import Path

from agent_video_pipeline.core.action import Action
from agent_video_pipeline.core.message import Message


class LipSyncAction(Action):
    """Action to apply lip-sync to video."""
    
    name: str = "LipSync"
    description: str = "Apply lip-sync to video using LatentSync"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from agent_video_pipeline.adapters.lipsync_stage import LatentSyncStage
        stage = LatentSyncStage()
        self.script_path = stage.script_path
        self.python_bin = stage.python_bin
        self.unet_config = stage.unet_config
    
    async def run(self, message: Message) -> Message:
        """Apply lip-sync to video.
        
        Args:
            message: Should have video_path and audio_path in metadata
            
        Returns:
            Message with video_path set to lip-synced video
        """
        video_path = message.video_path or message.get_meta("video_path")
        audio_path = message.audio_path or message.get_meta("audio_path")
        
        if not video_path or not audio_path:
            return Message(
                content="Error: video_path and audio_path required for lip-sync",
                role="assistant",
                cause_by=self.name,
            )
        
        output_dir = Path(message.get_meta("output_dir", "/tmp/agent_video"))
        output_path = output_dir / "lipsync_video.mp4"
        
        from agent_video_pipeline.adapters.lipsync_stage import LatentSyncStage
        
        stage = LatentSyncStage()
        
        try:
            stage.generate(
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=str(output_path),
            )
            
            result = Message(
                content=f"Applied lip-sync: {output_path}",
                video_path=str(output_path),
                role="assistant",
                cause_by=self.name,
            )
            result.set_meta("video_path", str(output_path))
            return result
            
        except Exception as e:
            return Message(
                content=f"Error applying lip-sync: {e}",
                role="assistant",
                cause_by=self.name,
            )


class MergeAudioAction(Action):
    """Action to merge multiple audio tracks."""
    
    name: str = "MergeAudio"
    description: str = "Merge TTS, BGM, and SFX into final audio"
    
    async def run(self, message: Message) -> Message:
        """Merge audio tracks onto video."""
        video_path = message.video_path or message.get_meta("video_path")
        
        # Collect audio tracks
        audio_specs = []
        
        tts_path = message.get_meta("tts_path")
        if tts_path:
            audio_specs.append((tts_path, 1.0))
        
        bgm_path = message.get_meta("bgm_path")
        if bgm_path:
            audio_specs.append((bgm_path, 0.2))
        
        sfx_path = message.get_meta("sfx_path")
        if sfx_path:
            audio_specs.append((sfx_path, 0.15))
        
        if not audio_specs:
            return Message(
                content="No audio tracks to merge",
                video_path=video_path,
                role="assistant",
                cause_by=self.name,
            )
        
        output_dir = Path(message.get_meta("output_dir", "/tmp/agent_video"))
        output_path = output_dir / "final.mp4"
        
        from agent_video_pipeline.editing.ffmpeg_editor import merge_audio
        
        try:
            merge_audio(video_path, audio_specs, str(output_path))
            
            result = Message(
                content=f"Merged audio: {output_path}",
                video_path=str(output_path),
                role="assistant",
                cause_by=self.name,
            )
            result.set_meta("video_path", str(output_path))
            return result
            
        except Exception as e:
            return Message(
                content=f"Error merging audio: {e}",
                role="assistant",
                cause_by=self.name,
            )


class ConcatenateVideosAction(Action):
    """Action to concatenate multiple video clips."""
    
    name: str = "ConcatenateVideos"
    description: str = "Concatenate multiple video clips into one"
    
    async def run(self, message: Message) -> Message:
        """Concatenate videos from message metadata."""
        video_paths = message.get_meta("video_paths", [])
        
        if len(video_paths) < 2:
            return Message(
                content="Need at least 2 videos to concatenate",
                video_path=video_paths[0] if video_paths else None,
                role="assistant",
                cause_by=self.name,
            )
        
        output_dir = Path(message.get_meta("output_dir", "/tmp/agent_video"))
        output_path = output_dir / "final_story.mp4"
        
        from agent_video_pipeline.editing.ffmpeg_editor import concat_videos
        
        try:
            concat_videos(video_paths, str(output_path))
            
            result = Message(
                content=f"Concatenated videos: {output_path}",
                video_path=str(output_path),
                role="assistant",
                cause_by=self.name,
            )
            result.set_meta("video_path", str(output_path))
            return result
            
        except Exception as e:
            return Message(
                content=f"Error concatenating videos: {e}",
                role="assistant",
                cause_by=self.name,
            )
