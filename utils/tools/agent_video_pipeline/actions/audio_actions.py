"""Audio generation actions."""
from __future__ import annotations

import subprocess
from pathlib import Path

from agent_video_pipeline.core.action import Action
from agent_video_pipeline.core.message import Message


class GenerateTTSAction(Action):
    """Action to generate TTS audio from text."""
    
    name: str = "GenerateTTS"
    description: str = "Generate speech audio from text using Hunyuan-TTS"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from agent_video_pipeline.adapters.tts_stage import HunyuanTTSStage
        stage = HunyuanTTSStage()
        self.script_path = stage.script_path
        self.python_bin = stage.python_bin
    
    async def run(self, message: Message) -> Message:
        """Generate TTS from message content.
        
        Args:
            message: Input with content as text to speak
            
        Returns:
            Message with audio_path set to generated audio
        """
        text = message.content
        
        output_dir = Path(message.get_meta("output_dir", "/tmp/agent_video"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "tts.wav"
        voice = message.get_meta("voice", "wenrouxuemei")
        
        cmd = [
            self.python_bin,
            str(self.script_path),
            "--text", text,
            "--output", str(output_path),
            "--voice", voice,
            "--stream",
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            result = Message(
                content=f"Generated TTS: {output_path}",
                audio_path=str(output_path),
                role="assistant",
                cause_by=self.name,
            )
            result.set_meta("audio_path", str(output_path))
            result.set_meta("audio_type", "tts")
            return result
            
        except subprocess.CalledProcessError as e:
            return Message(
                content=f"Error generating TTS: {e}",
                role="assistant",
                cause_by=self.name,
            )


class GenerateBGMAction(Action):
    """Action to generate background music."""
    
    name: str = "GenerateBGM"
    description: str = "Generate background music for video"
    
    async def run(self, message: Message) -> Message:
        """Generate BGM based on scene description."""
        # TODO: Integrate with music generation model
        # For now, return placeholder
        
        output_dir = Path(message.get_meta("output_dir", "/tmp/agent_video"))
        bgm_path = output_dir / "bgm.wav"
        
        # Create silent audio as placeholder
        from agent_video_pipeline.adapters.audio_stage import SFXStage
        
        sfx = SFXStage()
        duration = message.get_meta("duration_s", 5.0)
        sfx.generate_silent(duration, str(bgm_path))
        
        result = Message(
            content=f"Generated BGM: {bgm_path}",
            audio_path=str(bgm_path),
            role="assistant",
            cause_by=self.name,
        )
        result.set_meta("audio_path", str(bgm_path))
        result.set_meta("audio_type", "bgm")
        return result
