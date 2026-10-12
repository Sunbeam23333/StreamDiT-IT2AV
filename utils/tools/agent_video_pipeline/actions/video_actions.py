"""Video generation actions."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agent_video_pipeline.core.action import Action
from agent_video_pipeline.core.message import Message


class GenerateVideoAction(Action):
    """Action to generate video from image and prompt.
    
    Supports both HyVideo I2V and MagiHuman backends.
    """
    
    name: str = "GenerateVideo"
    description: str = "Generate video from image using I2V model"
    
    def __init__(
        self,
        backend: str = "hyvideo",  # "hyvideo" or "magihuman"
        script_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = backend
        
        if backend == "hyvideo":
            from agent_video_pipeline.adapters.video_stage import HyVideoStage
            stage = HyVideoStage()
            self.script_path = stage.script_path
            self.python_bin = stage.python_bin
            self.num_gpus = stage.num_gpus
        else:  # magihuman
            from agent_video_pipeline.adapters.video_stage import MagiHumanStage
            stage = MagiHumanStage()
            self.script_path = stage.script_path
            self.python_bin = stage.python_bin
    
    async def run(self, message: Message) -> Message:
        """Generate video from image.
        
        Args:
            message: Input with image_content as image path
            
        Returns:
            Message with video_path set to generated video
        """
        image_path = message.image_content or message.get_meta("image_path")
        prompt = message.content or "generate video"
        
        output_dir = Path(message.get_meta("output_dir", "/tmp/agent_video"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.backend == "hyvideo":
            return await self._run_hyvideo(image_path, prompt, output_dir)
        else:
            return await self._run_magihuman(image_path, prompt, output_dir)
    
    async def _run_hyvideo(
        self,
        image_path: str,
        prompt: str,
        output_dir: Path,
    ) -> Message:
        """Run HunyuanVideo I2V generation."""
        output_path = output_dir / "raw_video.mp4"
        
        cmd = [
            "torchrun",
            "--nproc_per_node", str(self.num_gpus),
            str(self.script_path),
            "--prompt", prompt,
            "--image_path", str(image_path),
            "--output_path", str(output_path),
            "--resolution", "720p",
            "--aspect_ratio", "9:16",
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            result = Message(
                content=f"Generated video: {output_path}",
                video_path=str(output_path),
                role="assistant",
                cause_by=self.name,
            )
            result.set_meta("video_path", str(output_path))
            return result
            
        except subprocess.CalledProcessError as e:
            return Message(
                content=f"Error generating video: {e}",
                role="assistant",
                cause_by=self.name,
            )
    
    async def _run_magihuman(
        self,
        image_path: str,
        prompt: str,
        output_dir: Path,
    ) -> Message:
        """Run MagiHuman video generation."""
        # Delegate to MagiHumanStage
        from agent_video_pipeline.adapters.video_stage import MagiHumanStage
        
        stage = MagiHumanStage()
        seconds = int(self.config.get("seconds", 5))
        
        try:
            stage.generate(
                image_path=image_path,
                prompt=prompt,
                output_dir=str(output_dir),
                seconds=seconds,
            )
            
            # Find generated video
            candidates = list(output_dir.glob("magihuman_video_*.mp4"))
            if candidates:
                video_path = sorted(candidates)[0]
                result = Message(
                    content=f"Generated video: {video_path}",
                    video_path=str(video_path),
                    role="assistant",
                    cause_by=self.name,
                )
                result.set_meta("video_path", str(video_path))
                return result
            else:
                return Message(
                    content="Video generation failed: no output file",
                    role="assistant",
                    cause_by=self.name,
                )
                
        except Exception as e:
            return Message(
                content=f"Error generating video: {e}",
                role="assistant",
                cause_by=self.name,
            )


class ExtendVideoAction(Action):
    """Action to extend video by generating continuation frames."""
    
    name: str = "ExtendVideo"
    description: str = "Extend video duration by generating continuation"
    
    async def run(self, message: Message) -> Message:
        """Extend existing video."""
        video_path = message.video_path or message.get_meta("video_path")
        
        # Extract last frame
        from agent_video_pipeline.editing.ffmpeg_editor import extract_last_frame
        
        output_dir = Path(message.get_meta("output_dir", "/tmp/agent_video"))
        last_frame = output_dir / "last_frame.png"
        
        extract_last_frame(video_path, str(last_frame))
        
        # Generate continuation
        gen_action = GenerateVideoAction()
        new_msg = Message(
            content="Continue the video",
            image_content=str(last_frame),
        )
        new_msg.metadata = message.metadata.copy()
        
        result = await gen_action.run(new_msg)
        
        # Concatenate videos
        # TODO: Implement video concatenation
        
        return result
