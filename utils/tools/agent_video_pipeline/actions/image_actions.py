"""Image generation actions."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agent_video_pipeline.core.action import Action
from agent_video_pipeline.core.message import Message


class GenerateImageAction(Action):
    """Action to generate an image from text prompt.
    
    Wraps the Hunyuan-Image generation script.
    """
    
    name: str = "GenerateImage"
    description: str = "Generate an image from text prompt using Hunyuan-Image"
    
    def __init__(
        self,
        script_path: str | None = None,
        python_bin: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Resolve script path
        if script_path is None:
            from agent_video_pipeline.adapters.image_stage import HunyuanImageStage
            stage = HunyuanImageStage()
            self.script_path = stage.script_path
            self.python_bin = stage.python_bin
        else:
            self.script_path = Path(script_path)
            self.python_bin = python_bin or sys.executable
    
    async def run(self, message: Message) -> Message:
        """Generate image from message content.
        
        Args:
            message: Input message with content as prompt
            
        Returns:
            Message with image_content set to generated image path
        """
        prompt = message.content
        output_dir = Path(message.get_meta("output_dir", "/tmp/agent_video"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "first_frame.png"
        size = message.get_meta("image_size", "768x1280")
        
        cmd = [
            self.python_bin,
            str(self.script_path),
            "--prompt", prompt,
            "--output", str(output_path),
            "--size", size,
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            result = Message(
                content=f"Generated image: {output_path}",
                image_content=str(output_path),
                role="assistant",
                cause_by=self.name,
            )
            result.set_meta("image_path", str(output_path))
            return result
            
        except subprocess.CalledProcessError as e:
            return Message(
                content=f"Error generating image: {e}",
                role="assistant",
                cause_by=self.name,
            )


class GenerateImageWithRefinement(Action):
    """Generate image with prompt refinement using LLM."""
    
    name: str = "GenerateImageWithRefinement"
    description: str = "Generate image with AI-refined prompt"
    
    async def run(self, message: Message) -> Message:
        """Refine prompt and generate image."""
        # TODO: Integrate with LLM for prompt refinement
        # For now, delegate to base action
        base_action = GenerateImageAction()
        return await base_action.run(message)
