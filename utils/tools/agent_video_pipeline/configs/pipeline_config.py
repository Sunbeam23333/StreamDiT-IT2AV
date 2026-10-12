"""Pipeline configuration management.

Centralized configuration for video generation pipeline settings.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from .llm_config import LLMConfig


class ModelPathsConfig(BaseModel):
    """Paths to model directories."""
    
    hunyuan_image: str = Field(default=os.path.join(os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace")), "models/Hunyuan-Image"))
    hunyuan_video: str = Field(default=os.path.join(os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace")), "models/HunyuanVideo-1.5"))
    hunyuan_tts: str = Field(default=os.path.join(os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace")), "models/Hunyuan-tts"))
    magihuman: str = Field(default=os.path.join(os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace")), "models/daVinci-MagiHuman"))
    latentsync: str = Field(default=os.path.join(os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace")), "models/LatentSync"))
    
    # Local cache paths
    local_base: str = Field(default=os.environ.get("IT2AV_LOCAL_ROOT", "/tmp/it2av"))


class GenerationConfig(BaseModel):
    """Generation hyperparameters."""
    
    # Image generation
    image_size: str = Field(default="768x1280")
    image_steps: int = Field(default=30)
    
    # Video generation
    video_resolution: str = Field(default="720p")
    video_aspect_ratio: str = Field(default="9:16")
    video_seed: int = Field(default=42)
    video_cfg_distilled: bool = Field(default=True)
    
    # TTS generation
    tts_voice: str = Field(default="wenrouxuemei")
    tts_sample_rate: int = Field(default=24000)
    
    # Lip-sync
    lipsync_steps: int = Field(default=25)
    lipsync_guidance_scale: float = Field(default=1.5)
    
    # MagiHuman
    magihuman_mode: str = Field(default="sr1080p")
    magihuman_seconds: int = Field(default=5)


class PipelineConfig(BaseModel):
    """Main pipeline configuration.
    
    Contains all settings for the video generation pipeline including:
    - Model paths
    - LLM configuration
    - Generation hyperparameters
    - Execution settings
    
    Example:
        config = PipelineConfig.from_yaml_file(Path("pipeline.yaml"))
        
        # Access settings
        print(config.models.hunyuan_image)
        print(config.generation.image_size)
    """
    
    # Project settings
    project_name: str = Field(default="agent_video")
    project_dir: str = Field(default="/tmp/agent_video")
    
    # Sub-configs
    models: ModelPathsConfig = Field(default_factory=ModelPathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    
    # Execution settings
    num_gpus: int = Field(default=8)
    conda_env: str = Field(default="it2av")
    skip_sync: bool = Field(default=False)
    dry_run: bool = Field(default=False)
    
    # Routes
    default_route: str = Field(default="auto")
    supported_routes: List[str] = Field(default=[
        "auto",
        "hyvideo_lipsync",
        "hyvideo_longform", 
        "magihuman_av",
    ])
    
    # Logging
    log_level: str = Field(default="INFO")
    save_intermediates: bool = Field(default=True)
    
    # Extra settings
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    def model_dump_yaml(self) -> str:
        """Export to YAML."""
        return yaml.safe_dump(self.model_dump(), allow_unicode=True, default_flow_style=False)
    
    def to_yaml_file(self, path: Path) -> None:
        """Save to YAML file."""
        path.write_text(self.model_dump_yaml(), encoding="utf-8")
    
    @classmethod
    def from_yaml_file(cls, path: Path) -> "PipelineConfig":
        """Load from YAML file."""
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**(data or {}))
    
    @classmethod
    def default(cls) -> "PipelineConfig":
        """Create default configuration."""
        return cls()


def create_default_config_file(path: Path) -> None:
    """Create a default configuration file.
    
    Args:
        path: Where to create the config file
    """
    config = PipelineConfig.default()
    config.to_yaml_file(path)


# Example default config content
DEFAULT_CONFIG_YAML = """
# Agent Video Pipeline Configuration

project_name: "agent_video"
project_dir: "/tmp/agent_video"

# Model paths (group disk)
models:
  hunyuan_image: "~/it2av_workspace/models/Hunyuan-Image"
  hunyuan_video: "~/it2av_workspace/models/HunyuanVideo-1.5"
  hunyuan_tts: "~/it2av_workspace/models/Hunyuan-tts"
  magihuman: "~/it2av_workspace/models/daVinci-MagiHuman"
  latentsync: "~/it2av_workspace/models/LatentSync"
  local_base: "/tmp/it2av"

# LLM configuration
llm:
  api_type: "hunyuan"
  api_key: "${HUNYUAN_API_KEY}"
  model: "hunyuan-lite"
  temperature: 0.7
  max_tokens: 4096

# Generation settings
generation:
  image_size: "768x1280"
  image_steps: 30
  video_resolution: "720p"
  video_aspect_ratio: "9:16"
  tts_voice: "wenrouxuemei"
  magihuman_mode: "sr1080p"

# Execution
num_gpus: 8
conda_env: "it2av"
log_level: "INFO"
""".strip()
