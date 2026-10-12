"""LLM configuration management.

Inspired by Mora's LLMConfig, supports multiple LLM providers.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class LLMType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GEMINI = "gemini"
    DASHSCOPE = "dashscope"  # Aliyun
    MOONSHOT = "moonshot"
    OLLAMA = "ollama"
    HUNYUAN = "hunyuan"  # Tencent


class LLMConfig(BaseModel):
    """Configuration for LLM API access.
    
    Example:
        config = LLMConfig(
            api_type=LLMType.OPENAI,
            api_key="sk-...",
            model="gpt-4",
        )
        
        # Or load from YAML
        config = LLMConfig.from_yaml_file(Path("config.yaml"))
    """
    
    # API credentials
    api_type: LLMType = LLMType.OPENAI
    api_key: str = Field(default="")
    api_version: Optional[str] = None
    base_url: str = Field(default="https://api.openai.com/v1")
    azure_endpoint: Optional[str] = None
    
    # Model settings
    model: Optional[str] = None
    
    # Generation parameters
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=1.0)
    top_k: int = Field(default=0)
    repetition_penalty: float = Field(default=1.0)
    stop: Optional[str] = None
    stream: bool = Field(default=False)
    timeout: int = Field(default=600)
    
    # Cost tracking
    calc_usage: bool = Field(default=True)
    
    # Network
    proxy: Optional[str] = None
    
    # Extra fields for provider-specific settings
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not empty placeholder."""
        if v in ["", None, "YOUR_API_KEY", "sk-"]:
            # Allow empty for cases where env var is used
            pass
        return v or ""
    
    def model_dump_yaml(self) -> str:
        """Export config to YAML string."""
        return yaml.safe_dump(self.model_dump(), allow_unicode=True)
    
    def to_yaml_file(self, path: Path) -> None:
        """Save config to YAML file."""
        path.write_text(self.model_dump_yaml(), encoding="utf-8")
    
    @classmethod
    def from_yaml_file(cls, path: Path) -> "LLMConfig":
        """Load config from YAML file."""
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**(data or {}))
    
    @classmethod
    def from_env(cls, prefix: str = "LLM_") -> "LLMConfig":
        """Load config from environment variables.
        
        Environment variables should be prefixed, e.g.:
        - LLM_API_TYPE=openai
        - LLM_API_KEY=sk-...
        - LLM_MODEL=gpt-4
        """
        import os
        
        kwargs = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                
                # Type conversion
                if config_key in ["max_tokens", "top_k", "timeout"]:
                    value = int(value)
                elif config_key in ["temperature", "top_p", "repetition_penalty"]:
                    value = float(value)
                elif config_key in ["stream", "calc_usage"]:
                    value = value.lower() in ("true", "1", "yes")
                
                kwargs[config_key] = value
        
        return cls(**kwargs)


def load_llm_config(
    file_path: Optional[Path] = None,
    use_env: bool = True,
) -> LLMConfig:
    """Load LLM config from file and/or environment.
    
    Priority: env vars > config file > defaults
    
    Args:
        file_path: Path to YAML config file
        use_env: Whether to override with environment variables
    
    Returns:
        Merged LLMConfig
    """
    # Start with file config if provided
    if file_path and file_path.exists():
        config = LLMConfig.from_yaml_file(file_path)
    else:
        config = LLMConfig()
    
    # Override with environment if enabled
    if use_env:
        env_config = LLMConfig.from_env()
        # Merge non-empty values
        for field_name, value in env_config.model_dump().items():
            if value:  # Only override if env var is set
                setattr(config, field_name, value)
    
    return config
