"""Action base class for agent behaviors.

Actions are executable units that process messages and produce results.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from .message import Message


class Action(BaseModel):
    """Base class for all agent actions.
    
    An Action represents a discrete unit of work that an agent can perform,
    such as generating an image, creating video, or editing content.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = ""
    prefix: str = ""
    description: str = ""
    
    # Execution metadata
    config: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.name:
            self.name = self.__class__.__name__
    
    def __str__(self) -> str:
        return self.name or self.__class__.__name__
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
    
    @abstractmethod
    async def run(self, *args, **kwargs) -> Message:
        """Execute the action.
        
        Must be implemented by subclasses.
        
        Returns:
            Message containing the action result.
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def set_config(self, key: str, value: Any) -> "Action":
        """Set configuration value."""
        self.config[key] = value
        return self
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)


class ActionResult:
    """Structured result from action execution.
    
    Provides standardized way to track action outcomes.
    """
    
    def __init__(
        self,
        success: bool,
        message: Message | None = None,
        error: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        self.success = success
        self.message = message
        self.error = error
        self.metadata = metadata or {}
    
    def __bool__(self) -> bool:
        return self.success
    
    @classmethod
    def ok(cls, message: Message, **metadata) -> "ActionResult":
        """Create successful result."""
        return cls(success=True, message=message, metadata=metadata)
    
    @classmethod
    def fail(cls, error: str, **metadata) -> "ActionResult":
        """Create failed result."""
        return cls(success=False, error=error, metadata=metadata)
