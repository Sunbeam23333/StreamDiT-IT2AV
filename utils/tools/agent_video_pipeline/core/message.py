"""Standardized message passing system for multi-agent communication.

Inspired by Mora's Message system, adapted for video generation pipeline.
"""
from __future__ import annotations

import uuid
from asyncio import Queue, QueueEmpty
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator


MESSAGE_ROUTE_TO_ALL = "<all>"


def get_class_name(cls) -> str:
    """Return class name with module path."""
    return f"{cls.__module__}.{cls.__name__}"


def any_to_str(val: Any) -> str:
    """Return class name or the class name of the object."""
    if isinstance(val, str):
        return val
    elif not callable(val):
        return get_class_name(type(val))
    else:
        return get_class_name(val)


def any_to_str_set(val) -> Set[str]:
    """Convert any type to string set."""
    res = set()
    if isinstance(val, (dict, list, set, tuple)):
        if isinstance(val, dict):
            val = val.values()
        for i in val:
            res.add(any_to_str(i))
    else:
        res.add(any_to_str(val))
    return res


class Message(BaseModel):
    """Standardized message for agent communication.
    
    Supports both text content and multimedia content (images, videos).
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: str = "user"  # system / user / assistant / tool
    sender: str = Field(default="")
    cause_by: str = Field(default="")
    content: str = Field(default="")
    
    # Multimedia content support
    image_content: Any = Field(default=None)  # PIL Image or list of frames
    video_path: Optional[str] = Field(default=None)  # Path to video file
    audio_path: Optional[str] = Field(default=None)  # Path to audio file
    
    # Routing
    send_to: Set[str] = Field(default={MESSAGE_ROUTE_TO_ALL})
    sent_from: str = Field(default="")
    
    # Metadata for pipeline tracking
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("sender", mode="before")
    @classmethod
    def check_sender(cls, sender: Any) -> str:
        return any_to_str(sender) if sender else ""
    
    @field_validator("sent_from", mode="before")
    @classmethod
    def check_sent_from(cls, sent_from: Any) -> str:
        return any_to_str(sent_from) if sent_from else ""
    
    @field_validator("send_to", mode="before")
    @classmethod
    def check_send_to(cls, send_to: Any) -> Set[str]:
        if not send_to:
            return {MESSAGE_ROUTE_TO_ALL}
        return any_to_str_set(send_to)
    
    def __str__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Message({self.role}, {self.sender}, {content_preview})"
    
    def set_meta(self, key: str, value: Any) -> "Message":
        """Set metadata key-value pair."""
        self.metadata[key] = value
        return self
    
    def get_meta(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)


class MessageQueue:
    """Asynchronous message queue for agent communication."""
    
    def __init__(self):
        self._queue: Queue = Queue()
    
    def pop(self) -> Message | None:
        """Pop one message from the queue (non-blocking)."""
        try:
            item = self._queue.get_nowait()
            if item:
                self._queue.task_done()
            return item
        except QueueEmpty:
            return None
    
    def pop_all(self) -> List[Message]:
        """Pop all messages from the queue."""
        ret = []
        while True:
            msg = self.pop()
            if not msg:
                break
            ret.append(msg)
        return ret
    
    def push(self, msg: Message) -> None:
        """Push a message into the queue."""
        self._queue.put_nowait(msg)
    
    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return self._queue.empty()
    
    def size(self) -> int:
        """Return the number of messages in the queue."""
        return self._queue.qsize()
