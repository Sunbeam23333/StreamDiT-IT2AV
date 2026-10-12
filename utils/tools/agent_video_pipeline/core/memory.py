"""Memory system for agent context retention and retrieval.

Inspired by Mora's Memory, provides indexed storage for conversation history.
"""
from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Iterable, List, Optional

from pydantic import BaseModel, Field, SerializeAsAny

from .message import Message


class Memory(BaseModel):
    """Indexed memory storage for agent conversation history.
    
    Features:
    - Storage with chronological order
    - Index by cause_by for quick retrieval
    - Keyword-based search
    """
    
    storage: List[SerializeAsAny[Message]] = Field(default_factory=list)
    index: DefaultDict[str, List[SerializeAsAny[Message]]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    ignore_id: bool = False
    
    class Config:
        arbitrary_types_allowed = True
    
    def add(self, message: Message) -> None:
        """Add a new message to storage and update index."""
        if message in self.storage:
            return
        self.storage.append(message)
        if message.cause_by:
            self.index[message.cause_by].append(message)
    
    def add_batch(self, messages: Iterable[Message]) -> None:
        """Add multiple messages."""
        for message in messages:
            self.add(message)
    
    def get_by_role(self, role: str) -> List[Message]:
        """Return all messages of a specified role."""
        return [m for m in self.storage if m.role == role]
    
    def get_by_content(self, content: str) -> List[Message]:
        """Return all messages containing specified content."""
        return [m for m in self.storage if content in m.content]
    
    def get_by_cause(self, cause_by: str) -> List[Message]:
        """Return all messages caused by specified action."""
        return self.index.get(cause_by, [])
    
    def delete_newest(self) -> Optional[Message]:
        """Delete and return the newest message."""
        if not self.storage:
            return None
        newest_msg = self.storage.pop()
        if newest_msg.cause_by and newest_msg in self.index[newest_msg.cause_by]:
            self.index[newest_msg.cause_by].remove(newest_msg)
        return newest_msg
    
    def delete(self, message: Message) -> None:
        """Delete a specific message."""
        if message in self.storage:
            self.storage.remove(message)
        if message.cause_by and message in self.index[message.cause_by]:
            self.index[message.cause_by].remove(message)
    
    def clear(self) -> None:
        """Clear all storage and index."""
        self.storage = []
        self.index = defaultdict(list)
    
    def count(self) -> int:
        """Return the number of messages."""
        return len(self.storage)
    
    def try_remember(self, keyword: str) -> List[Message]:
        """Find all messages containing keyword."""
        return [m for m in self.storage if keyword in m.content]
    
    def get(self, k: int = 0) -> List[Message]:
        """Return the most recent k messages, return all when k=0."""
        if k == 0:
            return self.storage.copy()
        return self.storage[-k:]
    
    def find_news(self, observed: List[Message], k: int = 0) -> List[Message]:
        """Find previously unseen messages from recent k memories."""
        already_observed = self.get(k)
        return [m for m in observed if m not in already_observed]
    
    def get_last_image(self) -> Optional[Message]:
        """Get the most recent message with image content."""
        for msg in reversed(self.storage):
            if msg.image_content is not None:
                return msg
        return None
    
    def get_last_video(self) -> Optional[Message]:
        """Get the most recent message with video path."""
        for msg in reversed(self.storage):
            if msg.video_path is not None:
                return msg
        return None
