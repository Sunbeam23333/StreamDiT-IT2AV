"""Role/Agent system for video generation pipeline.

Inspired by Mora's Role architecture with ReAct patterns.
"""
from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from .action import Action
from .memory import Memory
from .message import Message, MessageQueue


class RoleReactMode(str, Enum):
    """Reaction modes for role behavior."""
    REACT = "react"           # Standard think-act loop
    BY_ORDER = "by_order"     # Execute actions in order
    PLAN_AND_ACT = "plan_and_act"  # Plan first, then execute


class Role(BaseModel):
    """Base class for agent roles in the video pipeline.
    
    A Role represents an agent with specific capabilities (actions),
    memory, and behavior patterns.
    
    Example:
        class VideoProducer(Role):
            name: str = "Producer"
            profile: str = "Video Generation Expert"
            goal: str = "Generate high-quality videos from text"
            
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.set_actions([GenerateImageAction, GenerateVideoAction])
                self._set_react_mode(RoleReactMode.BY_ORDER)
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    # Role identity
    name: str = ""
    profile: str = ""
    goal: str = ""
    constraints: str = ""
    desc: str = ""
    
    # State management
    state: int = Field(default=-1)  # -1 = initial/terminated
    states: List[str] = Field(default_factory=list)
    SOPs: List[int] = Field(default_factory=list)  # Standard Operating Procedures
    
    # Actions
    actions: List[SerializeAsAny[Action]] = Field(default_factory=list)
    todo: Optional[Action] = Field(default=None, exclude=True)
    
    # Communication
    memory: Memory = Field(default_factory=Memory)
    msg_buffer: MessageQueue = Field(default_factory=MessageQueue, exclude=True)
    watch: set[str] = Field(default_factory=set)
    
    # Behavior settings
    react_mode: RoleReactMode = RoleReactMode.BY_ORDER
    max_react_loop: int = 10
    
    # Runtime state
    recovered: bool = False
    latest_observed_msg: Optional[Message] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._post_init()
    
    def _post_init(self):
        """Post-initialization setup."""
        if self.actions:
            self.set_actions(self.actions)
    
    def _reset(self):
        """Reset state and actions."""
        self.states = []
        self.actions = []
    
    def set_action(self, action: Type[Action] | Action) -> None:
        """Add a single action."""
        self.set_actions([action])
    
    def set_actions(self, actions: List[Type[Action] | Action]) -> None:
        """Set actions for this role.
        
        Args:
            actions: List of Action classes or instances.
        """
        self._reset()
        for action in actions:
            if isinstance(action, type):
                instance = action()
            else:
                instance = action
            self.actions.append(instance)
            self.states.append(f"{len(self.actions) - 1}. {instance.name}")
    
    def _set_react_mode(
        self,
        react_mode: RoleReactMode,
        max_react_loop: int = 10,
    ) -> None:
        """Set reaction mode.
        
        Args:
            react_mode: Mode for choosing actions.
            max_react_loop: Maximum react cycles to prevent infinite loops.
        """
        self.react_mode = react_mode
        self.max_react_loop = max_react_loop
    
    def _set_state(self, state: int) -> None:
        """Update current state."""
        self.state = state
        if state >= 0 and state < len(self.actions):
            self.todo = self.actions[state]
        else:
            self.todo = None
    
    def _watch(self, actions: List[Type[Action]]) -> None:
        """Watch for messages caused by specific actions."""
        from .message import any_to_str
        self.watch = {any_to_str(t) for t in actions}
    
    def put_message(self, message: Message) -> None:
        """Put message into buffer."""
        if message:
            self.msg_buffer.push(message)
    
    def get_memories(self, k: int = 0) -> List[Message]:
        """Get recent k messages from memory."""
        return self.memory.get(k)
    
    @property
    def history(self) -> List[Message]:
        """Get conversation history."""
        return self.memory.get()
    
    def _get_prefix(self) -> str:
        """Get role prefix for prompts."""
        if self.desc:
            return self.desc
        
        prefix = f"You are a {self.profile}, named {self.name}"
        if self.goal:
            prefix += f", your goal is {self.goal}"
        if self.constraints:
            prefix += f". The constraint is {self.constraints}"
        return prefix
    
    async def _think(self) -> bool:
        """Decide next action. Returns False if nothing to do."""
        if len(self.actions) == 1:
            self._set_state(0)
            return True
        
        if self.recovered and self.state >= 0:
            self._set_state(self.state)
            self.recovered = False
            return True
        
        # Simple round-robin for BY_ORDER mode
        next_state = self.state + 1
        if next_state >= len(self.actions):
            return False
        
        self._set_state(next_state)
        return True
    
    async def _act(self) -> Message:
        """Execute current todo action."""
        if not self.todo:
            return Message(content="No action to execute", role=self.profile)
        
        # Get context from memory
        context = self.get_memories(k=1)
        context_msg = context[0] if context else Message(content="")
        
        # Run action
        result = await self.todo.run(context_msg)
        
        if not isinstance(result, Message):
            result = Message(
                content=str(result),
                role=self.profile,
                cause_by=type(self.todo).__name__,
                sent_from=self.name,
            )
        
        self.memory.add(result)
        return result
    
    async def _observe(self) -> int:
        """Observe new messages from buffer."""
        news = self.msg_buffer.pop_all()
        
        if not news and self.recovered and self.latest_observed_msg:
            news = [self.latest_observed_msg]
        
        old_messages = self.memory.get()
        self.memory.add_batch(news)
        
        # Filter messages of interest
        from .message import any_to_str
        self.news = [
            n for n in news 
            if (any_to_str(n.cause_by) in self.watch or self.name in n.send_to)
            and n not in old_messages
        ]
        
        if self.news:
            self.latest_observed_msg = self.news[-1]
        
        return len(self.news)
    
    async def _react(self) -> Message:
        """Standard ReAct loop: think -> act -> think -> act..."""
        actions_taken = 0
        last_response = Message(content="No actions taken", role=self.profile)
        
        while actions_taken < self.max_react_loop:
            should_act = await self._think()
            if not should_act or self.todo is None:
                break
            
            last_response = await self._act()
            actions_taken += 1
        
        return last_response
    
    async def _act_by_order(self) -> Message:
        """Execute actions in predefined order."""
        start_idx = max(0, self.state) if self.state >= 0 else 0
        last_response = Message(content="No actions taken", role=self.profile)
        
        for i in range(start_idx, len(self.actions)):
            self._set_state(i)
            last_response = await self._act()
        
        return last_response
    
    async def react(self) -> Message:
        """Entry point for role reaction."""
        if self.react_mode == RoleReactMode.REACT:
            rsp = await self._react()
        elif self.react_mode == RoleReactMode.BY_ORDER:
            rsp = await self._act_by_order()
        else:
            raise ValueError(f"Unsupported react mode: {self.react_mode}")
        
        self._set_state(-1)
        return rsp
    
    async def run(self, with_message: str | Message | None = None) -> Message | None:
        """Main entry point for role execution.
        
        Args:
            with_message: Optional initial message to process.
        
        Returns:
            Final response message.
        """
        # Generate SOPs if needed
        await self._get_sops()
        
        # Process input message
        if with_message:
            if isinstance(with_message, str):
                msg = Message(content=with_message, role="user")
            else:
                msg = with_message
            self.put_message(msg)
        
        # Observe messages
        if not await self._observe():
            return None
        
        # React and produce response
        rsp = await self.react()
        self.todo = None
        
        return rsp
    
    async def _get_sops(self) -> None:
        """Generate Standard Operating Procedures.
        
        Determines the optimal action sequence based on goal.
        """
        if self.SOPs:
            return
        
        # Default: execute all actions in order
        self.SOPs = list(range(len(self.actions)))


# Convenience classes for common roles

class VideoProducer(Role):
    """Role for generating videos from text prompts."""
    
    name: str = "VideoProducer"
    profile: str = "Video Generation Expert"
    goal: str = "Generate high-quality videos from text descriptions"


class VideoEditor(Role):
    """Role for editing and post-processing videos."""
    
    name: str = "VideoEditor"
    profile: str = "Video Editing Expert"
    goal: str = "Edit and enhance videos with professional quality"


class PromptEngineer(Role):
    """Role for optimizing and refining prompts."""
    
    name: str = "PromptEngineer"
    profile: str = "Prompt Optimization Expert"
    goal: str = "Optimize prompts for better generation quality"
