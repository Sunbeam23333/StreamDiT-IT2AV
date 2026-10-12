"""Workflow state machine for managing video generation pipeline.

Provides structured workflow management with state transitions.
"""
from __future__ import annotations

import asyncio
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .message import Message
from .role import Role


class WorkflowState(Enum):
    """Standard workflow states for video generation."""
    IDLE = auto()
    PLANNING = auto()
    GENERATING_IMAGE = auto()
    GENERATING_AUDIO = auto()
    GENERATING_VIDEO = auto()
    LIPSYNCING = auto()
    EDITING = auto()
    CONCATENATING = auto()
    QC_CHECKING = auto()
    COMPLETED = auto()
    FAILED = auto()


class WorkflowTransition:
    """Represents a valid state transition."""
    
    def __init__(
        self,
        from_state: WorkflowState,
        to_state: WorkflowState,
        condition: Optional[Callable[[], bool]] = None,
        action: Optional[Callable[[], None]] = None,
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition
        self.action = action
    
    def can_transition(self) -> bool:
        """Check if transition is allowed."""
        return self.condition() if self.condition else True
    
    def execute(self) -> None:
        """Execute transition action."""
        if self.action:
            self.action()


class WorkflowStep(BaseModel):
    """A single step in the workflow."""
    
    name: str
    state: WorkflowState
    role: Optional[Role] = None
    action: Optional[str] = None
    description: str = ""
    enabled: bool = True
    dependencies: List[str] = Field(default_factory=list)
    
    # Execution results
    result: Optional[Message] = None
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class Workflow(BaseModel):
    """State machine driven workflow for video generation.
    
    Manages the execution flow of video generation tasks with:
    - State transitions
    - Step dependencies
    - Error handling and retry
    
    Example:
        workflow = Workflow(name="text_to_video")
        workflow.add_step(WorkflowStep(
            name="generate_image",
            state=WorkflowState.GENERATING_IMAGE,
            role=image_producer,
        ))
        workflow.add_step(WorkflowStep(
            name="generate_video",
            state=WorkflowState.GENERATING_VIDEO,
            role=video_producer,
            dependencies=["generate_image"],
        ))
        await workflow.execute()
    """
    
    name: str
    current_state: WorkflowState = WorkflowState.IDLE
    steps: Dict[str, WorkflowStep] = Field(default_factory=dict)
    transitions: List[WorkflowTransition] = Field(default_factory=list, exclude=True)
    
    # Execution tracking
    executed_steps: Set[str] = Field(default_factory=set)
    failed_steps: Set[str] = Field(default_factory=set)
    
    # Callbacks
    on_state_change: Optional[Callable[[WorkflowState, WorkflowState], None]] = None
    on_step_complete: Optional[Callable[[str, Message], None]] = None
    on_step_fail: Optional[Callable[[str, str], None]] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_step(self, step: WorkflowStep) -> "Workflow":
        """Add a step to the workflow."""
        self.steps[step.name] = step
        return self
    
    def add_transition(self, transition: WorkflowTransition) -> "Workflow":
        """Add a state transition."""
        self.transitions.append(transition)
        return self
    
    def can_transition_to(self, new_state: WorkflowState) -> bool:
        """Check if can transition to new state."""
        for t in self.transitions:
            if t.from_state == self.current_state and t.to_state == new_state:
                return t.can_transition()
        # Allow any transition if no explicit transition defined
        return True
    
    def transition_to(self, new_state: WorkflowState) -> bool:
        """Transition to new state."""
        if not self.can_transition_to(new_state):
            return False
        
        old_state = self.current_state
        self.current_state = new_state
        
        # Execute transition action
        for t in self.transitions:
            if t.from_state == old_state and t.to_state == new_state:
                t.execute()
        
        # Trigger callback
        if self.on_state_change:
            self.on_state_change(old_state, new_state)
        
        return True
    
    def get_ready_steps(self) -> List[WorkflowStep]:
        """Get steps ready for execution (dependencies satisfied)."""
        ready = []
        for name, step in self.steps.items():
            if not step.enabled or name in self.executed_steps:
                continue
            if all(dep in self.executed_steps for dep in step.dependencies):
                ready.append(step)
        return ready
    
    async def execute_step(self, step: WorkflowStep, context: Message) -> bool:
        """Execute a single workflow step."""
        if step.role:
            # Use role to execute
            result = await step.role.run(context)
            step.result = result
            
            if result and not result.content.startswith("Error"):
                self.executed_steps.add(step.name)
                if self.on_step_complete:
                    self.on_step_complete(step.name, result)
                return True
            else:
                step.error = result.content if result else "Unknown error"
                self.failed_steps.add(step.name)
                if self.on_step_fail:
                    self.on_step_fail(step.name, step.error)
                return False
        
        self.executed_steps.add(step.name)
        return True
    
    async def execute(self, initial_context: Optional[Message] = None) -> Dict[str, Any]:
        """Execute the complete workflow.
        
        Returns:
            Execution summary with results and statistics.
        """
        self.transition_to(WorkflowState.PLANNING)
        
        context = initial_context or Message(content="Start workflow")
        
        while True:
            ready_steps = self.get_ready_steps()
            
            if not ready_steps:
                # Check if all steps completed
                pending = [
                    name for name, step in self.steps.items()
                    if step.enabled and name not in self.executed_steps
                ]
                if not pending:
                    break
                else:
                    # Circular dependency or blocked
                    raise RuntimeError(f"Workflow blocked. Pending steps: {pending}")
            
            # Execute ready steps (can be parallelized)
            for step in ready_steps:
                self.transition_to(step.state)
                success = await self.execute_step(step, context)
                
                if success and step.result:
                    context = step.result
                else:
                    # Handle failure
                    self.transition_to(WorkflowState.FAILED)
                    return {
                        "success": False,
                        "failed_step": step.name,
                        "error": step.error,
                        "executed": list(self.executed_steps),
                    }
        
        self.transition_to(WorkflowState.COMPLETED)
        
        return {
            "success": True,
            "executed": list(self.executed_steps),
            "results": {
                name: step.result.content if step.result else None
                for name, step in self.steps.items()
            },
        }
    
    def reset(self) -> None:
        """Reset workflow to initial state."""
        self.current_state = WorkflowState.IDLE
        self.executed_steps.clear()
        self.failed_steps.clear()
        for step in self.steps.values():
            step.result = None
            step.error = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "name": self.name,
            "state": self.current_state.name,
            "total_steps": len(self.steps),
            "executed": len(self.executed_steps),
            "failed": len(self.failed_steps),
            "progress": len(self.executed_steps) / len(self.steps) if self.steps else 0,
        }


class VideoGenerationWorkflow(Workflow):
    """Pre-configured workflow for video generation pipeline."""
    
    def __init__(
        self,
        name: str = "video_generation",
        route: str = "hyvideo_lipsync",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.route = route
        self._setup_default_steps()
    
    def _setup_default_steps(self) -> None:
        """Setup default steps based on route."""
        # Common steps
        self.add_step(WorkflowStep(
            name="plan",
            state=WorkflowState.PLANNING,
            description="Analyze prompt and create generation plan",
        ))
        
        self.add_step(WorkflowStep(
            name="generate_image",
            state=WorkflowState.GENERATING_IMAGE,
            description="Generate first-frame image",
            dependencies=["plan"],
        ))
        
        if self.route in ["hyvideo_lipsync", "hyvideo_longform"]:
            self.add_step(WorkflowStep(
                name="generate_tts",
                state=WorkflowState.GENERATING_AUDIO,
                description="Generate TTS audio",
                dependencies=["plan"],
            ))
            
            self.add_step(WorkflowStep(
                name="generate_video",
                state=WorkflowState.GENERATING_VIDEO,
                description="Generate video from image",
                dependencies=["generate_image"],
            ))
            
            self.add_step(WorkflowStep(
                name="lipsync",
                state=WorkflowState.LIPSYNCING,
                description="Apply lip-sync",
                dependencies=["generate_video", "generate_tts"],
            ))
            
            self.add_step(WorkflowStep(
                name="edit",
                state=WorkflowState.EDITING,
                description="Mix audio and apply effects",
                dependencies=["lipsync"],
            ))
        
        elif self.route == "magihuman_av":
            self.add_step(WorkflowStep(
                name="generate_bgm",
                state=WorkflowState.GENERATING_AUDIO,
                description="Generate background music",
                dependencies=["plan"],
            ))
            
            self.add_step(WorkflowStep(
                name="generate_video",
                state=WorkflowState.GENERATING_VIDEO,
                description="Generate video with MagiHuman",
                dependencies=["generate_image"],
            ))
            
            self.add_step(WorkflowStep(
                name="edit",
                state=WorkflowState.EDITING,
                description="Mix BGM",
                dependencies=["generate_video", "generate_bgm"],
            ))
