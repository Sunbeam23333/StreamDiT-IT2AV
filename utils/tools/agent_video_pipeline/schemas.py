from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageCommand:
    stage: str
    command: str
    enabled: bool = True
    notes: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRoleSpec:
    name: str
    responsibility: str
    preferred_models: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChainStep:
    stage: str
    owner: str
    goal: str
    preferred_models: List[str] = field(default_factory=list)
    fallback_models: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewSpec:
    name: str
    stage: str
    reviewer: str
    objective: str
    checks: List[str] = field(default_factory=list)
    suggested_models: List[str] = field(default_factory=list)
    fail_action: str = "retry"
    blocking: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StoryboardBeat:
    beat_id: int
    summary: str
    duration_s: float
    shot_type: str
    source_mode: str
    continuity_mode: str
    suggested_route: str
    dialogue: str = ""
    visual_focus: str = ""
    transition_hint: str = ""
    needs_previous_frame: bool = False
    agent_notes: List[str] = field(default_factory=list)
    start_frame: str = ""
    end_frame: str = ""
    action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShotSpec:
    shot_id: int
    title: str
    description: str
    duration_s: float
    route: str
    aspect_ratio: str
    image_size: str
    dialogue: str = ""
    bgm_prompt: str = ""
    sfx_prompt: str = ""
    image_prompt: str = ""
    video_prompt: str = ""
    magihuman_prompt: str = ""
    end_frame_prompt: str = ""
    continuity_source: Optional[str] = None
    scenario: str = "generic"
    shot_type: str = ""
    source_mode: str = "direct_first_frame"
    continuity_notes: str = ""
    visual_focus: str = ""
    transition_hint: str = ""
    style_hint: str = ""
    motion_hint: str = ""
    start_frame_description: str = ""
    end_frame_description: str = ""
    motion_description: str = ""
    backend_candidates: List[str] = field(default_factory=list)
    agent_notes: List[str] = field(default_factory=list)
    commands: List[StageCommand] = field(default_factory=list)
    qc_checks: List[str] = field(default_factory=list)
    review_plan: List[ReviewSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectPlan:
    project_id: str
    user_prompt: str
    optimized_prompt_instruction: str
    route: str
    total_duration_s: float
    aspect_ratio: str
    image_size: str
    project_dir: str
    execution_mode: str = "gpu"
    planning_backend: str = "rule_based"
    scenario: str = "generic"
    scenario_display_name: str = "通用视频"
    storyboard_mode: str = "narrative_beats"
    quality_tier: str = "balanced"
    storyboard: List[StoryboardBeat] = field(default_factory=list)
    shots: List[ShotSpec] = field(default_factory=list)
    project_commands: List[StageCommand] = field(default_factory=list)
    agent_roles: List[AgentRoleSpec] = field(default_factory=list)
    generation_chain: List[ChainStep] = field(default_factory=list)
    review_plan: List[ReviewSpec] = field(default_factory=list)
    strategy_summary: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
