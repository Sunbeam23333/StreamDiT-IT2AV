"""
Advanced workflow example using the Role and Workflow framework.

This example demonstrates how to build custom agents and workflows
using the Mora-inspired multi-agent architecture.
"""
import asyncio
from pathlib import Path

from agent_video_pipeline.actions import (
    GenerateImageAction,
    GenerateTTSAction,
    GenerateVideoAction,
    LipSyncAction,
    MergeAudioAction,
)
from agent_video_pipeline.core import (
    Message,
    Role,
    RoleReactMode,
    VideoGenerationWorkflow,
    WorkflowState,
)


class VideoProducerRole(Role):
    """Custom role for video production."""
    
    name: str = "VideoProducer"
    profile: str = "Video Production Expert"
    goal: str = "Generate high-quality videos with synchronized audio"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set up actions in execution order
        self.set_actions([
            GenerateImageAction,
            GenerateTTSAction,
            GenerateVideoAction,
            LipSyncAction,
            MergeAudioAction,
        ])
        # Execute in order
        self._set_react_mode(RoleReactMode.BY_ORDER)


class QuickVideoRole(Role):
    """Role for quick video generation without lip-sync."""
    
    name: str = "QuickVideo"
    profile: str = "Fast Video Generator"
    goal: str = "Generate videos quickly for prototyping"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([
            GenerateImageAction,
            GenerateVideoAction,
        ])
        self._set_react_mode(RoleReactMode.BY_ORDER)


async def example_role_based():
    """Example using Role-based execution."""
    print("=" * 60)
    print("Example: Role-based Video Generation")
    print("=" * 60)
    
    # Create producer role
    producer = VideoProducerRole()
    
    # Create initial message
    msg = Message(
        content="一个可爱的机器人在跳舞",
        role="user",
    )
    msg.set_meta("output_dir", "/tmp/agent_video/role_example")
    msg.set_meta("image_size", "768x1280")
    msg.set_meta("voice", "wenrouxuemei")
    
    # Execute
    result = await producer.run(msg)
    
    print(f"Final result: {result}")
    print(f"Memory history ({producer.memory.count()} messages):")
    for m in producer.memory.get():
        print(f"  - {m}")


async def example_workflow_based():
    """Example using Workflow-based execution."""
    print("\n" + "=" * 60)
    print("Example: Workflow-based Video Generation")
    print("=" * 60)
    
    # Create workflow
    workflow = VideoGenerationWorkflow(
        name="text_to_video",
        route="hyvideo_lipsync",
    )
    
    # Create producer role
    producer = VideoProducerRole()
    
    # Assign role to workflow steps
    for step in workflow.steps.values():
        if step.state in [
            WorkflowState.GENERATING_IMAGE,
            WorkflowState.GENERATING_AUDIO,
            WorkflowState.GENERATING_VIDEO,
            WorkflowState.LIPSYNCING,
            WorkflowState.EDITING,
        ]:
            step.role = producer
    
    # Initial context
    context = Message(
        content="A beautiful sunset over the ocean with waves crashing",
        role="user",
    )
    context.set_meta("output_dir", "/tmp/agent_video/workflow_example")
    
    # Execute workflow
    result = await workflow.execute(context)
    
    print(f"Workflow completed: {result['success']}")
    print(f"Executed steps: {result['executed']}")


async def example_custom_workflow():
    """Example building custom workflow programmatically."""
    print("\n" + "=" * 60)
    print("Example: Custom Workflow")
    print("=" * 60)
    
    from agent_video_pipeline.core import Workflow, WorkflowStep
    
    # Create custom workflow
    workflow = Workflow(name="custom_image_to_video")
    
    # Add steps
    workflow.add_step(WorkflowStep(
        name="generate_image",
        state=WorkflowState.GENERATING_IMAGE,
        role=QuickVideoRole(),
        description="Generate first frame image",
    ))
    
    workflow.add_step(WorkflowStep(
        name="generate_video",
        state=WorkflowState.GENERATING_VIDEO,
        role=QuickVideoRole(),
        description="Generate video from image",
        dependencies=["generate_image"],
    ))
    
    # Execute
    context = Message(
        content="A cyberpunk city at night with neon lights",
        role="user",
    )
    context.set_meta("output_dir", "/tmp/agent_video/custom_workflow")
    
    result = await workflow.execute(context)
    
    print(f"Custom workflow completed: {result['success']}")
    print(f"Progress: {workflow.get_status()['progress']:.0%}")


async def main():
    """Run all examples."""
    # Note: These examples require the actual models to be available
    # They demonstrate the API usage
    
    print("Agent Video Pipeline - Advanced Examples")
    print("Note: These are API demonstrations. Actual execution requires models.\n")
    
    try:
        await example_role_based()
    except Exception as e:
        print(f"Role example skipped: {e}")
    
    try:
        await example_workflow_based()
    except Exception as e:
        print(f"Workflow example skipped: {e}")
    
    try:
        await example_custom_workflow()
    except Exception as e:
        print(f"Custom workflow example skipped: {e}")


if __name__ == "__main__":
    asyncio.run(main())
