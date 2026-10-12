"""
Simple pipeline usage example.

This example demonstrates the basic usage of the agent video pipeline
with the high-level run_pipeline API.
"""
import asyncio
from pathlib import Path

from agent_video_pipeline import run_pipeline


async def main():
    """Run a simple video generation pipeline."""
    
    # Example 1: Simple text-to-video with lip-sync
    print("=" * 60)
    print("Example 1: Text-to-Video with Lip-sync")
    print("=" * 60)
    
    result = run_pipeline(
        user_prompt="一个年轻的女主播在介绍新款智能手机，面带微笑，背景是现代科技风格的直播间",
        duration_s=5.0,
        project_id="demo_lipsync_001",
        project_dir="/tmp/agent_video/demo1",
        preferred_route="hyvideo_lipsync",
        aspect_ratio="9:16",
    )
    
    print(f"Result: {result}")
    
    # Example 2: MagiHuman route (for music/atmosphere videos)
    print("\n" + "=" * 60)
    print("Example 2: MagiHuman Video Generation")
    print("=" * 60)
    
    result = run_pipeline(
        user_prompt="Create an atmospheric video of a person walking in the rain with cinematic lighting",
        duration_s=8,
        project_id="demo_magihuman_001",
        project_dir="/tmp/agent_video/demo2",
        preferred_route="magihuman_av",
        aspect_ratio="9:16",
    )
    
    print(f"Result: {result}")
    
    # Example 3: Long-form video (auto-segmented)
    print("\n" + "=" * 60)
    print("Example 3: Long-form Video (15s)")
    print("=" * 60)
    
    result = run_pipeline(
        user_prompt="A cat exploring a garden. First the cat sniffs flowers, then plays with butterflies, and finally rests under a tree.",
        duration_s=15.0,
        project_id="demo_longform_001",
        project_dir="/tmp/agent_video/demo3",
        preferred_route="hyvideo_longform",
        aspect_ratio="9:16",
    )
    
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
