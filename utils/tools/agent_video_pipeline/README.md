# Agent Video Pipeline

A multi-agent video generation framework with multiple backend support for image generation, video generation, audio processing, and intelligent model selection.

## Features

### Multi-Backend Support
- **Image Generation**: Hunyuan-Image (local), Banana/FLUX (remote API)
- **Video Generation**:
  - HunyuanVideo 1.5 I2V
  - Wan2.2 MoE I2V via LightX2V
  - Wan2.2 MoE FLF2V (First-Last-Frame)
  - MagiHuman (unified TTS+Video)
- **Audio**: Hunyuan-TTS, BGM generation, SFX
- **Lip-sync**: LatentSync with automatic A/V alignment

### Smart Orchestration
- Automatic route detection based on prompt content and duration
- Fallback chain support for robust execution
- Multi-segment generation for long-form videos (>5s)
- Audio-video synchronization with speed adjustment

### Flexible Architecture
- Mora-inspired Role-Action-Workflow framework
- Configurable backend selection strategies
- Modular adapter design for easy extension

## Quick Start

```python
from agent_video_pipeline import run_pipeline

# Auto-select best backend
result = run_pipeline(
    user_prompt="一个女主播在介绍新款手机",
    duration_s=5.0,
    project_dir="/tmp/demo",
)

# Force specific backend
result = run_pipeline(
    user_prompt="A music video with ambient atmosphere",
    duration_s=8.0,
    preferred_route="magihuman_av",
    enable_av_sync=True,
)
```

## Available Routes

| Route | Backend | Description | Best For |
|-------|---------|-------------|----------|
| `hyvideo_lipsync` | HunyuanVideo 1.5 + LatentSync | I2V + lip-sync | ≤5s, dialogue |
| `hyvideo_longform` | HunyuanVideo 1.5 + continuity | Multi-segment | >5s content |
| `wan22_i2v` | Wan2.2 MoE I2V | High quality I2V | ≤5s, quality priority |
| `wan22_flf2v` | Wan2.2 MoE FLF2V | First-last-frame | >5s, controlled motion |
| `wan22_lipsync` | Wan2.2 + LatentSync | I2V + lip-sync | ≤5s, quality dialogue |
| `magihuman_av` | MagiHuman unified | TTS+Video together | Music videos, ambient |

## Project Structure

```
agent_video_pipeline/
├── core/              # Framework components
│   ├── message.py     # Message/Queue system
│   ├── memory.py      # Indexed memory storage
│   ├── action.py      # Action base class
│   ├── role.py        # Agent roles with React modes
│   └── workflow.py    # State machine workflows
├── adapters/          # Backend adapters
│   ├── image_stage.py        # Hunyuan-Image
│   ├── video_stage.py        # HunyuanVideo, MagiHuman
│   ├── wan_lightx2v_stage.py # Wan2.2 + LightX2V (NEW)
│   ├── tts_stage.py          # Hunyuan-TTS
│   ├── lipsync_stage.py      # LatentSync
│   └── audio_stage.py        # BGM, SFX
├── editing/           # Video editing & sync
│   ├── ffmpeg_editor.py  # concat, merge, subtitles
│   └── sync_editor.py    # A/V sync, speed adjust (NEW)
├── configs/           # Configuration
│   ├── llm_config.py
│   ├── pipeline_config.py
│   └── model_selector.py   # Backend selection (NEW)
├── actions/           # Built-in actions
├── schemas.py         # Data models
├── planner.py         # Route detection
├── orchestrator.py    # Main execution
└── main.py            # CLI
```

## New Features (v0.3.0)

### 0. Scenario-Aware Storyboarding & Review
```python
from agent_video_pipeline import run_pipeline

result = run_pipeline(
    user_prompt="先生成漫剧网格图，再拆成分镜格子做连续视频",
    duration_s=18.0,
    scenario_hint="comic_drama",
    quality_tier="quality",
    dry_run=True,
)

plan = result["plan"]
# plan 中将包含:
# - scenario / storyboard_mode
# - storyboard（分镜节拍）
# - agent_roles（Agent 分工）
# - generation_chain（生成链路）
# - review_plan（审核链路）
```


### 1. Wan2.2 + LightX2V Support
```python
from agent_video_pipeline.adapters import WanLightX2VStage

wan22 = WanLightX2VStage()

# I2V generation
wan22.generate_i2v(
    prompt="A cat playing piano",
    first_frame="first.png",
    output_path="output.mp4",
)

# FLF2V (first-last-frame) for controlled motion
wan22.generate_flf2v(
    prompt="Camera pan from left to right",
    first_frame="start.png",
    last_frame="end.png",
    output_path="output.mp4",
)
```

### 2. Model Selection Strategy
```python
from agent_video_pipeline.configs import ModelSelector, VideoBackend

selector = ModelSelector()

# Auto-detect best backend
backend = selector.select_video_backend(
    prompt="A talking head video with dialogue",
    duration=5.0,
    has_last_frame=False,
)

# Content-aware routing
content = selector.detect_content_type(prompt)
# Returns: {"has_dialogue": True, "is_music_video": False, "needs_precise_motion": False}
```

### 3. Audio-Video Synchronization
```python
from agent_video_pipeline.editing import (
    sync_video_audio,
    align_for_lipsync,
    adjust_video_speed,
)

# Auto-sync with strategy selection
output, info = sync_video_audio(
    video="video.mp4",
    audio="audio.wav",
    output="synced.mp4",
    strategy="auto",  # "stretch_video" | "stretch_audio" | "auto"
    max_speed_change=0.2,
)

# Lip-sync optimized
output, info = align_for_lipsync(
    video="lipsync.mp4",
    audio="tts.wav",
    output="final.mp4",
    tolerance=0.05,
)

# Video speed adjustment
adjust_video_speed(
    video="input.mp4",
    speed_factor=1.2,  # 20% faster
    output="faster.mp4",
    audio_pitch_correction=True,
)
```

## CLI Usage

```bash
# Full pipeline
python -m agent_video_pipeline run \
  --prompt "一个主播在讲解产品" \
  --duration 5 \
  --project-dir /tmp/demo

# Force Wan2.2 I2V
python -m agent_video_pipeline run \
  --prompt "A beautiful sunset" \
  --duration 5 \
  --route wan22_i2v

# Wan2.2 FLF2V (needs last frame)
python -m agent_video_pipeline run \
  --prompt "Camera movement" \
  --duration 8 \
  --route wan22_flf2v

# Image generation
python -m agent_video_pipeline image \
  --prompt "A red apple" \
  --output frame.png

# Lip-sync with A/V alignment
python -m agent_video_pipeline lipsync \
  --video raw.mp4 \
  --audio tts.wav \
  --output synced.mp4

# Video editing
python -m agent_video_pipeline edit concat \
  --video clip1.mp4 --video clip2.mp4 \
  --output merged.mp4
```

## Pipeline Execution

```python
from agent_video_pipeline import run_pipeline

result = run_pipeline(
    user_prompt="...",
    duration_s=10.0,
    
    # Backend selection
    preferred_route="auto",      # or specific route
    video_backend=None,          # or "wan22_i2v", "hyvideo", "magihuman"
    
    # Feature toggles
    enable_lipsync=True,         # Run LatentSync
    enable_av_sync=True,         # Auto A/V alignment
    
    # Execution options
    dry_run=False,               # Plan only
    skip_failed_shots=True,      # Continue on error
    
    # Output
    project_id="demo_001",
    project_dir="/tmp/demo",
)

# Result contains:
# - success: bool
# - output_video: str
# - project_dir: str
# - failed_shots: List[int]
# - plan: Dict
```

## Configuration

```yaml
# config.yaml
video_backends:
  - hyvideo_1.5
  - wan22_i2v
  - wan22_flf2v
  - magihuman

# Duration thresholds
max_hyvideo_duration: 5.0
max_wan22_duration: 5.0
use_flf2v_threshold: 10.0

# Quality settings
prefer_quality: true
enable_fallback: true
enable_av_sync: true

# Resource
available_gpus: 8
use_fast_sync: true
```

## Environment Variables

```bash
# Model paths
export WAN22_MODEL_PATH="/tmp/it2av/models/Wan2.2-I2V-A14B"
export LIGHTX2V_PATH="/path/to/LightX2V-main"
export HYVIDEO_MODEL_PATH="/path/to/HunyuanVideo-1.5"
export LATENTSYNC_CKPT_PATH="/path/to/latentsync_unet.pt"

# GPU settings
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NUM_GPUS=8
```

## Requirements

- Python 3.10+
- FFmpeg with libx264, aac
- CUDA 11.8+ (tested on 8x H20 95GB)
- Model weights synced via `fast_sync.py`

## Backend Comparison

| Backend | Speed | Quality | Duration | GPU Memory |
|---------|-------|---------|----------|------------|
| HunyuanVideo 1.5 | Medium | Good | ≤5s | ~40GB |
| Wan2.2 I2V | Medium | Excellent | ≤5s | ~60GB |
| Wan2.2 FLF2V | Medium | Excellent | ≤5s | ~60GB |
| MagiHuman | Fast | Good | ≤8s | ~50GB |

## License

Research use only.
