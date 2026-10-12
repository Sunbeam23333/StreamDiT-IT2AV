"""Model selection strategies and configuration for video generation pipeline.

Provides intelligent routing between different generation backends:
- Image: Hunyuan-Image (local) vs Banana (remote FLUX)
- Video: HyVideo-1.5 vs Wan2.2-LightX2V vs MagiHuman
- Audio: Hunyuan-TTS vs other TTS backends
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional


class VideoBackend(Enum):
    """Available video generation backends."""
    HYVIDEO_15 = "hyvideo_1.5"
    WAN22_I2V = "wan22_i2v"
    WAN22_FLF2V = "wan22_flf2v"
    MAGIHUMAN = "magihuman"


class ImageBackend(Enum):
    """Available image generation backends."""
    HUNYUAN_LOCAL = "hunyuan_local"
    BANANA_REMOTE = "banana_remote"


class AudioBackend(Enum):
    """Available audio generation backends."""
    HUNYUAN_TTS = "hunyuan_tts"
    COSYVOICE = "cosyvoice"


@dataclass
class ModelSelectionConfig:
    """Configuration for model selection strategy."""

    image_backends: List[ImageBackend] = field(
        default_factory=lambda: [ImageBackend.HUNYUAN_LOCAL, ImageBackend.BANANA_REMOTE]
    )
    video_backends: List[VideoBackend] = field(
        default_factory=lambda: [
            VideoBackend.HYVIDEO_15,
            VideoBackend.WAN22_I2V,
            VideoBackend.WAN22_FLF2V,
            VideoBackend.MAGIHUMAN,
        ]
    )
    audio_backends: List[AudioBackend] = field(
        default_factory=lambda: [AudioBackend.HUNYUAN_TTS]
    )

    max_hyvideo_duration: float = 5.0
    max_wan22_duration: float = 5.0
    use_flf2v_threshold: float = 10.0

    prefer_quality: bool = True
    enable_fallback: bool = True

    available_gpus: int = 8
    use_fast_sync: bool = True


class ModelSelector:
    """Intelligent model selector for video generation pipeline."""

    def __init__(self, config: Optional[ModelSelectionConfig] = None):
        self.config = config or ModelSelectionConfig()
        self._availability_cache: Dict[str, bool] = {}

    def detect_content_type(self, prompt: str) -> Dict[str, bool]:
        """Analyze prompt to detect content characteristics."""
        prompt_lower = prompt.lower()

        speaking_hints = [
            "口播", "对口型", "数字人", "主播", "讲解", "独白", "采访",
            "说话", "台词", "dialogue", "talking", "speaking", "narrator",
        ]
        music_hints = [
            "mv", "音乐视频", "氛围片", "bgm", "音乐", "表演", "performance",
            "music video", "concert", "singing",
        ]
        motion_hints = [
            "精确", "准确", "一致", "连续", "流畅", "smooth", "precise",
            "consistent", "continuous", "camera movement", "pan", "zoom",
        ]

        return {
            "has_dialogue": any(h in prompt_lower for h in speaking_hints),
            "is_music_video": any(h in prompt_lower for h in music_hints),
            "needs_precise_motion": any(h in prompt_lower for h in motion_hints),
        }

    def detect_scenario_type(self, prompt: str) -> str:
        """Infer a coarse scenario tag for downstream planning."""
        prompt_lower = prompt.lower()
        scenario_keywords = {
            "comic_drama": ["漫剧", "漫画", "网格图", "格子", "comic", "manga", "panel"],
            "beauty_glamour": ["美女", "美妆", "时尚", "写真", "beauty", "fashion", "glamour"],
            "historical_reenactment": ["历史", "古风", "朝代", "宫廷", "historical", "dynasty"],
            "narration_avatar": ["口播", "主播", "数字人", "讲解", "speaking", "narrator"],
        }
        best = "generic"
        best_score = 0
        for scenario, keywords in scenario_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > best_score:
                best = scenario
                best_score = score
        return best

    def select_video_backend(
        self,
        prompt: str,
        duration: float,
        has_last_frame: bool = False,
        quality_priority: Literal["speed", "balanced", "quality"] = "balanced",
    ) -> VideoBackend:
        """Select appropriate video generation backend."""
        content = self.detect_content_type(prompt)
        scenario = self.detect_scenario_type(prompt)

        if content["is_music_video"] and not content["has_dialogue"]:
            return VideoBackend.MAGIHUMAN

        if has_last_frame and (
            duration > self.config.use_flf2v_threshold
            or content["needs_precise_motion"]
            or scenario in {"comic_drama", "historical_reenactment"}
        ):
            return VideoBackend.WAN22_FLF2V

        if quality_priority == "quality" and scenario in {"comic_drama", "beauty_glamour"}:
            return VideoBackend.WAN22_I2V

        for backend in self.config.video_backends:
            if backend == VideoBackend.HYVIDEO_15 and duration <= self.config.max_hyvideo_duration:
                return backend
            if backend == VideoBackend.WAN22_I2V and duration <= self.config.max_wan22_duration:
                return backend
            if backend == VideoBackend.WAN22_FLF2V and has_last_frame:
                return backend
            if backend == VideoBackend.MAGIHUMAN:
                return backend

        return self.config.video_backends[0]

    def select_image_backend(
        self,
        prompt: str,
        require_fast: bool = False,
    ) -> ImageBackend:
        """Select appropriate image generation backend."""
        if require_fast and ImageBackend.BANANA_REMOTE in self.config.image_backends:
            return ImageBackend.BANANA_REMOTE
        return self.config.image_backends[0]

    def get_backend_for_route(self, route: str) -> VideoBackend:
        """Map route or backend alias to VideoBackend."""
        route_map = {
            "hyvideo": VideoBackend.HYVIDEO_15,
            "hyvideo_1.5": VideoBackend.HYVIDEO_15,
            "hyvideo_lipsync": VideoBackend.HYVIDEO_15,
            "hyvideo_longform": VideoBackend.HYVIDEO_15,
            "wan22_i2v": VideoBackend.WAN22_I2V,
            "wan22_lipsync": VideoBackend.WAN22_I2V,
            "wan22_flf2v": VideoBackend.WAN22_FLF2V,
            "magihuman": VideoBackend.MAGIHUMAN,
            "magihuman_av": VideoBackend.MAGIHUMAN,
        }
        return route_map.get(route, VideoBackend.HYVIDEO_15)

    def build_fallback_chain(self, primary: VideoBackend) -> List[VideoBackend]:
        """Build fallback chain for a primary backend."""
        chain = [primary]
        for backend in list(VideoBackend):
            if backend != primary:
                chain.append(backend)
        return chain

    def get_recommended_resolution(
        self,
        backend: VideoBackend,
        aspect_ratio: str = "9:16",
    ) -> str:
        """Get recommended resolution for a backend."""
        resolutions = {
            VideoBackend.HYVIDEO_15: "720p",
            VideoBackend.WAN22_I2V: "720p",
            VideoBackend.WAN22_FLF2V: "720p",
            VideoBackend.MAGIHUMAN: "1080p",
        }
        return resolutions.get(backend, "720p")


def create_selector_for_env() -> ModelSelector:
    """Create a model selector configured for the current environment."""
    import os

    config = ModelSelectionConfig()
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        config.available_gpus = len(cuda_visible.split(","))
    else:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                check=True,
            )
            config.available_gpus = len(result.stdout.strip().split("\n"))
        except Exception:
            config.available_gpus = 8

    return ModelSelector(config)
