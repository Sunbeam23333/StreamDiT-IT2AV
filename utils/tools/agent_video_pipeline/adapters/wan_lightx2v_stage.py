"""Adapter for Wan2.2 + LightX2V video generation (I2V & FLF2V modes)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional


class WanLightX2VStage:
    """Adapter for Wan2.2 MoE video generation via LightX2V framework.

    Supports two modes:
    - i2v: Image-to-Video (first-frame conditioned)
    - flf2v: First-Last-Frame-to-Video (start and end frame conditioned)

    Wraps the torchrun command for LightX2V-main inference.
    """

    def __init__(
        self,
        model_path: str | None = None,
        lightx2v_path: str | None = None,
        python_bin: str | None = None,
        num_gpus: int = 8,
        config_name: str = "wan_moe_i2v_h20_8gpu.json",
        skip_sync: bool = False,
    ):
        """Initialize Wan2.2 LightX2V adapter.

        Args:
            model_path: Path to Wan2.2-I2V-A14B model directory.
                       Defaults to env var WAN22_MODEL_PATH or auto-detected path.
            lightx2v_path: Path to LightX2V-main code directory.
                          Defaults to env var LIGHTX2V_PATH or auto-detected path.
            python_bin: Python executable path.
            num_gpus: Number of GPUs for torchrun (default 8).
            config_name: Config JSON filename under lightx2v_path/configs/wan22/.
            skip_sync: If True, skip model sync step (assume already cached).
        """
        tools_root = Path(__file__).resolve().parents[1]
        code_root = tools_root.parents[3]

        # Resolve model path
        if model_path:
            self.model_path = Path(model_path)
        elif os.getenv("WAN22_MODEL_PATH"):
            self.model_path = Path(os.getenv("WAN22_MODEL_PATH"))
        else:
            # Default to local cache path
            self.model_path = Path(os.environ.get("IT2AV_LOCAL_ROOT", "/tmp/it2av")) / "models" / "Wan2.2-I2V-A14B"

        # Resolve LightX2V path
        if lightx2v_path:
            self.lightx2v_path = Path(lightx2v_path)
        elif os.getenv("LIGHTX2V_PATH"):
            self.lightx2v_path = Path(os.getenv("LIGHTX2V_PATH"))
        else:
            self.lightx2v_path = code_root / "utils/video/LightX2V-main"

        self.python_bin = python_bin or sys.executable
        self.num_gpus = num_gpus
        self.config_name = config_name
        self.skip_sync = skip_sync

        # Validate paths
        if not self.lightx2v_path.exists():
            raise FileNotFoundError(f"LightX2V path not found: {self.lightx2v_path}")

    def _sync_model(self) -> None:
        """Sync model from group disk to local cache if needed."""
        if self.skip_sync or self.model_path.exists():
            return

        tools_root = Path(__file__).resolve().parents[1]
        code_root = tools_root.parents[3]
        fast_sync = code_root / "fast_sync.py"

        group_model = Path(os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace"))) / "models" / "Wan2.2-I2V-A14B"
        if not group_model.exists():
            print(f"Warning: Group model path not found: {group_model}, skipping sync")
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_bin,
            str(fast_sync),
            str(group_model),
            str(self.model_path),
            "--workers", "32",
            "--skip-existing",
        ]
        subprocess.run(cmd, check=True)

    def _get_config_path(self, task: Literal["i2v", "flf2v"]) -> Path:
        """Get appropriate config file path for the task."""
        if task == "flf2v":
            config_file = "wan_moe_flf2v_h20_8gpu.json"
        else:
            config_file = self.config_name
        return self.lightx2v_path / "configs" / "wan22" / config_file

    def generate_i2v(
        self,
        prompt: str,
        first_frame: str,
        output_path: str,
        seed: int = 42,
        negative_prompt: str | None = None,
        resolution: str = "720p",
    ) -> subprocess.CompletedProcess:
        """Generate video from first frame using I2V mode.

        Args:
            prompt: Text prompt describing the video motion.
            first_frame: Path to the first frame image.
            output_path: Path to save the generated video.
            seed: Random seed for reproducibility.
            negative_prompt: Negative prompt to avoid unwanted content.
            resolution: Target resolution (720p or 1080p).

        Returns:
            CompletedProcess object.
        """
        self._sync_model()

        config_path = self._get_config_path("i2v")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        default_negative = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
            "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )

        cmd = [
            "torchrun",
            "--nproc_per_node", str(self.num_gpus),
            "-m", "lightx2v.infer",
            "--model_cls", "wan2.2_moe",
            "--task", "i2v",
            "--model_path", str(self.model_path),
            "--config_json", str(config_path),
            "--seed", str(seed),
            "--prompt", prompt,
            "--negative_prompt", negative_prompt or default_negative,
            "--image_path", str(first_frame),
            "--save_result_path", str(output_path),
        ]

        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.num_gpus))

        return subprocess.run(cmd, check=True, env=env)

    def generate_flf2v(
        self,
        prompt: str,
        first_frame: str,
        last_frame: str,
        output_path: str,
        seed: int = 42,
        negative_prompt: str | None = None,
        resolution: str = "720p",
    ) -> subprocess.CompletedProcess:
        """Generate video from first and last frames using FLF2V mode.

        This mode provides better temporal consistency by conditioning on both
        start and end frames. Useful for controlled camera movements or
        character animations.

        Args:
            prompt: Text prompt describing the video motion.
            first_frame: Path to the first frame image.
            last_frame: Path to the last frame (target end frame) image.
            output_path: Path to save the generated video.
            seed: Random seed for reproducibility.
            negative_prompt: Negative prompt to avoid unwanted content.
            resolution: Target resolution (720p or 1080p).

        Returns:
            CompletedProcess object.
        """
        self._sync_model()

        config_path = self._get_config_path("flf2v")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        default_negative = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
            "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )

        cmd = [
            "torchrun",
            "--nproc_per_node", str(self.num_gpus),
            "-m", "lightx2v.infer",
            "--model_cls", "wan2.2_moe",
            "--task", "flf2v",
            "--model_path", str(self.model_path),
            "--config_json", str(config_path),
            "--seed", str(seed),
            "--prompt", prompt,
            "--negative_prompt", negative_prompt or default_negative,
            "--image_path", str(first_frame),
            "--last_frame_path", str(last_frame),
            "--save_result_path", str(output_path),
        ]

        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.num_gpus))

        return subprocess.run(cmd, check=True, env=env)

    def generate_last_frame(
        self,
        first_frame: str,
        prompt: str,
        output_path: str,
        seed: int = 42,
    ) -> str:
        """Generate a plausible last frame from first frame using image generation.

        This is a helper method that uses an image generation model to create
        an end frame for FLF2V mode. Falls back to using the first frame if
        image generation is not available.

        Args:
            first_frame: Path to first frame image.
            prompt: Text prompt (used to guide last frame generation).
            output_path: Path to save the generated last frame.
            seed: Random seed.

        Returns:
            Path to the generated last frame.
        """
        # For now, use simple image processing to create a variation
        # In practice, this could call an image-to-image model
        import shutil

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy first frame as last frame (simple fallback)
        # TODO: Integrate with image generation model for better results
        shutil.copy(first_frame, output_path)
        return str(output_path)
