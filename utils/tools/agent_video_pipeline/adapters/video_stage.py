"""Adapters for video generation backends (MagiHuman, HyVideo I2V)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


class MagiHumanStage:
    """Adapter for daVinci-MagiHuman video generation.

    Wraps ``script/digit/pipeline3_magihuman/run_single.sh``.
    """

    def __init__(
        self,
        script_path: str | None = None,
        python_bin: str | None = None,
        conda_env: str = "it2av",
        num_gpus: int = 8,
        skip_sync: bool = False,
    ):
        tools_root = Path(__file__).resolve().parents[1]
        code_root = tools_root.parents[3]
        self.script_path = Path(script_path or code_root / "script/digit/pipeline3_magihuman/run_single.sh")
        self.python_bin = python_bin or sys.executable
        self.conda_env = conda_env
        self.num_gpus = num_gpus
        self.skip_sync = skip_sync

    def generate(
        self,
        image_path: str,
        prompt: str,
        output_dir: str,
        seconds: int = 5,
        mode: str = "sr1080p",
        seed: int = 42,
        prompt_file: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run MagiHuman inference.

        Args:
            image_path: Path to source image (first frame).
            prompt: English prompt describing the video.
            output_dir: Directory for output files.
            seconds: Video duration in seconds.
            mode: One of "distill" (256p), "sr540p", "sr1080p" (default).
            seed: Random seed.
            prompt_file: Optional path to prompt text file (takes precedence).

        Returns:
            CompletedProcess object.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cmd = [
            "bash",
            str(self.script_path),
            "--image", str(image_path),
            "--mode", mode,
            "--seconds", str(seconds),
            "--seed", str(seed),
            "--num_gpus", str(self.num_gpus),
            "--output_dir", str(output_dir),
        ]
        if prompt_file:
            cmd += ["--prompt_file", str(prompt_file)]
        else:
            cmd += ["--prompt", prompt]
        if self.skip_sync:
            cmd.append("--skip_sync")
        return subprocess.run(cmd, check=True)


class HyVideoStage:
    """Adapter for HunyuanVideo-1.5 I2V generation.

    Wraps the ``torchrun`` command that calls generate.py inside
    ``utils/video/HunyuanVideo-1.5``.
    """

    def __init__(
        self,
        script_path: str | None = None,
        python_bin: str | None = None,
        num_gpus: int = 8,
        model_path_env: str = "HYVIDEO_MODEL_PATH",
    ):
        tools_root = Path(__file__).resolve().parents[1]
        code_root = tools_root.parents[3]
        self.script_path = Path(
            script_path or code_root / "utils/video/HunyuanVideo-1.5/generate.py"
        )
        self.python_bin = python_bin or sys.executable
        self.num_gpus = num_gpus
        self.model_path_env = model_path_env

    def generate(
        self,
        prompt: str,
        image_path: str,
        output_path: str,
        resolution: str = "720p",
        aspect_ratio: str = "9:16",
        seed: int = 42,
        cfg_distilled: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run HunyuanVideo-1.5 I2V inference.

        Args:
            prompt: Text prompt describing the video.
            image_path: Path to first-frame image.
            output_path: Where to save the raw video.
            resolution: "720p" or "1080p".
            aspect_ratio: "9:16" or "16:9".
            seed: Random seed.
            cfg_distilled: Whether to use CFG distillation.

        Returns:
            CompletedProcess object.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "torchrun",
            "--nproc_per_node", str(self.num_gpus),
            str(self.script_path),
            "--prompt", prompt,
            "--image_path", str(image_path),
            "--resolution", resolution,
            "--aspect_ratio", aspect_ratio,
            "--seed", str(seed),
            "--rewrite", "false",
            "--cfg_distilled", str(cfg_distilled).lower(),
            "--enable_step_distill", "false",
            "--sparse_attn", "false",
            "--use_sageattn", "false",
            "--offloading", "false",
            "--group_offloading", "false",
            "--overlap_group_offloading", "false",
            "--enable_cache", "true",
            "--cache_type", "deepcache",
            "--sr", "true",
            "--save_pre_sr_video", "true",
            "--output_path", str(output_path),
            "--model_path", f"${{{self.model_path_env}}}",
        ]
        return subprocess.run(cmd, check=True)
