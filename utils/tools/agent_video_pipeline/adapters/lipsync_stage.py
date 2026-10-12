"""Adapters for LatentSync lip-sync (audio-driven face animation)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


class LatentSyncStage:
    """Adapter for LatentSync audio-driven lip-sync.

    Wraps ``scripts/inference.py`` inside the LatentSync package.
    """

    def __init__(
        self,
        script_path: str | None = None,
        unet_config: str | None = None,
        inference_ckpt_env: str = "LATENTSYNC_CKPT_PATH",
        python_bin: str | None = None,
    ):
        tools_root = Path(__file__).resolve().parents[1]
        code_root = tools_root.parents[3]
        self.script_path = Path(
            script_path or code_root / "utils/video/LatentSync/scripts/inference.py"
        )
        self.unet_config = unet_config or "configs/unet/stage2_512.yaml"
        self.inference_ckpt_env = inference_ckpt_env
        self.python_bin = python_bin or sys.executable

    def generate(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        inference_steps: int = 25,
        guidance_scale: float = 1.5,
        temp_dir: str | None = None,
        enable_deepcache: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run LatentSync lip-sync inference.

        Args:
            video_path: Path to the source video (without audio).
            audio_path: Path to the TTS audio file to drive lip-sync.
            video_out_path: Where to save the lip-synced output video.
            inference_steps: Number of denoising steps (default 25).
            guidance_scale: Guidance scale (default 1.5).
            temp_dir: Temporary directory for intermediate files.
            enable_deepcache: Whether to enable deep-cache acceleration.

        Returns:
            CompletedProcess object.
        """
        out_dir = Path(video_out_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_bin,
            str(self.script_path),
            "--unet_config_path", self.unet_config,
            "--inference_ckpt_path", f"${{{self.inference_ckpt_env}}}",
            "--video_path", str(video_path),
            "--audio_path", str(audio_path),
            "--video_out_path", str(video_out_path),
            "--inference_steps", str(inference_steps),
            "--guidance_scale", str(guidance_scale),
        ]
        if temp_dir:
            cmd += ["--temp_dir", str(temp_dir)]
        if enable_deepcache:
            cmd.append("--enable_deepcache")
        return subprocess.run(cmd, check=True)
