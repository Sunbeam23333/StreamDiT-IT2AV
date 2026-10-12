"""Adapters for audio generation (BGM, SFX, voice-overs)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


class BananaImageStage:
    """Adapter for Banana (FLUX) image generation.

    Wraps ``utils/image/Banana/test_banana.py``.
    """

    def __init__(
        self,
        script_path: str | None = None,
        python_bin: str | None = None,
    ):
        tools_root = Path(__file__).resolve().parents[1]
        code_root = tools_root.parents[3]
        self.script_path = Path(script_path or code_root / "utils/image/Banana/test_banana.py")
        self.python_bin = python_bin or sys.executable

    def generate_single(
        self,
        prompt: str,
        output: str,
        size: str = "1024x1024",
        num_steps: int = 30,
        seed: int = 42,
    ) -> subprocess.CompletedProcess:
        """Generate an image via Banana/FLUX.

        Args:
            prompt: Text prompt.
            output: Output image path.
            size: Image resolution (e.g. "1024x1024", "768x1280").
            num_steps: Number of diffusion steps.
            seed: Random seed.

        Returns:
            CompletedProcess object.
        """
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_bin,
            str(self.script_path),
            "--prompt", prompt,
            "--output", str(output),
            "--size", size,
            "--num_steps", str(num_steps),
            "--seed", str(seed),
        ]
        return subprocess.run(cmd, check=True)

    def generate_from_prompt_file(
        self,
        prompt_file: str,
        output: str,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """Generate image reading prompt from a text file."""
        prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
        return self.generate_single(prompt=prompt, output=output, **kwargs)


class BGMStage:
    """Adapter for BGM generation via a text prompt.

    Currently wraps Hunyuan-TTS as a placeholder for the text→BGM task
    (produces a spoken narration that can serve as a vocal track placeholder).

    TODO: Replace with a dedicated BGM model such as ACE-Step or Stable Audio Open.
    """

    def __init__(
        self,
        script_path: str | None = None,
        python_bin: str | None = None,
    ):
        tools_root = Path(__file__).resolve().parents[1]
        code_root = tools_root.parents[3]
        self.script_path = Path(script_path or code_root / "utils/audio/Hunyuan-tts/hunyuan_tts.py")
        self.python_bin = python_bin or sys.executable

    def generate(
        self,
        text_or_prompt: str,
        output: str,
        voice: str = "wenrouxuemei",
        sample_rate: int = 24000,
    ) -> subprocess.CompletedProcess:
        """Placeholder BGM generation via TTS adapter.

        Args:
            text_or_prompt: Text to synthesise (used as placeholder).
            output: Output audio path.
            voice: TTS voice preset.
            sample_rate: Audio sample rate.

        Returns:
            CompletedProcess object.
        """
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_bin,
            str(self.script_path),
            "--text", text_or_prompt,
            "--output", str(output),
            "--voice", voice,
            "--sample_rate", str(sample_rate),
            "--stream",
        ]
        return subprocess.run(cmd, check=True)


class SFXStage:
    """Adapter for sound-effects generation.

    Placeholder that uses FFmpeg to generate a silent/ambient track.
    TODO: Replace with a dedicated SFX model.
    """

    def generate_silent(
        self,
        duration_s: float,
        output: str,
    ):
        """Create a silent audio track of the given duration."""
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-f", "lavfi",
            "-i", f"anullsrc=r=24000:cl=mono:d={duration_s}",
            "-c:a", "aac",
            "-y", str(out),
        ]
        return subprocess.run(cmd, check=False)
