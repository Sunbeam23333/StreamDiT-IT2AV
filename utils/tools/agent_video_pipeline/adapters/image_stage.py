from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class HunyuanImageStage:
    def __init__(self, script_path: str | None = None, python_bin: str | None = None):
        tools_root = Path(__file__).resolve().parents[1]
        code_root = tools_root.parents[3]
        self.script_path = Path(script_path or code_root / "utils/image/Hunyuan-Image/test_hy_t2i.py")
        self.python_bin = python_bin or sys.executable

    def generate_from_prompt_file(
        self,
        prompt_file: str,
        output: str,
        size: str = "768x1280",
        revise: bool = True,
        thinking: bool = False,
    ) -> subprocess.CompletedProcess:
        prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
        return self.generate_single(prompt=prompt, output=output, size=size, revise=revise, thinking=thinking)

    def generate_single(
        self,
        prompt: str,
        output: str,
        size: str = "768x1280",
        revise: bool = True,
        thinking: bool = False,
    ) -> subprocess.CompletedProcess:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_bin,
            str(self.script_path),
            "--prompt",
            prompt,
            "--output",
            str(output_path),
            "--size",
            size,
        ]
        if not revise:
            cmd.append("--no-revise")
        if thinking:
            cmd.append("--thinking")

        return subprocess.run(cmd, check=True)
