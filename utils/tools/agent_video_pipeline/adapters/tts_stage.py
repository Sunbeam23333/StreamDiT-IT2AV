from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class HunyuanTTSStage:
    def __init__(self, script_path: str | None = None, python_bin: str | None = None):
        tools_root = Path(__file__).resolve().parents[1]
        code_root = tools_root.parents[3]
        self.script_path = Path(script_path or code_root / "utils/audio/Hunyuan-tts/hunyuan_tts.py")
        self.python_bin = python_bin or sys.executable

    def generate_from_text_file(
        self,
        text_file: str,
        output: str,
        voice: str = "wenrouxuemei",
        sample_rate: int = 24000,
        stream: bool = True,
    ) -> subprocess.CompletedProcess:
        text = Path(text_file).read_text(encoding="utf-8").strip()
        return self.generate_single(text=text, output=output, voice=voice, sample_rate=sample_rate, stream=stream)

    def generate_single(
        self,
        text: str,
        output: str,
        voice: str = "wenrouxuemei",
        sample_rate: int = 24000,
        stream: bool = True,
    ) -> subprocess.CompletedProcess:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_bin,
            str(self.script_path),
            "--text",
            text,
            "--output",
            str(output_path),
            "--voice",
            voice,
            "--sample_rate",
            str(sample_rate),
        ]
        if stream:
            cmd.append("--stream")

        return subprocess.run(cmd, check=True)
