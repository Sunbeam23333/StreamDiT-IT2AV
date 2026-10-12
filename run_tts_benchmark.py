#!/usr/bin/env python3
"""
TTS Benchmark - 统一 TTS 推理脚本 (CosyVoice 备选方案).

支持模型:
  - f5tts        : F5-TTS (Flow Matching + DiT, 零样本克隆)
  - fish_speech  : Fish-Speech 1.5 (VQGAN + LLM, 极速推理)
  - spark_tts    : Spark-TTS (LLM-based, 音高/语速可控)
  - chattts      : ChatTTS (对话场景, 自然韵律)

GitHub:
  - https://github.com/SWivid/F5-TTS
  - https://github.com/fishaudio/fish-speech
  - https://github.com/SparkAudio/Spark-TTS
  - https://github.com/2noise/ChatTTS

用法:
    python run_tts_benchmark.py --model f5tts --model_path /path/to/F5-TTS
    python run_tts_benchmark.py --model fish_speech --model_path /path/to/Fish-Speech-1.5
    python run_tts_benchmark.py --model spark_tts --model_path /path/to/Spark-TTS-0.5B
    python run_tts_benchmark.py --model chattts --model_path /path/to/ChatTTS
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from importlib import util as importlib_util
from typing import Any

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

F5_DEFAULT_REF_TEXT = {
    "zh": "对，这就是我，万人敬仰的太乙真人。",
    "en": "Some call me nature, others call me mother nature.",
}


def _first_existing_path(candidates: list[str]) -> str | None:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _prepend_sys_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _build_repo_dir_candidates(code_dir: str, repo_dir_name: str, model_path: str) -> list[str]:
    script_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        code_dir,
        os.path.join(script_root, "utils", "audio", repo_dir_name),
        os.path.join(os.getcwd(), "utils", "audio", repo_dir_name),
    ]

    model_path_abs = os.path.abspath(model_path)
    model_marker = f"{os.sep}models{os.sep}Audio{os.sep}"
    if model_marker in model_path_abs:
        local_base = model_path_abs.split(model_marker, 1)[0]
        candidates.extend([
            os.path.join(local_base, "code", "utils", "audio", repo_dir_name),
            os.path.join(local_base, "distillation", "code", "utils", "audio", repo_dir_name),
        ])

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate_abs = os.path.abspath(candidate)
        if candidate_abs not in seen:
            seen.add(candidate_abs)
            unique_candidates.append(candidate_abs)
    return unique_candidates


def _resolve_code_dir(label: str, code_dir: str, repo_dir_name: str, model_path: str,
                      required_relpaths: list[str]) -> str:
    candidates = _build_repo_dir_candidates(code_dir=code_dir, repo_dir_name=repo_dir_name, model_path=model_path)
    for candidate in candidates:
        if all(os.path.exists(os.path.join(candidate, relpath)) for relpath in required_relpaths):
            return candidate

    required_desc = ", ".join(required_relpaths)
    checked_desc = "\n  - ".join(candidates)
    message = f"[{label}] 未找到可用代码目录，必须包含: {required_desc}\n[{label}] 已检查路径:\n  - {checked_desc}"
    raise FileNotFoundError(message)


# ============================================================================
# F5-TTS 推理
# ============================================================================
class F5TTSRunner:
    """
    F5-TTS: Flow Matching + DiT, 非自回归, 零样本3秒声音克隆.
    GitHub: https://github.com/SWivid/F5-TTS
    """

    def __init__(self, model_path: str, code_dir: str, device: str = "cuda:0", language: str = "zh"):
        self.model_path = model_path
        self.code_dir = _resolve_code_dir(
            label="F5-TTS",
            code_dir=code_dir,
            repo_dir_name="F5-TTS",
            model_path=model_path,
            required_relpaths=["src/f5_tts/api.py", "src/f5_tts/infer/examples/basic"],
        )
        self.device = device
        self.language = language

        src_dir = os.path.join(self.code_dir, "src")
        _prepend_sys_path(self.code_dir)
        _prepend_sys_path(src_dir)

        logging.info("[F5-TTS] Using code dir: %s", self.code_dir)
        logging.info("[F5-TTS] Loading model from %s", model_path)
        try:
            F5TTS = __import__("f5_tts.api", fromlist=["F5TTS"]).F5TTS
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"[F5-TTS] 无法导入 `f5_tts`。请确认代码目录完整，或执行 `pip install -e {self.code_dir}`。"
            ) from exc

        ckpt_file = _first_existing_path([
            os.path.join(model_path, "F5TTS_v1_Base", "model_1250000.safetensors"),
            os.path.join(model_path, "F5TTS_Base", "model_1200000.safetensors"),
        ])
        if ckpt_file is None:
            raise FileNotFoundError(
                f"[F5-TTS] 未找到 checkpoint。期望存在于 {model_path}/F5TTS_v1_Base 或 {model_path}/F5TTS_Base"
            )

        model_name = "F5TTS_v1_Base" if "F5TTS_v1_Base" in ckpt_file else "F5TTS_Base"
        vocoder_local_path = _first_existing_path([
            os.path.join(model_path, "vocos-mel-24khz"),
            os.path.join(model_path, "checkpoints", "vocos-mel-24khz"),
            os.path.join(self.code_dir, "checkpoints", "vocos-mel-24khz"),
        ])

        self.tts = F5TTS(
            model=model_name,
            ckpt_file=ckpt_file,
            vocoder_local_path=vocoder_local_path,
            device=device,
        )
        logging.info("[F5-TTS] Model loaded successfully (model=%s, ckpt=%s)", model_name, ckpt_file)

    def _get_default_ref_audio(self) -> str:
        ref_name = "basic_ref_zh.wav" if self.language == "zh" else "basic_ref_en.wav"
        return os.path.join(self.code_dir, "src", "f5_tts", "infer", "examples", "basic", ref_name)

    def synthesize(self, text: str, output_path: str, ref_audio: str | None = None,
                   ref_text: str | None = None) -> str:
        """合成语音."""
        ref_audio = ref_audio or self._get_default_ref_audio()
        ref_text = ref_text or F5_DEFAULT_REF_TEXT.get(self.language, F5_DEFAULT_REF_TEXT["zh"])

        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"[F5-TTS] 参考音频不存在: {ref_audio}")

        self.tts.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            file_wave=output_path,
            seed=None,
        )

        info = sf.info(output_path)
        logging.info("[F5-TTS] Generated: %s (%.1fs, ref=%s)", output_path, info.duration, os.path.basename(ref_audio))
        return output_path


# ============================================================================
# Fish-Speech 推理
# ============================================================================
class FishSpeechRunner:
    """
    Fish-Speech S2 当前 vendored 仓库与旧版 1.5 runner 不兼容。
    推荐改用官方 API Server / WebUI / 三段式 CLI。
    GitHub: https://github.com/fishaudio/fish-speech
    """

    def __init__(self, model_path: str, code_dir: str, device: str = "cuda:0"):
        raise RuntimeError(
            "[Fish-Speech] 当前仓库是 S2 版本，现有旧版 1.5 Python runner 已失效。请改用 `tools/api_server.py` + `tools/api_client.py`，或后续重写 S2 runner。"
        )

    def synthesize(self, text: str, output_path: str, ref_audio: str | None = None,
                   ref_text: str | None = None) -> str:
        raise RuntimeError("[Fish-Speech] 当前统一 benchmark 暂不直接支持，请使用官方 API Server 路线。")


# ============================================================================
# Spark-TTS 推理
# ============================================================================
class SparkTTSRunner:
    """
    Spark-TTS: LLM-based, 单流解耦语音令牌, 音高/语速可控.
    GitHub: https://github.com/SparkAudio/Spark-TTS
    """

    def __init__(self, model_path: str, code_dir: str, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.code_dir = _resolve_code_dir(
            label="Spark-TTS",
            code_dir=code_dir,
            repo_dir_name="Spark-TTS",
            model_path=model_path,
            required_relpaths=["cli/SparkTTS.py", "sparktts"],
        )

        _prepend_sys_path(self.code_dir)

        spark_cli_file = os.path.join(self.code_dir, "cli", "SparkTTS.py")
        if not os.path.isfile(spark_cli_file):
            raise FileNotFoundError(f"[Spark-TTS] 未找到入口文件: {spark_cli_file}")

        logging.info("[Spark-TTS] Loading model from %s", model_path)
        spec = importlib_util.spec_from_file_location("sparktts_cli_module", spark_cli_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"[Spark-TTS] 无法加载模块 spec: {spark_cli_file}")
        sparktts_module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(sparktts_module)
        SparkTTS = sparktts_module.SparkTTS

        self.model = SparkTTS(model_dir=model_path, device=device)
        logging.info("[Spark-TTS] Model loaded successfully")

    def synthesize(self, text: str, output_path: str, ref_audio: str | None = None,
                   ref_text: str | None = None, gender: str = "female",
                   pitch: str = "moderate", speed: str = "moderate") -> str:
        """合成语音."""
        with torch.no_grad():
            if ref_audio and os.path.exists(ref_audio):
                # 零样本声音克隆模式
                wav = self.model.inference(
                    text=text,
                    prompt_speech_path=ref_audio,
                    prompt_text=ref_text,
                )
            else:
                # 可控生成模式 (性别/音高/语速)
                wav = self.model.inference(
                    text=text,
                    gender=gender,
                    pitch=pitch,
                    speed=speed,
                )

        sr = 16000  # Spark-TTS 默认 16kHz
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        if wav.ndim > 1:
            wav = wav.squeeze()

        sf.write(output_path, wav, sr)
        logging.info("[Spark-TTS] Generated: %s (%.1fs)", output_path, len(wav) / sr)
        return output_path


# ============================================================================
# ChatTTS 推理
# ============================================================================
class ChatTTSRunner:
    """
    ChatTTS: 对话场景专用, 自然韵律, 支持停顿/笑声控制.
    GitHub: https://github.com/2noise/ChatTTS

    注意: ChatTTS 不支持零样本声音克隆, 但可通过 seed 固定音色.
    """

    def __init__(self, model_path: str, code_dir: str, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.code_dir = _resolve_code_dir(
            label="ChatTTS",
            code_dir=code_dir,
            repo_dir_name="ChatTTS",
            model_path=model_path,
            required_relpaths=["ChatTTS", "ChatTTS/norm.py"],
        )

        _prepend_sys_path(self.code_dir)

        logging.info("[ChatTTS] Loading model from %s", model_path)
        import ChatTTS as chattts_lib

        self.chat = chattts_lib.Chat()
        self.chat.load(
            source="custom",
            custom_path=model_path,
            device=device,
            compile=False,
        )
        logging.info("[ChatTTS] Model loaded successfully")

        # 固定一个好听的音色 seed
        self.speaker_seed = 42

    def synthesize(self, text: str, output_path: str, seed: int | None = None,
                   **kwargs) -> str:
        """合成语音."""
        seed = seed or self.speaker_seed

        # 设置音色
        torch.manual_seed(seed)
        params_infer = self.chat.InferCodeParams(
            spk_emb=self.chat.sample_random_speaker(),
            temperature=0.3,
            top_P=0.7,
            top_K=20,
        )

        # 设置 refine 参数 (控制停顿/口语化)
        params_refine = self.chat.RefineTextParams(
            prompt="[oral_2][laugh_0][break_4]",
        )

        wavs = self.chat.infer(
            [text],
            params_infer_code=params_infer,
            params_refine_text=params_refine,
        )

        audio = wavs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()

        sr = 24000  # ChatTTS 默认 24kHz
        sf.write(output_path, audio, sr)
        logging.info("[ChatTTS] Generated: %s (%.1fs, seed=%d)", output_path, len(audio) / sr, seed)
        return output_path


# ============================================================================
# 注册表
# ============================================================================
RUNNER_REGISTRY = {
    "f5tts": F5TTSRunner,
    "fish_speech": FishSpeechRunner,
    "spark_tts": SparkTTSRunner,
    "chattts": ChatTTSRunner,
}


# ============================================================================
# Benchmark 主流程
# ============================================================================
def load_prompts(prompts_file: str) -> list[dict[str, object]]:
    """加载 benchmark prompts."""
    with open(prompts_file) as f:
        data = json.load(f)
    return data.get("scenarios", [])


def run_benchmark(
    model_name: str,
    model_path: str,
    code_dir: str,
    prompts_file: str,
    output_dir: str,
    device: str = "cuda:0",
    language: str = "zh",
    scene_ids: list[str] | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    spark_gender: str = "female",
    spark_pitch: str = "moderate",
    spark_speed: str = "moderate",
    chattts_seed: int | None = 42,
) -> None:
    """运行 TTS benchmark."""
    os.makedirs(output_dir, exist_ok=True)

    runner_cls = RUNNER_REGISTRY.get(model_name)
    if runner_cls is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(RUNNER_REGISTRY.keys())}")

    # 初始化模型
    t0 = time.time()
    runner_init_kwargs = dict(model_path=model_path, code_dir=code_dir, device=device)
    if model_name == "f5tts":
        runner_init_kwargs["language"] = language
    runner: Any = runner_cls(**runner_init_kwargs)
    load_time = time.time() - t0
    logging.info("Model loaded in %.1fs", load_time)

    # 加载 prompts
    scenarios = load_prompts(prompts_file)
    if scene_ids:
        scenarios = [s for s in scenarios if s["id"] in scene_ids]

    logging.info("Running %d scenarios (language=%s)", len(scenarios), language)

    results = []
    total_duration = 0.0

    for idx, scenario in enumerate(scenarios, 1):
        scene_id = scenario["id"]

        # 获取 TTS 文本 (从 cosyvoice_tts 字段复用)
        raw_tts_info = scenario.get("cosyvoice_tts", {})
        tts_info = raw_tts_info if isinstance(raw_tts_info, dict) else {}
        text = tts_info.get(f"text_{language}", tts_info.get("text_zh", ""))

        if not text:
            logging.warning("[%d/%d] %s: no TTS text for language=%s, skip", idx, len(scenarios), scene_id, language)
            continue

        out_path = os.path.join(output_dir, f"{scene_id}_{model_name}.wav")

        logging.info("[%d/%d] %s: '%s'", idx, len(scenarios), scene_id, text[:50] + ("..." if len(text) > 50 else ""))

        t0 = time.time()
        try:
            if model_name == "spark_tts":
                runner.synthesize(
                    text=text,
                    output_path=out_path,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    gender=spark_gender,
                    pitch=spark_pitch,
                    speed=spark_speed,
                )
            elif model_name == "chattts":
                runner.synthesize(
                    text=text,
                    output_path=out_path,
                    seed=chattts_seed,
                )
            else:
                runner.synthesize(
                    text=text,
                    output_path=out_path,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
            gen_time = time.time() - t0

            # 读取生成的音频信息
            info = sf.info(out_path)
            audio_duration = info.duration

            total_duration += audio_duration
            results.append({
                "scene_id": scene_id,
                "text": text,
                "output": out_path,
                "duration": round(audio_duration, 2),
                "gen_time": round(gen_time, 2),
                "rtf": round(gen_time / max(audio_duration, 0.01), 3),  # Real-Time Factor
            })

            logging.info("  ✅ %.1fs audio in %.1fs (RTF=%.2f)", audio_duration, gen_time, gen_time / max(audio_duration, 0.01))

        except Exception as e:
            logging.error("  ❌ Failed: %s", e)
            results.append({
                "scene_id": scene_id,
                "text": text,
                "error": str(e),
            })

    # 保存结果摘要
    summary = {
        "model": model_name,
        "model_path": model_path,
        "language": language,
        "device": device,
        "model_load_time": round(load_time, 2),
        "total_audio_duration": round(total_duration, 2),
        "total_scenarios": len(scenarios),
        "successful": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "results": results,
    }

    summary_path = os.path.join(output_dir, f"summary_{model_name}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("")
    logging.info("=" * 60)
    logging.info("🎤 %s Benchmark 完成", model_name.upper())
    logging.info("  成功: %d / %d", summary["successful"], summary["total_scenarios"])
    logging.info("  总音频: %.1fs", total_duration)
    logging.info("  模型加载: %.1fs", load_time)
    if results:
        avg_rtf = np.mean([r["rtf"] for r in results if "rtf" in r])
        logging.info("  平均 RTF: %.3f (越小越快)", avg_rtf)
    logging.info("  结果: %s", summary_path)
    logging.info("=" * 60)

    # 清理显存
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTS Benchmark - 备选 TTS 方案统一推理")
    parser.add_argument("--model", required=True, choices=list(RUNNER_REGISTRY.keys()),
                        help="TTS model to use")
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    parser.add_argument("--code_dir", required=True, help="Path to TTS framework code")
    parser.add_argument("--prompts_file", required=True, help="JSON file with TTS prompts")
    parser.add_argument("--output_dir", required=True, help="Output directory for generated audio")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--language", default="zh", choices=["zh", "en"], help="Language (default: zh)")
    parser.add_argument("--scene_ids", nargs="*", help="Specific scene IDs to run")
    parser.add_argument("--ref_audio", default=None, help="Reference audio for voice cloning")
    parser.add_argument("--ref_text", default=None, help="Reference text (transcript of ref_audio)")
    parser.add_argument("--spark_gender", default="female", choices=["female", "male"], help="Spark-TTS voice creation gender")
    parser.add_argument("--spark_pitch", default="moderate", choices=["very_low", "low", "moderate", "high", "very_high"], help="Spark-TTS voice creation pitch")
    parser.add_argument("--spark_speed", default="moderate", choices=["very_low", "low", "moderate", "high", "very_high"], help="Spark-TTS voice creation speed")
    parser.add_argument("--chattts_seed", type=int, default=42, help="ChatTTS speaker seed")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.info("=" * 60)
    logging.info("🎤 TTS Benchmark Runner")
    logging.info("  Model:    %s", args.model)
    logging.info("  Weights:  %s", args.model_path)
    logging.info("  Code:     %s", args.code_dir)
    logging.info("  Language: %s", args.language)
    logging.info("  Device:   %s", args.device)
    logging.info("  Prompts:  %s", args.prompts_file)
    logging.info("  Output:   %s", args.output_dir)
    if args.model == "f5tts" and args.ref_audio is None:
        logging.info("  F5 ref:   使用仓库内置示例参考音频")
    if args.model == "spark_tts" and args.ref_audio is None:
        logging.info("  Spark:    使用可控造音色模式 (%s/%s/%s)", args.spark_gender, args.spark_pitch, args.spark_speed)
    if args.model == "chattts":
        logging.info("  Chat seed:%s", args.chattts_seed)
    logging.info("=" * 60)

    run_benchmark(
        model_name=args.model,
        model_path=args.model_path,
        code_dir=args.code_dir,
        prompts_file=args.prompts_file,
        output_dir=args.output_dir,
        device=args.device,
        language=args.language,
        scene_ids=args.scene_ids,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        spark_gender=args.spark_gender,
        spark_pitch=args.spark_pitch,
        spark_speed=args.spark_speed,
        chattts_seed=args.chattts_seed,
    )


if __name__ == "__main__":
    main()
