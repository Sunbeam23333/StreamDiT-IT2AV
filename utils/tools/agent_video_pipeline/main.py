"""
CLI entry point for the agent video pipeline.

Usage::

    # Full pipeline (auto-detect route)
    python main.py run --prompt "一个女主播在介绍新款手机" --duration 10 --project-dir /tmp/demo

    # Force a specific route
    python main.py run --prompt "..." --duration 10 --route magihuman_av

    # Dry run (generate plan JSON without executing)
    python main.py run --prompt "..." --duration 10 --dry-run

    # Direct tool invocations
    python main.py image --prompt-file prompts.txt --output img.png
    python main.py tts --text "你好世界" --output hello.wav
    python main.py lipsync --video in.mp4 --audio tts.wav --output out.mp4
    python main.py edit concat --video a.mp4 --video b.mp4 --output merged.mp4
    python main.py edit add-subtitles --video in.mp4 --srt subs.srt --output out.mp4
"""
from __future__ import annotations

import argparse
import re
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

TOOLS_ROOT = Path(__file__).resolve().parent
CODE_ROOT = TOOLS_ROOT.parents[3]

# Ensure package imports work both as module and as script.
sys.path.insert(0, str(TOOLS_ROOT.parent))


def _load_attr(module_path: str, attr_name: str) -> Any:
    return getattr(import_module(module_path), attr_name)


def _cmd_run(args: argparse.Namespace) -> int:
    run_pipeline = _load_attr("agent_video_pipeline.orchestrator", "run_pipeline")
    result = run_pipeline(
        user_prompt=args.prompt,
        duration_s=args.duration,
        project_id=args.project_id,
        project_dir=args.project_dir,
        preferred_route=args.route,
        aspect_ratio=args.aspect_ratio,
        scenario_hint=args.scenario,
        quality_tier=args.quality_tier,
        execution_mode=args.execution_mode,
        dry_run=args.dry_run,
    )
    print("Result:", result)
    return 0 if result.get("success") else 1


def _cmd_image(args: argparse.Namespace) -> int:
    image_stage_cls = _load_attr("agent_video_pipeline.adapters.image_stage", "HunyuanImageStage")
    stage = image_stage_cls()
    if args.prompt_file:
        result = stage.generate_from_prompt_file(args.prompt_file, args.output, size=args.size)
    else:
        result = stage.generate_single(args.prompt, args.output, size=args.size)
    print("Done:", result)
    return 0


def _cmd_tts(args: argparse.Namespace) -> int:
    tts_stage_cls = _load_attr("agent_video_pipeline.adapters.tts_stage", "HunyuanTTSStage")
    stage = tts_stage_cls()
    text = Path(args.text_file).read_text(encoding="utf-8").strip() if args.text_file else args.text
    result = stage.generate_single(text=text, output=args.output, voice=args.voice)
    print("Done:", result)
    return 0


def _cmd_lipsync(args: argparse.Namespace) -> int:
    lipsync_stage_cls = _load_attr("agent_video_pipeline.adapters.lipsync_stage", "LatentSyncStage")
    stage = lipsync_stage_cls()
    result = stage.generate(args.video, args.audio, args.output)
    print("Done:", result)
    return 0


def _cmd_edit(args: argparse.Namespace) -> int:
    concat_videos = _load_attr("agent_video_pipeline.editing.ffmpeg_editor", "concat_videos")
    merge_audio = _load_attr("agent_video_pipeline.editing.ffmpeg_editor", "merge_audio")
    add_subtitles = _load_attr("agent_video_pipeline.editing.ffmpeg_editor", "add_subtitles")
    apply_crossfade = _load_attr("agent_video_pipeline.editing.ffmpeg_editor", "apply_crossfade")

    sub = args.edit_command
    if sub == "concat":
        concat_videos(args.video, args.output)
    elif sub == "merge-audio":
        specs = []
        for spec in args.audio or []:
            matched = re.match(r"(.+):([\d.]+)$", spec)
            specs.append((matched.group(1), float(matched.group(2))) if matched else (spec, 1.0))
        merge_audio(args.video, specs, args.output)
    elif sub == "add-subtitles":
        add_subtitles(args.video, args.srt, args.output)
    elif sub == "apply-transitions":
        apply_crossfade(args.video, args.transition_s, args.output)
    else:
        print(f"Unknown edit command: {sub}")
        return 1
    print("Done.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent_video_pipeline", description="End-to-end agent video pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the full pipeline")
    r.add_argument("--prompt", required=True, help="User story / prompt")
    r.add_argument("--duration", type=float, required=True, help="Target duration in seconds")
    r.add_argument("--project-id", help="Project identifier")
    r.add_argument("--project-dir", help="Root directory for project files")
    r.add_argument(
        "--route",
        default="auto",
        choices=["auto", "hyvideo_lipsync", "hyvideo_longform", "magihuman_av", "wan22_i2v", "wan22_flf2v", "wan22_lipsync"],
    )
    r.add_argument(
        "--scenario",
        default="auto",
        choices=["auto", "comic_drama", "beauty_glamour", "historical_reenactment", "narration_avatar", "generic"],
        help="显式指定场景策略；默认自动识别。",
    )
    r.add_argument(
        "--quality-tier",
        default="balanced",
        choices=["speed", "balanced", "quality"],
        help="规划时偏速度、平衡还是偏质量。",
    )
    r.add_argument("--aspect-ratio", default="9:16", choices=["9:16", "16:9"])
    r.add_argument(
        "--execution-mode",
        default="gpu",
        choices=["gpu", "cpu"],
        help="`gpu` 走本地推理框架，`cpu` 走远程 API / 调试链路。",
    )
    r.add_argument("--dry-run", action="store_true", help="Generate plan without executing")
    r.set_defaults(func=_cmd_run)

    img = sub.add_parser("image", help="Generate a single image (Hunyuan-Image)")
    img.add_argument("--prompt", help="Inline prompt")
    img.add_argument("--prompt-file", help="Path to prompt text file")
    img.add_argument("--output", required=True, help="Output image path")
    img.add_argument("--size", default="768x1280", help="Image resolution")
    img.set_defaults(func=_cmd_image)

    t = sub.add_parser("tts", help="Generate TTS audio (Hunyuan-TTS)")
    t.add_argument("--text", help="Inline text")
    t.add_argument("--text-file", help="Path to text file")
    t.add_argument("--output", required=True, help="Output audio path")
    t.add_argument("--voice", default="wenrouxuemei")
    t.set_defaults(func=_cmd_tts)

    l = sub.add_parser("lipsync", help="Run LatentSync lip-sync")
    l.add_argument("--video", required=True, help="Input video (no audio)")
    l.add_argument("--audio", required=True, help="TTS audio to drive lip-sync")
    l.add_argument("--output", required=True, help="Output video path")
    l.set_defaults(func=_cmd_lipsync)

    e = sub.add_parser("edit", help="FFmpeg editing primitives")
    e.add_argument("edit_command", choices=["concat", "merge-audio", "add-subtitles", "apply-transitions"])
    e.add_argument("--video", action="append", help="Video file (repeatable for concat/xfade)")
    e.add_argument("--audio", action="append", help="audio_path:volume pairs for merge-audio")
    e.add_argument("--srt", help="Subtitle file for add-subtitles")
    e.add_argument("--output", required=True, help="Output path")
    e.add_argument("--transition-s", type=float, default=0.5, help="Crossfade duration for apply-transitions")
    e.set_defaults(func=_cmd_edit)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
