"""
Main orchestration entry point for the agent video pipeline.

Usage::

    from agent_video_pipeline import run_pipeline

    result = run_pipeline(
        user_prompt="一个主播在讲解最新的手机产品",
        duration_s=10.0,
        project_id="demo_001",
        project_dir="/tmp/demo",
        route="auto",
        aspect_ratio="9:16",
    )
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters.image_stage import HunyuanImageStage
from .adapters.tts_stage import HunyuanTTSStage
from .adapters.video_stage import HyVideoStage, MagiHumanStage
from .adapters.wan_lightx2v_stage import WanLightX2VStage
from .adapters.lipsync_stage import LatentSyncStage
from .adapters.audio_stage import BananaImageStage, BGMStage, SFXStage
from .editing.ffmpeg_editor import (
    add_subtitles,
    apply_crossfade,
    concat_videos,
    extract_last_frame,
    merge_audio,
)
from .editing.sync_editor import align_for_lipsync, get_media_duration
from .configs.model_selector import ModelSelector, VideoBackend
from .cpu_pipeline import run_cpu_pipeline
from .planner import build_project_plan
from .prompting import split_story_units, extract_dialogue, normalize_text
from .schemas import ProjectPlan, ShotSpec, StageCommand

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


TOOLS_ROOT = Path(__file__).resolve().parent
CODE_ROOT = TOOLS_ROOT.parents[3]
FFMPEG_EDITOR = TOOLS_ROOT / "editing/ffmpeg_editor.py"


# ---------------------------------------------------------------------------
# Per-stage runners
# ---------------------------------------------------------------------------

def _run_command(cmd: str | list[str], cwd: str | None = None, env: Dict[str, str] | None = None, enabled: bool = True) -> bool:
    """Execute a shell command, optionally skipping if disabled."""
    if not enabled:
        return True
    if isinstance(cmd, list):
        logger.info("Running: %s", " ".join(cmd))
    else:
        logger.info("Running: %s", cmd)
    merged_env = {**os.environ, **(env or {})}
    try:
        if isinstance(cmd, str):
            subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=merged_env)
        else:
            subprocess.run(cmd, check=True, cwd=cwd, env=merged_env)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s", e)
        return False


def _write_prompts(shot_dir: Path, shot: ShotSpec) -> None:
    """Write all prompt text files for a shot."""
    prompts_dir = shot_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "image_prompt.txt": shot.image_prompt,
        "i2v_prompt.txt": shot.video_prompt,
        "dialogue.txt": shot.dialogue,
        "bgm_prompt.txt": shot.bgm_prompt,
        "sfx_prompt.txt": shot.sfx_prompt,
        "magihuman_prompt.txt": shot.magihuman_prompt,
    }
    for fname, content in mapping.items():
        if content:
            (prompts_dir / fname).write_text(content, encoding="utf-8")


def _stage_image(shot: ShotSpec, shot_dir: Path, hy_image: HunyuanImageStage, banana: BananaImageStage) -> bool:
    """Generate the first-frame image for a shot."""
    img_path = shot_dir / "assets/first_frame.png"

    # Continuity: use previous segment's tail frame
    if shot.continuity_source:
        prev_video = Path(shot.continuity_source)
        if prev_video.exists():
            logger.info("Extracting tail frame from %s", prev_video)
            extract_last_frame(str(prev_video), str(img_path))
            return True

    # Generate via Hunyuan-Image (default) or Banana
    logger.info("Generating first-frame image → %s", img_path)
    try:
        hy_image.generate_single(
            prompt=shot.image_prompt,
            output=str(img_path),
            size=shot.image_size,
        )
        return True
    except Exception:
        logger.warning("Hunyuan-Image failed, falling back to Banana")
        try:
            banana.generate_single(
                prompt=shot.image_prompt,
                output=str(img_path),
            )
            return True
        except Exception as e:
            logger.error("Banana also failed: %s", e)
            return False


def _stage_tts(shot: ShotSpec, shot_dir: Path, tts: HunyuanTTSStage) -> bool:
    """Generate TTS audio for a shot's dialogue."""
    if not shot.dialogue:
        return True
    out = shot_dir / "audio/tts.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Generating TTS → %s", out)
    try:
        tts.generate_single(
            text=shot.dialogue,
            output=str(out),
        )
        return True
    except Exception as e:
        logger.error("TTS failed: %s", e)
        return False


def _stage_bgm_sfx(shot: ShotSpec, shot_dir: Path, bgm: BGMStage, sfx: SFXStage) -> bool:
    """Generate BGM and SFX for a shot."""
    out_dir = shot_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    if shot.bgm_prompt:
        out = out_dir / "bgm.wav"
        try:
            logger.info("Generating BGM → %s", out)
            bgm.generate(text_or_prompt=shot.bgm_prompt, output=str(out))
        except Exception as e:
            logger.warning("BGM generation failed (non-critical): %s", e)
            ok = False

    if shot.sfx_prompt:
        out = out_dir / "sfx.wav"
        try:
            logger.info("Generating SFX → %s", out)
            sfx.generate_silent(duration_s=shot.duration_s, output=str(out))
        except Exception as e:
            logger.warning("SFX generation failed (non-critical): %s", e)

    return ok


def _stage_video_hyvideo(shot: ShotSpec, shot_dir: Path, hy_video: HyVideoStage, seed: int) -> bool:
    """Run HunyuanVideo-1.5 I2V generation."""
    out = shot_dir / "video/raw_video.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Generating I2V video → %s", out)
    try:
        hy_video.generate(
            prompt=shot.video_prompt,
            image_path=str(shot_dir / "assets/first_frame.png"),
            output_path=str(out),
            aspect_ratio=shot.aspect_ratio,
            seed=seed,
        )
        return True
    except Exception as e:
        logger.error("HyVideo generation failed: %s", e)
        return False


def _stage_video_magihuman(shot: ShotSpec, shot_dir: Path, magihuman: MagiHumanStage) -> bool:
    """Run MagiHuman video generation."""
    out_dir = shot_dir / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running MagiHuman → %s", out_dir)
    try:
        magihuman.generate(
            image_path=str(shot_dir / "assets/first_frame.png"),
            prompt=shot.magihuman_prompt,
            output_dir=str(out_dir),
            seconds=max(1, int(round(shot.duration_s))),
        )
        return True
    except Exception as e:
        logger.error("MagiHuman failed: %s", e)
        return False


def _stage_video_wan22_i2v(shot: ShotSpec, shot_dir: Path, wan22: WanLightX2VStage, seed: int) -> bool:
    """Run Wan2.2 MoE I2V generation via LightX2V."""
    out = shot_dir / "video/raw_video.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Generating Wan2.2 I2V video → %s", out)
    try:
        wan22.generate_i2v(
            prompt=shot.video_prompt,
            first_frame=str(shot_dir / "assets/first_frame.png"),
            output_path=str(out),
            seed=seed,
        )
        return True
    except Exception as e:
        logger.error("Wan2.2 I2V generation failed: %s", e)
        return False


def _stage_video_wan22_flf2v(
    shot: ShotSpec,
    shot_dir: Path,
    wan22: WanLightX2VStage,
    seed: int,
    last_frame: str | None = None,
) -> bool:
    """Run Wan2.2 MoE FLF2V (first-last-frame) generation."""
    out = shot_dir / "video/raw_video.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    first_frame = str(shot_dir / "assets/first_frame.png")

    # Generate or use provided last frame
    if last_frame and Path(last_frame).exists():
        final_last_frame = last_frame
    else:
        # Generate a target last frame from first frame
        final_last_frame = str(shot_dir / "assets/last_frame_generated.png")
        logger.info("Generating last frame for FLF2V → %s", final_last_frame)
        wan22.generate_last_frame(first_frame, shot.video_prompt, final_last_frame, seed=seed)

    logger.info("Generating Wan2.2 FLF2V video → %s", out)
    try:
        wan22.generate_flf2v(
            prompt=shot.video_prompt,
            first_frame=first_frame,
            last_frame=final_last_frame,
            output_path=str(out),
            seed=seed,
        )
        return True
    except Exception as e:
        logger.error("Wan2.2 FLF2V generation failed: %s", e)
        return False


def _stage_lipsync_with_sync(shot: ShotSpec, shot_dir: Path, lipsync: LatentSyncStage) -> bool:
    """Run LatentSync lip-sync with automatic audio-video alignment."""
    if not shot.dialogue:
        return True

    raw_video = shot_dir / "video/raw_video.mp4"
    tts_audio = shot_dir / "audio/tts.wav"
    out = shot_dir / "video/lipsync_video.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Check duration mismatch before lipsync
    try:
        video_duration = get_media_duration(str(raw_video))
        audio_duration = get_media_duration(str(tts_audio))
        duration_diff = abs(video_duration - audio_duration)

        if duration_diff > 0.5:  # Significant mismatch
            logger.warning(
                "Duration mismatch detected: video=%.2fs, audio=%.2fs, diff=%.2fs",
                video_duration, audio_duration, duration_diff
            )
            logger.info("Will perform alignment after lip-sync")
    except Exception as e:
        logger.warning("Could not check duration: %s", e)

    logger.info("Running LatentSync → %s", out)
    try:
        lipsync.generate(
            video_path=str(raw_video),
            audio_path=str(tts_audio),
            video_out_path=str(out),
        )

        # Post-lipsync alignment if needed
        try:
            aligned_out = shot_dir / "video/lipsync_aligned.mp4"
            aligned_path, info = align_for_lipsync(
                str(out),
                str(tts_audio),
                str(aligned_out),
            )
            if info.get("action") != "direct_merge":
                # Replace original with aligned version
                shutil.move(aligned_path, out)
                logger.info("Applied alignment: %s", info)
        except Exception as align_err:
            logger.warning("Alignment step failed (non-critical): %s", align_err)

        return True
    except Exception as e:
        logger.error("LatentSync failed: %s", e)
        return False


def _stage_lipsync(shot: ShotSpec, shot_dir: Path, lipsync: LatentSyncStage) -> bool:
    """Run LatentSync lip-sync."""
    if not shot.dialogue:
        return True
    out = shot_dir / "video/lipsync_video.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running LatentSync → %s", out)
    try:
        lipsync.generate(
            video_path=str(shot_dir / "video/raw_video.mp4"),
            audio_path=str(shot_dir / "audio/tts.wav"),
            video_out_path=str(out),
        )
        return True
    except Exception as e:
        logger.error("LatentSync failed: %s", e)
        return False


def _stage_edit_mix(shot: ShotSpec, shot_dir: Path) -> bool:
    """Mix TTS + BGM + SFX onto the lip-synced video."""
    video_in = str(shot_dir / "video/lipsync_video.mp4")
    if not Path(video_in).exists():
        video_in = str(shot_dir / "video/raw_video.mp4")

    audio_specs: list[tuple[str, float]] = []
    tts = shot_dir / "audio/tts.wav"
    if tts.exists():
        audio_specs.append((str(tts), 1.0))
    bgm = shot_dir / "audio/bgm.wav"
    if bgm.exists():
        audio_specs.append((str(bgm), 0.2))
    sfx = shot_dir / "audio/sfx.wav"
    if sfx.exists():
        audio_specs.append((str(sfx), 0.15))

    if not audio_specs:
        return True

    out = shot_dir / "video/final.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Mixing audio → %s", out)
    try:
        merge_audio(video_in, audio_specs, str(out))
        return True
    except Exception as e:
        logger.error("merge_audio failed: %s", e)
        return False


def _stage_edit_magihuman(shot: ShotSpec, shot_dir: Path) -> bool:
    """Merge BGM onto MagiHuman output."""
    video_dir = shot_dir / "video"
    # Find the generated magihuman video file
    candidates = list(video_dir.glob("magihuman_video_*.mp4"))
    candidates = [c for c in candidates if "_video.mp4" not in c.name and "_audio" not in c.name]
    if not candidates:
        logger.warning("No MagiHuman video found in %s", video_dir)
        return False
    video_in = str(sorted(candidates)[0])

    bgm = shot_dir / "audio/bgm.wav"
    audio_specs = [(str(bgm), 0.2)] if bgm.exists() else []

    out = shot_dir / "video/final.mp4"
    if audio_specs:
        merge_audio(video_in, audio_specs, str(out))
    else:
        shutil.copy(video_in, str(out))
    return True


def _shot_final_file(shot: ShotSpec, shot_dir: Path) -> Optional[str]:
    """Return the best available final video for a shot."""
    for name in ["video/final.mp4", "video/lipsync_video.mp4", "video/raw_video.mp4"]:
        p = shot_dir / name
        if p.exists():
            return str(p)
    candidates = list((shot_dir / "video").glob("magihuman_video_*.mp4"))
    if candidates:
        return str(sorted(candidates)[0])
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    user_prompt: str,
    duration_s: float,
    project_id: str | None = None,
    project_dir: str | None = None,
    preferred_route: str = "auto",
    aspect_ratio: str = "9:16",
    video_backend: str | None = None,
    scenario_hint: str | None = None,
    quality_tier: str = "balanced",
    execution_mode: str = "gpu",
    enable_lipsync: bool = True,
    enable_av_sync: bool = True,
    dry_run: bool = False,
    save_plan: bool = True,
    skip_failed_shots: bool = True,
) -> dict[str, Any]:
    """Run the full prompt→video pipeline.

    Parameters
    ----------
    user_prompt : str
        Original user prompt / story description.
    duration_s : float
        Target total video duration in seconds.
    project_id : str, optional
        Project identifier. Auto-generated if not provided.
    project_dir : str, optional
        Root directory for all project files. Auto-created under /tmp if not provided.
    preferred_route : str
        One of "auto", "hyvideo_lipsync", "hyvideo_longform", "magihuman_av",
        "wan22_i2v", "wan22_flf2v", "wan22_lipsync".
    aspect_ratio : str
        One of "9:16" (portrait) or "16:9" (landscape).
    video_backend : str, optional
        Force specific video backend ("hyvideo", "wan22_i2v", "wan22_flf2v", "magihuman").
        If None, auto-selected based on route and content.
    scenario_hint : str, optional
        显式指定场景策略（如 comic_drama / beauty_glamour / historical_reenactment / narration_avatar）。
    quality_tier : str
        规划时偏速度、平衡还是偏质量，支持 "speed" / "balanced" / "quality"。
    enable_lipsync : bool
        Whether to run LatentSync for lip-syncing (default True).
    enable_av_sync : bool
        Whether to enable automatic audio-video duration alignment (default True).
    dry_run : bool
        If True, only build and save the plan without executing.
    save_plan : bool
        If True, write the plan JSON next to the project dir.
    skip_failed_shots : bool
        If True, continue with remaining shots when one fails.

    Returns
    -------
    dict
        Summary dict with ``success``, ``project_id``, ``project_dir``,
        ``output_video``, ``failed_shots``, ``plan``.
    """
    project_id = project_id or f"proj_{datetime.now():%Y%m%d_%H%M%S}"
    project_dir_path = Path(project_dir or f"/tmp/agent_video/{project_id}")
    project_dir_path.mkdir(parents=True, exist_ok=True)

    if execution_mode == "cpu":
        return run_cpu_pipeline(
            user_prompt=user_prompt,
            duration_s=duration_s,
            project_id=project_id,
            project_dir=str(project_dir_path),
            preferred_route=preferred_route,
            aspect_ratio=aspect_ratio,
            scenario_hint=scenario_hint,
            quality_tier=quality_tier,
            dry_run=dry_run,
            save_plan=save_plan,
        )

    logger.info("=" * 60)
    logger.info(
        "Building project plan  (id=%s, route=%s, scenario=%s, execution=%s)",
        project_id,
        preferred_route,
        scenario_hint or "auto",
        execution_mode,
    )
    logger.info("=" * 60)

    plan = build_project_plan(
        user_prompt=user_prompt,
        duration_s=duration_s,
        project_id=project_id,
        project_dir=str(project_dir_path),
        preferred_route=preferred_route,
        aspect_ratio=aspect_ratio,
        scenario_hint=scenario_hint,
        quality_tier=quality_tier,
        execution_mode=execution_mode,
        planning_backend="rule_based",
    )

    if save_plan:
        plan_path = project_dir_path / "plan.json"
        plan_path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Plan saved → %s", plan_path)

    if dry_run:
        logger.info("Dry-run: skipping execution")
        return {
            "success": True,
            "project_id": project_id,
            "project_dir": str(project_dir_path),
            "dry_run": True,
            "plan": plan.to_dict(),
        }

    hy_image = HunyuanImageStage()
    banana = BananaImageStage()
    tts = HunyuanTTSStage()
    bgm_adapter = BGMStage()
    sfx_adapter = SFXStage()
    hy_video = HyVideoStage()
    wan22_video = WanLightX2VStage()
    magihuman = MagiHumanStage()
    lipsync = LatentSyncStage()
    model_selector = ModelSelector()

    failed_shots: list[int] = []

    for shot in plan.shots:
        shot_dir = project_dir_path / "shots" / f"shot_{shot.shot_id:03d}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        assets_dir = shot_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        for sub in ["video", "audio", "prompts"]:
            (shot_dir / sub).mkdir(exist_ok=True)

        _write_prompts(shot_dir, shot)

        logger.info("")
        logger.info("--- Shot %d ---  route=%s  duration=%.1fs", shot.shot_id, shot.route, shot.duration_s)

        ok = True
        if not _stage_image(shot, shot_dir, hy_image, banana):
            ok = False

        if shot.route != "magihuman_av" and not _stage_tts(shot, shot_dir, tts):
            ok = False
        _ = _stage_bgm_sfx(shot, shot_dir, bgm_adapter, sfx_adapter)

        use_flf2v = shot.route == "wan22_flf2v" or (
            shot.route == "hyvideo_longform" and shot.duration_s > 5 and shot.shot_id > 1
        )
        backend = model_selector.get_backend_for_route(video_backend) if video_backend else model_selector.get_backend_for_route(shot.route)

        if backend == VideoBackend.MAGIHUMAN or shot.route == "magihuman_av":
            if not _stage_video_magihuman(shot, shot_dir, magihuman):
                ok = False
            if _shot_final_file(shot, shot_dir):
                _ = _stage_edit_magihuman(shot, shot_dir)
        elif backend == VideoBackend.WAN22_I2V or shot.route in ("wan22_i2v", "wan22_lipsync"):
            if not _stage_video_wan22_i2v(shot, shot_dir, wan22_video, seed=42 + shot.shot_id):
                ok = False
            if enable_lipsync and shot.dialogue:
                lipsync_ok = _stage_lipsync_with_sync(shot, shot_dir, lipsync) if enable_av_sync else _stage_lipsync(shot, shot_dir, lipsync)
                if not lipsync_ok:
                    ok = False
                if not _stage_edit_mix(shot, shot_dir):
                    ok = False
        elif backend == VideoBackend.WAN22_FLF2V or use_flf2v:
            last_frame = None
            if shot.continuity_source and shot.shot_id > 1:
                prev_shot_dir = project_dir_path / "shots" / f"shot_{shot.shot_id - 1:03d}"
                last_frame = str(prev_shot_dir / "assets/last_frame_target.png")
            if not _stage_video_wan22_flf2v(shot, shot_dir, wan22_video, seed=42 + shot.shot_id, last_frame=last_frame):
                ok = False
            if enable_lipsync and shot.dialogue:
                lipsync_ok = _stage_lipsync_with_sync(shot, shot_dir, lipsync) if enable_av_sync else _stage_lipsync(shot, shot_dir, lipsync)
                if not lipsync_ok:
                    ok = False
                if not _stage_edit_mix(shot, shot_dir):
                    ok = False
            elif shot.shot_id < len(plan.shots):
                tail = shot_dir / "assets/last_frame.png"
                raw = shot_dir / "video/raw_video.mp4"
                if raw.exists():
                    extract_last_frame(str(raw), str(tail))
        else:
            if not _stage_video_hyvideo(shot, shot_dir, hy_video, seed=42 + shot.shot_id):
                ok = False
            if enable_lipsync and shot.dialogue:
                lipsync_ok = _stage_lipsync_with_sync(shot, shot_dir, lipsync) if enable_av_sync else _stage_lipsync(shot, shot_dir, lipsync)
                if not lipsync_ok:
                    ok = False
                if not _stage_edit_mix(shot, shot_dir):
                    ok = False
            elif shot.shot_id < len(plan.shots):
                tail = shot_dir / "assets/last_frame.png"
                raw = shot_dir / "video/raw_video.mp4"
                if raw.exists():
                    extract_last_frame(str(raw), str(tail))

        if not ok:
            failed_shots.append(shot.shot_id)
            if skip_failed_shots:
                logger.warning("Shot %d failed, continuing", shot.shot_id)
            else:
                break

    if plan.project_commands and failed_shots and skip_failed_shots:
        logger.info("Skipping concat because some shots failed")
    elif plan.project_commands:
        for cmd in plan.project_commands:
            if cmd.stage == "concat_story" and cmd.enabled:
                final = project_dir_path / "final_story.mp4"
                video_paths: list[str] = []
                for planned_shot in plan.shots:
                    current_shot_dir = project_dir_path / "shots" / f"shot_{planned_shot.shot_id:03d}"
                    final_file = _shot_final_file(planned_shot, current_shot_dir)
                    if final_file:
                        video_paths.append(final_file)
                concat_videos(video_paths=video_paths, output=str(final))
                logger.info("Final story → %s", final)

    output_video = str(project_dir_path / "final_story.mp4")
    if not Path(output_video).exists():
        for shot in reversed(plan.shots):
            shot_output_dir = project_dir_path / "shots" / f"shot_{shot.shot_id:03d}"
            final_file = _shot_final_file(shot, shot_output_dir)
            if final_file:
                output_video = final_file
                break

    return {
        "success": len(failed_shots) == 0,
        "project_id": project_id,
        "project_dir": str(project_dir_path),
        "output_video": output_video,
        "failed_shots": failed_shots,
        "plan": plan.to_dict(),
    }
