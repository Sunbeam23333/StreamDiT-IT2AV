from __future__ import annotations

from pathlib import Path
from typing import List

from .configs.model_selector import ModelSelector
from .prompting import (
    extract_dialogue,
    make_bgm_prompt,
    make_i2v_prompt,
    make_image_prompt,
    make_last_frame_prompt,
    make_magihuman_prompt,
    make_sfx_prompt,
    make_text_agent_instruction,
    normalize_text,
)
from .scenario_framework import (
    build_agent_roles,
    build_generation_chain,
    build_project_review_plan,
    build_shot_review_plan,
    build_storyboard_beats,
    build_strategy_summary,
    detect_scenario,
)
from .schemas import ProjectPlan, ShotSpec, StageCommand, StoryboardBeat


TOOLS_ROOT = Path(__file__).resolve().parent
CODE_ROOT = TOOLS_ROOT.parents[3]
LATENTSYNC_ROOT = CODE_ROOT / "utils/video/LatentSync"
MAGIHUMAN_RUNNER = CODE_ROOT / "script/digit/pipeline3_magihuman/run_single.sh"
FFMPEG_EDITOR = TOOLS_ROOT / "editing/ffmpeg_editor.py"
MAIN_ENTRY = TOOLS_ROOT / "main.py"


def detect_route(
    user_prompt: str,
    duration_s: float,
    preferred_route: str = "auto",
    scenario: str = "generic",
) -> str:
    """Detect a project-level primary route based on scenario, prompt and duration."""
    if preferred_route != "auto":
        return preferred_route

    prompt = user_prompt.lower()
    speaking_hints = [
        "口播", "对口型", "数字人", "主播", "讲解", "独白", "采访",
        "说话", "台词", "dialogue", "talking", "speaking", "narrator",
    ]
    music_hints = [
        "magi", "magi human", "氛围片", "mv", "音乐视频",
        "环境音", "bgm", "表演片段", "music video", "concert", "singing",
    ]
    precise_motion_hints = [
        "精确", "准确", "一致", "连续", "流畅", "smooth", "precise",
        "consistent", "continuous", "camera movement", "pan", "zoom",
    ]

    if scenario == "comic_drama":
        if any(h in prompt for h in speaking_hints):
            return "wan22_lipsync"
        return "wan22_flf2v" if duration_s > 4 else "wan22_i2v"

    if scenario == "beauty_glamour":
        if any(h in prompt for h in speaking_hints):
            return "wan22_lipsync"
        return "wan22_flf2v" if duration_s > 5 else "wan22_i2v"

    if scenario == "historical_reenactment":
        if any(h in prompt for h in speaking_hints):
            return "hyvideo_lipsync"
        return "wan22_flf2v" if duration_s > 5 else "wan22_i2v"

    if scenario == "narration_avatar":
        return "hyvideo_lipsync"

    if any(h in prompt for h in music_hints) and not any(h in prompt for h in speaking_hints):
        return "magihuman_av"
    if duration_s > 5 and any(h in prompt for h in precise_motion_hints):
        return "wan22_flf2v"
    if duration_s > 5:
        return "hyvideo_longform"
    if any(h in prompt for h in speaking_hints):
        return "hyvideo_lipsync"
    return "hyvideo_lipsync"


def image_size_for_aspect(aspect_ratio: str) -> str:
    return "768x1280" if aspect_ratio == "9:16" else "1280x768"


def _shot_dir(project_dir: Path, shot_id: int) -> Path:
    return project_dir / "shots" / f"shot_{shot_id:03d}"


def _build_hyvideo_command(shot_dir: Path, shot: ShotSpec, seed: int) -> str:
    return (
        f'cd "{CODE_ROOT / "utils/video/HunyuanVideo-1.5"}" && '
        f'torchrun --nproc_per_node=${{NUM_GPUS:-8}} generate.py '
        f'--prompt "$(cat \"{shot_dir / "prompts/i2v_prompt.txt"}\")" '
        f'--image_path "{shot_dir / "assets/first_frame.png"}" '
        f'--resolution 720p '
        f'--aspect_ratio {shot.aspect_ratio} '
        f'--seed {seed} '
        f'--rewrite false '
        f'--cfg_distilled true '
        f'--enable_step_distill false '
        f'--sparse_attn false '
        f'--use_sageattn false '
        f'--offloading false '
        f'--group_offloading false '
        f'--overlap_group_offloading false '
        f'--enable_cache true '
        f'--cache_type deepcache '
        f'--sr true '
        f'--save_pre_sr_video true '
        f'--output_path "{shot_dir / "video/raw_video.mp4"}" '
        f'--model_path "${{HYVIDEO_MODEL_PATH}}"'
    )


def _build_wan22_i2v_command(shot_dir: Path, seed: int) -> str:
    return (
        f'orchestrator::wan22_i2v prompt_file="{shot_dir / "prompts/i2v_prompt.txt"}" '
        f'first_frame="{shot_dir / "assets/first_frame.png"}" '
        f'output="{shot_dir / "video/raw_video.mp4"}" seed={seed}'
    )


def _build_wan22_flf2v_command(shot_dir: Path, seed: int) -> str:
    return (
        f'orchestrator::wan22_flf2v prompt_file="{shot_dir / "prompts/i2v_prompt.txt"}" '
        f'first_frame="{shot_dir / "assets/first_frame.png"}" '
        f'last_frame="{shot_dir / "assets/last_frame_target.png"}" '
        f'output="{shot_dir / "video/raw_video.mp4"}" seed={seed}'
    )


def _build_lipsync_command(shot_dir: Path) -> str:
    return (
        f'cd "{LATENTSYNC_ROOT}" && '
        f'python3 scripts/inference.py '
        f'--unet_config_path configs/unet/stage2_512.yaml '
        f'--inference_ckpt_path "${{LATENTSYNC_CKPT_PATH:-checkpoints/latentsync_unet.pt}}" '
        f'--video_path "{shot_dir / "video/raw_video.mp4"}" '
        f'--audio_path "{shot_dir / "audio/tts.wav"}" '
        f'--video_out_path "{shot_dir / "video/lipsync_video.mp4"}" '
        f'--inference_steps 25 '
        f'--guidance_scale 1.5 '
        f'--temp_dir "{shot_dir / "tmp_latentsync"}" '
        f'--enable_deepcache'
    )


def _build_edit_command(shot_dir: Path, use_bgm: bool = True) -> str:
    audio_args = [f'--audio "{shot_dir / "audio/tts.wav"}:1.0"']
    if use_bgm:
        audio_args.append(f'--audio "{shot_dir / "audio/bgm.wav"}:0.2"')
    audio_args.append(f'--audio "{shot_dir / "audio/sfx.wav"}:0.15"')
    args = " ".join(audio_args)
    return (
        f'python3 "{FFMPEG_EDITOR}" merge-audio '
        f'--video "{shot_dir / "video/lipsync_video.mp4"}" '
        f'{args} '
        f'--output "{shot_dir / "video/final.mp4"}" '
        f'--keep-orig 0'
    )


def _build_magihuman_merge_command(shot_dir: Path) -> str:
    return (
        f'MAGI_VIDEO=$(find "{shot_dir / "video"}" -maxdepth 1 -name "magihuman_video_*.mp4" '
        f'| grep -v "_video.mp4" | head -n 1) && '
        f'python3 "{FFMPEG_EDITOR}" merge-audio '
        f'--video "$MAGI_VIDEO" '
        f'--audio "{shot_dir / "audio/bgm.wav"}:0.2" '
        f'--output "{shot_dir / "video/final.mp4"}" '
        f'--keep-orig 1'
    )


def _refine_route_for_quality(route: str, scenario: str, quality_tier: str, has_dialogue: bool) -> str:
    if quality_tier == "speed":
        speed_map = {
            "wan22_i2v": "hyvideo_lipsync" if has_dialogue else "hyvideo_lipsync",
            "wan22_lipsync": "hyvideo_lipsync",
            "wan22_flf2v": "hyvideo_longform" if not has_dialogue else "hyvideo_lipsync",
        }
        return speed_map.get(route, route)

    if quality_tier == "quality":
        if route == "hyvideo_lipsync" and scenario in {"comic_drama", "beauty_glamour"}:
            return "wan22_lipsync" if has_dialogue else "wan22_i2v"
        if route == "hyvideo_longform" and scenario in {"comic_drama", "historical_reenactment", "beauty_glamour"}:
            return "wan22_flf2v"

    return route


def _select_shot_route(
    beat: StoryboardBeat,
    preferred_route: str,
    project_route: str,
    scenario: str,
    quality_tier: str,
) -> str:
    if preferred_route != "auto":
        return preferred_route

    has_dialogue = bool(beat.dialogue)
    route = beat.suggested_route or project_route
    if beat.needs_previous_frame and route == "wan22_i2v":
        route = "wan22_flf2v"
    if beat.duration_s > 5 and route in {"hyvideo_lipsync", "wan22_lipsync", "wan22_i2v"}:
        route = "wan22_flf2v" if scenario in {"comic_drama", "beauty_glamour", "historical_reenactment"} else "hyvideo_longform"
    return _refine_route_for_quality(route, scenario, quality_tier, has_dialogue)


def _build_shot_commands(project_dir: Path, shot: ShotSpec, seed: int) -> List[StageCommand]:
    shot_dir = _shot_dir(project_dir, shot.shot_id)
    commands: List[StageCommand] = []

    if shot.continuity_source:
        commands.append(
            StageCommand(
                stage="extract_previous_tail",
                command=(
                    f'python3 "{FFMPEG_EDITOR}" extract-last-frame '
                    f'--video "{project_dir / shot.continuity_source}" '
                    f'--output "{shot_dir / "assets/first_frame.png"}"'
                ),
                notes="从上一镜尾帧提取当前镜的首帧锚点。",
                outputs=[str(shot_dir / "assets/first_frame.png")],
            )
        )
    else:
        notes = "首帧生图入口，默认复用混元 T2I。"
        if shot.source_mode == "comic_grid_panel":
            notes = "建议先生成漫画网格图并拆格，当前首帧可作为关键格替代。"
        elif shot.source_mode == "historical_reference_pack":
            notes = "生成前建议先核验时代参考包，再出首帧。"
        commands.append(
            StageCommand(
                stage="generate_first_frame",
                command=(
                    f'python3 "{MAIN_ENTRY}" image '
                    f'--prompt-file "{shot_dir / "prompts/image_prompt.txt"}" '
                    f'--output "{shot_dir / "assets/first_frame.png"}" '
                    f'--size "{shot.image_size}"'
                ),
                notes=notes,
                outputs=[str(shot_dir / "assets/first_frame.png")],
            )
        )

    if shot.route == "magihuman_av":
        commands.append(
            StageCommand(
                stage="generate_bgm",
                command=(
                    f'python3 "{MAIN_ENTRY}" tts '
                    f'--text-file "{shot_dir / "prompts/bgm_prompt.txt"}" '
                    f'--output "{shot_dir / "audio/bgm.wav"}" '
                    f'--voice wenrouxuemei --stream'
                ),
                enabled=False,
                notes="这里先占位，后续建议替换为专门的 BGM 生成 adapter。",
            )
        )
        commands.append(
            StageCommand(
                stage="generate_magihuman_video",
                command=(
                    f'bash "{MAGIHUMAN_RUNNER}" '
                    f'--image "{shot_dir / "assets/first_frame.png"}" '
                    f'--prompt_file "{shot_dir / "prompts/magihuman_prompt.txt"}" '
                    f'--seconds {int(max(1, round(shot.duration_s)))} '
                    f'--mode sr1080p '
                    f'--output_dir "{shot_dir / "video"}"'
                ),
                notes="MagiHuman 路线直接由首帧+增强 prompt 产出音视频。",
            )
        )
        commands.append(
            StageCommand(
                stage="merge_bgm",
                command=_build_magihuman_merge_command(shot_dir),
                enabled=False,
                notes="可选：给 MagiHuman 结果叠额外 BGM。",
            )
        )
        return commands

    if shot.dialogue:
        commands.append(
            StageCommand(
                stage="generate_tts",
                command=(
                    f'python3 "{MAIN_ENTRY}" tts '
                    f'--text-file "{shot_dir / "prompts/dialogue.txt"}" '
                    f'--output "{shot_dir / "audio/tts.wav"}" '
                    f'--voice wenrouxuemei --stream'
                ),
                outputs=[str(shot_dir / "audio/tts.wav")],
            )
        )

    if shot.route in {"wan22_i2v", "wan22_lipsync"}:
        commands.append(
            StageCommand(
                stage="generate_video",
                command=_build_wan22_i2v_command(shot_dir, seed=seed),
                notes="Wan2.2 I2V 路线，由 orchestrator 内部 adapter 执行。",
                outputs=[str(shot_dir / "video/raw_video.mp4")],
            )
        )
    elif shot.route == "wan22_flf2v":
        commands.append(
            StageCommand(
                stage="prepare_last_frame_target",
                command=(
                    f'orchestrator::wan22_generate_last_frame prompt_file="{shot_dir / "prompts/last_frame_prompt.txt"}" '
                    f'input="{shot_dir / "assets/first_frame.png"}" output="{shot_dir / "assets/last_frame_target.png"}" seed={seed}'
                ),
                notes="生成或提供目标尾帧，用于 FLF2V 控制动作收束。",
                outputs=[str(shot_dir / "assets/last_frame_target.png")],
            )
        )
        commands.append(
            StageCommand(
                stage="generate_video",
                command=_build_wan22_flf2v_command(shot_dir, seed=seed),
                notes="Wan2.2 FLF2V 路线，由 orchestrator 内部 adapter 执行。",
                outputs=[str(shot_dir / "video/raw_video.mp4")],
            )
        )
    else:
        commands.append(
            StageCommand(
                stage="generate_video",
                command=_build_hyvideo_command(shot_dir, shot, seed=seed),
                notes="HunyuanVideo-1.5 I2V 路线，依赖外部注入 HYVIDEO_MODEL_PATH。",
                outputs=[str(shot_dir / "video/raw_video.mp4")],
            )
        )

    if shot.dialogue:
        commands.append(
            StageCommand(
                stage="lipsync",
                command=_build_lipsync_command(shot_dir),
                outputs=[str(shot_dir / "video/lipsync_video.mp4")],
            )
        )
        commands.append(
            StageCommand(
                stage="edit_mix",
                command=_build_edit_command(shot_dir, use_bgm=True),
                outputs=[str(shot_dir / "video/final.mp4")],
            )
        )
    else:
        commands.append(
            StageCommand(
                stage="extract_tail_for_next_segment",
                command=(
                    f'python3 "{FFMPEG_EDITOR}" extract-last-frame '
                    f'--video "{shot_dir / "video/raw_video.mp4"}" '
                    f'--output "{shot_dir / "assets/last_frame.png"}"'
                ),
                notes="保留尾帧给下一镜继续生成或做 continuity 锚点。",
                outputs=[str(shot_dir / "assets/last_frame.png")],
            )
        )
    return commands


def _fallback_chain_strings(model_selector: ModelSelector, route: str) -> List[str]:
    primary_backend = model_selector.get_backend_for_route(route)
    return [backend.value for backend in model_selector.build_fallback_chain(primary_backend)]


def _resolve_project_route(shots: List[ShotSpec], primary_route: str) -> str:
    routes = {shot.route for shot in shots}
    return primary_route if len(routes) == 1 else f"mixed:{primary_route}"


def _expected_final_video(shot: ShotSpec, shot_dir: Path) -> str:
    if shot.route == "magihuman_av" or shot.dialogue:
        return str(shot_dir / "video/final.mp4")
    return str(shot_dir / "video/raw_video.mp4")


def build_project_plan(
    user_prompt: str,
    duration_s: float,
    project_id: str,
    project_dir: str,
    preferred_route: str = "auto",
    aspect_ratio: str = "9:16",
    scenario_hint: str | None = None,
    quality_tier: str = "balanced",
    storyboard_override: list[StoryboardBeat] | None = None,
    execution_mode: str = "gpu",
    planning_backend: str = "rule_based",
) -> ProjectPlan:
    strategy = detect_scenario(user_prompt, scenario_hint=scenario_hint)
    image_size = image_size_for_aspect(aspect_ratio)
    primary_route = detect_route(
        user_prompt,
        duration_s,
        preferred_route=preferred_route,
        scenario=strategy.key,
    )
    storyboard = storyboard_override or build_storyboard_beats(user_prompt, duration_s, strategy)
    model_selector = ModelSelector()

    shots: List[ShotSpec] = []
    for beat in storyboard:
        shot_index = beat.beat_id
        shot_route = _select_shot_route(
            beat=beat,
            preferred_route=preferred_route,
            project_route=primary_route,
            scenario=strategy.key,
            quality_tier=quality_tier,
        )
        dialogue = beat.dialogue if shot_route != "magihuman_av" else ""
        style_hint = strategy.style_hint
        motion_hint = strategy.motion_hint
        continuity_source = None
        if beat.needs_previous_frame and shot_index > 1:
            continuity_source = f"shots/shot_{shot_index - 1:03d}/video/raw_video.mp4"

        start_frame_description = normalize_text(beat.start_frame or beat.summary)
        motion_description = normalize_text(beat.action or beat.summary)
        # 如果 end_frame 未指定，基于 motion_description 推导 end_frame 描述
        if beat.end_frame:
            end_frame_description = normalize_text(beat.end_frame)
        elif beat.start_frame and motion_description != beat.summary:
            # 有明确的 start_frame 和 motion，组合推导 end_frame
            end_frame_description = normalize_text(
                f"{beat.start_frame}, 动作完成后: {motion_description}"
            )
        elif motion_description != beat.summary:
            # 只有 motion，基于 summary 推导 end_frame
            end_frame_description = normalize_text(
                f"{beat.summary}, 动作完成状态: {motion_description}"
            )
        else:
            # 完全没有动作描述，添加一个默认的变化提示
            end_frame_description = normalize_text(
                f"{beat.summary} (动作完成后的状态)"
            )
        shot = ShotSpec(
            shot_id=shot_index,
            title=f"shot_{shot_index:03d}",
            description=normalize_text(beat.summary),
            duration_s=beat.duration_s,
            route=shot_route,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            dialogue=extract_dialogue(beat.summary) if beat.dialogue else dialogue,
            bgm_prompt=make_bgm_prompt(beat.summary),
            sfx_prompt=make_sfx_prompt(beat.summary),
            image_prompt=make_image_prompt(
                start_frame_description,
                aspect_ratio=aspect_ratio,
                style_hint=style_hint,
                visual_focus=beat.visual_focus,
            ),
            video_prompt=make_i2v_prompt(
                motion_description,
                speaking=bool(dialogue) or shot_route in {"hyvideo_lipsync", "wan22_lipsync"},
                motion_hint=motion_hint,
                quality_hint=strategy.continuity_focus,
                start_frame=start_frame_description,
                end_frame=end_frame_description,
                visual_focus=beat.visual_focus,
            ),
            magihuman_prompt=make_magihuman_prompt(
                motion_description,
                dialogue,
                make_bgm_prompt(beat.summary),
                style_hint=style_hint,
                motion_hint=motion_hint,
            ),
            end_frame_prompt=make_last_frame_prompt(
                end_frame_description,
                aspect_ratio=aspect_ratio,
                style_hint=style_hint,
                visual_focus=beat.visual_focus,
            ),
            continuity_source=continuity_source,
            scenario=strategy.key,
            shot_type=beat.shot_type,
            source_mode=beat.source_mode,
            continuity_notes=strategy.continuity_focus,
            visual_focus=beat.visual_focus,
            transition_hint=beat.transition_hint,
            style_hint=style_hint,
            motion_hint=motion_hint,
            start_frame_description=start_frame_description,
            end_frame_description=end_frame_description,
            motion_description=motion_description,
            backend_candidates=_fallback_chain_strings(model_selector, shot_route),
            agent_notes=list(beat.agent_notes),
            qc_checks=list(build_shot_review_plan(strategy, has_dialogue=bool(dialogue), shot_type=beat.shot_type)[0].checks),
            review_plan=build_shot_review_plan(strategy, has_dialogue=bool(dialogue), shot_type=beat.shot_type),
        )
        shot.commands = _build_shot_commands(Path(project_dir), shot, seed=42 + shot_index)
        shots.append(shot)

    project_commands: List[StageCommand] = []
    if len(shots) > 1:
        final_inputs = []
        for shot in shots:
            shot_dir = _shot_dir(Path(project_dir), shot.shot_id)
            final_inputs.append(_expected_final_video(shot, shot_dir))
        concat_args = " ".join(f'--video "{item}"' for item in final_inputs)
        project_commands.append(
            StageCommand(
                stage="concat_story",
                command=(
                    f'python3 "{FFMPEG_EDITOR}" concat {concat_args} '
                    f'--output "{Path(project_dir) / "final_story.mp4"}"'
                ),
                notes="将分镜片段拼接成成片。",
                outputs=[str(Path(project_dir) / "final_story.mp4")],
            )
        )

    strategy_summary = build_strategy_summary(strategy)
    notes = [
        "当前 plan 已支持按题材切换分镜策略，而不是统一平均切段。",
        "每个 shot 都带有 route、fallback backends、审核链路和 continuity 提示。",
        "VLM / 审美 / 口型审核目前先体现在计划书中，后续可以接成实际执行节点。",
        "如需更强控制，可把 storyboard 阶段替换成 LLM/VLM 联合规划或训练专用审核器。",
    ]

    review_plan = build_project_review_plan(strategy)
    agent_roles = build_agent_roles(strategy)
    generation_chain = build_generation_chain(strategy, quality_tier=quality_tier)
    project_route = _resolve_project_route(shots, primary_route)

    return ProjectPlan(
        project_id=project_id,
        user_prompt=user_prompt,
        optimized_prompt_instruction=make_text_agent_instruction(
            user_prompt=user_prompt,
            route=project_route,
            duration_s=duration_s,
            aspect_ratio=aspect_ratio,
            scenario_name=strategy.display_name,
            storyboard_mode=strategy.storyboard_mode,
            review_focus=list(strategy.review_focus),
        ),
        route=project_route,
        total_duration_s=duration_s,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        project_dir=project_dir,
        execution_mode=execution_mode,
        planning_backend=planning_backend,
        scenario=strategy.key,
        scenario_display_name=strategy.display_name,
        storyboard_mode=strategy.storyboard_mode,
        quality_tier=quality_tier,
        storyboard=storyboard,
        shots=shots,
        project_commands=project_commands,
        agent_roles=agent_roles,
        generation_chain=generation_chain,
        review_plan=review_plan,
        strategy_summary=strategy_summary,
        notes=notes,
    )
