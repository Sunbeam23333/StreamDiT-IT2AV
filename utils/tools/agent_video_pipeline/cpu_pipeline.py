from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from .api_clients import VenusProxyClient, VenusVideoClient
from .editing.ffmpeg_editor import concat_videos, extract_frame, extract_last_frame, make_story_motion_video_from_frames, trim_video
from .planner import build_project_plan
from .scenario_framework import detect_scenario
from .schemas import ShotSpec, StoryboardBeat

logger = logging.getLogger(__name__)

STORYBOARD_FALLBACK_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]
DEFAULT_IMAGE_MODEL = "gemini-3-pro-image"
DEFAULT_VIDEO_MODEL = "veo-3.1-fast"
IMAGE_MIN_REVIEW_SCORE = 6.0
VIDEO_MIN_REVIEW_SCORE = 7.0
IMAGE_FAILURE_KEYWORDS = ("多人", "重复", "拼贴", "分屏", "多镜头", "多时刻", "duplicate", "collage", "split screen")
VIDEO_FAILURE_KEYWORDS = (
    "静态",
    "平移",
    "缩放",
    "无动作",
    "首尾",
    "重复",
    "still",
    "pan only",
    "zoom only",
    "no motion",
    "duplicate",
)


def _extract_json_block(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM 返回中未找到 JSON: {text[:500]}")
    return json.loads(candidate[start : end + 1])


def _normalize_durations(raw_shots: list[dict[str, Any]], duration_s: float) -> list[float]:
    raw_values = [max(float(shot.get("duration_s", 0) or 0), 0.2) for shot in raw_shots]
    total = sum(raw_values) or float(len(raw_shots))
    normalized: list[float] = []
    remaining = round(duration_s, 2)
    for index, value in enumerate(raw_values, start=1):
        if index == len(raw_values):
            shot_duration = max(0.6, round(remaining, 2))
        else:
            shot_duration = max(0.6, round(duration_s * value / total, 2))
            remaining = round(remaining - shot_duration, 2)
        normalized.append(shot_duration)
    return normalized


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _default_source_mode(scenario_key: str) -> str:
    mapping = {
        "comic_drama": "comic_grid_panel",
        "beauty_glamour": "hero_reference_frame",
        "historical_reenactment": "historical_reference_pack",
        "narration_avatar": "speaker_anchor_frame",
    }
    return mapping.get(scenario_key, "direct_first_frame")


def _default_continuity_mode(scenario_key: str) -> str:
    mapping = {
        "comic_drama": "panel_to_panel",
        "beauty_glamour": "look_anchor",
        "historical_reenactment": "era_consistency",
        "narration_avatar": "identity_lock",
    }
    return mapping.get(scenario_key, "story_unit")


def _default_route(scenario_key: str, has_dialogue: bool, needs_previous_frame: bool) -> str:
    if scenario_key == "comic_drama":
        if has_dialogue:
            return "wan22_lipsync"
        return "wan22_flf2v" if needs_previous_frame else "wan22_i2v"
    if scenario_key == "beauty_glamour":
        return "wan22_flf2v" if needs_previous_frame else "wan22_i2v"
    if scenario_key == "historical_reenactment":
        if has_dialogue:
            return "hyvideo_lipsync"
        return "wan22_flf2v" if needs_previous_frame else "wan22_i2v"
    if scenario_key == "narration_avatar":
        return "hyvideo_lipsync"
    return "hyvideo_longform" if needs_previous_frame else ("hyvideo_lipsync" if has_dialogue else "wan22_i2v")


def build_llm_storyboard(
    proxy_client: VenusProxyClient,
    user_prompt: str,
    duration_s: float,
    aspect_ratio: str,
    scenario_key: str,
    scenario_name: str,
    quality_tier: str,
    preferred_model: str = "gemini-3-pro",
) -> tuple[list[StoryboardBeat], str, dict[str, Any]]:
    system_prompt = f"""你是资深视频分镜导演和 Agent 编排器。

请为一个 {duration_s:.1f} 秒、画幅 {aspect_ratio}、场景 `{scenario_name}` 的短视频任务输出严格 JSON。
目标是让后续 CPU/API 调试链路可以直接执行：先做首帧，再做尾帧目标，再做视频，再做 Gemini 质检。

请遵守：
1. 只返回 JSON，不要加解释。
2. shots 数量控制在 1-4 个，优先可执行、可验证，而不是空泛文学描述。
3. 每个 shot 需要这些字段：
   - summary: 镜头目标画面与动作，中文，简洁但具体
   - start_frame: 只描述镜头开始时的单一静态瞬间，必须是“一张图”能成立的内容，禁止写多个时刻、多个机位、拼贴、分屏、同一人物重复出现
   - end_frame: 只描述镜头结束时的单一静态瞬间，必须与 start_frame 保持同一主体/服装/场景连续性
   - action: 只描述从 start_frame 到 end_frame 的连续动作弧线，不能把多个镜头拼进一条 action
   - duration_s: 数字
   - shot_type: 如 establish_panel / dialogue_panel / payoff_panel / hero_closeup / historical_action / speaker_segment
   - source_mode: 如 comic_grid_panel / hero_reference_frame / historical_reference_pack / speaker_anchor_frame / direct_first_frame
   - continuity_mode: 如 panel_to_panel / look_anchor / era_consistency / identity_lock / story_unit
   - suggested_route: 从 [hyvideo_lipsync, hyvideo_longform, wan22_i2v, wan22_flf2v, wan22_lipsync, magihuman_av] 中选择
   - dialogue: 没有则空字符串
   - visual_focus: 该镜头最关键的视觉关注点
   - transition_hint: 与下一镜怎么接
   - needs_previous_frame: true/false
   - agent_notes: 字符串数组
4. start_frame 和 end_frame 必须满足：
   - 只允许一个主镜头、一个主时刻、一个主主体（除非用户明确要求多人）
   - 禁止出现“从A到B”“同时出现三种姿态”“多格漫画拼图”“同一人物多次入镜”这类表述
5. action 必须明确动作推进，不能只写“镜头感很好”“高级感氛围”。
6. 尽量体现该场景的特征：
   - comic_drama: 网格格子、对白格、承接格、拆格连续性
   - beauty_glamour: 脸、皮肤、妆造、眼神、轻运动、高级感
   - historical_reenactment: 年代、服化道、器物、空间和历史感
   - narration_avatar: 口播分段、嘴型、稳定机位
7. 总时长约等于 {duration_s:.1f} 秒，质量偏好为 `{quality_tier}`。

返回格式：
{{
  "title": "...",
  "scenario": "{scenario_key}",
  "shots": [{{...}}]
}}"""
    user_message = f"请为以下需求出分镜 JSON：\n{user_prompt}"
    text, resolved_model = proxy_client.chat_text(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        model=preferred_model,
        fallback_models=STORYBOARD_FALLBACK_MODELS,
        max_tokens=4096,
        temperature=0.5,
    )
    payload = _extract_json_block(text)
    raw_shots = list(payload.get("shots", []))
    if not raw_shots:
        raise ValueError(f"LLM 分镜结果为空: {text}")

    durations = _normalize_durations(raw_shots, duration_s)
    beats: list[StoryboardBeat] = []
    for index, (raw_shot, shot_duration) in enumerate(zip(raw_shots, durations), start=1):
        dialogue = _normalize_text(raw_shot.get("dialogue", ""))
        has_dialogue = bool(dialogue.strip())
        needs_previous_frame = _coerce_bool(raw_shot.get("needs_previous_frame"), default=index > 1)
        summary = _normalize_text(raw_shot.get("summary") or f"第{index}镜头")
        start_frame = _normalize_text(raw_shot.get("start_frame") or summary)
        end_frame = _normalize_text(raw_shot.get("end_frame") or start_frame)
        action = _normalize_text(raw_shot.get("action") or summary)
        beats.append(
            StoryboardBeat(
                beat_id=index,
                summary=summary,
                duration_s=shot_duration,
                shot_type=_normalize_text(raw_shot.get("shot_type") or f"{scenario_key}_shot"),
                source_mode=_normalize_text(raw_shot.get("source_mode") or _default_source_mode(scenario_key)),
                continuity_mode=_normalize_text(raw_shot.get("continuity_mode") or _default_continuity_mode(scenario_key)),
                suggested_route=_normalize_text(
                    raw_shot.get("suggested_route")
                    or _default_route(scenario_key, has_dialogue, needs_previous_frame)
                ),
                dialogue=dialogue,
                visual_focus=_normalize_text(raw_shot.get("visual_focus", "")),
                transition_hint=_normalize_text(raw_shot.get("transition_hint", "")),
                needs_previous_frame=needs_previous_frame,
                agent_notes=[_normalize_text(item) for item in raw_shot.get("agent_notes", []) if _normalize_text(item)],
                start_frame=start_frame,
                end_frame=end_frame,
                action=action,
            )
        )
    return beats, resolved_model, payload


def _write_prompts(shot_dir: Path, shot: ShotSpec) -> None:
    prompts_dir = shot_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "image_prompt.txt": shot.image_prompt,
        "i2v_prompt.txt": shot.video_prompt,
        "last_frame_prompt.txt": shot.end_frame_prompt,
        "dialogue.txt": shot.dialogue,
        "bgm_prompt.txt": shot.bgm_prompt,
        "sfx_prompt.txt": shot.sfx_prompt,
        "magihuman_prompt.txt": shot.magihuman_prompt,
    }
    for filename, content in mapping.items():
        if content:
            (prompts_dir / filename).write_text(content, encoding="utf-8")


def _image_review_prompt(shot: ShotSpec, frame_role: str = "first") -> str:
    target = shot.start_frame_description if frame_role == "first" else shot.end_frame_description
    checks = list(shot.qc_checks or ["主体一致性", "构图清晰", "风格正确"])
    checks.extend(
        [
            "是否只有一个主主体，而不是同一人物重复出现",
            "是否不是拼贴/分屏/多格画面",
            "是否不是把多个时刻压进同一张图",
            "是否是单一静态瞬间，而非多格漫画/连环画/分镜板",
        ]
    )
    return (
        f"你是{frame_role}帧质检审核员。请基于上传图片，判断它是否符合目标镜头，并直接返回 JSON。\n"
        f"目标{frame_role}帧：{target}\n"
        f"镜头总目标：{shot.description}\n"
        f"场景：{shot.scenario}\n"
        f"视觉重点：{shot.visual_focus}\n"
        f"审核点：{'；'.join(checks)}\n"
        "\n【关键要求】"
        "\n- 这是视频生成的首帧/尾帧，必须是单一静态画面"
        "\n- 绝对禁止：三联画、分格图、分镜板、连环画、同一人物多次出现"
        "\n- 绝对禁止：左右/上下拼接的多个画面"
        "\n- 必须是：单一构图、单一时刻、单一主体的一张完整图片"
        "\n\n如果发现多格拼接、三联画、分镜板、同一人物重复出现，必须判定为不合格（pass=false, score<5）"
        "\n返回格式：{\"pass\": true/false, \"score\": 0-10, \"issues\": [\"...\"], \"suggestions\": [\"...\"]}"
    )


def _video_frame_review_prompt(shot: ShotSpec) -> str:
    checks = list(shot.qc_checks or ["动作自然", "主体稳定", "连贯性"])
    checks.extend(
        [
            "是否只是静态图片平移/缩放，而没有动作推进",
            "首帧、中间帧、尾帧是否体现从开始状态到结束状态的变化",
            "首尾帧是否明显不同且符合动作收束",
        ]
    )
    return (
        "你会收到同一镜头的关键帧序列（首帧 / 中间帧 / 尾帧）。"
        "请判断这个视频片段是否符合目标镜头，并直接返回 JSON。\n"
        f"镜头总目标：{shot.description}\n"
        f"开始状态：{shot.start_frame_description}\n"
        f"动作弧线：{shot.motion_description}\n"
        f"结束状态：{shot.end_frame_description}\n"
        f"镜头类型：{shot.shot_type}\n"
        f"转场提示：{shot.transition_hint}\n"
        f"审核点：{'；'.join(checks)}\n"
        "请重点关注：主体身份是否稳定、动作是否真实推进、是否只是平移缩放假动作、尾帧是否符合预期。\n"
        '返回格式：{"pass": true/false, "score": 0-10, "issues": ["..."], "suggestions": ["..."]}'
    )


def _safe_review(proxy_client: VenusProxyClient, file_path: Path, prompt: str) -> str:
    try:
        return proxy_client.analyze_file(file_path=file_path, prompt=prompt, model="gemini-2.5-flash")
    except Exception as exc:
        logger.warning("Review skipped for %s: %s", file_path, exc)
        return json.dumps(
            {
                "pass": False,
                "score": 0,
                "issues": [f"review_unavailable: {exc}"],
                "suggestions": ["检查当前代理 token 与多模态接口权限。"],
            },
            ensure_ascii=False,
            indent=2,
        )


def _safe_video_frame_review(
    proxy_client: VenusProxyClient,
    shot: ShotSpec,
    first_frame_path: Path,
    video_path: Path,
    review_dir: Path,
) -> str:
    frame_dir = review_dir / "video_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = [first_frame_path]

    mid_frame_path = frame_dir / "middle_frame.png"
    last_frame_path = frame_dir / "last_frame.png"
    try:
        extract_frame(video=str(video_path), output=str(mid_frame_path), time_s=max(shot.duration_s / 2.0, 0.1))
        if mid_frame_path.exists():
            frame_paths.append(mid_frame_path)
    except Exception as exc:
        logger.warning("Middle frame extraction skipped for shot %s: %s", shot.shot_id, exc)

    try:
        extract_last_frame(video=str(video_path), output=str(last_frame_path))
        if last_frame_path.exists():
            frame_paths.append(last_frame_path)
    except Exception as exc:
        logger.warning("Last frame extraction skipped for shot %s: %s", shot.shot_id, exc)

    try:
        return proxy_client.analyze_files(
            file_paths=frame_paths,
            prompt=_video_frame_review_prompt(shot),
            model="gemini-2.5-flash",
        )
    except Exception as exc:
        logger.warning("Frame-based video review skipped for %s: %s", video_path, exc)
        return json.dumps(
            {
                "pass": False,
                "score": 0,
                "issues": [f"video_review_unavailable: {exc}"],
                "suggestions": ["检查关键帧抽取、上传权限或 Gemini 多图审核能力。"],
            },
            ensure_ascii=False,
            indent=2,
        )


def _normalize_review_result(review_text: str) -> dict[str, Any]:
    try:
        payload = _extract_json_block(review_text)
    except Exception:
        return {
            "pass": False,
            "score": 0.0,
            "issues": [f"invalid_review_json: {review_text[:240]}"],
            "suggestions": [],
        }

    issues = [_normalize_text(item) for item in payload.get("issues", []) if _normalize_text(item)]
    suggestions = [_normalize_text(item) for item in payload.get("suggestions", []) if _normalize_text(item)]
    try:
        score = float(payload.get("score", 0) or 0)
    except Exception:
        score = 0.0

    passed_raw = payload.get("pass")
    if isinstance(passed_raw, bool):
        passed = passed_raw
    elif isinstance(passed_raw, (int, float)):
        passed = bool(passed_raw)
    else:
        lowered = _normalize_text(passed_raw).lower()
        passed = lowered in {"true", "pass", "passed", "yes", "ok"}
    if passed_raw is None:
        passed = score >= VIDEO_MIN_REVIEW_SCORE and not issues

    return {
        "pass": passed,
        "score": score,
        "issues": issues,
        "suggestions": suggestions,
    }


def _review_has_keywords(review: dict[str, Any], keywords: tuple[str, ...]) -> bool:
    haystack = " ".join(review.get("issues", []) + review.get("suggestions", [])).lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def _image_matches_size(image_path: Path, size: str) -> bool:
    try:
        import struct

        with image_path.open("rb") as handle:
            header = handle.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            return False
        width, height = struct.unpack(">II", header[16:24])
        expected_width, expected_height = (int(part) for part in size.lower().split("x", 1))
        return width == expected_width and height == expected_height
    except Exception:
        return False


def _postprocess_image_review(review: dict[str, Any], image_path: Path, shot: ShotSpec) -> dict[str, Any]:
    processed = {
        "pass": bool(review.get("pass", False)),
        "score": float(review.get("score", 0) or 0),
        "issues": list(review.get("issues", [])),
        "suggestions": list(review.get("suggestions", [])),
    }
    if _image_matches_size(image_path, shot.image_size):
        ratio_markers = ("9:16", "2:3", "画幅", "比例", "竖屏", "横屏", "aspect ratio")
        processed["issues"] = [
            issue for issue in processed["issues"] if not any(marker.lower() in issue.lower() for marker in ratio_markers)
        ]
        processed["suggestions"] = [
            item for item in processed["suggestions"] if not any(marker.lower() in item.lower() for marker in ratio_markers)
        ]

    if not _review_has_keywords(processed, IMAGE_FAILURE_KEYWORDS) and processed["score"] >= IMAGE_MIN_REVIEW_SCORE:
        if len(processed["issues"]) <= 1:
            processed["pass"] = True
    return processed


def _review_failed(review: dict[str, Any], keywords: tuple[str, ...], minimum_score: float) -> bool:
    if not review.get("pass", False):
        return True
    if float(review.get("score", 0) or 0) < minimum_score:
        return True
    return _review_has_keywords(review, keywords)


def _refine_prompt_with_review(base_prompt: str, review: dict[str, Any], strict_fix: str) -> str:
    feedback_items = [*review.get("issues", [])[:3], *review.get("suggestions", [])[:3]]
    feedback = "；".join(item for item in feedback_items if item)
    suffix = f" Reviewer feedback: {feedback}." if feedback else ""
    return f"{base_prompt}. STRICT FIX: {strict_fix}.{suffix}"


def _normalize_image_size(image_path: Path, size: str) -> None:
    width, height = (int(part) for part in size.lower().split("x", 1))
    normalized_path = image_path.with_name(f"{image_path.stem}_normalized{image_path.suffix}")
    cmd = [
        "ffmpeg",
        "-i",
        str(image_path),
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
        "-frames:v",
        "1",
        "-y",
        str(normalized_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    normalized_path.replace(image_path)



def _generate_reviewed_image(
    proxy_client: VenusProxyClient,
    shot: ShotSpec,
    output_path: Path,
    review_dir: Path,
    base_prompt: str,
    review_prompt: str,
    review_stem: str,
    strict_fix: str,
    image_model: str,
    reference_paths: list[Path],
    minimum_score: float = IMAGE_MIN_REVIEW_SCORE,
    max_attempts: int = 2,
) -> tuple[str, dict[str, Any]]:
    current_prompt = base_prompt
    last_review_text = ""
    last_review_result: dict[str, Any] = {"pass": False, "score": 0.0, "issues": ["not_started"], "suggestions": []}

    for attempt in range(1, max_attempts + 1):
        prompt_path = review_dir / f"{review_stem}_prompt_attempt_{attempt}.txt"
        prompt_path.write_text(current_prompt, encoding="utf-8")
        proxy_client.generate_image(
            prompt=current_prompt,
            output_path=output_path,
            model=image_model,
            size=shot.image_size,
            reference_image_paths=reference_paths,
        )
        _normalize_image_size(output_path, shot.image_size)
        review_text = _safe_review(proxy_client, output_path, review_prompt)
        review_path = review_dir / f"{review_stem}_attempt_{attempt}.json"
        review_path.write_text(review_text, encoding="utf-8")

        parsed = _postprocess_image_review(_normalize_review_result(review_text), output_path, shot)
        last_review_text = review_text
        last_review_result = parsed
        if not _review_failed(parsed, IMAGE_FAILURE_KEYWORDS, minimum_score=minimum_score):
            (review_dir / f"{review_stem}.json").write_text(review_text, encoding="utf-8")
            return review_text, parsed
        if attempt < max_attempts:
            current_prompt = _refine_prompt_with_review(base_prompt, parsed, strict_fix)

    (review_dir / f"{review_stem}.json").write_text(last_review_text, encoding="utf-8")
    raise RuntimeError(
        f"{review_stem} review failed: score={last_review_result.get('score')} issues={last_review_result.get('issues')}"
    )


def _build_end_frame_reference_paths(shot: ShotSpec, image_outputs: dict[int, Path]) -> list[Path]:
    references: list[Path] = []
    hero_image = image_outputs.get(1)
    if hero_image and hero_image.exists() and shot.shot_id != 1:
        references.append(hero_image)
    current_image = image_outputs.get(shot.shot_id)
    if current_image and current_image.exists():
        references.append(current_image)
    previous_image = image_outputs.get(shot.shot_id - 1)
    if previous_image and previous_image.exists() and previous_image not in references:
        references.append(previous_image)
    return references


def _build_middle_frame_prompt(shot: ShotSpec) -> str:
    return _normalize_text(
        "midpoint action frame, same character, same outfit, same camera distance, same scene continuity, "
        f"halfway through the motion. start state: {shot.start_frame_description}. "
        f"motion arc: {shot.motion_description}. target end state: {shot.end_frame_description}. "
        f"visual focus: {shot.visual_focus}. one exact moment only, no collage, no split screen, no duplicate subject"
    )


def run_cpu_pipeline(
    user_prompt: str,
    duration_s: float,
    project_id: str,
    project_dir: str,
    preferred_route: str = "auto",
    aspect_ratio: str = "9:16",
    scenario_hint: str | None = None,
    quality_tier: str = "balanced",
    dry_run: bool = False,
    save_plan: bool = True,
    planning_model: str = "gemini-3-pro",
    image_model: str = DEFAULT_IMAGE_MODEL,
    video_model: str = DEFAULT_VIDEO_MODEL,
) -> dict[str, Any]:
    project_dir_path = Path(project_dir)
    project_dir_path.mkdir(parents=True, exist_ok=True)

    proxy_client = VenusProxyClient.from_env()
    if proxy_client is None:
        raise RuntimeError(
            "CPU/API 模式缺少 `VENUS_PROXY_TOKEN` 或 `ENV_VENUS_OPENAPI_SECRET_ID`，无法调用 Gemini/NanoBanana。"
        )

    strategy = detect_scenario(user_prompt, scenario_hint=scenario_hint)
    storyboard, resolved_planning_model, storyboard_payload = build_llm_storyboard(
        proxy_client=proxy_client,
        user_prompt=user_prompt,
        duration_s=duration_s,
        aspect_ratio=aspect_ratio,
        scenario_key=strategy.key,
        scenario_name=strategy.display_name,
        quality_tier=quality_tier,
        preferred_model=planning_model,
    )

    plan = build_project_plan(
        user_prompt=user_prompt,
        duration_s=duration_s,
        project_id=project_id,
        project_dir=str(project_dir_path),
        preferred_route=preferred_route,
        aspect_ratio=aspect_ratio,
        scenario_hint=scenario_hint,
        quality_tier=quality_tier,
        storyboard_override=storyboard,
        execution_mode="cpu",
        planning_backend=resolved_planning_model,
    )
    plan.notes.insert(0, "CPU 模式优先走 Gemini/NanoBanana/Veo 等远程 API；视频接口不可用时回退为首尾关键帧动作视频，而不是纯静态平移。")
    plan.notes.insert(1, "CPU 模式已启用审核阻断：首帧/尾帧/视频不合格时不会直接拼接成片。")
    plan.notes.insert(2, f"LLM 分镜采用 `{resolved_planning_model}`；若 `gemini-3-pro` 不可用会自动回退。")

    storyboard_path = project_dir_path / "storyboard_llm.json"
    storyboard_path.write_text(json.dumps(storyboard_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if save_plan:
        plan_path = project_dir_path / "plan.json"
        plan_path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    if dry_run:
        return {
            "success": True,
            "project_id": project_id,
            "project_dir": str(project_dir_path),
            "dry_run": True,
            "plan": plan.to_dict(),
            "storyboard_path": str(storyboard_path),
        }

    image_outputs: dict[int, Path] = {}
    failed_shots: list[int] = []
    review_summaries: dict[int, dict[str, str]] = {}

    for shot in plan.shots:
        shot_dir = project_dir_path / "shots" / f"shot_{shot.shot_id:03d}"
        for sub in ["assets", "video", "audio", "prompts", "reviews"]:
            (shot_dir / sub).mkdir(parents=True, exist_ok=True)
        _write_prompts(shot_dir, shot)

        reference_paths: list[Path] = []
        hero_image = image_outputs.get(1)
        previous_image = image_outputs.get(shot.shot_id - 1)
        if hero_image and hero_image.exists() and shot.shot_id != 1:
            reference_paths.append(hero_image)
        if previous_image and previous_image.exists() and previous_image not in reference_paths:
            reference_paths.append(previous_image)

        first_frame_path = shot_dir / "assets/first_frame.png"
        try:
            image_review, _ = _generate_reviewed_image(
                proxy_client=proxy_client,
                shot=shot,
                output_path=first_frame_path,
                review_dir=shot_dir / "reviews",
                base_prompt=shot.image_prompt,
                review_prompt=_image_review_prompt(shot, frame_role="first"),
                review_stem="first_frame_review",
                strict_fix=(
                    "Generate exactly one primary subject in one single static moment, no collage, no split screen, "
                    "no repeated person, no multi-time-step composition"
                ),
                image_model=image_model,
                reference_paths=reference_paths,
            )
            image_outputs[shot.shot_id] = first_frame_path
            review_summaries.setdefault(shot.shot_id, {})["image_review"] = str(shot_dir / "reviews/first_frame_review.json")
        except Exception as exc:
            failed_shots.append(shot.shot_id)
            (shot_dir / "reviews/first_frame_error.txt").write_text(str(exc), encoding="utf-8")
            logger.error("CPU image generation failed for shot %s: %s", shot.shot_id, exc)
            continue

    video_client = VenusVideoClient.from_env()
    
    # 定义哪些 route 支持首尾帧生成
    ROUTES_SUPPORT_LAST_FRAME = {
        "veo3", "veo-3.1-fast", "veo-3.1",  # Veo3 系列
        "wan22_flf2v",  # Wan2.2 的 first-last-frame 模式
    }
    
    for shot in plan.shots:
        shot_dir = project_dir_path / "shots" / f"shot_{shot.shot_id:03d}"
        if shot.shot_id not in image_outputs or shot.shot_id in failed_shots:
            continue

        image_path = image_outputs[shot.shot_id]
        last_frame_target = shot_dir / "assets/last_frame_target.png"
        raw_video_path = shot_dir / "video/raw_video.mp4"
        final_video_path = shot_dir / "video/final.mp4"

        # 根据 route 决定是否生成 last_frame
        route_name = shot.route.lower() if shot.route else preferred_route.lower()
        supports_last_frame = route_name in ROUTES_SUPPORT_LAST_FRAME
        
        if supports_last_frame:
            try:
                _generate_reviewed_image(
                    proxy_client=proxy_client,
                    shot=shot,
                    output_path=last_frame_target,
                    review_dir=shot_dir / "reviews",
                    base_prompt=shot.end_frame_prompt,
                    review_prompt=_image_review_prompt(shot, frame_role="end"),
                    review_stem="last_frame_review",
                    strict_fix=(
                        "Generate the same character and scene continuity, but as the ending moment only; no repeated subject, "
                        "no collage, no split screen, no mixed time steps"
                    ),
                    image_model=image_model,
                    reference_paths=_build_end_frame_reference_paths(shot, image_outputs),
                )
                review_summaries.setdefault(shot.shot_id, {})["last_frame_review"] = str(shot_dir / "reviews/last_frame_review.json")
            except Exception as exc:
                failed_shots.append(shot.shot_id)
                (shot_dir / "reviews/last_frame_error.txt").write_text(str(exc), encoding="utf-8")
                logger.error("CPU last-frame generation failed for shot %s: %s", shot.shot_id, exc)
                continue
        else:
            logger.info("Shot %s route=%s 不支持首尾帧生成，跳过 last_frame 生成", shot.shot_id, route_name)

        middle_frame_target = shot_dir / "assets/middle_frame_target.png"
        middle_frame_prompt = _build_middle_frame_prompt(shot)
        (shot_dir / "prompts/middle_frame_prompt.txt").write_text(middle_frame_prompt, encoding="utf-8")

        generation_modes: list[str] = []
        if video_client and video_client.available:
            generation_modes.append("remote")
        generation_modes.append("keyframe_fallback")

        video_success = False
        current_video_prompt = shot.video_prompt
        for mode in generation_modes:
            try:
                if mode == "remote":
                    # Veo3 支持首尾帧，只在有 last_frame_target 时传入
                    video_client.generate_veo_image2video(
                        prompt=current_video_prompt,
                        first_frame_path=image_path,
                        last_frame_path=last_frame_target if (supports_last_frame and last_frame_target.exists()) else None,
                        output_path=raw_video_path,
                        model=video_model,
                        aspect_ratio=shot.aspect_ratio,
                        resolution="720p",
                        generate_audio=False,
                        duration_s=shot.duration_s,
                    )
                else:
                    # keyframe_fallback：优先保证主体一致性，默认仅使用首帧+尾帧，不强依赖 middle_frame
                    if supports_last_frame and last_frame_target.exists():
                        middle_image_for_fallback = None
                        try:
                            proxy_client.generate_image(
                                prompt=middle_frame_prompt,
                                output_path=middle_frame_target,
                                model=image_model,
                                size=shot.image_size,
                                reference_image_paths=[image_path, last_frame_target],
                            )
                            _normalize_image_size(middle_frame_target, shot.image_size)
                            # 为了稳定性，默认不启用 middle_frame 参与合成，避免身份漂移。
                            # 如需更强动作，可改为 middle_image_for_fallback = str(middle_frame_target)
                        except Exception as middle_exc:
                            (shot_dir / "reviews/middle_frame_error.txt").write_text(str(middle_exc), encoding="utf-8")
                            logger.warning("Middle-frame generation failed for shot %s: %s", shot.shot_id, middle_exc)

                        make_story_motion_video_from_frames(
                            first_image=str(image_path),
                            middle_image=middle_image_for_fallback,
                            last_image=str(last_frame_target),
                            output=str(raw_video_path),
                            duration_s=shot.duration_s,
                            size=shot.image_size,
                        )
                    else:
                        raise RuntimeError(
                            f"Route {route_name} 不支持首尾帧生成，且 remote 模式不可用。"
                            "请使用支持的模型（veo3/wan22_flf2v）或启用 video_client。"
                        )
            except Exception as exc:
                error_file = shot_dir / "reviews" / f"video_{mode}_error.txt"
                error_file.write_text(str(exc), encoding="utf-8")
                logger.warning("Video generation failed for shot %s mode=%s: %s", shot.shot_id, mode, exc)
                continue

            if mode == "remote":
                trim_video(video=str(raw_video_path), output=str(final_video_path), duration_s=shot.duration_s)
            else:
                final_video_path.write_bytes(raw_video_path.read_bytes())
            video_review = _safe_video_frame_review(
                proxy_client=proxy_client,
                shot=shot,
                first_frame_path=image_path,
                video_path=final_video_path,
                review_dir=shot_dir / "reviews",
            )
            review_file = shot_dir / "reviews" / f"video_review_{mode}.json"
            review_file.write_text(video_review, encoding="utf-8")
            review_summaries.setdefault(shot.shot_id, {})[f"video_review_{mode}"] = str(review_file)

            parsed_video_review = _normalize_review_result(video_review)
            if not _review_failed(parsed_video_review, VIDEO_FAILURE_KEYWORDS, minimum_score=VIDEO_MIN_REVIEW_SCORE):
                (shot_dir / "reviews/video_review.json").write_text(video_review, encoding="utf-8")
                review_summaries.setdefault(shot.shot_id, {})["video_review"] = str(shot_dir / "reviews/video_review.json")
                video_success = True
                break

            current_video_prompt = _refine_prompt_with_review(
                shot.video_prompt,
                parsed_video_review,
                "Video must show real action progression from the start frame to the end frame, not a static pan/zoom-only animation, and identity must remain stable",
            )
            (shot_dir / "reviews/video_retry_prompt.txt").write_text(current_video_prompt, encoding="utf-8")

        if not video_success:
            # 稳定性兜底：若已产出可播放片段，但审核未通过，仍纳入最终拼接，避免整镜头丢失
            if final_video_path.exists() and final_video_path.stat().st_size > 0:
                logger.warning(
                    "Shot %s quality gate failed, but keep best-effort clip for concat: %s",
                    shot.shot_id,
                    final_video_path,
                )
                (shot_dir / "reviews/video_quality_gate_failed.txt").write_text(
                    "Quality gate failed after retries; keep best-effort clip to avoid missing shot in final video.",
                    encoding="utf-8",
                )
                review_summaries.setdefault(shot.shot_id, {})["video_quality_gate_failed"] = str(
                    shot_dir / "reviews/video_quality_gate_failed.txt"
                )
                video_success = True
            else:
                failed_shots.append(shot.shot_id)
                logger.error("CPU video generation failed quality gate for shot %s", shot.shot_id)
                continue

    available_videos: list[str] = []
    for shot in plan.shots:
        if shot.shot_id in failed_shots:
            continue
        final_video_path = project_dir_path / "shots" / f"shot_{shot.shot_id:03d}" / "video/final.mp4"
        if final_video_path.exists():
            available_videos.append(str(final_video_path))

    output_video = project_dir_path / "final_story.mp4"
    if available_videos:
        logger.info("[CPU] 拼接成片：clips=%d", len(available_videos))
        concat_videos(video_paths=available_videos, output=str(output_video))

    return {
        "success": bool(available_videos) and not failed_shots,
        "project_id": project_id,
        "project_dir": str(project_dir_path),
        "output_video": str(output_video if output_video.exists() else Path(available_videos[-1]) if available_videos else project_dir_path),
        "failed_shots": failed_shots,
        "plan": plan.to_dict(),
        "storyboard_path": str(storyboard_path),
        "review_summaries": review_summaries,
    }
