from __future__ import annotations

import re
from typing import List, Optional


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SENTENCE_SPLIT_RE = re.compile(r"[\n。！？!?；;]+")
_DIALOGUE_RE = re.compile(r"[“\"]([^”\"]+)[”\"]")


_SINGLE_SUBJECT_GUARDRAILS = (
    "single exact moment captured in one static photograph, one camera angle only, exactly one primary subject, "
    "strictly no collage, no split screen, no multi-panel layout, no triptych, no storyboard grid, "
    "no repeated person, no duplicate face, no same person appearing multiple times, "
    "no multi-exposure, no motion trail, no mirrored clone, "
    "单一静态画面，禁止三联画分格，禁止分镜板格式，禁止同一人物重复出现"
)


_VIDEO_GUARDRAILS = (
    "single continuous shot, real motion progression from the reference start frame to the target end state, "
    "not a slideshow, not a static image pan only, not a pure zoom only, no duplicated subject, "
    "no abrupt shot change, no montage, no split screen"
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_story_units(text: str) -> List[str]:
    parts = [normalize_text(part) for part in _SENTENCE_SPLIT_RE.split(text) if normalize_text(part)]
    return parts or [normalize_text(text)]


def extract_dialogue(text: str) -> str:
    matches = _DIALOGUE_RE.findall(text)
    if matches:
        return " ".join(normalize_text(item) for item in matches)
    for marker in ("台词:", "对白:", "dialogue:", "speech:"):
        if marker in text:
            return normalize_text(text.split(marker, 1)[1])
    return ""


def space_cjk_text(text: str) -> str:
    spaced = " ".join(char for char in text.strip())
    spaced = re.sub(r"\s+", " ", spaced)
    return spaced.strip()


def make_text_agent_instruction(
    user_prompt: str,
    route: str,
    duration_s: float,
    aspect_ratio: str,
    scenario_name: str = "通用视频",
    storyboard_mode: str = "narrative_beats",
    review_focus: Optional[List[str]] = None,
) -> str:
    review_items = "、".join(review_focus or ["主体一致性", "镜头连贯性", "音画同步", "审美质量"])
    return f"""你是视频创作规划 Agent。请先把用户原始需求改写成可执行的生成计划，不要改变核心意图。

场景类型: {scenario_name}
目标线路: {route}
目标时长: {duration_s:.1f}s
目标画幅: {aspect_ratio}
分镜模式: {storyboard_mode}
重点核验: {review_items}

请输出时务必覆盖以下字段：
1. subject（主体设定与人物锚点）
2. scene（场景/光线/镜头语言/题材风格）
3. continuity（角色、服装、机位、时空连续性）
4. storyboard（如何拆镜、每一镜的目标画面与时长）
5. audio（对白 / BGM / SFX 拆分）
6. generation_chain（每一阶段用什么后端、何时回退）
7. review_chain（每一阶段的 VLM / 审核 / 审美核验点）
8. retry_policy（失败后是重试 prompt、重抽首帧、换模型还是人工兜底）

用户原始需求：
{user_prompt}
""".strip()


def make_image_prompt(
    description: str,
    aspect_ratio: str,
    style_hint: str = "",
    visual_focus: str = "",
) -> str:
    orientation = "portrait composition" if aspect_ratio == "9:16" else "landscape composition"
    style_clause = f", {normalize_text(style_hint)}" if style_hint else ""
    focus_clause = f", visual focus: {normalize_text(visual_focus)}" if visual_focus else ""
    return normalize_text(
        f"a single high-quality photograph, {description}, {_SINGLE_SUBJECT_GUARDRAILS}, {orientation}, cinematic lighting, "
        f"natural skin details, high detail, clean face, clean hands, no watermark, no text overlay{focus_clause}{style_clause}"
    )


def make_last_frame_prompt(
    description: str,
    aspect_ratio: str,
    style_hint: str = "",
    visual_focus: str = "",
) -> str:
    orientation = "portrait composition" if aspect_ratio == "9:16" else "landscape composition"
    style_clause = f", {normalize_text(style_hint)}" if style_hint else ""
    focus_clause = f", visual focus: {normalize_text(visual_focus)}" if visual_focus else ""
    return normalize_text(
        f"target ending frame, {description}, {_SINGLE_SUBJECT_GUARDRAILS}, same person, same outfit, same scene continuity, "
        f"{orientation}, cinematic lighting, high detail, clean face, clean hands, no watermark, no text overlay{focus_clause}{style_clause}"
    )


def make_i2v_prompt(
    description: str,
    speaking: bool,
    motion_hint: str = "",
    quality_hint: str = "",
    start_frame: str = "",
    end_frame: str = "",
    visual_focus: str = "",
) -> str:
    motion = (
        "subtle head movement, natural blinking, mild body sway, speaking to camera"
        if speaking
        else "coherent action progression, natural body movement, cinematic motion"
    )
    extra_parts = [
        normalize_text(motion_hint),
        normalize_text(quality_hint),
        f"start frame: {normalize_text(start_frame)}" if start_frame else "",
        f"target end frame: {normalize_text(end_frame)}" if end_frame else "",
        f"visual focus: {normalize_text(visual_focus)}" if visual_focus else "",
    ]
    extra = ", ".join(item for item in extra_parts if item)
    suffix = f", {extra}" if extra else ""
    return normalize_text(
        f"{description}, {motion}, {_VIDEO_GUARDRAILS}, preserve identity consistency, preserve outfit consistency, "
        f"realistic motion, stable composition{suffix}"
    )


def make_bgm_prompt(description: str) -> str:
    return normalize_text(
        f"instrumental background music for scene: {description}, cinematic, emotionally aligned, no vocals, clean mix, suitable for short video editing"
    )


def make_sfx_prompt(description: str) -> str:
    return normalize_text(
        f"environmental sound design for scene: {description}, realistic ambience, clean layers, short-video friendly, no speech"
    )


def make_magihuman_prompt(
    description: str,
    dialogue: str,
    bgm_prompt: str,
    style_hint: str = "",
    motion_hint: str = "",
) -> str:
    dialogue = normalize_text(dialogue) or "保持自然口型与细微表情变化，不需要额外对白。"
    bgm_prompt = normalize_text(bgm_prompt)
    dialogue_for_prompt = space_cjk_text(dialogue) if _CJK_RE.search(dialogue) else dialogue
    extra = ", ".join(item for item in [normalize_text(style_hint), normalize_text(motion_hint)] if item)
    extra_clause = f" Additional style guidance: {extra}." if extra else ""
    return f"""A single continuous shot video of {normalize_text(description)}. Keep the character identity, clothes, hairstyle and facial structure consistent with the reference image. Prioritize expressive face performance, believable lip motion, stable upper-body motion, natural blinking, subtle shoulder movement, realistic skin texture, clean background separation, and cinematic but production-safe lighting. Avoid abrupt camera switches, broken anatomy, text overlays, extra fingers, duplicated limbs, identity drift, heavy motion blur, split screens, collages, repeated subjects, and frozen pan-only shots.{extra_clause}

Dialogue: {dialogue_for_prompt}
Background Sound: {bgm_prompt}
""".strip()
