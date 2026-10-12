from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .prompting import extract_dialogue, normalize_text, split_story_units
from .schemas import AgentRoleSpec, ChainStep, ReviewSpec, StoryboardBeat


@dataclass(frozen=True)
class ScenarioStrategy:
    key: str
    display_name: str
    description: str
    storyboard_mode: str
    default_route: str
    style_hint: str
    motion_hint: str
    continuity_focus: str
    detection_keywords: tuple[str, ...]
    review_focus: tuple[str, ...]


_SCENARIO_ALIASES: Dict[str, str] = {
    "auto": "auto",
    "comic": "comic_drama",
    "comic_drama": "comic_drama",
    "manga": "comic_drama",
    "manhua": "comic_drama",
    "beauty": "beauty_glamour",
    "beauty_glamour": "beauty_glamour",
    "glamour": "beauty_glamour",
    "historical": "historical_reenactment",
    "history": "historical_reenactment",
    "historical_reenactment": "historical_reenactment",
    "avatar": "narration_avatar",
    "narration_avatar": "narration_avatar",
    "talking_head": "narration_avatar",
    "generic": "generic",
}


_SCENARIOS: Dict[str, ScenarioStrategy] = {
    "comic_drama": ScenarioStrategy(
        key="comic_drama",
        display_name="漫剧 / 漫画分镜视频",
        description="先生成网格分镜或关键格，再拆格作为首尾帧，适合剧情推进与强事件连续性。",
        storyboard_mode="comic_grid_panels",
        default_route="wan22_flf2v",
        style_hint="graphic novel panel style, comic layout readability, strong silhouette, clear panel composition",
        motion_hint="panel-to-panel continuity, motion bridge between key poses, clear action readability",
        continuity_focus="角色身份、服装、发型、关键道具在格间保持一致；优先保留上一格尾帧作为下一格起点。",
        detection_keywords=("漫剧", "漫画", "网格图", "格子", "分镜", "二次元", "番剧", "comic", "manga", "panel"),
        review_focus=("格间连贯", "角色一致性", "动作承接", "字幕与对白节奏", "审美统一"),
    ),
    "beauty_glamour": ScenarioStrategy(
        key="beauty_glamour",
        display_name="美女 / 颜值 / 时尚氛围视频",
        description="以脸、发丝、皮肤、妆造和动作优雅度为核心，减少无意义大动作，强调审美稳定。",
        storyboard_mode="hero_shot_sequence",
        default_route="wan22_i2v",
        style_hint="beauty editorial, flattering portrait lighting, premium skin texture, elegant fashion framing",
        motion_hint="gentle camera move, subtle pose transition, elegant gaze, smooth hair and cloth motion",
        continuity_focus="统一人物 look、妆容、服装、景别和镜头节奏，优先固定人设锚点后再扩动作。",
        detection_keywords=("美女", "颜值", "时尚", "穿搭", "氛围感", "美妆", "写真", "beauty", "fashion", "glamour"),
        review_focus=("脸部稳定", "皮肤与发丝质量", "神态和眼神", "审美得分", "服装连续性"),
    ),
    "historical_reenactment": ScenarioStrategy(
        key="historical_reenactment",
        display_name="历史 / 古风 / 时代再现场景",
        description="强调时代考据和服化道一致性，先锁定世界观和器物，再做镜头衔接与动作生成。",
        storyboard_mode="historical_beats",
        default_route="wan22_flf2v",
        style_hint="historical reenactment, era-accurate costume and props, cinematic world-building, grounded realism",
        motion_hint="measured motion, stable camera grammar, respect for scene geography, prop-aware action",
        continuity_focus="服装、发饰、建筑、器物、时代纹理必须保持一致，必要时用参考包先做核验。",
        detection_keywords=("历史", "古代", "宫廷", "朝代", "将军", "皇帝", "民国", "古风", "historical", "dynasty"),
        review_focus=("年代准确性", "服化道一致性", "建筑与器物合理性", "人物身份稳定", "镜头庄重感"),
    ),
    "narration_avatar": ScenarioStrategy(
        key="narration_avatar",
        display_name="数字人 / 口播讲解视频",
        description="优先保证口型、台词可懂度和主体稳定，再兼顾镜头节奏与包装元素。",
        storyboard_mode="speaker_segments",
        default_route="hyvideo_lipsync",
        style_hint="clean studio framing, presenter-centric composition, readable gestures, production-safe lighting",
        motion_hint="talking-head motion, subtle gesture, consistent eye line, stable upper-body framing",
        continuity_focus="主体身份、人脸结构、机位和收音风格必须稳定，台词与口型对齐优先级最高。",
        detection_keywords=("口播", "主播", "数字人", "讲解", "解说", "采访", "独白", "talking head", "narrator", "speaking"),
        review_focus=("口型同步", "ASR 对白一致", "脸部稳定", "镜头抖动", "字幕可读性"),
    ),
    "generic": ScenarioStrategy(
        key="generic",
        display_name="通用视频",
        description="按叙事单元拆镜，结合时长、对白和动作强度选择后端。",
        storyboard_mode="narrative_beats",
        default_route="hyvideo_lipsync",
        style_hint="cinematic coherence, stable identity, readable composition",
        motion_hint="natural motion, clear action intention, coherent scene continuity",
        continuity_focus="在同一故事单元内保持主体、服装和场景连续；跨单元允许合理转场。",
        detection_keywords=(),
        review_focus=("主体一致性", "动作连贯性", "音画同步", "构图与审美"),
    ),
}

_ACTION_HINTS = ("追", "跑", "冲", "打", "拥抱", "转身", "推门", "fight", "run", "chase", "jump")


def available_scenarios() -> List[str]:
    return list(_SCENARIOS.keys())


def _normalize_hint(scenario_hint: Optional[str]) -> Optional[str]:
    if not scenario_hint:
        return None
    return _SCENARIO_ALIASES.get(scenario_hint.strip().lower())


def detect_scenario(prompt: str, scenario_hint: Optional[str] = None) -> ScenarioStrategy:
    normalized_hint = _normalize_hint(scenario_hint)
    if normalized_hint and normalized_hint != "auto" and normalized_hint in _SCENARIOS:
        return _SCENARIOS[normalized_hint]

    prompt_lower = prompt.lower()
    best_key = "generic"
    best_score = 0

    for key, strategy in _SCENARIOS.items():
        if key == "generic":
            continue
        score = sum(1 for token in strategy.detection_keywords if token in prompt_lower)
        if score > best_score:
            best_key = key
            best_score = score

    return _SCENARIOS[best_key]


def build_agent_roles(strategy: ScenarioStrategy) -> List[AgentRoleSpec]:
    common_roles = [
        AgentRoleSpec(
            name="ScenarioRouterAgent",
            responsibility="识别题材并选择合适的生成与审核链路。",
            preferred_models=["通用 LLM", "轻量分类器"],
            deliverables=["scenario_tag", "route_policy", "retry_policy"],
        ),
        AgentRoleSpec(
            name="StoryboardAgent",
            responsibility="把剧本或需求拆成可执行分镜，并给出镜头目标、节奏和首尾帧锚点。",
            preferred_models=["剧本规划 LLM", "多模态规划模型"],
            deliverables=["storyboard", "shot_duration", "camera_notes"],
        ),
        AgentRoleSpec(
            name="ContinuityAgent",
            responsibility="维护角色、服装、道具、机位与时空连续性。",
            preferred_models=["VLM", "一致性打分模型"],
            deliverables=["continuity_notes", "reference_pack", "fallback_hint"],
        ),
        AgentRoleSpec(
            name="QCReviewAgent",
            responsibility="在首帧、视频片段和成片阶段进行质量审核、审美打分与回退决策。",
            preferred_models=["视频 VLM", "审美评分模型", "ASR 对齐模型"],
            deliverables=["review_report", "retry_decision", "risk_flags"],
        ),
    ]

    scenario_roles: Dict[str, List[AgentRoleSpec]] = {
        "comic_drama": [
            AgentRoleSpec(
                name="ComicGridAgent",
                responsibility="先生成漫画网格图/关键格，再拆格为首尾帧锚点。",
                preferred_models=["图像 LLM", "布局规划模型"],
                deliverables=["grid_layout", "panel_prompts", "panel_crop_plan"],
            )
        ],
        "beauty_glamour": [
            AgentRoleSpec(
                name="AestheticDirectorAgent",
                responsibility="统一脸部、妆造、姿态和镜头审美，压制畸形与廉价感。",
                preferred_models=["审美评分模型", "人脸细节检测模型"],
                deliverables=["aesthetic_guardrails", "look_anchor", "pose_plan"],
            )
        ],
        "historical_reenactment": [
            AgentRoleSpec(
                name="HistoricalResearchAgent",
                responsibility="检查年代、服化道、建筑和器物是否符合设定。",
                preferred_models=["知识增强 LLM", "图文检索/VLM"],
                deliverables=["era_constraints", "prop_reference", "world_consistency_report"],
            )
        ],
        "narration_avatar": [
            AgentRoleSpec(
                name="SpeechDirectorAgent",
                responsibility="围绕台词、口型和镜头稳定性组织口播片段。",
                preferred_models=["ASR 模型", "口型评分模型"],
                deliverables=["speech_segments", "lipsync_policy", "subtitle_rules"],
            )
        ],
    }
    return common_roles + scenario_roles.get(strategy.key, [])


def build_generation_chain(strategy: ScenarioStrategy, quality_tier: str = "balanced") -> List[ChainStep]:
    common_prefix = [
        ChainStep(
            stage="intent_parse",
            owner="ScenarioRouterAgent",
            goal="识别题材、主角设定、时长约束和后端策略。",
            preferred_models=["通用 LLM"],
            fallback_models=["规则路由器"],
            success_criteria=["场景分类正确", "约束提取完整", "已给出回退顺序"],
        ),
        ChainStep(
            stage="storyboard_plan",
            owner="StoryboardAgent",
            goal="输出分镜、时长、首尾帧锚点和镜头目标。",
            preferred_models=["剧本规划 LLM", "VLM"],
            fallback_models=["规则模板"],
            success_criteria=["每镜时长可执行", "镜头目标清晰", "转场逻辑合理"],
        ),
    ]

    scenario_middle: Dict[str, List[ChainStep]] = {
        "comic_drama": [
            ChainStep(
                stage="comic_grid_design",
                owner="ComicGridAgent",
                goal="生成漫画网格图或关键格，并产出拆格/裁切规划。",
                preferred_models=["图像布局模型", "一致性生图模型"],
                fallback_models=["单格关键帧生成"],
                success_criteria=["格子语义完整", "关键动作已覆盖", "主角形象稳定"],
            ),
            ChainStep(
                stage="panel_to_video",
                owner="ContinuityAgent",
                goal="按格间连续性把关键格扩展为短视频片段。",
                preferred_models=["Wan2.2 FLF2V", "Wan2.2 I2V"],
                fallback_models=["HyVideo 1.5"],
                success_criteria=["格间动作承接自然", "角色与道具一致", "镜头节奏符合剧情"],
            ),
        ],
        "beauty_glamour": [
            ChainStep(
                stage="look_anchor_pack",
                owner="AestheticDirectorAgent",
                goal="先锁定人设 look、妆造和镜头审美，再放大动作。",
                preferred_models=["高质量 T2I", "审美评分模型"],
                fallback_models=["通用首帧生成"],
                success_criteria=["脸部稳定", "妆造统一", "审美噪声受控"],
            ),
            ChainStep(
                stage="hero_video_generation",
                owner="ContinuityAgent",
                goal="生成小幅但高质的镜头运动，避免廉价和畸形。",
                preferred_models=["Wan2.2 I2V", "Wan2.2 FLF2V"],
                fallback_models=["HyVideo 1.5"],
                success_criteria=["动作优雅", "脸和手稳定", "光影统一"],
            ),
        ],
        "historical_reenactment": [
            ChainStep(
                stage="world_reference_check",
                owner="HistoricalResearchAgent",
                goal="核定时代设定、服化道和器物边界，再进入镜头生成。",
                preferred_models=["知识增强 LLM", "检索/VLM"],
                fallback_models=["人工参考包"],
                success_criteria=["时代约束明确", "关键道具可追踪", "场景风格一致"],
            ),
            ChainStep(
                stage="historical_video_generation",
                owner="ContinuityAgent",
                goal="生成镜头时保持时代感和世界观稳定。",
                preferred_models=["Wan2.2 FLF2V", "Wan2.2 I2V"],
                fallback_models=["HyVideo 1.5"],
                success_criteria=["服化道连续", "空间关系清楚", "运动不过度浮夸"],
            ),
        ],
        "narration_avatar": [
            ChainStep(
                stage="speech_layout",
                owner="SpeechDirectorAgent",
                goal="按台词时长拆段并决定嘴型、字幕和机位节奏。",
                preferred_models=["ASR", "口播规划 LLM"],
                fallback_models=["规则切段"],
                success_criteria=["段落长度合理", "每段口型任务清晰", "机位不漂移"],
            ),
            ChainStep(
                stage="lipsync_video_generation",
                owner="ContinuityAgent",
                goal="在保证脸部稳定的前提下生成口播视频。",
                preferred_models=["HyVideo 1.5 + LatentSync", "Wan2.2 + LatentSync"],
                fallback_models=["静态口播模板"],
                success_criteria=["口型同步", "台词可懂度高", "镜头稳定"],
            ),
        ],
    }

    suffix = [
        ChainStep(
            stage="segment_review",
            owner="QCReviewAgent",
            goal="对首帧、片段和音频进行审核，并决定是否重试。",
            preferred_models=["视频 VLM", "审美评分模型", "ASR 对齐模型"],
            fallback_models=["人工抽检"],
            success_criteria=["关键缺陷被发现", "给出明确重试动作", "保留失败原因"],
        ),
        ChainStep(
            stage="final_edit_and_review",
            owner="QCReviewAgent",
            goal="合成成片并进行最终审查、节奏和包装检查。",
            preferred_models=["视频 VLM", "审美评分模型"],
            fallback_models=["人工终审"],
            success_criteria=["成片节奏完整", "音画同步", "无明显安全或审美问题"],
        ),
    ]

    quality_overlay = {
        "speed": "偏重速度，优先减少二次核验与大模型重试。",
        "balanced": "平衡质量与成本，关键节点核验。",
        "quality": "偏重质量，允许更多一致性检查和审美复核。",
    }.get(quality_tier, "平衡质量与成本，关键节点核验。")

    chain = common_prefix + scenario_middle.get(strategy.key, []) + suffix
    chain.append(
        ChainStep(
            stage="quality_policy",
            owner="ScenarioRouterAgent",
            goal=quality_overlay,
            preferred_models=[quality_tier],
            fallback_models=[],
            success_criteria=["质量策略已显式声明"],
        )
    )
    return chain


def build_project_review_plan(strategy: ScenarioStrategy) -> List[ReviewSpec]:
    return [
        ReviewSpec(
            name="storyboard_review",
            stage="planning",
            reviewer="StoryboardAgent",
            objective="在真正生成前审查分镜是否适合该题材。",
            checks=[
                f"是否符合 {strategy.display_name} 的分镜模式",
                "每镜时长是否可执行",
                "是否给出明确首尾帧/参考图锚点",
                f"是否覆盖重点核验：{'、'.join(strategy.review_focus)}",
            ],
            suggested_models=["多模态 LLM", "VLM"],
            fail_action="rewrite_storyboard",
            blocking=True,
        ),
        ReviewSpec(
            name="final_delivery_review",
            stage="delivery",
            reviewer="QCReviewAgent",
            objective="成片交付前做合规、质量和审美的最终确认。",
            checks=[
                "音画是否一致",
                "是否存在明显畸形、闪烁、错位",
                "题材风格是否统一",
                "是否需要人工终审或抽检",
            ],
            suggested_models=["视频 VLM", "审美评分模型", "审核模型"],
            fail_action="human_review",
            blocking=True,
        ),
    ]


def build_shot_review_plan(strategy: ScenarioStrategy, has_dialogue: bool, shot_type: str) -> List[ReviewSpec]:
    checks = [
        "主体身份是否稳定",
        "服装、场景和镜头语言是否连续",
        f"是否满足题材重点：{'、'.join(strategy.review_focus)}",
    ]
    if has_dialogue:
        checks.append("口型、语音、字幕是否对齐")

    shot_specific = {
        "comic_drama": ["关键格之间是否可读", "动作桥接是否自然"],
        "beauty_glamour": ["脸、皮肤、发丝是否自然", "动作和眼神是否高级"],
        "historical_reenactment": ["服化道和器物是否符合设定", "时代氛围是否统一"],
        "narration_avatar": ["口播机位是否稳定", "手势和头部运动是否克制"],
    }
    checks.extend(shot_specific.get(strategy.key, []))
    if shot_type:
        checks.append(f"镜头类型 `{shot_type}` 是否达到预期")

    review_specs = [
        ReviewSpec(
            name="frame_and_prompt_review",
            stage="image_generation",
            reviewer="QCReviewAgent",
            objective="首帧生成后立即检查人设、风格和关键道具。",
            checks=checks,
            suggested_models=["图像 VLM", "审美评分模型"],
            fail_action="retry_image_prompt",
            blocking=True,
        ),
        ReviewSpec(
            name="clip_review",
            stage="video_generation",
            reviewer="QCReviewAgent",
            objective="短视频片段生成后检查动作、连贯性和明显缺陷。",
            checks=checks + ["是否存在闪烁、断帧、肢体畸形或镜头跳变"],
            suggested_models=["视频 VLM", "动作质量打分模型"],
            fail_action="retry_video_with_fallback",
            blocking=True,
        ),
    ]
    if has_dialogue:
        review_specs.append(
            ReviewSpec(
                name="speech_alignment_review",
                stage="audio_video_sync",
                reviewer="QCReviewAgent",
                objective="对白场景额外检查 ASR、口型和情绪匹配。",
                checks=["ASR 文本与原对白是否一致", "口型和音频是否同步", "情绪表达是否自然"],
                suggested_models=["ASR 模型", "口型评分模型", "视频 VLM"],
                fail_action="retry_lipsync_or_trim_audio",
                blocking=True,
            )
        )
    return review_specs


def build_strategy_summary(strategy: ScenarioStrategy) -> List[str]:
    return [
        strategy.description,
        f"分镜模式采用 `{strategy.storyboard_mode}`。",
        f"连续性重点：{strategy.continuity_focus}",
        f"重点核验：{'、'.join(strategy.review_focus)}。",
    ]


def build_storyboard_beats(prompt: str, duration_s: float, strategy: ScenarioStrategy) -> List[StoryboardBeat]:
    units = split_story_units(prompt)
    if strategy.key == "comic_drama":
        drafts = _build_comic_drafts(units, duration_s)
    elif strategy.key == "beauty_glamour":
        drafts = _build_beauty_drafts(units, duration_s)
    elif strategy.key == "historical_reenactment":
        drafts = _build_history_drafts(units, duration_s)
    elif strategy.key == "narration_avatar":
        drafts = _build_avatar_drafts(units, duration_s)
    else:
        drafts = _build_generic_drafts(units, duration_s)
    return _materialize_beats(drafts, duration_s)


def _build_generic_drafts(units: List[str], duration_s: float) -> List[Dict[str, object]]:
    base_duration = duration_s / max(len(units), 1)
    drafts: List[Dict[str, object]] = []
    for unit in units:
        segment_count = max(1, int(round(base_duration / 4.5))) if base_duration > 5 else 1
        segment_count = min(segment_count, 3)
        for index in range(segment_count):
            drafts.append(
                {
                    "summary": normalize_text(unit) if segment_count == 1 else f"{normalize_text(unit)}；第{index + 1}段，保持人物与动作连续",
                    "shot_type": "narrative",
                    "source_mode": "direct_first_frame",
                    "continuity_mode": "story_unit",
                    "suggested_route": "hyvideo_longform" if base_duration > 5 else "hyvideo_lipsync",
                    "dialogue": extract_dialogue(unit),
                    "visual_focus": "按照叙事单元生成首帧，再扩展短视频片段。",
                    "transition_hint": "叙事连贯转场",
                    "needs_previous_frame": index > 0,
                    "agent_notes": ["如果单段过长则自动切片，并沿用上一段尾帧。"],
                    "weight": 1.0,
                }
            )
    return drafts


def _build_comic_drafts(units: List[str], duration_s: float) -> List[Dict[str, object]]:
    drafts: List[Dict[str, object]] = []
    base_duration = duration_s / max(len(units), 1)
    for unit in units:
        clean = normalize_text(unit)
        dialogue = extract_dialogue(clean)
        panel_count = 1
        if dialogue:
            panel_count += 1
        if _contains_keywords(clean, _ACTION_HINTS) or base_duration > 4.0:
            panel_count += 1
        panel_count = min(panel_count, 3)
        for panel_idx in range(panel_count):
            if panel_idx == 0:
                shot_type = "establish_panel"
                summary = f"{clean}；漫画网格第1格，建立角色、场景与情绪"
                route = "wan22_i2v"
            elif dialogue and panel_idx == 1:
                shot_type = "dialogue_panel"
                summary = f"{clean}；漫画网格第{panel_idx + 1}格，聚焦对白与表情变化"
                route = "wan22_lipsync"
            else:
                shot_type = "payoff_panel"
                summary = f"{clean}；漫画网格第{panel_idx + 1}格，承接上一格并给出动作结果"
                route = "wan22_flf2v"
            drafts.append(
                {
                    "summary": summary,
                    "shot_type": shot_type,
                    "source_mode": "comic_grid_panel",
                    "continuity_mode": "panel_to_panel",
                    "suggested_route": route,
                    "dialogue": dialogue if shot_type == "dialogue_panel" else "",
                    "visual_focus": "优先从漫画网格图拆格得到关键格，再把格子扩展成视频。",
                    "transition_hint": "切格过渡或速度桥接",
                    "needs_previous_frame": panel_idx > 0,
                    "agent_notes": [
                        "先做网格图/关键格，再拆格作为首尾帧。",
                        "对白格优先做表情和嘴型稳定。",
                    ],
                    "weight": 1.2 if shot_type == "dialogue_panel" else 1.0,
                }
            )
    return drafts


def _build_beauty_drafts(units: List[str], duration_s: float) -> List[Dict[str, object]]:
    source = [normalize_text(unit) for unit in units] or ["单人审美镜头"]
    base_duration = duration_s / max(len(source), 1)
    drafts: List[Dict[str, object]] = []
    for unit in source:
        shot_count = 2 if base_duration > 4.0 else 1
        shot_count = 3 if len(source) == 1 and duration_s >= 9.0 else shot_count
        templates = [
            ("hero_open", "建立人设与氛围，锁定脸部和妆造", "wan22_i2v"),
            ("hero_closeup", "拉近脸部、眼神和发丝细节", "wan22_i2v"),
            ("hero_motion", "做轻运动与姿态转折，保持高级感", "wan22_flf2v"),
        ]
        for idx in range(shot_count):
            shot_type, extra, route = templates[idx]
            drafts.append(
                {
                    "summary": f"{unit}；{extra}",
                    "shot_type": shot_type,
                    "source_mode": "hero_reference_frame",
                    "continuity_mode": "look_anchor",
                    "suggested_route": route,
                    "dialogue": extract_dialogue(unit),
                    "visual_focus": "先锁定脸部、皮肤、发丝和服装质感，再扩动作。",
                    "transition_hint": "柔和视线/机位切换",
                    "needs_previous_frame": idx > 0,
                    "agent_notes": [
                        "首帧阶段就做审美筛查，低分样本不进入视频生成。",
                        "优先使用同一套人设锚图维持 look。",
                    ],
                    "weight": 1.1 if shot_type == "hero_closeup" else 1.0,
                }
            )
    return drafts


def _build_history_drafts(units: List[str], duration_s: float) -> List[Dict[str, object]]:
    source = [normalize_text(unit) for unit in units] or ["历史场景再现"]
    if len(source) == 1 and duration_s >= 9.0:
        unit = source[0]
        source = [
            f"{unit}；先建立时代环境与空间关系",
            f"{unit}；再推进人物动作与关键事件",
            f"{unit}；最后给出结果和氛围余韵",
        ]
    drafts: List[Dict[str, object]] = []
    for idx, unit in enumerate(source):
        dialogue = extract_dialogue(unit)
        route = "hyvideo_lipsync" if dialogue else ("wan22_flf2v" if idx > 0 else "wan22_i2v")
        shot_type = "historical_dialogue" if dialogue else ("historical_establish" if idx == 0 else "historical_action")
        drafts.append(
            {
                "summary": unit,
                "shot_type": shot_type,
                "source_mode": "historical_reference_pack",
                "continuity_mode": "era_consistency",
                "suggested_route": route,
                "dialogue": dialogue,
                "visual_focus": "优先检查年代、服化道、器物和建筑，再做镜头运动。",
                "transition_hint": "庄重、清晰的叙事转场",
                "needs_previous_frame": idx > 0,
                "agent_notes": [
                    "先构建时代参考包，再生成首帧与视频。",
                    "服化道错误优先回退到首帧或参考包阶段修正。",
                ],
                "weight": 1.15 if shot_type == "historical_action" else 1.0,
            }
        )
    return drafts


def _build_avatar_drafts(units: List[str], duration_s: float) -> List[Dict[str, object]]:
    drafts: List[Dict[str, object]] = []
    base_duration = duration_s / max(len(units), 1)
    for unit in units:
        clean = normalize_text(unit)
        segment_count = max(1, int(round(base_duration / 4.0))) if base_duration > 5 else 1
        segment_count = min(segment_count, 3)
        for idx in range(segment_count):
            drafts.append(
                {
                    "summary": clean if segment_count == 1 else f"{clean}；口播第{idx + 1}段，保持机位和口型稳定",
                    "shot_type": "speaker_segment",
                    "source_mode": "speaker_anchor_frame",
                    "continuity_mode": "identity_lock",
                    "suggested_route": "hyvideo_lipsync",
                    "dialogue": extract_dialogue(clean) or clean,
                    "visual_focus": "主体稳定、嘴型清晰、机位克制。",
                    "transition_hint": "字幕或包装辅助转场",
                    "needs_previous_frame": idx > 0,
                    "agent_notes": ["对白主导时长规划，视频时长服从语音和口型。"],
                    "weight": 1.0,
                }
            )
    return drafts


def _materialize_beats(drafts: List[Dict[str, object]], duration_s: float) -> List[StoryboardBeat]:
    if not drafts:
        return []

    total_weight = sum(float(draft.get("weight", 1.0)) for draft in drafts)
    remaining = round(duration_s, 2)
    beats: List[StoryboardBeat] = []

    for index, draft in enumerate(drafts, start=1):
        if index == len(drafts):
            shot_duration = max(0.6, round(remaining, 2))
        else:
            shot_duration = max(0.6, round(duration_s * float(draft.get("weight", 1.0)) / total_weight, 2))
            remaining = round(remaining - shot_duration, 2)

        beats.append(
            StoryboardBeat(
                beat_id=index,
                summary=str(draft["summary"]),
                duration_s=shot_duration,
                shot_type=str(draft["shot_type"]),
                source_mode=str(draft["source_mode"]),
                continuity_mode=str(draft["continuity_mode"]),
                suggested_route=str(draft["suggested_route"]),
                dialogue=str(draft.get("dialogue", "")),
                visual_focus=str(draft.get("visual_focus", "")),
                transition_hint=str(draft.get("transition_hint", "")),
                needs_previous_frame=bool(draft.get("needs_previous_frame", False)),
                agent_notes=list(draft.get("agent_notes", [])),
            )
        )
    return beats


def _contains_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)
