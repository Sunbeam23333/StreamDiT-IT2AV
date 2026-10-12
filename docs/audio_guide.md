# 音频使用指南：TTS · 音效 · BGM · 歌曲

> 这是本仓库的重点模块。四类音频各司其职：**TTS 负责台词，TangoFlux 负责环境音 / 拟音，ACE-Step 负责背景音乐，SongGeneration 负责带人声的完整歌曲**。
> 环境问题请先看 [环境搭建与踩坑全记录](environment_pitfalls.md)，模型原理见 [参考框架详解](reference_frameworks.md)。

- [0. 准备工作](#0-准备工作)
- [1. TTS（台词 / 旁白）](#1-tts台词--旁白)
- [2. 音效 · TangoFlux](#2-音效--tangoflux)
- [3. 背景音乐 · ACE-Step](#3-背景音乐--ace-step)
- [4. 歌曲 · SongGeneration](#4-歌曲--songgeneration)
- [5. 一键 benchmark 与 prompt 文件格式](#5-一键-benchmark-与-prompt-文件格式)
- [6. 混音建议](#6-混音建议)

## 能力一览

| 模型 | 类型 | 输入 | 语言 | 时长控制 | 采样率 | 显存 |
|---|---|---|---|---|---|---|
| Spark-TTS | TTS | 文本 + 性别 / 音高 / 语速，或参考音频 | 中 / 英 | 由文本决定 | 16 kHz | ~4 GB |
| CosyVoice 2 / 3 | TTS | 文本 + 预置音色 / 参考音频 / instruct | 9 种语言 + 18 种方言 | 由文本决定 | 24 kHz | ~4 GB |
| F5-TTS | TTS | 文本 + 参考音频 + 参考文本 | 中 / 英 | 由文本决定（可 `fix_duration`） | 24 kHz | ~4 GB |
| ChatTTS | TTS | 文本 + speaker seed + 口语标记 | 中 / 英 | 由文本决定 | 24 kHz | ~4 GB |
| 混元 TTS 2.5 | TTS（API） | 文本 + 音色 | 中 / 英 | 由文本决定 | 24 kHz | 0 |
| TangoFlux | 音效 | 英文自然语言描述 | 英 | ✅ 1–30 s | 44.1 kHz 立体声 | ~4 GB |
| ACE-Step | BGM / 歌曲 | 风格 tag + 结构化歌词 | 17 种 | ✅ ≤240 s | 48 kHz | ~16 GB |
| SongGeneration v2 | 歌曲 | 描述 + 结构化歌词 + 预置风格 | 中 / 英 | ❌ 由歌词长度决定（≤4 分 30 秒） | 48 kHz | ~32 GB |

> 显存一列是单卡的粗略参考（上游文档 + 我们实际能跑起来的配置），不是 profiler 测得的峰值。

---

## 0. 准备工作

```bash
# 1) 拉取音频框架 (固定 commit; SongGeneration 已 vendored)
bash third_party/fetch_third_party.sh audio

# 2) 下载权重到 ${IT2AV_WORKSPACE}/models/Audio
python download/download_audio_models.py --models all           # 全部音频模型
python download/download_audio_models.py --models audio music   # 音效 (TangoFlux + flan-t5) + 音乐 (ACE-Step + SongGeneration)
python download/download_audio_models.py --models tts           # 全部 TTS (CosyVoice2/3 + ttsfrd + F5 + Fish + Spark + ChatTTS)
#   单个 key: tangoflux flan_t5_large ace_step songgen_v2_large songgen_runtime
#             cosyvoice2 cosyvoice3 cosyvoice_ttsfrd f5tts fish_speech spark_tts chattts
#   分组:     all / audio / music / tts / tts_alt (只含 F5 / Fish / Spark / ChatTTS)

# 3) 环境 (二选一, 见 docs/environment_pitfalls.md#0-推荐环境布局)
bash script/audio/setup_audio_envs.sh all          # 分环境 (省心)
bash script/audio/setup_tts_alternatives.sh        # TTS 备选框架的环境
```

---

## 1. TTS（台词 / 旁白）

### 1.1 统一 TTS benchmark

```bash
# 默认对比 CosyVoice2 / Spark-TTS / ChatTTS / F5-TTS
bash script/audio/run_tts_benchmark.sh

bash script/audio/run_tts_benchmark.sh --models cosyvoice2,spark_tts,chattts,f5tts --gpu 0 --language zh
bash script/audio/run_tts_benchmark.sh --models f5tts --ref_audio /path/ref.wav --ref_text "参考音频的逐字转写"
bash script/audio/run_tts_benchmark.sh --scene_ids scene_01_pure_music scene_04_anime
```

底层是 `run_tts_benchmark.py`，单模型也可以直接调：

```bash
python run_tts_benchmark.py --model spark_tts \
    --model_path ${IT2AV_WORKSPACE}/models/Audio/Spark-TTS-0.5B \
    --code_dir utils/audio/Spark-TTS \
    --prompts_file script/audio/prompts_tts_voice_style_compare.json \
    --output_dir outputs/tts_spark \
    --spark_gender female --spark_pitch moderate --spark_speed moderate
```

`script/audio/prompts_tts_voice_style_compare.json` 内置 8 种风格（新闻播报、纪录片旁白、高端广告、直播带货、睡前故事、动漫角色、激励演讲、客服接待），用来对比不同模型的音色和情绪表现；每个模型能做什么、怎么调用，机器可读的版本在 `script/audio/tts_capabilities.json`。

### 1.2 各模型的控制方式

| 模型 | 音色来源 | 风格 / 效果控制 | 关键参数 |
|---|---|---|---|
| **Spark-TTS** | 不给参考音频时"造音色"；给参考音频时克隆 | `gender=female/male`、`pitch/speed=very_low…very_high` | `--spark_gender --spark_pitch --spark_speed` |
| **CosyVoice2** | SFT 预置音色（`cosyvoice.list_available_spks()`，如 `中文女` / `中文男`）、zero-shot、cross-lingual | 语言标签 `<\|zh\|>` `<\|en\|>`；事件标记 `[breath]` `[noise]` `[laughter]` `[cough]` `[sigh]`；`<laughter>…</laughter>` `<strong>…</strong>` | `--cosyvoice_speaker`、`--cosyvoice_ref_audio/--cosyvoice_ref_text`、`--cosyvoice_speed` |
| **CosyVoice3** | zero-shot + instruct2 | instruct：18 种方言（粤语、四川话、东北话……）、大声 / 轻声、快 / 慢、开心 / 伤心 / 生气、机器人风格等 | `--tts_engine cosyvoice3`（默认不推荐） |
| **F5-TTS** | 参考音频（必需） | 多说话人 TOML 配置 | `nfe_step`、`cfg_strength`、`sway_sampling_coef`、`speed`、`fix_duration`、`seed` |
| **ChatTTS** | 随机 speaker embedding（固定 seed 才能复现） | `[speed_5]` `[oral_2]` `[laugh_0]` `[break_4]` `[uv_break]` `[laugh]` `[lbreak]` | `--chattts_seed` |
| **混元 TTS 2.5** | 平台音色 | 语速 / 音量 | `utils/audio/Hunyuan-tts/hunyuan_tts.py --sample_rate` |

### 1.3 经验

- 数字人口播首选 **Spark-TTS**（稳定，不需要参考音频）或 **混元 TTS**（云端，质量最好）；
- 要克隆某个具体声音用 **F5-TTS**，参考音频 5–10 秒、干净无 BGM，参考文本要和音频逐字一致；
- **CosyVoice 在我们的环境里会输出"非人类语言"**，用之前请先跑 `bash script/digit/run_debug_tts.sh` 自检，原理见 [TTS 评测](tts_evaluation.md)；
- TTS 时长决定整条视频的时长（pipeline2 会把视频慢放到 TTS 长度、把 BGM / SFX 截到 TTS 长度）。

---

## 2. 音效 · TangoFlux

```bash
bash script/audio/run_tangoflux.sh                         # 独立运行 (~4 GB 显存)
bash script/audio/run_audio_benchmark.sh tangoflux 0       # 在 GPU 0 上跑 benchmark
bash script/audio/run_audio_benchmark.sh tangoflux 0 zh 10 # 所有场景只生成 10 s, 快速验证
```

**prompt 写法**：纯英文自然语言，描述"有什么声音、在什么环境、什么质感"，不需要 tag 或结构：

```text
Gentle mountain wind blowing through valleys, distant bird songs echoing, soft flowing river water sounds, peaceful morning nature ambience
Hammer slowly hitting the wooden table
Rain falling on a metal roof with occasional thunder
Crowd cheering in a basketball arena, sneakers squeaking on the hardwood floor
```

- 时长 1–30 s（`duration`），推荐 `steps=50, guidance_scale=4.5`；
- 中文 prompt 会明显变差（文本编码器是 flan-t5）；
- 不要在 prompt 里写音乐或人声，那是 ACE-Step 和 TTS 的活。

---

## 3. 背景音乐 · ACE-Step

```bash
bash script/audio/run_ace_step.sh                          # 独立运行 (~16 GB 显存)
bash script/audio/run_audio_benchmark.sh ace_step 1 zh 60  # GPU 1, 每段 60 s
```

**输入 = 风格 tag（`prompt`）+ 结构化歌词（`lyrics`）**。

### 3.1 风格 tag

逗号分隔，通常包含：流派、乐器、BPM、情绪、是否有人声。官方预设示例：

| 风格 | tag |
|---|---|
| Modern Pop | `pop, synth, drums, guitar, 120 bpm, upbeat, catchy, vibrant, female vocals` |
| Rock | `rock, electric guitar, drums, bass, 130 bpm, energetic, rebellious, gritty, male vocals` |
| Hip Hop | `hip hop, 808 bass, hi-hats, synth, 90 bpm, bold, urban, intense, male vocals` |
| EDM | `edm, synth, bass, kick drum, 128 bpm, euphoric, pulsating, energetic, instrumental` |
| Classical | `classical, orchestral, strings, piano, 60 bpm, elegant, emotive, timeless` |
| Jazz | `jazz, saxophone, piano, double bass, 110 bpm, smooth, improvisational, soulful` |
| 电影感 BGM（我们的用法） | `epic cinematic, orchestral, instrumental, majestic, sunrise, inspirational, strings and brass, slow tempo, 80 bpm` |

### 3.2 歌词结构标签

| 标签 | 含义 |
|---|---|
| `[verse]` / `[chorus]` / `[bridge]` | 主歌 / 副歌 / 桥段 |
| `[intro-short]` `[intro-medium]` | 前奏 |
| `[outro-short]` `[outro-medium]` | 尾奏 |
| `[inst-long]` | 纯器乐段 |

**纯音乐 BGM**：`lyrics = "[inst-long] ; [inst-long] ; [inst-long]"`，同时 tag 里写 `instrumental`。

### 3.3 推理参数

| 参数 | 值 | 说明 |
|---|---|---|
| `audio_duration` | ≤240 | ≤0 时在 30–240 s 间随机 |
| `infer_step` | 100 | |
| `guidance_scale` | 15 | |
| `scheduler_type` | `euler` | |
| `cfg_type` | `apg` | Adaptive Projected Guidance，高 CFG 下不容易过饱和 |
| `omega_scale` | 10 | |
| `guidance_interval` | 0.5 | 只在中间一段时间步施加 guidance |
| `min_guidance_scale` | 3 | |

支持语言：en, de, fr, es, it, pt, pl, tr, ru, cs, nl, ar, zh, ja, hu, ko, hi。

---

## 4. 歌曲 · SongGeneration

```bash
bash script/audio/run_songgen.sh                           # 独立运行 (~32 GB 显存)
bash script/audio/run_audio_benchmark.sh songgen 2
```

> ⚠️ 需要 `tools/new_auto_prompt.pt` 的真实文件（上游已下线），见 [踩坑 E3](environment_pitfalls.md#e3-new_auto_promptpt-是-git-lfs-指针)。SongGeneration **仅限非商业用途**。

**输入（JSONL 每行一条）**：

```json
{"idx": "scene_01", "descriptions": "female, synth-pop, sweet, synthesizer, drum machine, bass",
 "gt_lyric": "[intro-medium] ; [verse] 凌晨三点的便利店.冰柜发出持续的嗡鸣 ; [chorus] 这里是城市的守夜人 ; [outro-medium]",
 "auto_prompt_audio_type": "Pop"}
```

### 4.1 `descriptions`：四个维度，逗号分隔

| 维度 | 可选值示例 | 数量 |
|---|---|---|
| 性别 | `male`, `female`, `mixed` | 3 |
| 流派 | `pop`, `rock`, `hip-hop`, `jazz`, `ballad`, `edm` … | ~300 |
| 情绪 | `energetic`, `sad`, `romantic`, `upbeat` … | ~296 |
| 乐器 | `piano`, `guitar`, `drums`, `synthesizer` … | ~133 |

### 4.2 `gt_lyric`：格式非常严格

- 段落之间用 **`;`** 分隔，句子之间用 **`.`** 分隔；
- **只能用英文标点**，不能出现 `。，！？`；
- 中文歌词每段最后一句**不加** `.`，英文歌词每段最后一句**必须加** `.`；
- 结构标签与 ACE-Step 相同，另有 `[intro-long]` `[outro-long]` `[inst-short]` `[inst-medium]` `[silence]`。

```text
中文: [intro-medium] ; [verse] 凌晨三点的便利店.冰柜发出持续的嗡鸣 ; [chorus] 这里是城市的守夜人 ; [outro-medium]
英文: [intro-medium] ; [verse] Trails wind through the forest. Trees stand tall. ; [chorus] Forest is the sanctuary. ; [outro-medium]
纯音乐: [inst-long] ; [inst-long] ; [inst-long]
```

### 4.3 预置风格与生成模式

- `auto_prompt_audio_type`：`Pop` `Latin` `Rock` `Electronic` `Metal` `Country` `R&B/Soul` `Ballad` `Jazz` `World` `Hip-Hop` `Funk` `Soundtrack` `Auto`（这就是 `new_auto_prompt.pt` 里存的风格 prompt）；
- `--generate_type`：`mixed`（默认，人声 + 伴奏）/ `vocal`（纯人声）/ `bgm`（纯伴奏）/ `separate`（双轨分别输出）。

---

## 5. 一键 benchmark 与 prompt 文件格式

```bash
bash script/audio/run_audio_benchmark.sh all          # 依次跑 4 个模型
bash script/audio/run_audio_benchmark.sh parallel     # 4 个模型分别放在 GPU 0–3 并行
bash script/audio/run_audio_benchmark.sh <model> <gpu> <zh|en> <duration_override>
```

等价的 Python 入口：

```bash
python run_audio_benchmark.py --model all \
    --prompts_file script/audio/prompts_audio_benchmark.json \
    --output_dir outputs/audio_benchmark \
    --duration_override 30 --tts_engine cosyvoice2
```

输出结构：

```text
outputs/audio_benchmark/
├── tangoflux_sfx/scene_xx_sfx.wav
├── ace_step_bgm/scene_xx_bgm.wav
├── songgen_song/          (benchmark_input.jsonl + 生成结果)
├── cosyvoice_tts/scene_xx_tts_{zh|en}_{i}.wav
└── benchmark_report.json  (每条的音频时长 / 推理耗时 / RTF, 以及总耗时)
```

**prompt 文件格式**（`script/audio/prompts_audio_benchmark.json`，10 个场景：纯音乐 3 个、动漫 3 个、真人运动 4 个）。一个场景里同时给出四个模型各自的输入，保证"同一画面"下四类音频语义一致：

```json
{
  "id": "scene_01_pure_music",
  "category": "纯音乐 (Pure Music / Instrumental)",
  "video_prompt": "A cinematic timelapse of sunrise over misty mountains ...",
  "tangoflux_sfx": {"prompt": "Gentle mountain wind blowing through valleys ...", "duration": 30, "steps": 50},
  "ace_step_bgm":  {"prompt": "epic cinematic, orchestral, instrumental, ...", "lyrics": "[inst-long] ; [inst-long] ; [inst-long]", "duration": 120},
  "songgen_song":  {"descriptions": "instrumental, orchestral, cinematic, ...", "gt_lyric": "[inst-long] ; [inst-long] ; [inst-long]", "auto_prompt_audio_type": "Soundtrack"},
  "cosyvoice_tts": {"text_zh": "晨曦初露，薄雾笼罩着连绵的群山……", "text_en": "As dawn breaks over the misty mountains ..."}
}
```

`--duration_override` 只影响 TangoFlux（上限 30）和 ACE-Step（上限 240）；SongGeneration 和 TTS 的时长由歌词 / 文本长度决定。

---

## 6. 混音建议

pipeline2 的默认混音（`script/digit/pipeline2/step3_lipsync_merge.sh` 里拼出的 filter）：

```bash
ffmpeg -y -i lipsync_video.mp4 -i bgm.wav -i sfx.wav -t ${DUR} -filter_complex \
  "[0:a]atrim=0:${DUR},asetpts=PTS-STARTPTS,volume=1.0[tts]; \
   [1:a]atrim=0:${DUR},asetpts=PTS-STARTPTS,volume=0.15[bgm]; \
   [2:a]atrim=0:${DUR},asetpts=PTS-STARTPTS,volume=0.2[sfx]; \
   [tts][bgm][sfx]amix=inputs=3:duration=first:dropout_transition=0[aout]" \
  -map 0:v -map "[aout]" -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k final.mp4
```

| 轨道 | 音量 | 说明 |
|---|---|---|
| TTS | 1.0 | 以 TTS 时长为准 |
| BGM | 0.15 | 台词清晰度优先；纯 BGM 视频可调到 0.6–0.8 |
| SFX | 0.2 | 环境感；动作片段可适当调高 |

`--vol_tts / --vol_bgm / --vol_sfx` 调整音量，`--no_bgm / --no_sfx` 关闭对应轨道，`--only merge` 只重做混音。
