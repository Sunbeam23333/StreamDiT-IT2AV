# 参考框架详解

> 本仓库**不重新实现**任何生成模型，而是把一批开源 SOTA 框架以"固定 commit + 兼容补丁"的方式接入，再用脚本 / Agent 把它们串成 IT2AV（Image + Text → Audio + Video）流水线。
> 本文逐一说明每个框架：**解决什么问题、模型结构、我们怎么用、踩过什么坑、固定在哪个版本、许可证是什么**。
> 拉取方式见 [`third_party/README.md`](../third_party/README.md)；所有 commit 都记录在 [`third_party/manifest.tsv`](../third_party/manifest.tsv)。

- [1. 总览](#1-总览)
- [2. 音频：TTS](#2-音频tts)
- [3. 音频：音效 / 背景音乐 / 歌曲](#3-音频音效--背景音乐--歌曲)
- [4. 视频：图生视频](#4-视频图生视频)
- [5. 视频：说话人 / 对口型 / 音视频联合生成](#5-视频说话人--对口型--音视频联合生成)
- [6. 图像：首帧生成](#6-图像首帧生成)
- [7. Agent 设计参考](#7-agent-设计参考)
- [8. 选型结论](#8-选型结论)

---

## 1. 总览

| 类别 | 框架 | 在流水线中的角色 | 状态 |
|---|---|---|---|
| TTS | **Spark-TTS** | pipeline2 默认台词配音（可控性别 / 音高 / 语速） | ✅ 主力 |
| TTS | **CosyVoice 2 / 3** | 零样本克隆 / 方言 / instruct；benchmark 对照 | ⚠️ 有"非人话"问题 |
| TTS | **F5-TTS** | 参考音频驱动的高保真克隆 | ✅ benchmark |
| TTS | **ChatTTS** | 对话口语风格、笑声 / 停顿控制 | ✅ benchmark |
| TTS | **Fish-Speech (S2)** | 多语种克隆，独立服务化 | 🧪 experimental |
| TTS | **混元 TTS 2.5（云端 API）** | 不占 GPU 的高质量中文配音 | ✅ 可选 |
| SFX | **TangoFlux** | 环境音 / 拟音 | ✅ 主力 |
| BGM | **ACE-Step v1** | 纯音乐 BGM（也能生成带人声的歌） | ✅ 主力 |
| Song | **SongGeneration v2 (LeVo)** | 带人声的完整歌曲 | ✅（vendored） |
| I2V | **HunyuanVideo-1.5** | pipeline2 首帧生视频（720p + SR 1080p） | ✅ 主力 |
| I2V | **Wan 2.1 / 2.2（经 LightX2V）** | I2V / FLF2V / VACE / TI2V benchmark | ✅ |
| I2V | **LingBot-World（经 LightX2V）** | 相机位姿控制 I2V | ✅ benchmark |
| Lip-sync | **LatentSync 1.6** | 给 I2V 视频对口型 | ✅ 主力 |
| Avatar | **LiveAvatar** | 音频驱动的流式数字人（无限长度） | ✅ |
| AV | **daVinci-MagiHuman** | 首帧 + 文本一步生成音视频 | ✅ |
| T2I | **HunyuanImage 3.0 / Nano-Banana Pro（API）、FLUX / SD3.5（diffusers / xDiT）** | 首帧 / 人像 | ✅ |

---

## 2. 音频：TTS

### 2.1 Spark-TTS

| 项目 | 内容 |
|---|---|
| 论文 | [Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens](https://arxiv.org/abs/2503.01710) |
| 代码 / 权重 | [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) · [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) |
| 固定版本 | `2f1ea9082400547242641f5271b6f941c9f439d1` |
| 许可证 | 代码 Apache-2.0；权重 CC BY-NC-SA 4.0 |

**结构**：以 Qwen2.5-0.5B 为骨干的纯 LLM TTS。核心是 **BiCodec**：把语音拆成两路离散 token——低码率的**语义 token**（50 TPS，管内容）和固定长度的**全局 token**（管音色 / 说话人属性），LLM 直接自回归生成这两类 token，再由 BiCodec 解码器还原波形。因为"音色"被解耦成独立 token，所以不给参考音频也能用 `gender / pitch / speed` 这样的粗粒度属性"造"一个虚拟说话人。

**我们怎么用**：pipeline2 的台词配音（`step1_generate_audio.sh`，GPU 0），参数 `female / moderate / moderate`；`run_tts_benchmark.py --model spark_tts --spark_gender --spark_pitch --spark_speed`。输出 **16 kHz**。

### 2.2 CosyVoice 2 / 3

| 项目 | 内容 |
|---|---|
| 论文 | [CosyVoice](https://arxiv.org/abs/2407.05407) · [CosyVoice 2: Scalable Streaming Speech Synthesis with LLMs](https://arxiv.org/abs/2412.10117) · [CosyVoice 3: Towards In-the-wild Speech Generation](https://arxiv.org/abs/2505.17589) |
| 代码 / 权重 | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) · [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) · [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) |
| 依赖 | [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)（[论文](https://arxiv.org/abs/2309.03199)，提供 `BASECFM` 条件流匹配基类）@ `bd4d90d9` |
| 固定版本 | `04bcadc6340e266b4ba09f4474b4668c444aa063` |
| 许可证 | Apache-2.0 |

**结构**（两段式）：

```
text + prompt speech token ──► LLM (Qwen2.5-0.5B 初始化) ──► speech token (FSQ, 6561 码本, 25 Hz)
                                                                 │
                         prompt mel + speaker embedding ──► chunk-aware causal Flow Matching ──► mel ──► HiFT vocoder ──► 24 kHz wav
```

- **监督语义 token**：在 ASR 模型编码器里插入 FSQ 量化层得到 speech tokenizer（`speech_tokenizer_v2.onnx`），token 天然对齐文本内容；
- **LLM**：把文本 token 和 speech token 放在同一序列里自回归，支持流式；
- **Flow Matching**：基于 Matcha-TTS 的 CFM，chunk-aware 因果掩码让它也能流式解码；
- CosyVoice 3 扩大了数据规模，并加入 post-training（DiffRO）和更多 instruct（18 种方言、情绪、语速）。

**推理模式**：SFT（预置音色 `中文女` / `中文男`）、zero-shot（3 秒参考音频 + 参考文本）、cross-lingual、instruct（自然语言控制风格）。文本里可插 `[breath]`、`[laughter]`、`<strong>…</strong>` 等标记。

**我们怎么用**：`run_audio_benchmark.py --model cosyvoice --tts_engine cosyvoice2`（默认）/ `cosyvoice3`；`run_tts_benchmark.py --model cosyvoice2`。`fp16=False`、`onnxruntime-gpu==1.20.1`。**我们的环境里一直输出"非人类语言"，没彻底解决**，见 [TTS 评测与 CosyVoice 排查](tts_evaluation.md)。

### 2.3 F5-TTS

| 项目 | 内容 |
|---|---|
| 论文 | [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/abs/2410.06885) |
| 代码 / 权重 | [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) · [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) |
| 固定版本 | `623c96c294965be60246f9c3679298f1d466189e` |
| 许可证 | 代码 MIT；权重 CC BY-NC 4.0（训练数据 Emilia） |

**结构**：完全非自回归。文本字符序列直接 pad 到与语音帧等长，经 ConvNeXt V2 提取文本特征后，与"被 mask 的参考 mel"一起送进 **DiT**，用 flow matching 做 infilling；推理时用 **Sway Sampling**（把采样步集中在早期时间步）提升少步质量，最后用 Vocos 声码器出 24 kHz 波形。不需要时长模型、音素对齐和 G2P。

**我们怎么用**：`run_tts_benchmark.py --model f5tts --ref_audio <wav> --ref_text "<转写>"`，可调 `nfe_step / cfg_strength / sway_sampling_coef / speed`。

### 2.4 ChatTTS

| 项目 | 内容 |
|---|---|
| 代码 / 权重 | [2noise/ChatTTS](https://github.com/2noise/ChatTTS) · [2Noise/ChatTTS](https://huggingface.co/2Noise/ChatTTS) |
| 固定版本 | `c26573a61ebde14ac456d8ed4b9f96908d3dd8fa` |
| 许可证 | 代码 AGPL-3.0；权重 CC BY-NC 4.0 |

**结构**：GPT 式自回归模型生成离散语音 token（DVAE 码本），再由 DVAE 解码 + Vocos 声码器输出 24 kHz。专为对话场景训练，能生成笑声、停顿、口语填充词。

**我们怎么用**：`run_tts_benchmark.py --model chattts --chattts_seed 42`（固定 seed 才能固定音色）；文本控制标记 `[oral_2] [laugh_0] [break_4] [uv_break] [laugh] [lbreak] [speed_5]`。

### 2.5 Fish-Speech (S2)

| 项目 | 内容 |
|---|---|
| 论文 | [Fish-Speech: Leveraging LLMs for Advanced Multilingual TTS](https://arxiv.org/abs/2411.01156) · [Fish Audio S2 Technical Report](https://arxiv.org/abs/2603.08823) |
| 代码 | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) |
| 固定版本 | `49985a34a7041bb350338735e2cb3291eef045d4` |
| 许可证 | Fish Audio Research License（研究用途） |

**结构**：双自回归（Dual-AR）LLM + 多码本音频 codec，无需 G2P 的多语种克隆。

**状态**：S2 代码绑定 `torch==2.8.0`，和主环境冲突；下载脚本拉的是 `fish-speech-1.5` 权重，与 S2 runner 不匹配。因此 `tts_capabilities.json` 里标为 `benchmark_ready: false`，推荐单独环境 + 官方 API server 接入（`script/audio/run_fish_speech.sh`）。

### 2.6 混元 TTS 2.5（云端 API）

- 接口封装：`utils/audio/Hunyuan-tts/hunyuan_tts.py`（同步 / 流式两种调用，PCM → wav），`tts_generate.py` 批量生成；
- 模型 `hunyuan-tts-2.5`，24 kHz 单声道 PCM；
- 需要 `HUNYUAN_API_KEY` + `HUNYUAN_API_BASE`（内部服务，开源版本不提供默认值）。我们在 CosyVoice 出问题后，用它作为数字人链路的默认 TTS（`step2_generate_audio.sh`、`step2a_hunyuan_tts.sh`），不占 GPU。

---

## 3. 音频：音效 / 背景音乐 / 歌曲

### 3.1 TangoFlux —— 音效（SFX）

| 项目 | 内容 |
|---|---|
| 论文 | [TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and CLAP-Ranked Preference Optimization](https://arxiv.org/abs/2412.21037) |
| 代码 / 权重 | [declare-lab/TangoFlux](https://github.com/declare-lab/TangoFlux) · [declare-lab/TangoFlux](https://huggingface.co/declare-lab/TangoFlux) · 文本编码器 [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) |
| 固定版本 | `164420faa2b3f8a28bb013ccde7006422a616fc7` |
| 许可证 | 非商用研究（权重受 Stability AI Community License 约束，VAE 来自 Stable Audio Open） |

**结构**：515M 参数，Stable Audio Open 的 VAE 把 44.1 kHz 立体声压到 latent，主干是 FLUX 式的 **MMDiT + DiT** 混合 transformer，文本条件来自 flan-t5-large，时长作为额外条件，用 **rectified flow matching** 训练。论文的另一个贡献 **CRPO**（CLAP-Ranked Preference Optimization）：用 CLAP 给同一 prompt 的多个采样打分、构造偏好对，迭代做 DPO 式对齐。单张 A40 约 3.7 s 生成 30 s 音频，是 AudioLDM 2 的上位替代（AudioLDM 2：712M、10 s、200 步、24.8 s）。

**我们怎么用**：`run_audio_benchmark.py --model tangoflux`、`step1_generate_audio.sh` GPU 2；`steps=50, guidance_scale=4.5, duration≤30`，英文 prompt，显式传入本地 flan-t5 路径。

### 3.2 ACE-Step v1 —— 背景音乐（BGM）

| 项目 | 内容 |
|---|---|
| 论文 | [ACE-Step: A Step Towards Music Generation Foundation Model](https://arxiv.org/abs/2506.00045) |
| 代码 / 权重 | [ace-step/ACE-Step](https://github.com/ace-step/ACE-Step) · [ACE-Step/ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) |
| 固定版本 | `1bee4c9f5b43e30995f8d4d33b3919197ce1bd68` |
| 许可证 | Apache-2.0 |

**结构**：3.5B 参数的扩散音乐基础模型。音频先经 **DCAE**（Sana 式深度压缩自编码器，作用在 mel 上）压缩，再由 **linear-attention DiT** 用 flow matching 生成；文本 tag 用 UMT5 编码，歌词用多语种 lyric tokenizer（这就是它需要 `pypinyin` / `hangul_romanize` / `spacy` 的原因）；训练时用 **REPA** 把中间表征对齐到 MERT 和 mHuBERT，提升音乐结构和歌词对齐。支持 19 种语言，A100 上约 20 s 生成 4 分钟音乐，还支持 retake / repaint / edit / LoRA。

**我们怎么用**：BGM 主力（GPU 1）。`infer_step=100, guidance_scale=15, euler, cfg_type=apg, omega_scale=10, guidance_interval=0.5, min_guidance_scale=3`；纯音乐歌词 `[inst-long] ; [inst-long] ; [inst-long]`；最长 240 s，48 kHz。

### 3.3 SongGeneration v2 (LeVo) —— 歌曲

| 项目 | 内容 |
|---|---|
| 论文 | [LeVo: High-Quality Song Generation with Multi-Preference Alignment](https://arxiv.org/abs/2506.07520) |
| 代码 | 原 `tencent-ailab/SongGeneration`（**已下线**），本仓库 vendored 在 `utils/audio/SongGeneration` |
| 权重 | [lglg666/SongGeneration-v2-large](https://huggingface.co/lglg666/SongGeneration-v2-large) · 运行时 [lglg666/SongGeneration-Runtime](https://huggingface.co/lglg666/SongGeneration-Runtime) |
| 许可证 | **仅限学术 / 研究 / 教育用途，禁止任何商业或生产用途**（见 `utils/audio/SongGeneration/LICENSE`） |

**结构**：语言模型 **LeLM** 并行建模两类 token——**混合 token**（人声 + 伴奏混在一起，保证整体和谐）和**双轨 token**（人声、伴奏分开，保证音质）；音频 codec 负责编码 / 解码，Demucs 用于分轨，Qwen2 tokenizer 处理歌词。之后用 **DPO 做多偏好对齐**（歌词对齐、音乐性、prompt 一致性）。输入是"结构化歌词 + 文本描述（性别 / 流派 / 情绪 / 乐器）+ 可选参考音频或预置风格"。

**我们怎么用**：`run_audio_benchmark.py --model songgen` 生成 JSONL 后以子进程调用 `generate.py --generate_type mixed --use_flash_attn`；支持 `mixed / vocal / bgm / separate`。为了在 transformers 4.57 + torch 2.6 + Python 3.12 下运行，我们改了 18 个文件（E1–E6），见 [踩坑记录](environment_pitfalls.md#e-songgeneration--levo歌曲)。

### 3.4 Stable Audio Open（可选）

[stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)（[论文](https://arxiv.org/abs/2407.14358)）：47 s / 44.1 kHz 的 latent diffusion 文生音频。我们提供了 `download/download_stable_audio.py`，它也是 daVinci-MagiHuman 的音频 VAE 来源。

---

## 4. 视频：图生视频

### 4.1 HunyuanVideo-1.5

| 项目 | 内容 |
|---|---|
| 论文 | [HunyuanVideo 1.5 Technical Report](https://arxiv.org/abs/2511.18870) |
| 代码 / 权重 | [Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) · [tencent/HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) |
| 附加权重 | Qwen2.5-VL-7B-Instruct（多模态文本编码器）、byT5-small + Glyph-SDXL-v2（ModelScope，文字渲染）、SigLIP（取自 FLUX.1-Redux-dev，图像条件） |
| 固定版本 | `2641c0de73da0a2d9682fed24af2c0a516527cc1` |
| 许可证 | Tencent Hunyuan Community License |

**结构**：8.3B 参数 DiT + 高压缩 3D causal VAE；**SSTA**（selective and sliding tile attention）降低长序列注意力开销；Qwen2.5-VL 做多模态文本理解，byT5 + Glyph 负责画面中的文字；原生支持 480p / 720p 的 T2V / I2V，另带一个**8 步蒸馏的视频超分网络**输出 1080p；支持 CFG 蒸馏、step 蒸馏、DeepCache、SageAttention 加速。

**我们怎么用**：pipeline2 Step 2，`torchrun --nproc_per_node=8 generate.py --resolution 720p --sr true`，121 帧 / 24 fps，关闭 offload，开启 CFG 蒸馏 + DeepCache。另有原生 / LightX2V 两套 benchmark（`script/video/run_hyvideo15_*.sh`）。

### 4.2 LightX2V + Wan 2.1 / 2.2 + LingBot-World

| 项目 | 内容 |
|---|---|
| 推理框架 | [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) @ `cffcc57b`（+ `LightX2V.patch`） |
| Wan 论文 | [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314) |
| 权重 | Wan2.2-I2V-A14B、Wan2.2-TI2V-5B、Wan2.2-T2V-A14B-Diffusers、Wan2.1-FLF2V-14B-720P、Wan2.1-VACE-14B、Wan2.2-VACE-Fun-A14B、robbyant/lingbot-world-base-cam |
| 许可证 | Apache-2.0 |

- **LightX2V**：轻量视频生成推理框架，统一封装 Wan / HunyuanVideo 等模型，提供 step 蒸馏、量化、offload、序列并行、特征缓存等加速；
- **Wan 2.2**：A14B 是 **MoE** 结构——高噪声专家负责早期去噪（布局 / 运动），低噪声专家负责后期细节，总参数 27B、每步激活 14B；TI2V-5B 是高压缩 VAE 的小模型；
- **FLF2V / VACE**：首尾帧控制、多条件视频编辑；
- **LingBot-World**：基于 Wan 2.2 的相机位姿控制 I2V（`generate_camera_poses.py` 生成轨迹）。

**我们的补丁**：新增 LingBot 的 model / runner / weights / infer，以及 H20 上 `hunyuan_video_15` / `wan` / `wan22` / `wan22_vace` / `lingbot` 的配置；对应 benchmark 在 `script/video/`。Agent 里的 `wan22_i2v` / `wan22_flf2v` 路由也走这里（`adapters/wan_lightx2v_stage.py`）。

---

## 5. 视频：说话人 / 对口型 / 音视频联合生成

### 5.1 LatentSync 1.6 —— 对口型

| 项目 | 内容 |
|---|---|
| 论文 | [LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision](https://arxiv.org/abs/2412.09262) |
| 代码 / 权重 | [bytedance/LatentSync](https://github.com/bytedance/LatentSync) · [ByteDance/LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6) |
| 固定版本 | `a229c3948406bc2cf6eaf4873e662e70c6a04746`（+ `LatentSync.patch`） |
| 许可证 | Apache-2.0 |

**结构**：端到端、不经过 3D / 2D 中间表示。在 SD 1.5 的 latent 空间里做 inpainting：输入是"嘴部被 mask 的帧 + 参考帧"，Whisper 提取的音频特征通过 cross-attention 注入 UNet；训练时用 **StableSyncNet** 做唇音同步监督，用 **TREPA**（基于视频表征的时序对齐损失）提升时间一致性。1.6 版在 512×512 上训练，牙齿 / 嘴唇更清晰。

**我们怎么用**：pipeline2 Step 3（先 `setpts` 慢放到 TTS 时长、转 25 fps），`stage2_512.yaml, inference_steps=25, guidance_scale=1.5, --enable_deepcache`，约 18 GB 显存。补丁让检测不到人脸的帧复用相邻帧，不再整段崩溃。

### 5.2 LiveAvatar —— 音频驱动流式数字人

| 项目 | 内容 |
|---|---|
| 论文 | [Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length](https://arxiv.org/abs/2512.04677) |
| 代码 / 权重 | [Alibaba-Quark/LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar) · [Quark-Vision/Live-Avatar](https://huggingface.co/Quark-Vision/Live-Avatar)（基座 [Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)） |
| 固定版本 | `775e3630b9ac956f3ed3be4c8d1932c5a15d7677`（+ `LiveAvatar.patch`） |
| 许可证 | Apache-2.0 |

**结构**：在 Wan2.2-S2V-14B 基础上做**分块因果自回归**蒸馏，按块生成视频并维护 KV cache，从而支持无限长度；多卡用**时间步流水线并行（TPP）**把不同去噪步放到不同 GPU 上，实现实时流式输出（默认 5 卡）。

**我们怎么用**：老版数字人流水线（`run_digit_pipeline.sh`：Banana 生人像 → TTS → LiveAvatar）。补丁新增 `generate_ref_portraits.py`（生成参考人像）、`prepare_digit_samples.py`（准备 TTS 样本），并修复了 cond KV 缓冲区写死 720×400 的问题。

### 5.3 daVinci-MagiHuman —— 首帧 + 文本 → 音视频一步生成

| 项目 | 内容 |
|---|---|
| 论文 | [Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model](https://arxiv.org/abs/2603.21986) |
| 代码 / 权重 | [GAIR-NLP/daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman) · [GAIR/daVinci-MagiHuman](https://huggingface.co/GAIR/daVinci-MagiHuman)（文本编码器 t5gemma-9b，音频 VAE 来自 Stable Audio Open） |
| 可选依赖 | [SandAI-org/MagiCompiler](https://github.com/SandAI-org/MagiCompiler) @ `b4cc3161` |
| 固定版本 | `bd8196473342c58ddd5786af07d8b6a5becfed2b`（+ `daVinci-MagiHuman.patch`） |
| 许可证 | Apache-2.0 |

**结构**：约 15B 的**单流（single-stream）** transformer，把视频 latent 和音频 latent 拼进同一个 token 序列联合去噪，同一次采样同时得到画面和声音（口型与人声天然同步），省掉 TTS → I2V → 对口型的多级串联。提供蒸馏版（8 步、无 CFG、256p）和 base + 超分版（540p / 1080p）。

**我们怎么用**：pipeline3（`script/digit/pipeline3_magihuman/run_magihuman.sh --mode distill|sr540p|sr1080p`），Agent 的 `magihuman_av` 路由。补丁主要解决显存峰值（MLP 分块、VAE 输出 offload、SR 前清缓存）和缺 `magi_compiler` / flash-attn 时的回退。

---

## 6. 图像：首帧生成

| 框架 | 链接 | 用途 |
|---|---|---|
| **HunyuanImage 3.0**（API） | [技术报告](https://arxiv.org/abs/2509.23951) | 首帧 / 人像（`utils/image/Hunyuan-Image/`，需 `HUNYUAN_API_*`） |
| **Nano-Banana Pro / gemini-3-pro-image**（API） | — | 首帧、参考人像、图像编辑（`utils/image/Banana/`，需 `VENUS_*`） |
| **diffusers** | [huggingface/diffusers](https://github.com/huggingface/diffusers) @ `07a63e19` | FLUX.1-dev / schnell、SD 3.5 Large / Medium、PixArt-Σ（`run_t2i_diffusers.py`） |
| **xDiT** | [xdit-project/xDiT](https://github.com/xdit-project/xDiT) @ `84813c97` · [论文](https://arxiv.org/abs/2411.01738) | DiT 多卡并行推理（USP / PipeFusion / CFG 并行，`run_t2i_xdit.py`） |

---

## 7. Agent 设计参考

`utils/tools/agent_video_pipeline` 的设计参考了以下工作（**均未直接引入代码**）：

| 工作 | 链接 | 借鉴点 |
|---|---|---|
| **Mora** | [arXiv 2403.13248](https://arxiv.org/abs/2403.13248) · [lichao-sun/Mora](https://github.com/lichao-sun/Mora) | `core/` 下的 Message / Memory / Action / Role / Workflow 抽象 |
| **MovieAgent** | [arXiv 2503.07314](https://arxiv.org/abs/2503.07314) | 导演 / 编剧 / 分镜多 Agent 分工、分层 CoT 规划 |
| **VQQA** | [arXiv 2603.12310](https://arxiv.org/abs/2603.12310) | 把 VLM 批评当作"语义梯度"做闭环 prompt 优化 → 我们的审核-重试链路 |
| **LAVE** | [arXiv 2402.10294](https://arxiv.org/abs/2402.10294) | LLM 驱动的视频剪辑交互范式 |

---

## 8. 选型结论

| 需求 | 首选 | 备选 | 理由 |
|---|---|---|---|
| 中文台词配音（本地） | Spark-TTS | F5-TTS（有参考音频时） | 稳定、可控、不依赖参考音频 |
| 中文台词配音（云端） | 混元 TTS 2.5 | — | 质量高、不占 GPU |
| 声音克隆 | F5-TTS | CosyVoice2 | F5 只需几秒参考音频 + 转写 |
| 环境音 / 拟音 | TangoFlux | Stable Audio Open | 快、质量高、30 s |
| BGM | ACE-Step | SongGeneration `--bgm` | Apache-2.0、240 s、生成快 |
| 带人声的歌 | SongGeneration | ACE-Step | 歌词对齐更好，但**仅限非商用** |
| 首帧生视频 | HunyuanVideo-1.5 | Wan2.2-I2V-A14B | 720p + SR 1080p，8 卡快 |
| 对口型 | LatentSync 1.6 | — | 视频 + 音频直接对齐，SOTA |
| 一步音视频 | daVinci-MagiHuman | LiveAvatar（音频驱动） | 无需多级串联 |
