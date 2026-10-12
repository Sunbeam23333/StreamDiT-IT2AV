<div align="center">

# StreamDiT-IT2AV

**图像 + 文本 → 音画同步的视频。**
把开源的 TTS、音效、背景音乐、歌曲、图生视频和口型同步模型串成端到端流水线的可复现工具集。

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![开发环境 8×H20](https://img.shields.io/badge/%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83-8%C3%97H20_95GB-555555.svg)](#证据边界)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-0B6B45.svg)](LICENSE)

[English](README.md) · [音频使用指南](docs/audio_guide.md) · [环境搭建与踩坑](docs/environment_pitfalls.md) · [参考框架详解](docs/reference_frameworks.md) · [流水线](docs/pipelines.md) · [TTS 评测](docs/tts_evaluation.md) · [第三方代码](third_party/README.md)

</div>

StreamDiT-IT2AV 不训练新的生成模型，做的是模型外围的工程层：16 个上游框架的固定版本与兼容补丁，统一的推理和 benchmark 脚本，一个 Agent 规划器，以及在单一 Python 3.12 环境里把这些框架跑通的过程中踩过的每一个坑。**重点在音频侧：台词（TTS）、环境音效（SFX）、背景音乐（BGM）和完整歌曲。**

> <a id="证据边界"></a>**证据边界。** 仓库里的每个脚本在开发阶段都在 8×H20（95 GB）服务器上跑过。为了公开发布，内部路径、服务地址和密钥都换成了环境变量，第三方代码改成了"固定 commit + 补丁"的形式。发布版本通过了以下静态检查：55 个 shell 脚本的 `bash -n`、238 个 Python 文件的编译检查，以及每个上游补丁对其固定 commit 的 `git apply --check`。但**整理之后没有在 GPU 上重新端到端跑过**。仓库不给出任何定量的质量指标。已知两个缺口：CosyVoice 在我们的环境里输出听不清的语音（尚未解决，[详情](docs/tts_evaluation.md)）；SongGeneration 上游仓库已下线，其中一个资源文件（`new_auto_prompt.pt`）需要用户自行提供。

## 整体架构

<div align="center">
  <img src="assets/it2av_overview.svg" alt="StreamDiT-IT2AV：输入、Agent 规划、音频分支（TTS / 音效 / BGM / 歌曲）、视觉分支（首帧、图生视频、数字人）与 FFmpeg 合成" width="100%">
</div>

输入一张图和一段脚本。规划器（LLM Agent 或固定的 shell 流水线）把任务拆成**音频分支**和**视觉分支**，其中台词音频驱动口型同步。最后由 FFmpeg 把视频调速到台词长度，把 BGM 和音效裁到同样的长度，再混音合成。

## 音频模型

| 任务 | 模型 | 参数量 | 输出 | 时长控制 | 显存 | 状态 |
|---|---|---|---|---|---|---|
| **TTS** | [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) | 0.5B | 16 kHz | 由文本决定 | ~4 GB | ✅ 流水线 2 默认 |
| | [CosyVoice 2 / 3](https://github.com/FunAudioLLM/CosyVoice) | 0.5B | 24 kHz | 由文本决定 | ~4 GB | ⚠️ 在我们环境中输出不可懂 |
| | [F5-TTS](https://github.com/SWivid/F5-TTS) | 0.3B | 24 kHz | 由文本决定 | ~4 GB | ✅ 声音克隆 |
| | [ChatTTS](https://github.com/2noise/ChatTTS) | — | 24 kHz | 由文本决定 | ~4 GB | ✅ 口语对话风格 |
| | [Fish-Speech S2](https://github.com/fishaudio/fish-speech) | — | — | 由文本决定 | — | 🧪 需独立环境 / API server |
| | 混元 TTS 2.5（API） | — | 24 kHz | 由文本决定 | 0 | ✅ 云端方案 |
| **音效** | [TangoFlux](https://github.com/declare-lab/TangoFlux) | 515M | 44.1 kHz 立体声 | 1–30 s | ~4 GB | ✅ |
| **BGM** | [ACE-Step v1](https://github.com/ace-step/ACE-Step) | 3.5B | 48 kHz | ≤ 240 s | ~16 GB | ✅ |
| **歌曲** | SongGeneration v2-large（[LeVo](https://arxiv.org/abs/2506.07520)） | — | 48 kHz | 由歌词长度决定（≤ 4 分 30 秒） | ~32 GB | ✅ 已 vendored，仅限非商用 |

<sub>显存一列是单卡的粗略参考（上游文档 + 我们实际能跑起来的配置），不是 profiler 测得的峰值。Prompt 写法、参数、歌词格式规则和混音配方见 [音频使用指南](docs/audio_guide.md)。</sub>

## 视频与图像模型

| 阶段 | 模型 | 运行方式 |
|---|---|---|
| 首帧 | 混元生图 3.0（API）、Nano-Banana Pro（API），以及通过 [diffusers](https://github.com/huggingface/diffusers) / [xDiT](https://github.com/xdit-project/xDiT) 运行的 FLUX.1 / SD 3.5 | `utils/image/`、`run_t2i_*.py` |
| 图生视频 | [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | 720p + 自带超分到 1080p，8 卡 |
| 图生视频 | Wan 2.1 / 2.2（I2V、TI2V、FLF2V、VACE）、LingBot-World，通过 [LightX2V](https://github.com/ModelTC/LightX2V) | `script/video/` 下的 benchmark |
| 口型同步 | [LatentSync 1.6](https://github.com/bytedance/LatentSync) | 512×512，25 步，约 18 GB |
| 数字人 | [LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar) | 5 卡 timestep 流水线并行 |
| 音视频联合生成 | [daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman) | `distill` / `sr540p` / `sr1080p` 三套配置 |

每个框架的结构、论文、固定 commit 和许可证见 [docs/reference_frameworks.md](docs/reference_frameworks.md)。

## 快速开始

```bash
git clone https://github.com/Sunbeam23333/StreamDiT-IT2AV.git && cd StreamDiT-IT2AV
cp .env.example .env                     # 填写 IT2AV_WORKSPACE、HF_TOKEN，以及可选的代理 / API 密钥

# 1. 按我们实际使用的 commit 拉取第三方框架（并打上补丁）
bash third_party/fetch_third_party.sh audio          # TTS / 音效 / BGM；视频和图像再加 `video image`

# 2. 环境（选方案前先看 docs/environment_pitfalls.md）
bash script/audio/setup_audio_envs.sh all            # 每个音频模型一个 conda 环境（最省心）
bash script/audio/setup_tts_alternatives.sh          # Spark-TTS / F5-TTS / ChatTTS / Fish-Speech
bash setup_env.sh                                    # 视频 / 口型同步 / MagiHuman 的主环境

# 3. 权重 -> ${IT2AV_WORKSPACE}/models
python download/download_audio_models.py --models all     # 全部音频模型；也可按组选：audio music tts
python download/download_models.py                   # 视频模型（体积很大）

# 4. 生成
bash script/audio/run_audio_benchmark.sh parallel    # TangoFlux / ACE-Step / SongGeneration / CosyVoice 分别跑在 GPU 0-3
bash script/audio/run_tts_benchmark.sh               # CosyVoice2 / Spark-TTS / ChatTTS / F5-TTS
bash script/digit/pipeline2/step1_generate_audio.sh  # 流水线 2：音频 -> 视频 -> 口型同步 -> 混音
bash script/digit/pipeline2/step2_generate_video.sh
bash script/digit/pipeline2/step3_lipsync_merge.sh
```

所有脚本都会 source [`script/common/env.sh`](script/common/env.sh)。它负责读取 `.env`，再由 `IT2AV_WORKSPACE`（共享存储）、`IT2AV_LOCAL_ROOT`（本地高速缓存）和 `IT2AV_CONDA_ENV` 推出所有路径，不依赖任何特定机器。

## 流水线

| 流水线 | 入口 | 流程 |
|---|---|---|
| **流水线 2**（推荐） | `script/digit/pipeline2/` | Spark-TTS ∥ ACE-Step ∥ TangoFlux → HunyuanVideo-1.5 图生视频 → `setpts` 调速到台词长度 → LatentSync → `amix` 混音 |
| **流水线 3** | `script/digit/pipeline3_magihuman/` | 首帧 + 增强后的 prompt → daVinci-MagiHuman（一次生成音频和视频） |
| **数字人 v1** | `script/digit/step*.sh` | 人像 → TTS → LiveAvatar 说话视频 → Wan 2.2 动作续写 → 拼接 |
| **Agent** | `utils/tools/agent_video_pipeline/` | LLM 分镜 → 按场景选择策略和路由 → 分阶段生成 → VLM 审核与重试 → 拼接 |

参数和目录结构见 [docs/pipelines.md](docs/pipelines.md)。

## 环境踩坑精选

[环境搭建与踩坑全记录](docs/environment_pitfalls.md) 收录了 40 多条问题，每条都按"现象 → 根因 → 解决"来写，并可以按报错关键字查找。其中最耗时间的几类：

- **pip `resolution-too-deep` 和 `OpenBLAS not found`。** 不设上限的 `wandb` / `tensorboard`，加上锁死 `numpy<2` 的老包（`onnxruntime-gpu==1.18`、`descript-audio-codec`、`k-diffusion`），会不断把 numpy 和 scipy 拉回源码编译。解决办法是给所有依赖加上版本上界，并把这几个包单独用 `--no-deps` 安装。
- **SongGeneration 跑在新版本栈上。** `is_flash_attn_available` 被改名，`SequenceSummary` 被移除，`torch.load` 默认变成 `weights_only=True`，代码里还有一个硬性的 `transformers<4.40` 断言。另外 `new_auto_prompt.pt` 只是一个 Git LFS 指针，`hf_hub_download` 下载它时会"成功"返回。
- **CosyVoice 的 import 问题。** 依次缺 `whisper`、`pyworld`（只在解析 yaml 时用到），以及空的 `Matcha-TTS` 子模块。
- **在新服务器上跑 LatentSync。** 系统自带的 ffmpeg 没有 libx264，结果报出 `stack expects a non-empty TensorList`。另外，以前只要有一帧人脸侧过去，整段视频就会失败。
- **Shell 细节。** `$(sync_model ...)` 把进度日志和路径一起捕获了进去；非交互式 shell 里如果没有先加载 conda 的 shell hook，`conda activate` 会直接报错。

## 目录结构

```text
StreamDiT-IT2AV/
├── run_audio_benchmark.py        # TangoFlux / ACE-Step / SongGeneration / CosyVoice 推理（含 RTF 报告）
├── run_tts_benchmark.py          # CosyVoice2/3、Spark-TTS、F5-TTS、ChatTTS、Fish-Speech 推理
├── run_t2i_diffusers.py · run_t2i_xdit.py · fast_sync.py · setup_env.sh
├── script/
│   ├── common/env.sh             # 路径 / 环境 / 代理的唯一配置来源
│   ├── audio/                    # 音频与 TTS 脚本、prompt 集、requirements、能力矩阵
│   ├── digit/                    # 数字人流水线（pipeline2、pipeline3_magihuman、LiveAvatar）
│   ├── video/ · image/           # 各模型 benchmark
├── download/                     # Hugging Face / ModelScope 下载脚本
├── utils/
│   ├── audio/SongGeneration/     # 已 vendored（上游下线），已打补丁
│   ├── audio/Hunyuan-tts/        # 混元 TTS 2.5 API 客户端
│   ├── image/{Hunyuan-Image,Banana}/
│   └── tools/agent_video_pipeline/
├── third_party/                  # manifest.tsv、fetch_third_party.sh、patches/
├── docs/                         # 中文文档
└── assets/
```

## 致谢

本项目完全建立在开源社区的工作之上：[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)、[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)、[Spark-TTS](https://github.com/SparkAudio/Spark-TTS)、[F5-TTS](https://github.com/SWivid/F5-TTS)、[ChatTTS](https://github.com/2noise/ChatTTS)、[Fish-Speech](https://github.com/fishaudio/fish-speech)、[TangoFlux](https://github.com/declare-lab/TangoFlux)、[ACE-Step](https://github.com/ace-step/ACE-Step)、SongGeneration / LeVo（腾讯 AI Lab）、[HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)、[LightX2V](https://github.com/ModelTC/LightX2V)、[Wan](https://github.com/Wan-Video/Wan2.2)、[LatentSync](https://github.com/bytedance/LatentSync)、[LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar)、[daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman)、[diffusers](https://github.com/huggingface/diffusers)、[xDiT](https://github.com/xdit-project/xDiT) 和 [Mora](https://github.com/lichao-sun/Mora)。使用时请引用它们的论文（列表见 [docs/reference_frameworks.md](docs/reference_frameworks.md)）。

## 许可证

本仓库代码采用 [Apache 2.0](LICENSE) 许可。第三方框架和模型权重沿用各自的许可证，其中不少**仅限非商用**，例如 SongGeneration、TangoFlux，以及 ChatTTS / F5-TTS / Spark-TTS 的权重。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 引用

```bibtex
@misc{streamdit_it2av2026,
  title        = {StreamDiT-IT2AV: An Orchestration Toolkit for Image+Text to Audio-Video Generation},
  year         = {2026},
  url          = {https://github.com/Sunbeam23333/StreamDiT-IT2AV}
}
```
