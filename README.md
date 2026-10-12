<div align="center">

# StreamDiT-IT2AV

**Image + Text → synchronized Audio + Video.**
A reproducible toolkit that wires open-source TTS, sound-effect, music, song, image-to-video and lip-sync models into end-to-end pipelines.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Developed on 8×H20](https://img.shields.io/badge/Developed_on-8%C3%97H20_95GB-555555.svg)](#evidence-boundary)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-0B6B45.svg)](LICENSE)

[中文](README_zh.md) · [Audio guide](docs/audio_guide.md) · [Environment & pitfalls](docs/environment_pitfalls.md) · [Reference frameworks](docs/reference_frameworks.md) · [Pipelines](docs/pipelines.md) · [TTS evaluation](docs/tts_evaluation.md) · [Third-party](third_party/README.md)

</div>

StreamDiT-IT2AV does not train a new generator. It is the engineering layer around one: pinned versions of 16 upstream frameworks, compatibility patches, unified runners and benchmark scripts, an agent planner, and a long, honest record of every environment problem we hit while getting them to run together on a single Python 3.12 stack. **The audio side is the focus: speech (TTS), sound effects (SFX), background music (BGM) and full songs.**

> <a id="evidence-boundary"></a>**Evidence boundary.** Every script in this repository was run on an 8×H20 (95 GB) server during development. For the public release, internal paths, endpoints and credentials were replaced by environment variables and the third-party code was converted to pinned commits plus patches. The released tree passes static checks: `bash -n` on all 55 shell scripts, a compile pass over all 238 Python files, and `git apply --check` of each upstream patch against its pinned commit. However, **it has not been re-run end-to-end on GPU after this reorganization**. The repo reports no quantitative quality numbers. Two known gaps: CosyVoice produces unintelligible speech in our environment (unresolved, [details](docs/tts_evaluation.md)), and SongGeneration's upstream repository is offline, so one asset (`new_auto_prompt.pt`) has to be supplied by the user.

## Architecture

<div align="center">
  <img src="assets/it2av_overview.svg" alt="StreamDiT-IT2AV: inputs, agent planner, audio branch (TTS / SFX / BGM / song), visual branch (first frame, image-to-video, talking human) and FFmpeg composition" width="100%">
</div>

An image and a script go in. A planner (either the LLM agent or a fixed shell pipeline) splits the work into an **audio branch** and a **visual branch**. Speech drives the lip-sync stage. FFmpeg retimes the video to the speech length, trims BGM and SFX to match, and mixes the final track.

## Audio models

| Task | Model | Params | Output | Length control | VRAM | Status |
|---|---|---|---|---|---|---|
| **TTS** | [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) | 0.5B | 16 kHz | text length | ~4 GB | ✅ default in pipeline 2 |
| | [CosyVoice 2 / 3](https://github.com/FunAudioLLM/CosyVoice) | 0.5B | 24 kHz | text length | ~4 GB | ⚠️ unintelligible output in our env |
| | [F5-TTS](https://github.com/SWivid/F5-TTS) | 0.3B | 24 kHz | text length | ~4 GB | ✅ voice cloning |
| | [ChatTTS](https://github.com/2noise/ChatTTS) | — | 24 kHz | text length | ~4 GB | ✅ conversational |
| | [Fish-Speech S2](https://github.com/fishaudio/fish-speech) | — | — | text length | — | 🧪 separate env / API server |
| | Hunyuan TTS 2.5 (API) | — | 24 kHz | text length | 0 | ✅ cloud option |
| **SFX** | [TangoFlux](https://github.com/declare-lab/TangoFlux) | 515M | 44.1 kHz stereo | 1–30 s | ~4 GB | ✅ |
| **BGM** | [ACE-Step v1](https://github.com/ace-step/ACE-Step) | 3.5B | 48 kHz | ≤ 240 s | ~16 GB | ✅ |
| **Song** | SongGeneration v2-large ([LeVo](https://arxiv.org/abs/2506.07520)) | — | 48 kHz | lyric length (≤ 4.5 min) | ~32 GB | ✅ vendored, non-commercial |

<sub>VRAM values are rough single-GPU guidance (upstream docs plus what fit during our runs), not profiled peaks. Prompt formats, parameters, lyric rules and the mixing recipe are in the [audio guide](docs/audio_guide.md).</sub>

## Video and image models

| Stage | Model | How we run it |
|---|---|---|
| First frame | HunyuanImage 3.0 (API), Nano-Banana Pro (API), FLUX.1 / SD 3.5 via [diffusers](https://github.com/huggingface/diffusers) and [xDiT](https://github.com/xdit-project/xDiT) | `utils/image/`, `run_t2i_*.py` |
| Image → video | [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | 720p + built-in SR to 1080p, 8 GPUs |
| Image → video | Wan 2.1 / 2.2 (I2V, TI2V, FLF2V, VACE), LingBot-World via [LightX2V](https://github.com/ModelTC/LightX2V) | `script/video/` benchmarks |
| Lip-sync | [LatentSync 1.6](https://github.com/bytedance/LatentSync) | 512×512, 25 steps, ~18 GB |
| Talking avatar | [LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar) | 5-GPU timestep pipeline parallelism |
| Joint audio-video | [daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman) | `distill` / `sr540p` / `sr1080p` configs |

Architectures, papers, pinned commits and licenses for each framework are in [docs/reference_frameworks.md](docs/reference_frameworks.md).

## Quick start

```bash
git clone https://github.com/Sunbeam23333/StreamDiT-IT2AV.git && cd StreamDiT-IT2AV
cp .env.example .env                     # set IT2AV_WORKSPACE, HF_TOKEN, optional proxy / API keys

# 1. Third-party frameworks at the exact commits we used (+ our patches)
bash third_party/fetch_third_party.sh audio          # TTS / SFX / BGM; add `video image` for the rest

# 2. Environments (see docs/environment_pitfalls.md before choosing)
bash script/audio/setup_audio_envs.sh all            # one conda env per audio model (least friction)
bash script/audio/setup_tts_alternatives.sh          # Spark-TTS / F5-TTS / ChatTTS / Fish-Speech
bash setup_env.sh                                    # main env for video / lip-sync / MagiHuman

# 3. Weights -> ${IT2AV_WORKSPACE}/models
python download/download_audio_models.py --models all     # every audio model; or pick: audio music tts
python download/download_models.py                   # video models (large)

# 4. Generate
bash script/audio/run_audio_benchmark.sh parallel    # TangoFlux / ACE-Step / SongGeneration / CosyVoice on GPU 0-3
bash script/audio/run_tts_benchmark.sh               # CosyVoice2 / Spark-TTS / ChatTTS / F5-TTS
bash script/digit/pipeline2/step1_generate_audio.sh  # pipeline 2: audio -> video -> lip-sync -> mix
bash script/digit/pipeline2/step2_generate_video.sh
bash script/digit/pipeline2/step3_lipsync_merge.sh
```

Each script sources [`script/common/env.sh`](script/common/env.sh), which loads `.env` and resolves every path from `IT2AV_WORKSPACE` (shared storage), `IT2AV_LOCAL_ROOT` (fast local cache) and `IT2AV_CONDA_ENV`. Nothing is hard-coded to a particular machine.

## Pipelines

| Pipeline | Entry | Flow |
|---|---|---|
| **Pipeline 2** (recommended) | `script/digit/pipeline2/` | Spark-TTS ∥ ACE-Step ∥ TangoFlux → HunyuanVideo-1.5 I2V → `setpts` retime to speech length → LatentSync → `amix` |
| **Pipeline 3** | `script/digit/pipeline3_magihuman/` | first frame + enhanced prompt → daVinci-MagiHuman (audio and video in one pass) |
| **Digital human v1** | `script/digit/step*.sh` | portrait → TTS → LiveAvatar talking video → Wan 2.2 action continuation → merge |
| **Agent** | `utils/tools/agent_video_pipeline/` | LLM storyboard → per-scenario strategy and route → stage generation → VLM review and retry → concat |

See [docs/pipelines.md](docs/pipelines.md) for options and data layout.

## Lessons from the environment work

The [environment and pitfalls log](docs/environment_pitfalls.md) has 40+ entries, each written as symptom → root cause → fix and indexed by error string. The ones that cost us the most:

- **pip `resolution-too-deep` and `OpenBLAS not found`.** Unbounded `wandb` / `tensorboard`, and old packages that pin `numpy<2` (`onnxruntime-gpu==1.18`, `descript-audio-codec`, `k-diffusion`), keep dragging numpy and scipy back to source builds. Fix: bound everything and install those packages separately with `--no-deps`.
- **SongGeneration on a modern stack.** `is_flash_attn_available` was renamed, `SequenceSummary` was removed, `torch.load` now defaults to `weights_only=True`, there is a hard assert on `transformers<4.40`, and `new_auto_prompt.pt` is a Git LFS pointer that `hf_hub_download` happily "succeeds" on.
- **CosyVoice imports.** `whisper`, `pyworld` (needed only to parse the yaml) and an empty `Matcha-TTS` submodule.
- **LatentSync on a fresh server.** The system ffmpeg lacks libx264, so the run dies with `stack expects a non-empty TensorList`. Separately, a single turned-away face used to abort the whole clip.
- **Shell plumbing.** `$(sync_model ...)` captured progress logs along with the path. Separately, `conda activate` fails in a non-interactive shell unless the conda shell hook is loaded first.

## Repository layout

```text
StreamDiT-IT2AV/
├── run_audio_benchmark.py        # TangoFlux / ACE-Step / SongGeneration / CosyVoice runner (+ RTF report)
├── run_tts_benchmark.py          # CosyVoice2/3, Spark-TTS, F5-TTS, ChatTTS, Fish-Speech runner
├── run_t2i_diffusers.py · run_t2i_xdit.py · fast_sync.py · setup_env.sh
├── script/
│   ├── common/env.sh             # single source of paths / env / proxy
│   ├── audio/                    # audio + TTS scripts, prompt sets, requirements, capability matrix
│   ├── digit/                    # digital-human pipelines (pipeline2, pipeline3_magihuman, LiveAvatar)
│   ├── video/ · image/           # model benchmarks
├── download/                     # Hugging Face / ModelScope downloaders
├── utils/
│   ├── audio/SongGeneration/     # vendored (upstream offline), patched
│   ├── audio/Hunyuan-tts/        # Hunyuan TTS 2.5 API client
│   ├── image/{Hunyuan-Image,Banana}/
│   └── tools/agent_video_pipeline/
├── third_party/                  # manifest.tsv, fetch_third_party.sh, patches/
├── docs/                         # guides (Chinese)
└── assets/
```

## Acknowledgements

This project is built entirely on the work of the open-source community: [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS), [Spark-TTS](https://github.com/SparkAudio/Spark-TTS), [F5-TTS](https://github.com/SWivid/F5-TTS), [ChatTTS](https://github.com/2noise/ChatTTS), [Fish-Speech](https://github.com/fishaudio/fish-speech), [TangoFlux](https://github.com/declare-lab/TangoFlux), [ACE-Step](https://github.com/ace-step/ACE-Step), SongGeneration / LeVo (Tencent AI Lab), [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), [LightX2V](https://github.com/ModelTC/LightX2V), [Wan](https://github.com/Wan-Video/Wan2.2), [LatentSync](https://github.com/bytedance/LatentSync), [LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar), [daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman), [diffusers](https://github.com/huggingface/diffusers), [xDiT](https://github.com/xdit-project/xDiT) and [Mora](https://github.com/lichao-sun/Mora). Please cite their papers (listed in [docs/reference_frameworks.md](docs/reference_frameworks.md)) if you use them.

## License

Code in this repository is licensed under [Apache 2.0](LICENSE). Third-party frameworks and model weights keep their own licenses, and several are **non-commercial**: SongGeneration, TangoFlux, and the ChatTTS / F5-TTS / Spark-TTS weights, among others. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{streamdit_it2av2026,
  title        = {StreamDiT-IT2AV: An Orchestration Toolkit for Image+Text to Audio-Video Generation},
  year         = {2026},
  url          = {https://github.com/Sunbeam23333/StreamDiT-IT2AV}
}
```
