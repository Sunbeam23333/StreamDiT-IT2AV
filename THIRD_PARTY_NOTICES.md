# Third-Party Notices

StreamDiT-IT2AV's own code (scripts, runners, agent pipeline, patches) is released under the Apache License 2.0 (see `LICENSE`).
It orchestrates the third-party projects listed below. They are **not** relicensed by this repository: each one stays under its upstream license, and model weights often carry terms that differ from the code license. You are responsible for complying with all of them.

## Vendored code

| Component | Location | License |
|---|---|---|
| SongGeneration (LeVo), © 2025 Tencent | `utils/audio/SongGeneration/` | **SongGeneration License Terms: academic, research and education use only; no commercial or production use.** See `utils/audio/SongGeneration/LICENSE`, which also lists the licenses of its bundled components. Modified for compatibility with transformers ≥ 4.47 and PyTorch ≥ 2.6 (see `third_party/patches/SongGeneration.patch`). |

## Fetched at pinned commits (`third_party/manifest.tsv`)

| Component | Upstream | Code license | Weights / notes |
|---|---|---|---|
| ACE-Step | ace-step/ACE-Step | Apache-2.0 | Apache-2.0 |
| ChatTTS | 2noise/ChatTTS | AGPL-3.0 | CC BY-NC 4.0 (non-commercial) |
| CosyVoice | FunAudioLLM/CosyVoice | Apache-2.0 | Apache-2.0 |
| F5-TTS | SWivid/F5-TTS | MIT | CC BY-NC 4.0 (non-commercial) |
| Matcha-TTS | shivammehta25/Matcha-TTS | MIT | — |
| Spark-TTS | SparkAudio/Spark-TTS | Apache-2.0 | CC BY-NC-SA 4.0 (non-commercial) |
| TangoFlux | declare-lab/TangoFlux | see upstream | Non-commercial research use; Stability AI Community License applies to components from Stable Audio Open |
| Fish-Speech | fishaudio/fish-speech | Fish Audio Research License | research use; see upstream |
| HunyuanVideo-1.5 | Tencent-Hunyuan/HunyuanVideo-1.5 | Tencent Hunyuan Community License | same; check territory and usage restrictions |
| LatentSync | bytedance/LatentSync | Apache-2.0 | see upstream model card (patched) |
| LiveAvatar | Alibaba-Quark/LiveAvatar | Apache-2.0 | based on Wan2.2-S2V-14B (Apache-2.0) (patched) |
| daVinci-MagiHuman | GAIR-NLP/daVinci-MagiHuman | Apache-2.0 | see upstream model card (patched) |
| LightX2V | ModelTC/LightX2V | Apache-2.0 | Wan 2.x weights: Apache-2.0 (patched) |
| MagiCompiler | SandAI-org/MagiCompiler | Apache-2.0 | — |
| diffusers | huggingface/diffusers | Apache-2.0 | FLUX.1-dev: FLUX.1 [dev] Non-Commercial License; SD 3.5: Stability AI Community License |
| xDiT | xdit-project/xDiT | Apache-2.0 | — |

Patches in `third_party/patches/` are derivative works of the respective upstream projects and are provided under the same license as the file they modify.

## External services

Hunyuan TTS / HunyuanImage and the Venus-style gateway (Gemini, Nano-Banana, Veo, Kling) are accessed over HTTP with **your own** credentials. No credentials or endpoints are shipped with this repository. Usage is governed by the provider's terms.
