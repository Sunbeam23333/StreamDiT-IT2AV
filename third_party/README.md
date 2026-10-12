# third_party

本仓库不直接提交第三方框架的源码，而是记录**我们实验时实际使用的 commit**，再加上我们的兼容补丁。这样既能精确复现实验环境，又不会把十几个上游仓库塞进本仓库。

```text
third_party/
├── manifest.tsv            # name / group / dest / repo / commit / patch / license
├── fetch_third_party.sh    # 按 commit 拉取到 utils/{audio,video,image}/ 并打补丁
└── patches/
    ├── LatentSync.patch          # 人脸检测失败帧复用相邻帧; 依赖与主环境对齐
    ├── LiveAvatar.patch          # cond KV 缓冲区按分辨率自动推算; dashscope 延迟导入; 人像 / TTS 样本准备脚本
    ├── daVinci-MagiHuman.patch   # MLP 分块防 OOM; VAE 输出 offload; magi_compiler / flash-attn 回退
    ├── LightX2V.patch            # LingBot-World 接入; H20 上的 HunyuanVideo-1.5 / Wan 2.x / VACE 配置
    └── SongGeneration.patch      # 仅作改动记录 (代码已 vendored 在 utils/audio/SongGeneration)
```

## 用法

```bash
bash third_party/fetch_third_party.sh --list               # 查看清单
bash third_party/fetch_third_party.sh                      # 默认: audio 组 (TTS / SFX / BGM)
bash third_party/fetch_third_party.sh video image          # 按组: audio | video | image | optional | all
bash third_party/fetch_third_party.sh CosyVoice Matcha-TTS # 按名字
bash third_party/fetch_third_party.sh --force LatentSync   # 删除后重新拉取
bash third_party/fetch_third_party.sh --no-patch LiveAvatar
```

脚本对每个条目执行：`git fetch --depth 1 <commit>`（失败则完整 fetch）→ `checkout` → `git apply patches/<name>.patch`，并在目标目录写入 `.it2av_fetched` 标记。已存在但不是脚本拉取的目录会被跳过（防止覆盖你的本地修改），需要覆盖时加 `--force`。`all` 不包含 `optional` 组（MagiCompiler）。

## 清单

| 名称 | 组 | 目标目录 | 上游 | commit | 补丁 |
|---|---|---|---|---|---|
| ACE-Step | audio | `utils/audio/ACE-Step` | [ace-step/ACE-Step](https://github.com/ace-step/ACE-Step) | `1bee4c9f` | — |
| ChatTTS | audio | `utils/audio/ChatTTS` | [2noise/ChatTTS](https://github.com/2noise/ChatTTS) | `c26573a6` | — |
| CosyVoice | audio | `utils/audio/CosyVoice` | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | `04bcadc6` | — |
| F5-TTS | audio | `utils/audio/F5-TTS` | [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) | `623c96c2` | — |
| Matcha-TTS | audio | `utils/audio/Matcha-TTS` | [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) | `bd4d90d9` | — |
| Spark-TTS | audio | `utils/audio/Spark-TTS` | [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) | `2f1ea908` | — |
| TangoFlux | audio | `utils/audio/TangoFlux` | [declare-lab/TangoFlux](https://github.com/declare-lab/TangoFlux) | `164420fa` | — |
| fish-speech | audio | `utils/audio/fish-speech` | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) | `49985a34` | — |
| HunyuanVideo-1.5 | video | `utils/video/HunyuanVideo-1.5` | [Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | `2641c0de` | — |
| LatentSync | video | `utils/video/LatentSync` | [bytedance/LatentSync](https://github.com/bytedance/LatentSync) | `a229c394` | ✅ |
| LiveAvatar | video | `utils/video/LiveAvatar` | [Alibaba-Quark/LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar) | `775e3630` | ✅ |
| daVinci-MagiHuman | video | `utils/video/daVinci-MagiHuman` | [GAIR-NLP/daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman) | `bd819647` | ✅ |
| LightX2V | video | `utils/video/LightX2V-main` | [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) | `cffcc57b` | ✅ |
| MagiCompiler | optional | `utils/video/MagiCompiler` | [SandAI-org/MagiCompiler](https://github.com/SandAI-org/MagiCompiler) | `b4cc3161` | — |
| diffusers | image | `utils/image/diffusers` | [huggingface/diffusers](https://github.com/huggingface/diffusers) | `07a63e19` | — |
| xDiT | image | `utils/image/xDiT` | [xdit-project/xDiT](https://github.com/xdit-project/xDiT) | `84813c97` | — |
| **SongGeneration** | audio | `utils/audio/SongGeneration` | ~~tencent-ailab/SongGeneration~~（已下线） | vendored | 已应用 |

## 这些 commit 是怎么确定的

实验时各框架是以 GitHub 源码包的形式下载的，没有保留 `.git`。我们用"本地目录的文件时间戳 ↔ 上游 commit 历史"反查出对应的 commit，再拉取该 commit 与实验副本做 `diff -rq`，确认未修改的仓库完全一致；有修改的仓库则把差异导出为补丁。

## 关于 CosyVoice 的 Matcha-TTS

CosyVoice 的 `third_party/Matcha-TTS` 是 git submodule，浅拉取后是空目录，会导致 `No module named 'matcha'`。`fetch_third_party.sh` 会把空目录替换成指向 `utils/audio/Matcha-TTS` 的软链接，所以拉取 CosyVoice 时请同时拉取 Matcha-TTS（`audio` 组已包含）。

## 许可证

每个框架遵循其上游许可证，详见 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。其中 SongGeneration、TangoFlux、ChatTTS / F5-TTS / Spark-TTS 的权重、Fish-Speech 等**带有非商用限制**。
