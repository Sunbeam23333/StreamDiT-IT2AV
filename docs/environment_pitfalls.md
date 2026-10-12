# 环境搭建与踩坑全记录

> 本文整理了我们在 8×H20 (95 GB) 服务器、Python 3.12、CUDA 12.x 上把 TTS / 音效 / BGM / 歌曲 / 视频 / 对口型整条链路跑通时踩过的**所有**坑。
> 每一条都按 **现象 → 根因 → 解决** 组织，并注明本仓库中已经落地修复的位置。遇到报错时，先在下面的 **速查表** 里 `Ctrl+F` 搜报错关键字。

- [0. 推荐环境布局](#0-推荐环境布局)
- [速查表：报错关键字 → 条目](#速查表报错关键字--条目)
- [A. 通用 / 基础设施](#a-通用--基础设施)
- [B. pip 依赖解析 (requirements_audio.txt 的六轮报错)](#b-pip-依赖解析-requirements_audiotxt-的六轮报错)
- [C. TangoFlux（音效 SFX）](#c-tangoflux音效-sfx)
- [D. ACE-Step（背景音乐 BGM）](#d-ace-step背景音乐-bgm)
- [E. SongGeneration / LeVo（歌曲）](#e-songgeneration--levo歌曲)
- [F. CosyVoice 2 / 3（TTS）](#f-cosyvoice-2--3tts)
- [G. 其它 TTS：Spark-TTS / F5-TTS / ChatTTS / Fish-Speech / 混元 TTS](#g-其它-ttsspark-tts--f5-tts--chattts--fish-speech--混元-tts)
- [H. 视频生成与对口型](#h-视频生成与对口型)
- [I. Agent / 云端 API](#i-agent--云端-api)
- [J. 开源整理时的额外处理](#j-开源整理时的额外处理)

---

## 0. 推荐环境布局

各音频框架对 `diffusers` / `transformers` 的版本要求互相冲突，我们最终采用 **"一个主环境 + 按需拆分"** 的方式：

| 环境 | 用途 | 关键版本 | 安装方式 |
|---|---|---|---|
| `it2av`（主环境，`IT2AV_CONDA_ENV`） | 视频 / 对口型 / MagiHuman / 统一 TTS benchmark | Python 3.12, torch 2.5.1, transformers 4.57.1 | `bash setup_env.sh` + `pip install -r script/audio/requirements_tts_unified.txt` |
| `audio_tangoflux` | TangoFlux 音效 | diffusers 0.30.0, transformers 4.44.0（~4 GB 显存） | `bash script/audio/setup_audio_envs.sh tangoflux` |
| `audio_ace_step` | ACE-Step BGM | diffusers ≥0.33.0, transformers 4.50.0（~16 GB） | `bash script/audio/setup_audio_envs.sh ace_step` |
| `audio_songgen` | SongGeneration 歌曲 | diffusers 0.27.2, transformers 4.37.2（~32 GB） | `bash script/audio/setup_audio_envs.sh songgen` |
| `audio_cosyvoice` | CosyVoice 2/3 | diffusers 0.29.0, transformers 4.51.3（~4 GB） | `bash script/audio/setup_audio_envs.sh cosyvoice` |
| `spark_tts` / `f5tts` / `chattts` / `fish_speech` | TTS 备选 | 各仓库原生依赖 | `bash script/audio/setup_tts_alternatives.sh <name>` |

> **实际上我们最终把 TangoFlux / ACE-Step / SongGeneration / CosyVoice 全塞进了同一个 Python 3.12 环境**（`requirements_audio.txt`，transformers 4.57.1 + numpy 2.x），代价就是下面 B、E 两节的那一长串兼容修补。如果你不想重走一遍，直接按上表分环境装最省事。

`requirements_audio.txt` 统一环境的安装顺序（**顺序不能乱**，原因见 B 节）：

```bash
conda activate it2av
pip install -r script/audio/requirements_audio.txt      # 1. 主依赖 (numpy 2.x / scipy>=1.14)
pip install descript-audio-codec --no-deps               # 2. SongGeneration 用, 不能参与依赖解析
pip install onnxruntime-gpu==1.20.1 --no-deps            # 3. CosyVoice 用, 必须 >=1.19 (numpy 2 ABI)
pip install descript-audiotools argbind k-diffusion      # 4. SongGeneration 运行时隐式依赖 (见 E5)
pip install openai-whisper pyworld                       # 5. CosyVoice 运行时隐式依赖 (见 F1/F3)
```

---

## 速查表：报错关键字 → 条目

| 报错 / 现象关键字 | 条目 |
|---|---|
| `File name too long`（路径里混着 `[SCAN]` `[COPY]` 日志） | [A1](#a1-sync_model-的返回值被日志污染--file-name-too-long) |
| `torchrun: command not found` / `conda activate` 无效 | [A2](#a2-非交互-shell-里-conda-没初始化) |
| `Unrecognized option 'crf'` / `stack expects a non-empty TensorList` | [A3](#a3-系统-ffmpeg-没有-libx264) |
| `Network is unreachable` 访问 huggingface.co | [A4](#a4-服务器无外网--模型在推理时在线下载) |
| 改了代码但服务器上跑的还是旧逻辑 | [A5](#a5-代码没同步--本地盘缓存过期) |
| `error: resolution-too-deep` | [B1](#b1-resolution-too-deep) |
| `Could not configure working env. Have you installed MeCab?` | [B2](#b2-fugashi-回退到源码包要求系统-mecab) |
| `ERROR: Dependency "OpenBLAS" not found` | [B3](#b3-onnxruntime-gpu1180-把-numpy-降级--scipy-源码编译失败) / [B4](#b4-descript-audio-codec-的依赖链引发回溯) |
| `module 'pkgutil' has no attribute 'ImpImporter'` | [B5](#b5-k-diffusion-拉出-scikit-learn-121-源码包python-312-构建失败) |
| `AttributeError: _ARRAY_API not found` | [B6](#b6-_array_api-not-found) |
| `peft>=0.17.0 is required ... found peft==0.15.0` | [B7](#b7-peft0170-is-required) |
| `numpy.dtype size changed ... Expected 96 from C header, got 88` | [B8](#b8-numpydtype-size-changed) |
| `HTTPSConnectionPool(host='huggingface.co') ... flan-t5-large` | [C1](#c1-flan-t5-large-在推理时在线下载) |
| ACE-Step `Unsupported format` | [D1](#d1-保存路径没有-wav-扩展名--unsupported-format) |
| `No module named 'hangul_romanize'` | [D2](#d2-ace-step-推理链缺依赖) |
| `cannot import name 'is_flash_attn_available'` | [E1](#e1-is_flash_attn_available-被改名) |
| `WeightsUnpickler error` / `Unsupported operand 118` | [E2](#e2-pytorch-26-torchload-默认-weights_onlytrue) |
| `_pickle.UnpicklingError: invalid load key, 'v'` | [E3](#e3-new_auto_promptpt-是-git-lfs-指针) |
| `cannot import name 'SequenceSummary'` | [E4](#e4-sequencesummary-在-transformers-447-中被移除) |
| `No module named 'audiotools'` / `'argbind'` / `'k_diffusion'` | [E5](#e5-songgeneration-的隐式依赖) |
| `AssertionError`（`levo.py` 的 transformers 版本断言） | [E6](#e6-levopy-断言-transformers--440) |
| `No module named 'whisper'` | [F1](#f1-no-module-named-whisper) |
| `No module named 'matcha'` | [F2](#f2-no-module-named-matcha) |
| `No module named 'pyworld'` | [F3](#f3-no-module-named-pyworld) |
| TTS 输出"像人声但不是人话" | [F4](#f4-未解决cosyvoice-输出非人类语言) |
| 缺少 `Glyph-SDXL-v2` / `byt5_model.pt` | [H1](#h1-hunyuanvideo-15-缺少-glyph-sdxl-v2) |
| `--resolution 1080p` 不被接受 | [H2](#h2-hunyuanvideo-15-没有原生-1080p) |
| LatentSync `Face not detected` | [H4](#h4-latentsync-转头--遮挡帧检测不到人脸直接崩溃) |
| MagiHuman `CUDA out of memory` / `No module named 'magi_compiler'` | [H6](#h6-davinci-magihuman-显存峰值与-magi_compiler) |
| LiveAvatar 高分辨率下 cond cache 越界 | [H7](#h7-liveavatar-cond-kv-缓冲区写死为-720400) |
| Venus 视频接口 401 / 鉴权失败 | [I1](#i1-venus-视频接口必须用-sdk-签名) |

---

## A. 通用 / 基础设施

我们的服务器形态是：**共享盘（NFS，慢）存模型和代码 + 每台机器的本地盘（`/tmp`，快）做缓存**。本仓库中对应 `IT2AV_WORKSPACE`（共享）和 `IT2AV_LOCAL_ROOT`（本地），`fast_sync.py` 负责多线程增量同步，所有脚本都会先 `sync_model` / `sync_code` 再推理。

### A1. `sync_model` 的返回值被日志污染 → `File name too long`

- **现象**：`mkdir: cannot create directory '...[SCAN] 扫描源目录 ... [DONE] ...': File name too long`。
- **根因**：shell 函数用 `echo "${DST}"` 返回路径、调用方用 `$(sync_model ...)` 捕获；但函数里调用的 `python fast_sync.py` 把进度日志打到了 **stdout**，被 `$()` 一起捕获，变量里变成"几百字符日志 + 路径"。
- **解决**：函数内部所有日志都走 stderr：`python "${SYNC_SCRIPT}" "${SRC}" "${DST}" --workers 32 --skip-existing >&2`，`echo` 日志也加 `>&2`。
- **已修**：`script/digit/pipeline2/step{1,3}_*.sh`、`script/digit/step_digit_beauty_pipeline.sh`。

### A2. 非交互 shell 里 conda 没初始化

- **现象**：`bash xxx.sh` 时 `torchrun: command not found`，或 `conda activate` 报 `CondaError: Run 'conda init' before 'conda activate'`。
- **根因**：非交互 shell 不读 `~/.bashrc`，`conda` 函数不存在；另外有脚本把 `conda activate` 写在变量定义之前。
- **解决**：
  ```bash
  eval "$(conda shell.bash hook)" 2>/dev/null || true   # set -e 下不要因为 hook 的警告退出
  conda activate "${IT2AV_CONDA_ENV}"
  ```
  环境名统一由 `script/common/env.sh` 里的 `IT2AV_CONDA_ENV`（默认 `it2av`）提供，所有脚本开头都会 `source env.sh`。

### A3. 系统 ffmpeg 没有 libx264

- **现象**（LatentSync）：
  ```
  Unrecognized option 'crf'.
  Error splitting the argument list: Option not found
  Error: Could not open video.
  Affine transforming 0 faces...
  RuntimeError: stack expects a non-empty TensorList
  ```
- **根因**：LatentSync 在 `latentsync/utils/util.py::read_video` 和 `lipsync_pipeline.py` 里用 `subprocess.run("ffmpeg ... -crf 18 ...")` 调**系统** ffmpeg；服务器自带的 ffmpeg 没编译 libx264，不认 `-crf` → 视频预处理失败 → 0 张人脸 → `torch.stack([])` 崩溃。真正的错误在第一行，最后的 Tensor 报错只是连锁反应。
- **解决**：下载一个静态版 ffmpeg（我们用的是 `ffmpeg-7.0.2-amd64-static`），在 `.env` 里设 `IT2AV_FFMPEG_DIR=/path/to/ffmpeg-7.0.2-amd64-static`，`env.sh` 会把它放到 `PATH` 最前面，子进程里的 `ffmpeg` 也会用到它。

### A4. 服务器无外网 / 模型在推理时在线下载

- **现象**：`HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded ... [Errno 101] Network is unreachable`。
- **根因**：很多框架在 `from_pretrained("org/name")` 时默认联网（TangoFlux 的 flan-t5、HunyuanVideo 的 Glyph 等）。
- **解决**：
  1. 在有网的机器上用 `download/*.py` 预下载到 `${IT2AV_WORKSPACE}/models/`；
  2. 推理时显式传入本地路径（见 [C1](#c1-flan-t5-large-在推理时在线下载)）；
  3. 需要代理时在 `.env` 里设 `IT2AV_PROXY_URL`（`env.sh` 只在没有 `HTTP(S)_PROXY` 时才导出），下载脚本也认 `IT2AV_PROXY_HOST/PORT/USER/PASS` 和 `HF_TOKEN`、`HF_ENDPOINT`。
- **注意**：代理账号密码**只放 `.env`**（已在 `.gitignore`），不要写进脚本。

### A5. 代码没同步 / 本地盘缓存过期

- **现象**：明明改了代码，日志里还在打印旧的提示语（我们在 SongGeneration 上被这个坑了三轮）；或者 LatentSync 的补丁"不生效"。
- **根因**：脚本从共享盘读代码，再 `fast_sync` 到本地盘；本地盘有 `.sync_done` 标记时会跳过同步。共享盘没更新、或本地盘有旧缓存，跑的都是旧代码。
- **解决**：改完代码先确认共享盘上的文件已更新；需要刷新本地缓存时加 `--force_sync`，或删除 `${IT2AV_LOCAL_ROOT}/code/.../.sync_done`；调试时可用 `--skip_sync` 直接跑共享盘上的代码。

### A6. 服务器没有 git-lfs

- **现象**：`git lfs pull` → `git: 'lfs' is not a git command`；clone 下来的 `.pt` 只有 133 字节。
- **解决**：用 `curl -L https://media.githubusercontent.com/media/<org>/<repo>/<branch>/<path>` 直接拉 LFS 实体文件，并校验大小；或者在有 git-lfs 的机器上拉好再同步。SongGeneration 的情况见 [E3](#e3-new_auto_promptpt-是-git-lfs-指针)。

---

## B. pip 依赖解析 (requirements_audio.txt 的六轮报错)

把 TangoFlux / ACE-Step / SongGeneration / CosyVoice 的依赖合进一个 Python 3.12 环境时，pip 先后炸了六次。核心教训只有两条：

1. **所有大包都要有上下界**，否则 pip 会把几十个历史版本挨个试一遍；
2. **"老旧且约束保守"的包（onnxruntime-gpu 1.18、descript-audio-codec、k-diffusion）不要参与依赖解析**，单独 `--no-deps` 装。

### B1. `resolution-too-deep`

- **现象**：`error: resolution-too-deep × Dependency resolution exceeded maximum depth`。
- **根因**：`wandb`（0.19.6 → 0.9.7，60+ 版本）、`tensorboard`（2.20 → 1.13，40+）、`tensorboardX`、`deepspeed` 都没有上界，每个版本又牵出 `google-auth → cachetools → rsa` 的递归回溯，组合爆炸。
- **解决**：加上下界：`tensorboard>=2.18,<2.21`、`tensorboardX>=2.6,<2.7`、`wandb>=0.19,<0.20`、`deepspeed>=0.15.1,<0.17`、`pytorch_lightning>=2.5.1,<2.7`、`lightning>=2.2.4,<2.7`、`spacy>=3.8,<3.9`、`modelscope>=1.20,<1.35`。

### B2. fugashi 回退到源码包，要求系统 MeCab

- **现象**：`RuntimeError: Could not configure working env. Have you installed MeCab?`。
- **根因**：pip 把 `fugashi` 从 1.5.2 一路回退到 1.3.1；1.4.0 以上有预编译 wheel，1.3.x 是源码包，需要系统装 `mecab` + `libmecab-dev`。同时 `click` 也在 8.3 → 7.1 之间回溯。
- **解决**：`fugashi[unidic-lite]>=1.4.0`、`click>=8.1.0,<9`，并给 `gradio` / `fastapi` / `uvicorn` / `rich` 加上界。

### B3. `onnxruntime-gpu==1.18.0` 把 numpy 降级 → scipy 源码编译失败

- **现象**：`ERROR: Dependency "OpenBLAS" not found`。
- **根因链**：
  ```
  onnxruntime-gpu==1.18.0 要求 numpy<2
    → pip 把 numpy 2.2.6 降到 1.26.4
      → scipy 跟着回溯 1.17.1 → 1.11.1（下载 14 个版本，~500 MB）
        → scipy 1.11.1 没有 cp312 wheel，只能源码编译 → 找不到 OpenBLAS
  ```
- **解决**：`numpy>=2.0.0`、`scipy>=1.14.0`（1.14 起才有 cp312 wheel），并把 onnxruntime-gpu 移出 requirements，最后单独 `--no-deps` 安装。

### B4. descript-audio-codec 的依赖链引发回溯

- **现象**：同样是 `OpenBLAS not found`，但这次 onnxruntime 已经移除了。
- **根因**：`descript-audio-codec → descript-audiotools → pyloudnorm / pystoi / torch-stoi / flatten-dict / randomname / markdown2 ...`，这些包互相约束，迫使 pip 再次把 numpy 降到 1.26.4。
- **解决**：`descript-audio-codec` 也移出 requirements，`pip install descript-audio-codec --no-deps`（只有 SongGeneration 的 `mert_model.py` 用它，真正的运行时依赖 torch / numpy / scipy 已经装好）。

### B5. k-diffusion 拉出 scikit-learn 1.2.1 源码包，Python 3.12 构建失败

- **现象**：`AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`。
- **根因**：`k-diffusion → clean-fid（回溯 20 个版本）/ dctorch` → 波及 `soxr`（回退到 0.3.2 源码包）和 `scikit-learn`（回退到 1.2.1 源码包）→ 老 setuptools 用了 Python 3.12 已删除的 `pkgutil.ImpImporter`。
- **解决**：从 requirements 里删掉 `k-diffusion`。注意：SongGeneration 运行时（`third_party/stable_audio_tools/.../sampling.py`）其实还要 `import k_diffusion`，所以装完主依赖后再单独 `pip install k-diffusion`（见 [E5](#e5-songgeneration-的隐式依赖)）。

### B6. `_ARRAY_API not found`

- **现象**：
  ```
  from .onnxruntime_pybind11_state import *
  AttributeError: _ARRAY_API not found
  ```
- **根因**：`onnxruntime-gpu 1.18.0` 的 pybind11 扩展是针对 numpy 1.x C ABI 编译的，numpy 2.x 改了 ABI。`--no-deps` 只能绕过安装期检查，绕不过运行期 ABI。
- **解决**：升级到 `onnxruntime-gpu==1.20.1`（≥1.19 支持 numpy 2.x）。CosyVoice 只用 `InferenceSession` / `SessionOptions`，完全兼容。

### B7. `peft>=0.17.0 is required`

- **现象**：TangoFlux 加载时报 `peft>=0.17.0 is required for a normal functioning of this module, but found peft==0.15.0`。
- **根因**：`diffusers>=0.33` 硬性要求 `peft>=0.17`，而 requirements 里写的是 `peft>=0.10`。
- **解决**：`peft>=0.17.0,<1`。

### B8. `numpy.dtype size changed`

- **现象**（LatentSync 环境）：`ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88`，`insightface` 导入失败。
- **根因**：LatentSync 锁了 `numpy==1.26.4`，但装上的 `scikit-image` wheel 是针对 numpy 2.x 编译的。
- **解决**：`pip install --force-reinstall --no-cache-dir "scikit-image==0.22.0" "numpy==1.26.4"`。顺带出现的 `bitsandbytes ... compiled without GPU support` 只是警告，不影响推理。

---

## C. TangoFlux（音效 SFX）

### C1. flan-t5-large 在推理时在线下载

- **现象**：`Max retries exceeded with url: /google/flan-t5-large/resolve/main/config.json`。
- **根因**：`TangoFlux.__init__()` 在 `text_encoder_dir=None` 时用 `config["text_encoder_name"]`（`google/flan-t5-large`）在线 `from_pretrained`。
- **解决**：`python download/download_audio_models.py --models flan_t5_large` 预下载，`run_audio_benchmark.py` 里改为 `TangoFluxModel(config, text_encoder_dir=sync_model("flan_t5_large"))`。

### C2. 使用约束

- 文本编码器是 flan-t5，**prompt 只用英文**，中文 prompt 效果明显变差；
- `duration` 上限 **30 s**，输出 44.1 kHz 立体声；推荐 `steps=50, guidance_scale=4.5`；
- 直接加载本地 safetensors，不走 `snapshot_download`。

---

## D. ACE-Step（背景音乐 BGM）

### D1. 保存路径没有 `.wav` 扩展名 → `Unsupported format`

- **根因**：`save_path = f"{scene_id}_bgm"`，ACE-Step 的 `save_wav_file()` 发现它不是目录就直接当文件名用，`torchaudio.save()` 按扩展名推断格式失败。
- **解决**：路径加上 `.wav`。

### D2. ACE-Step 推理链缺依赖

- **现象**：`ModuleNotFoundError: No module named 'hangul_romanize'`（之后还会一个接一个地报）。
- **解决**：一次装齐歌词前端依赖：
  ```bash
  pip install hangul-romanize num2words spacy pypinyin py3langid opencc-python-reimplemented loguru tokenizers
  ```
  ACE-Step 只用 `spacy.lang.zh/ja/en` 的轻量分词器，**不需要** `python -m spacy download xx_core_web_sm`。

### D3. 使用约束

- 最长 **240 s**（`audio_duration<=0` 时会在 30–240 s 间随机）；输出 48 kHz；
- 我们的推理参数：`infer_step=100, guidance_scale=15, scheduler=euler, cfg_type=apg, omega_scale=10, guidance_interval=0.5, min_guidance_scale=3`；
- 纯音乐 BGM 的歌词写 `[inst-long] ; [inst-long] ; [inst-long]`，详见 [音频使用指南](audio_guide.md)。

---

## E. SongGeneration / LeVo（歌曲）

SongGeneration 是整条链路里最难伺候的：它的代码锁定在 `transformers==4.37.2` / `torch<2.6` 的年代，而我们的环境是 transformers 4.57 + torch 2.6+ + Python 3.12。下面每一条在 `utils/audio/SongGeneration`（vendored）里都已经修好，完整 diff 见 `third_party/patches/SongGeneration.patch`。

### E1. `is_flash_attn_available` 被改名

- **根因**：`codeclm/models/llama/modeling_llama.py` 拷贝自旧版 transformers，新版改名为 `is_flash_attn_2_available`。
- **解决**：
  ```python
  try:
      from transformers.utils import is_flash_attn_2_available as is_flash_attn_available
  except ImportError:
      from transformers.utils import is_flash_attn_available
  ```

### E2. PyTorch 2.6 `torch.load` 默认 `weights_only=True`

- **现象**：`_pickle.UnpicklingError: Weights only load failed ... Unsupported operand 118`。
- **解决**：`generate.py`（4 处）和 `tools/gradio/`（3 处）的 `torch.load` 全部显式加 `weights_only=False`。只对可信的官方权重这样做。

### E3. `new_auto_prompt.pt` 是 Git LFS 指针

- **现象**：`_pickle.UnpicklingError: invalid load key, 'v'`。
- **根因**：上游仓库 `.gitattributes` 里 `*.pt filter=lfs`，clone 下来的 `tools/new_auto_prompt.pt` 只有 133 字节的指针（`version https://git-lfs...`，真实文件 14,959,842 字节，sha256 `616dbe27...134e7`）。我们还踩了两个连环坑：
  1. HF 上的 `SongGeneration-Runtime` 只有 `ckpt/` 和 `third_party/`，**没有 `tools/`**，所以靠 symlink 替换 `tools/` 不起作用；
  2. `hf_hub_download` 发现本地已有同名文件时直接返回缓存，**下载"成功"了但拿到的还是 133 字节的指针**。
- **解决**：`generate.py` 新增 `_resolve_lfs_file()`：检测指针 → 下载到 `.downloading` 临时文件 → 校验大小和文件头 → 原子替换，失败直接 `raise`，不静默继续。
- **⚠️ 现状**：`tencent-ailab/SongGeneration` 的 GitHub 仓库目前已下线（404），默认的 LFS 地址可能拿不到文件。请自行准备该文件，最直接的做法是把它覆盖到 `utils/audio/SongGeneration/tools/new_auto_prompt.pt`；也可以用环境变量：
  ```bash
  export SONGGEN_AUTO_PROMPT=/path/to/new_auto_prompt.pt            # run_audio_benchmark.py 会拷贝到 tools/
  export SONGGEN_AUTO_PROMPT_URL=https://your.mirror/new_auto_prompt.pt
  export SONGGEN_LFS_BASE_URL=https://your.mirror/SongGeneration     # generate.py 自身的 LFS 下载基址
  ```

### E4. `SequenceSummary` 在 transformers ≥4.47 中被移除

- **现象**：`ImportError: cannot import name 'SequenceSummary' from 'transformers.modeling_utils'`。
- **解决**：新增 `codeclm/tokenizer/Flow1dVAE/models_gpt/models/_compat.py`（`SequenceSummary` 的完整实现），12 个 `gpt2*.py` 改为 `try: from transformers.modeling_utils import SequenceSummary / except ImportError: from ._compat import ...`。

### E5. SongGeneration 的隐式依赖

- **现象**：依次报 `No module named 'audiotools'` → `'argbind'` → `'k_diffusion'`。
- **根因**：调用链 `generate.py → audio_tokenizer → Flow1dVAE → stable_audio_tools → dac → audiotools`，这些包都不在官方 `requirements.txt` 里。注意 PyPI 名和 import 名不同：`descript-audiotools` → `audiotools`，`k-diffusion` → `k_diffusion`。
- **解决**：`pip install descript-audiotools argbind k-diffusion`（`run_audio_benchmark.py::_ensure_songgen_deps()` 也会自动检查并安装）。

### E6. `levo.py` 断言 transformers < 4.40

- **现象**：`assert version.parse(transformers.__version__) < version.parse("4.40")` 直接失败。
- **根因与解决**：SongGeneration 自带完整的 llama 实现（`codeclm/models/llama/`），并不依赖 transformers 内置的 LlamaModel，这个断言只是作者的保守锁版本。已注释掉，同环境的其它模型可以继续用新版 transformers。

### E7. 运行时资源

- 需要 HF 上的 `SongGeneration-Runtime`（`ckpt/` + `third_party/`，含 Qwen2-7B tokenizer、demucs、stable_audio_tools）和 `SongGeneration-v2-large` 权重：`python download/download_audio_models.py --models songgen_v2_large songgen_runtime`；`run_audio_benchmark.py` 会把 `ckpt/`、`third_party/` 软链接进代码目录（已在 `.gitignore`）。
- 以子进程方式运行 `generate.py`（`cwd` 必须是 SongGeneration 根目录）；日志里的 `reserved memory: 95GB` 只是打印的整卡容量，v2-large 建议单卡显存约 32 GB 以上。
- 歌词格式非常严格（英文标点、段落用 `;`、句子用 `.`），见 [音频使用指南](audio_guide.md#4-歌曲--songgeneration)。

---

## F. CosyVoice 2 / 3（TTS）

### F1. `No module named 'whisper'`

- **根因**：`cosyvoice/cli/frontend.py` 顶部 `import whisper`（用于提取 mel）。
- **解决**：`pip install openai-whisper`（主环境里用 `--no-deps` 装，避免它改动 torch / numba 版本）。

### F2. `No module named 'matcha'`

- **根因**：CosyVoice 的 `third_party/Matcha-TTS` 是 git submodule，浅 clone 或复制后是**空目录**；`flow_matching.py` / `decoder.py` 需要 `from matcha.models.components.flow_matching import BASECFM`。原来的代码 `if os.path.exists(matcha_path)` 对空目录也成立，于是静默跳过、到 import 才报错。
- **解决**：单独拉取 Matcha-TTS（`utils/audio/Matcha-TTS`），`third_party/fetch_third_party.sh` 会自动把空 submodule 目录替换为指向它的软链接；`run_audio_benchmark.py` 用"目录里有没有 `setup.py`"来判断是否真正就绪。

### F3. `No module named 'pyworld'`

- **根因**：`hyperpyyaml` 解析 yaml 时通过 `pydoc.locate` 间接 import 了 `cosyvoice.dataset.processor`，它顶部 `import pyworld as pw`——**推理根本用不到，但不装就加载不了配置**。
- **解决**：`pip install pyworld`（`setup_env.sh` Step 5b 已包含）。

### F4. 【未解决】CosyVoice 输出"非人类语言"

- **现象**：音频有能量、有共振峰、音色也像参考人，但听起来是"外星语"。
- **排查过程**：

  | 轮次 | 怀疑点 | 结论 |
  |---|---|---|
  | 1 | `fp16=True` 数值问题 | 改 `fp16=False`，**无效** |
  | 2 | CosyVoice3 本身质量问题（上游 issue #1703） | 退回 CosyVoice2，**无效** |
  | 3 | CPU 版 onnxruntime 让 `speech_tokenizer_v2.onnx` 静默回退到 CPU | 换 `onnxruntime-gpu` 1.18 → 1.20.1，**无效** |
  | 4 | whisper 版本导致 `mel_filters` 不一致 | 读源码后**排除** |
  | 5 | 写了 `debug_cosyvoice_weights.py` / `debug_cosyvoice_tts.py` 逐级诊断 | 信号层全部正常，**LLM 输出的 speech token 唯一值率 <5%、相邻重复率 >50%** |

- **当前结论**：问题出在"prompt speech tokenizer → LLM"这一段——LLM 退化成在少数几个 token 之间打转，下游 Flow + HiFT 忠实地把垃圾 token 解码成"频谱像人声但不是人话"的音频。具体是 ONNX tokenizer 输出异常、输入 mel 数值范围不对，还是权重文件不完整，**没能拿到确凿证据**。
- **规避**：数字人主链路改用 Spark-TTS（本地）或混元 TTS（云端 API）；`run_audio_benchmark.py` 默认 `--tts_engine cosyvoice2`。如果要继续排查，跑 `bash script/digit/run_debug_tts.sh`，重点看 Step 4/5 的 token 统计。详细分析和"为什么榜单分数高但我们跑出来是乱码"见 [TTS 评测与 CosyVoice 排查](tts_evaluation.md)。

### F5. 其它注意事项

- 采样率是 **24 kHz**（CosyVoice2/3 的 yaml 都是 `sample_rate: 24000`）；
- `ttsfrd`（文本正则化）是可选包，没有它会退回 `wetext`；
- CosyVoice3 被我们标记为 `recommended: false`，默认 benchmark 只跑 CosyVoice2。

---

## G. 其它 TTS：Spark-TTS / F5-TTS / ChatTTS / Fish-Speech / 混元 TTS

| 框架 | 坑 / 注意事项 |
|---|---|
| **Spark-TTS** | 输出 **16 kHz**（其它 TTS 都是 24 kHz），混音前不必手动重采样，`ffmpeg amix` 会处理；不给参考音频时用 `gender / pitch / speed` 造音色（我们的数字人用 `female / moderate / moderate`）。 |
| **F5-TTS** | **必须**有参考音频 + 参考文本（`--ref_audio --ref_text`），参考文本必须和参考音频内容一致，否则会出现漏读 / 重复；权重是 CC BY-NC。 |
| **ChatTTS** | 音色是随机采样的 speaker embedding，**不固定 seed 每次音色都不一样**（`--chattts_seed 42`）；`pynini` / `WeTextProcessing` 在 Python 3.12 上经常退回源码编译失败，统一环境里没装，需要文本正则化时 `conda install -c conda-forge pynini=2.1.5`。 |
| **Fish-Speech** | vendored 的是 **S2** 代码，官方 `pyproject` 绑定 `torch==2.8.0`，和主环境冲突；而下载脚本拉的是 1.5 权重，旧 runner 与 S2 代码脱节。因此标记为 `experimental`，推荐单独环境 + API server 方式接入。 |
| **混元 TTS 2.5（云端）** | 需要 `HUNYUAN_API_KEY` + `HUNYUAN_API_BASE`；`step2_generate_audio.sh` 只有在两者都设置时才默认走混元，否则自动回退到 CosyVoice。返回 24 kHz PCM，脚本负责封装成 wav。 |

---

## H. 视频生成与对口型

### H1. HunyuanVideo-1.5 缺少 Glyph-SDXL-v2

- **现象**：加载 pipeline 时找不到 `<model_root>/text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt`。
- **根因**：Glyph-SDXL-v2（byT5 glyph 文字渲染编码器）托管在 **ModelScope**，不在 HunyuanVideo-1.5 的 HF 仓库里；单独下载到共享盘后，`fast_sync` 只同步主模型目录，本地盘依然缺这个子目录。
- **解决**：`python download/download_models.py --models glyph_sdxl`（或 `modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir .../text_encoder/Glyph-SDXL-v2`）；`step2_generate_video.sh` 会检查本地盘，缺失时自动从共享盘补同步，都没有才报错退出。

### H2. HunyuanVideo-1.5 没有原生 1080p

- `--resolution` 只接受 `480p` / `720p`。1080p 的正确做法是 **720p 生成（50 步）+ 内置 SR（720→1080，8 步蒸馏）**：`--resolution 720p --sr true`，加 `--save_pre_sr_video true` 同时保留 720p 版本。
- 8×H20 下关闭 offloading（总显存 760 GB 足够），开启 CFG 蒸馏（约 2 倍加速）和 DeepCache；SageAttention 需要额外安装（`--sage_attn`）。

### H3. 视频时长与 TTS 时长不一致

- HunyuanVideo-1.5 默认 121 帧 / 24 fps ≈ 5 s，而 TTS 台词长短不一、BGM / SFX 又明显更长。
- **做法**（`step3_lipsync_merge.sh`）：
  1. `ffmpeg -vf "setpts=PTS*${TTS_DUR}/${VIDEO_DUR}" -r 25 -an` 把视频**慢放到 TTS 时长**，同时转 **25 fps**（LatentSync 要求 25 fps；音频长于视频时它会正→反→正循环视频，效果很怪）；
  2. LatentSync 对口型；
  3. `atrim=0:${TTS_DUR}` 把 BGM / SFX 截到 TTS 时长，再 `amix`（TTS 1.0 / BGM 0.15 / SFX 0.2，可用 `--vol_*` 调整，`--no_bgm` / `--no_sfx` 关闭）。

### H4. LatentSync 转头 / 遮挡帧检测不到人脸直接崩溃

- **根因**：`affine_transform_video` 逐帧检测人脸，任何一帧失败就 `raise`；而 I2V 生成的视频里人物经常转头。
- **解决**（`third_party/patches/LatentSync.patch`）：检测失败的帧复用**上一帧**的 face / box / affine matrix；开头连续失败的帧先占位、等第一个成功帧出现后回填；只有全部帧都失败才报错。日志会打印 `X/N frames had no face detected, reused nearest neighbor`。

### H5. LatentSync 依赖与资源

- 官方 requirements 锁的 `transformers==4.48.0`、`diffusers==0.32.2`、`accelerate==0.26.1` 跟主环境冲突，补丁统一为 `transformers==4.57.1`、`diffusers>=0.33,<1`、`accelerate>=0.34.2`；
- 1.6 版（512×512，`configs/unet/stage2_512.yaml`）需要约 **18 GB** 显存；推理参数 `--inference_steps 25 --guidance_scale 1.5 --enable_deepcache`；
- `python download/download_latentsync.py --inference-only --link-checkpoints` 只下载推理必需的 ~5.7 GB 并自动链接 `checkpoints/`。

### H6. daVinci-MagiHuman 显存峰值与 `magi_compiler`

- **现象**：长视频 / SR 阶段 `CUDA out of memory`；未安装 MagiCompiler 时 `No module named 'magi_compiler'`；没有 flash-attn 的机器上 import 失败。
- **解决**（`third_party/patches/daVinci-MagiHuman.patch`）：
  - `dit_module.py`：MLP 的 fp32 激活按 **8192 token 分块**计算（单块峰值约 2 GB，原实现一次性分配整段 fp32 中间结果）；`magi_compiler` 缺失时退化为 no-op 装饰器；`flash_attn` 缺失时回退；
  - `turbo_vaed`：`output_offload` 写入预分配的 CPU tensor，避免解码时显存翻倍；
  - `video_generate.py`：SR / decode 之前 `gc.collect()` + `torch.cuda.synchronize()` + `empty_cache()`；
  - 新增 `magi_compiler/` stub（`__init__` / `api` / `config`），完整版可用 `fetch_third_party.sh optional` 拉取 MagiCompiler。
- `script/digit/pipeline3_magihuman/config_*.json` 是**模板**，`run_magihuman.sh` 运行时会把模型路径改写成实际路径，不要手动填绝对路径。

### H7. LiveAvatar cond KV 缓冲区写死为 720×400

- **根因**：`causal_s2v_pipeline_tpp.py::_initialize_kv_cache` 里 `cond_k / cond_v` 固定 3000 token（注释写着 "hard-coded for 720*400"），1280×720 需要约 8800 token，直接越界。
- **解决**：`cond_cache_size = int(kv_cache_size * 0.7) + 500`，按 KV cache 大小自动推算；另外把 `prompt_extend.py` 里的 `dashscope` 改成延迟导入（不用 prompt 扩写时不需要装）。
- LiveAvatar 的 TPP 流水线默认 **5 卡**（`run_liveavatar_5gpu.sh`），单卡用 `run_liveavatar_quick_test.sh`。

### H8. 其它

- `run_wan22_diffusers_benchmark.sh` 原文件在 `echo "Average ` 处被截断，导致 `bash -n` 语法错误，已补全结尾；
- LightX2V 补丁新增了 H20 的 `hunyuan_video_15` / `wan` / `wan22` / `wan22_vace` / `lingbot` 配置和 LingBot-World（相机位姿控制 I2V）的 model / runner / weights；
- `generate_camera_poses.py` 生成 LingBot 所需的相机轨迹，`prepare_vace_flf2v.py` 生成 VACE 首尾帧输入。

---

## I. Agent / 云端 API

### I1. Venus 视频接口必须用 SDK 签名

- **现象**：LLM / 生图接口用 `Bearer <token>@<app_group>` 能通，视频接口（Veo / Kling）同样的 token 返回鉴权失败。
- **根因**：`/llmproxy/*`（LLM、生图）同时支持 Bearer token 和 SDK 签名；`/video/*` **只支持** `secret_id + secret_key` 的 SDK 签名，且视频和 LLM 用的是不同的应用组。
- **解决**：`api_clients.py` 分别读取 `ENV_VENUS_OPENAPI_SECRET_ID` / `ENV_VENUS_OPENAPI_SECRET_KEY`、`VENUS_APP_GROUP_ID`（LLM / 生图）和 `VENUS_VIDEO_APP_GROUP_ID`（视频）；没有视频密钥时自动回退到本地关键帧视频。

### I2. 生图接口偶发空响应

- `gemini-3-pro-image` 偶尔返回空内容或结构不同的响应：解析兼容多种返回格式、空响应重试，多次失败后回退到 `gpt-image-1`。

### I3. 多模态审核上传超时

- 把视频上传到 GCS 再让 VLM 审核经常超时 → 改为 **inline 图片审核 + 视频首 / 中 / 尾关键帧审核**。

### I4. "假视频"与"一图多镜头"

- **现象**：早期 smoke 测试里首尾帧完全一样、一张图里同时出现同一人物的三个时刻、视频只是首帧平移缩放。
- **根因**：旧的 CPU smoke 走的是"首帧 + zoom/pan"假动画分支；首帧 prompt 直接吃了整段动作描述，模型把多个时刻画进了同一张图。
- **解决**：分镜拆成 `start_frame / end_frame / action` 三段，首帧 prompt 只描述起始时刻；本地回退视频改为首 → 中 → 尾三关键帧过渡；首帧、尾帧、视频任一审核不通过就阻断拼接；生成后强制裁成目标画幅。

---

## J. 开源整理时的额外处理

- **固定版本**：`third_party/manifest.tsv` 里的 commit 是用"本地代码目录时间戳 ↔ GitHub 历史"精确反查出来的实验版本，并用 `diff -rq` 验证过与实验副本一致；我们的修改以 `git apply` 补丁形式提供。
- **SongGeneration 直接 vendored**：上游仓库已下线，无法按 commit 拉取，所以代码（含全部兼容修复）直接放在 `utils/audio/SongGeneration`；`third_party/patches/SongGeneration.patch` 仅作为改动记录。
- **脱敏**：所有内部路径改为 `IT2AV_*` 环境变量，所有密钥 / 代理 / 内部域名改为从环境变量读取（见 `.env.example`），没有默认值。
