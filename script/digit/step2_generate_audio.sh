#!/bin/bash
# ============================================================================
# Step 2: 生成音频素材 (TTS + BGM + 音效, 多卡并行)
#
# 输入: digit_beauty_20260316.tsv 中的 5 个 case 台词/场景
# 输出:
#   ${IT2AV_WORKSPACE}/data/audio/digit/20260316/
#     case_{id}/
#       tts.wav          - CosyVoice3 台词语音 (~10s)
#       bgm.wav          - ACE-Step 背景音乐 (15s)
#       sfx.wav          - TangoFlux 环境音效 (15s)
#
# 多卡分配:
#   GPU 5: CosyVoice3 TTS (台词, 串行 5 个 case)
#   GPU 6: ACE-Step BGM (背景音乐, 串行 5 个 case)
#   GPU 7: TangoFlux SFX (音效, 串行 5 个 case)
#
# 说明:
#   - TTS 只生成台词部分 (~10s), 不控制时长
#   - BGM 和 SFX 严格控制 15s (覆盖完整 15s 视频)
#   - 三个模型并行跑在不同 GPU 上, 互不干扰
#
# 用法:
#   bash step2_generate_audio.sh                    # 默认 GPU 5,6,7
#   GPU_TTS=5 GPU_BGM=6 GPU_SFX=7 bash step2_generate_audio.sh
#
# 依赖: CosyVoice3, ACE-Step, TangoFlux 模型已同步
# ============================================================================

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"
conda activate ${IT2AV_CONDA_ENV}

# ===== 路径配置 =====
GROUP_BASE=${IT2AV_WORKSPACE}
GROUP_CODE=${IT2AV_CODE_ROOT}
LOCAL_BASE=${IT2AV_LOCAL_ROOT}
SYNC_SCRIPT=${GROUP_CODE}/fast_sync.py

DATE_TAG="20260316"
OUTPUT_BASE=${GROUP_BASE}/data/audio/digit/${DATE_TAG}

# GPU 分配 (默认: LiveAvatar 用 0-4, 音频用 5-7)
GPU_TTS=${GPU_TTS:-5}
GPU_BGM=${GPU_BGM:-6}
GPU_SFX=${GPU_SFX:-7}

echo "============================================================"
echo "🎵 Step 2: 生成音频素材 (多卡并行)"
echo "  输出目录: ${OUTPUT_BASE}"
echo "  GPU 分配: TTS=${GPU_TTS}, BGM=${GPU_BGM}, SFX=${GPU_SFX}"
echo "============================================================"

# ===== 创建输出目录 =====
for CASE_ID in 1 2 3 4 5; do
    mkdir -p ${OUTPUT_BASE}/case_${CASE_ID}
done

# ===== Step 2a: 同步模型到本地盘 =====
echo ""
echo ">>> 同步模型..."

sync_model() {
    local MODEL_NAME=$1
    local SRC=${GROUP_BASE}/models/Audio/${MODEL_NAME}
    local DST=${LOCAL_BASE}/models/Audio/${MODEL_NAME}
    if [ -d "$SRC" ] && [ ! -f "${DST}/.sync_done" ]; then
        echo "  同步 ${MODEL_NAME}..."
        mkdir -p ${DST}
        python ${SYNC_SCRIPT} ${SRC} ${DST} --workers 32 --skip-existing
        touch ${DST}/.sync_done
    else
        echo "  ${MODEL_NAME} 已缓存"
    fi
}

# TTS 模型
sync_model "Fun-CosyVoice3-0.5B"
sync_model "CosyVoice2-0.5B"
sync_model "CosyVoice-ttsfrd"

# BGM 模型
sync_model "ACE-Step-v1-3.5B"

# SFX 模型
sync_model "TangoFlux"
sync_model "flan-t5-large"

# 同步代码
for CODE_DIR in CosyVoice Matcha-TTS ACE-Step TangoFlux; do
    SRC=${GROUP_CODE}/utils/audio/${CODE_DIR}
    DST=${LOCAL_BASE}/code/utils/audio/${CODE_DIR}
    if [ -d "$SRC" ]; then
        mkdir -p ${DST}
        python ${SYNC_SCRIPT} ${SRC} ${DST} --workers 32 --skip-existing
    fi
done

# ===== 关键: 确保 Matcha-TTS 在 CosyVoice/third_party/ 下也有一份 =====
# CosyVoice 的 YAML 配置通过 hyperpyyaml 加载时会 import matcha 模块,
# 它期望 Matcha-TTS 位于 CosyVoice/third_party/Matcha-TTS
COSYVOICE_MATCHA=${LOCAL_BASE}/code/utils/audio/CosyVoice/third_party/Matcha-TTS
MATCHA_SRC=${LOCAL_BASE}/code/utils/audio/Matcha-TTS
if [ -d "${MATCHA_SRC}" ]; then
    # 检查目标是否已就绪 (含 setup.py)
    if [ ! -f "${COSYVOICE_MATCHA}/setup.py" ]; then
        echo "  链接/同步 Matcha-TTS -> CosyVoice/third_party/Matcha-TTS"
        mkdir -p $(dirname ${COSYVOICE_MATCHA})
        # 优先用软链接, 失败则用 rsync
        if [ -L "${COSYVOICE_MATCHA}" ] || [ -d "${COSYVOICE_MATCHA}" ]; then
            rm -rf ${COSYVOICE_MATCHA}
        fi
        ln -sf ${MATCHA_SRC} ${COSYVOICE_MATCHA} 2>/dev/null || \
            cp -r ${MATCHA_SRC} ${COSYVOICE_MATCHA}
        echo "  ✅ Matcha-TTS 已就绪: ${COSYVOICE_MATCHA}"
    else
        echo "  Matcha-TTS 已就绪于 CosyVoice/third_party/"
    fi
fi

# ===== 安装所有音频模型缺失依赖 =====
# 三个模型的依赖版本有冲突, 这里选择一组能让三者同时工作的中间版本:
#   - CosyVoice 要 transformers==4.51.3, diffusers==0.29.0, x-transformers==2.11.24
#   - ACE-Step  要 transformers==4.50.0,  diffusers>=0.33.0, py3langid==0.3.0
#   - TangoFlux 要 transformers==4.44.0,  diffusers==0.30.0
# 策略: transformers 4.x 系列差异小, 选 4.51.3 (CosyVoice 需要);
#        diffusers>=0.33.0 (ACE-Step 强需求); 补装所有缺失包
echo ">>> 安装/修复音频模型依赖..."
# openai-whisper: CosyVoice requirements.txt 指定 ==20231117,
# 但经验证 v20231117 和最新版的 mel_filters/log_mel_spectrogram 实现完全相同
# (都从 assets/mel_filters.npz 加载预计算矩阵), 因此不强制降级.
# 注: 如果需要降级, 先安装 setuptools (Python 3.12 移除了 pkg_resources):
#   pip install setuptools && pip install "openai-whisper==20231117"
pip install -q \
    "openai-whisper" \
    "transformers==4.57.1" \
    "diffusers>=0.33.0,<1" \
    "accelerate>=0.34.2" \
    "x-transformers>=2.3.25,<3" \
    "py3langid>=0.3.0,<1" \
    "pyworld" \
    "HyperPyYAML>=1.2.3" \
    "conformer>=0.3.2" \
    "pypinyin>=0.53.0" \
    "hangul-romanize>=0.1.0" \
    "num2words>=0.5.14" \
    "peft>=0.17.0" \
    "spacy>=3.7.5" \
    "pydantic>=2.7.4" \
    2>&1 | tail -5
echo "  ✅ 依赖安装完成"

# 验证关键导入
echo ">>> 验证关键模块..."
python3 -c "
import sys
ok = True

# ===== 关键: monkey-patch pydantic.v1 的 ForwardRef 兼容性 =====
# Python 3.12.4+ 改了 ForwardRef._evaluate() 签名, 添加了必须的 recursive_guard kwarg.
# pydantic v2 的 v1 兼容层 (pydantic/v1/typing.py) 在 <2.7.4 中未修复此问题.
# 这里做 runtime patch 作为保险, 即使 pip 升级 pydantic>=2.7.4 失败也能工作.
import typing
if sys.version_info >= (3, 12, 4):
    try:
        import pydantic.v1.typing as _pv1t
        _orig_eval = _pv1t.evaluate_forwardref
        import functools, inspect
        # 检查是否已修复 (原始版本不传 recursive_guard 关键字参数)
        _src = inspect.getsource(_orig_eval)
        if 'recursive_guard' not in _src:
            def _patched_evaluate_forwardref(ref, globalns, localns):
                return typing.cast(typing.Any, ref)._evaluate(
                    globalns, localns, frozenset(), recursive_guard=frozenset()
                )
            _pv1t.evaluate_forwardref = _patched_evaluate_forwardref
            print('  🔧 已 monkey-patch pydantic.v1 ForwardRef (Python 3.12.4+)')
    except (ImportError, AttributeError):
        pass  # pydantic v1 独立安装, 无 pydantic.v1 模块

checks = [
    ('x_transformers', 'x-transformers (CosyVoice DiT)'),
    ('py3langid', 'py3langid (ACE-Step)'),
    ('diffusers', 'diffusers'),
    ('transformers', 'transformers'),
    ('pyworld', 'pyworld (CosyVoice)'),
    ('pypinyin', 'pypinyin (ACE-Step)'),
]
for mod, desc in checks:
    try:
        __import__(mod)
        print(f'  ✅ {desc}')
    except ImportError as e:
        print(f'  ❌ {desc}: {e}')
        ok = False

import transformers, diffusers
print(f'  📦 transformers=={transformers.__version__}, diffusers=={diffusers.__version__}')

# 测试 spacy + pydantic 兼容性
try:
    from spacy.lang.en import English
    print('  ✅ spacy ↔ pydantic 兼容')
except TypeError as e:
    if 'ForwardRef' in str(e) or 'recursive_guard' in str(e):
        print(f'  ❌ spacy/pydantic ForwardRef 仍不兼容: {e}')
        ok = False
    else:
        raise

# 测试 diffusers + transformers 兼容性
try:
    from diffusers.loaders import SD3LoraLoaderMixin
    print('  ✅ diffusers ↔ transformers 兼容')
except Exception as e:
    print(f'  ❌ diffusers ↔ transformers 不兼容: {e}')
    ok = False

if not ok:
    print('  ⚠️  部分依赖缺失, 可能导致运行失败')
    sys.exit(1)
" 2>&1

echo ">>> 模型同步 & 依赖检查完成"

# ===== 定义 5 个 case 的台词/BGM/SFX prompt =====
# Case 1: 温暖的卧室睡裙
# Case 2: 夜晚樱花树小道 制服短裙
# Case 3: 安静的校园小径 校园风格子裙
# Case 4: 城市街道 高跟鞋短裙
# Case 5: 有灯光的小桥上 旗袍

# ========= 生成 audio prompts JSON =========
AUDIO_PROMPTS=${OUTPUT_BASE}/audio_prompts.json

cat > ${AUDIO_PROMPTS} << 'AUDIO_EOF'
{
  "cases": [
    {
      "id": "case_1",
      "tts_text": "晚上好，生活再忙再累，也别忘了好好爱自己。",
      "bgm_prompt": "warm cozy ambient, soft piano and strings, gentle, relaxing, bedroom atmosphere, intimate, lo-fi, 70 bpm, instrumental",
      "bgm_lyrics": "[inst-long]",
      "bgm_duration": 15,
      "sfx_prompt": "Quiet bedroom ambience at night, soft wind outside window, gentle fabric rustling sounds, warm cozy indoor atmosphere, peaceful nighttime",
      "sfx_duration": 15
    },
    {
      "id": "case_2",
      "tts_text": "晚上好，黑夜无论怎样悠长，白昼总会到来，一切都会慢慢变好。",
      "bgm_prompt": "dreamy night walk, cherry blossom, japanese ambient, soft piano and acoustic guitar, melancholic yet hopeful, 75 bpm, instrumental, peaceful",
      "bgm_lyrics": "[inst-long]",
      "bgm_duration": 15,
      "sfx_prompt": "Night cherry blossom garden ambience, gentle wind through trees, flower petals falling softly, distant crickets, quiet nighttime nature sounds",
      "sfx_duration": 15
    },
    {
      "id": "case_3",
      "tts_text": "晚上好，世上只有一种英雄主义，就是在认清生活真相后依然热爱生活。",
      "bgm_prompt": "campus youth, acoustic guitar, warm, inspirational, folk style, gentle drums, 80 bpm, instrumental, youthful energy",
      "bgm_lyrics": "[inst-long]",
      "bgm_duration": 15,
      "sfx_prompt": "Quiet campus pathway ambience, gentle breeze through trees, leaves rustling, distant birds singing, footsteps on stone path, peaceful school atmosphere",
      "sfx_duration": 15
    },
    {
      "id": "case_4",
      "tts_text": "晚上好，不必行色匆匆，不必光芒四射，只需做自己，早点休息吧。",
      "bgm_prompt": "urban night chill, lo-fi beats, city pop, smooth bass, soft synth pad, 85 bpm, instrumental, sophisticated night walk",
      "bgm_lyrics": "[inst-long]",
      "bgm_duration": 15,
      "sfx_prompt": "City street night ambience, traffic light clicking, distant car sounds, heels clicking on crosswalk, gentle spring evening wind, urban nightlife atmosphere",
      "sfx_duration": 15
    },
    {
      "id": "case_5",
      "tts_text": "晚上好，万物皆有裂痕，那是光照进来的地方，愿你今夜好梦。",
      "bgm_prompt": "traditional Chinese, guzheng and bamboo flute, elegant, flowing water, peaceful bridge, 65 bpm, instrumental, ancient beauty",
      "bgm_lyrics": "[inst-long]",
      "bgm_duration": 15,
      "sfx_prompt": "Small bridge over stream at night, flowing water sounds, lantern lights crackling softly, gentle wind, peaceful Chinese garden atmosphere, distant temple bells",
      "sfx_duration": 15
    }
  ]
}
AUDIO_EOF

echo "✅ Audio prompts JSON: ${AUDIO_PROMPTS}"

# ===== 生成 Python 脚本: 统一入口 =====
AUDIO_GEN_SCRIPT=${OUTPUT_BASE}/run_audio_gen.py

cat > ${AUDIO_GEN_SCRIPT} << 'PYSCRIPT_EOF'
#!/usr/bin/env python3
"""
音频生成脚本 - 支持 TTS / BGM / SFX 三种模式
用法:
    python run_audio_gen.py --mode tts --prompts audio_prompts.json --output_dir /path/to/output
    python run_audio_gen.py --mode bgm --prompts audio_prompts.json --output_dir /path/to/output
    python run_audio_gen.py --mode sfx --prompts audio_prompts.json --output_dir /path/to/output
"""
import argparse
import gc
import json
import os
import sys
import time
import typing

# ===== Monkey-patch: pydantic.v1 ForwardRef 兼容 Python 3.12.4+ =====
# Python 3.12.4 改了 ForwardRef._evaluate() 签名, 添加了必须的 recursive_guard kwarg.
# pydantic v2 的 v1 兼容层 (pydantic/v1/typing.py) 在 <2.7.4 中未修复.
# 必须在 import spacy (ACE-Step BGM) 之前执行!
if sys.version_info >= (3, 12, 4):
    try:
        import pydantic.v1.typing as _pv1t
        import inspect as _inspect
        _src = _inspect.getsource(_pv1t.evaluate_forwardref)
        if 'recursive_guard' not in _src:
            def _patched_evaluate_forwardref(ref, globalns, localns):
                return typing.cast(typing.Any, ref)._evaluate(
                    globalns, localns, frozenset(), recursive_guard=frozenset()
                )
            _pv1t.evaluate_forwardref = _patched_evaluate_forwardref
            print("🔧 已 monkey-patch pydantic.v1 ForwardRef (Python 3.12.4+)")
    except (ImportError, AttributeError, OSError):
        pass  # pydantic v1 独立安装或无 v1 兼容层

import torch

GROUP_BASE = os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace"))
GROUP_CODE = os.environ["IT2AV_CODE_ROOT"]
LOCAL_BASE = os.environ.get("IT2AV_LOCAL_ROOT", "/tmp/it2av")
LOCAL_MODELS = f"{LOCAL_BASE}/models/Audio"
LOCAL_CODE = f"{LOCAL_BASE}/code/utils/audio"


def ensure_cosyvoice_ttsfrd_resource(model_path, cosyvoice_code):
    ttsfrd_dir = f"{LOCAL_MODELS}/CosyVoice-ttsfrd"
    resource_src = os.path.join(ttsfrd_dir, "resource")
    resource_dst = os.path.join(cosyvoice_code, "pretrained_models", "CosyVoice-ttsfrd", "resource")

    if not os.path.isdir(resource_src):
        print(f"  ⚠️  未找到 ttsfrd resource: {resource_src}")
        return False

    os.makedirs(os.path.dirname(resource_dst), exist_ok=True)
    if os.path.islink(resource_dst) and os.path.realpath(resource_dst) == os.path.realpath(resource_src):
        print(f"  ✅ ttsfrd 资源已就绪: {resource_dst}")
        return True

    if os.path.lexists(resource_dst):
        if os.path.islink(resource_dst):
            os.unlink(resource_dst)
        elif os.path.isdir(resource_dst) and not os.listdir(resource_dst):
            os.rmdir(resource_dst)
        else:
            print(f"  ⚠️  保留已有非空 frontend 目录: {resource_dst}")
            return True

    os.symlink(resource_src, resource_dst)
    print(f"  🔗 挂载 ttsfrd 资源: {resource_dst} -> {resource_src}")
    return True


def run_tts(cases, output_dir):
    """CosyVoice3 TTS 台词语音生成"""
    import torchaudio

    cosyvoice_code = f"{LOCAL_CODE}/CosyVoice"
    # Matcha-TTS 路径: 优先 CosyVoice/third_party/Matcha-TTS, 其次独立同步的
    matcha_path = os.path.join(cosyvoice_code, "third_party", "Matcha-TTS")
    matcha_standalone = f"{LOCAL_CODE}/Matcha-TTS"
    if not os.path.isfile(os.path.join(matcha_path, "setup.py")):
        # third_party 下没有 Matcha-TTS (git submodule 未初始化), 尝试创建软链接
        if os.path.isdir(matcha_standalone):
            os.makedirs(os.path.dirname(matcha_path), exist_ok=True)
            if os.path.isdir(matcha_path) and not os.listdir(matcha_path):
                os.rmdir(matcha_path)
            if not os.path.exists(matcha_path):
                os.symlink(matcha_standalone, matcha_path)
                print(f"  🔗 创建软链接: {matcha_path} -> {matcha_standalone}")
        # 如果软链接也失败, 直接用独立目录
        if not os.path.isfile(os.path.join(matcha_path, "setup.py")):
            matcha_path = matcha_standalone
            print(f"  ⚠️  使用独立 Matcha-TTS: {matcha_path}")

    for p in [cosyvoice_code, matcha_path]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # 验证 matcha 可导入
    try:
        import matcha  # noqa: F401
        print("  ✅ matcha 模块可用")
    except ImportError:
        print(f"  ⚠️  matcha 不可导入, 当前 sys.path 中的 Matcha-TTS: {matcha_path}")
        print(f"  检查 {matcha_path} 内容: {os.listdir(matcha_path) if os.path.isdir(matcha_path) else 'N/A'}")

    # 改用 CosyVoice2: CosyVoice3 存在严重的已知质量问题
    # (GitHub Issue #1703: "效果简直就是噩梦" —— 乱读/重复/非人类语言)
    # CosyVoice2 更稳定可靠
    model_path = f"{LOCAL_MODELS}/CosyVoice2-0.5B"
    if not os.path.isdir(model_path):
        # 回退到 CosyVoice3 (如果 CosyVoice2 未下载)
        model_path = f"{LOCAL_MODELS}/Fun-CosyVoice3-0.5B"
        is_v3 = True
        print(f"⚠️  CosyVoice2 不可用, 回退到 CosyVoice3: {model_path}")
    else:
        is_v3 = False
        print(f"📦 加载 CosyVoice2: {model_path}")

    # wetext 包初始化会从 modelscope 下载模型 (pengzhendong/wetext),
    # 服务器无法连通 modelscope 会卡住 2-4 分钟.
    # 设置极短超时让它快速 fallback 到无 frontend 模式.
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_TIMEOUT", "5")
    os.environ.setdefault("MODELSCOPE_HUB_TIMEOUT", "5")

    ensure_cosyvoice_ttsfrd_resource(model_path, cosyvoice_code)

    from cosyvoice.cli.cosyvoice import AutoModel
    # fp16=False 是关键: CosyVoice 2/3 在 fp16 下可能导致音频出现
    # 前后较长空白、重复、乱读等问题 (参见 GitHub issues)
    cosyvoice = AutoModel(model_dir=model_path, fp16=False)
    print(f"  ℹ️  frontend: {getattr(cosyvoice.frontend, 'text_frontend', 'unknown')}")
    try:
        import onnxruntime
        print(f"  ℹ️  onnxruntime={onnxruntime.__version__}, providers={onnxruntime.get_available_providers()}")
    except Exception as e:
        print(f"  ⚠️  无法读取 onnxruntime 信息: {e}")
    if not is_v3:
        print("  ℹ️  CosyVoice2 将使用 text_frontend=False (官方推荐 zero-shot 方式)")

    ref_audio = os.path.join(cosyvoice_code, "asset", "zero_shot_prompt.wav")
    # prompt_text 格式:
    #   CosyVoice3: 需要 "You are a helpful assistant.<|endofprompt|>" 前缀
    #   CosyVoice2: 不需要前缀, 直接用参考音频对应文本
    if is_v3:
        prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    else:
        prompt_text = "希望你以后能够做的比我还好呦。"

    for case in cases:
        case_id = case["id"]
        text = case["tts_text"]
        save_path = os.path.join(output_dir, case_id, "tts.wav")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
            print(f"  ⏭️  {case_id} TTS 已存在")
            continue

        print(f"\n🎤 [{case_id}] TTS: {text}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        t0 = time.time()

        try:
            all_speech = []
            use_text_frontend = is_v3
            for i, chunk in enumerate(cosyvoice.inference_zero_shot(
                text, prompt_text, ref_audio, stream=False,
                text_frontend=use_text_frontend
            )):
                all_speech.append(chunk["tts_speech"])

            if all_speech:
                speech = torch.cat(all_speech, dim=-1)
                torchaudio.save(save_path, speech, cosyvoice.sample_rate)
                elapsed = time.time() - t0
                duration = speech.shape[-1] / cosyvoice.sample_rate
                print(f"  ✅ 保存: {save_path} ({duration:.1f}s, 耗时 {elapsed:.1f}s)")
        except Exception as e:
            print(f"  ❌ TTS 失败: {e}")
            # 回退到 cross_lingual
            try:
                all_speech = []
                cross_prefix = "You are a helpful assistant.<|endofprompt|>" if is_v3 else ""
                for i, chunk in enumerate(cosyvoice.inference_cross_lingual(
                    f"{cross_prefix}{text}",
                    ref_audio, stream=False,
                    text_frontend=use_text_frontend
                )):
                    all_speech.append(chunk["tts_speech"])
                if all_speech:
                    speech = torch.cat(all_speech, dim=-1)
                    torchaudio.save(save_path, speech, cosyvoice.sample_rate)
                    print(f"  ✅ 保存 (fallback): {save_path}")
            except Exception as e2:
                print(f"  ❌ TTS fallback 也失败: {e2}")

        gc.collect()
        torch.cuda.empty_cache()

    del cosyvoice
    gc.collect()
    torch.cuda.empty_cache()


def run_bgm(cases, output_dir):
    """ACE-Step BGM 背景音乐生成"""
    ace_code = f"{LOCAL_CODE}/ACE-Step"
    if ace_code not in sys.path:
        sys.path.insert(0, ace_code)

    # 检查 diffusers/transformers 兼容性
    try:
        from diffusers.loaders import SD3LoraLoaderMixin  # noqa: F401
        print("  ✅ diffusers/transformers 兼容")
    except Exception as e:
        print(f"  ❌ diffusers/transformers 不兼容: {e}")
        print(f"  请先运行: pip install 'diffusers>=0.33.0' 'transformers>=4.44.0,<5'")
        sys.exit(1)

    model_path = f"{LOCAL_MODELS}/ACE-Step-v1-3.5B"
    print(f"📦 加载 ACE-Step: {model_path}")

    from acestep.pipeline_ace_step import ACEStepPipeline
    pipeline = ACEStepPipeline(
        checkpoint_dir=model_path,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=False,
        overlapped_decode=False,
    )

    for case in cases:
        case_id = case["id"]
        prompt = case["bgm_prompt"]
        lyrics = case.get("bgm_lyrics", "[inst-long]")
        duration = case.get("bgm_duration", 15)
        save_path = os.path.join(output_dir, case_id, "bgm.wav")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
            print(f"  ⏭️  {case_id} BGM 已存在")
            continue

        print(f"\n🎵 [{case_id}] BGM ({duration}s): {prompt[:60]}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        t0 = time.time()

        pipeline(
            audio_duration=duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=100,
            guidance_scale=15.0,
            scheduler_type="euler",
            cfg_type="apg",
            omega_scale=10.0,
            manual_seeds="42",
            guidance_interval=0.5,
            guidance_interval_decay=0.0,
            min_guidance_scale=3.0,
            use_erg_tag=True,
            use_erg_lyric=False,
            use_erg_diffusion=True,
            oss_steps="",
            guidance_scale_text=0.0,
            guidance_scale_lyric=0.0,
            save_path=save_path,
        )

        elapsed = time.time() - t0
        print(f"  ✅ 保存: {save_path} ({duration}s, 耗时 {elapsed:.1f}s)")

        gc.collect()
        torch.cuda.empty_cache()

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def run_sfx(cases, output_dir):
    """TangoFlux SFX 环境音效生成"""
    import torchaudio

    tangoflux_code = f"{LOCAL_CODE}/TangoFlux"
    if tangoflux_code not in sys.path:
        sys.path.insert(0, tangoflux_code)

    model_path = f"{LOCAL_MODELS}/TangoFlux"
    print(f"📦 加载 TangoFlux: {model_path}")

    # NOTE: TangoFluxInference 内部用 snapshot_download(repo_id=name),
    #       不接受本地路径, 会报 HFValidationError.
    #       改为手动加载 (与 run_audio_benchmark.py 一致).
    from diffusers import AutoencoderOobleck
    from safetensors.torch import load_file
    from tangoflux.model import TangoFlux as TangoFluxModel

    device = "cuda:0"
    vae = AutoencoderOobleck()
    vae_weights = load_file(os.path.join(model_path, "vae.safetensors"))
    vae.load_state_dict(vae_weights)

    with open(os.path.join(model_path, "config.json"), "r") as f:
        import json as _json
        config = _json.load(f)

    # TangoFlux text encoder (flan-t5-large) 路径:
    # 优先用独立下载的 flan-t5-large, 否则用 TangoFlux 目录自带的
    text_encoder_path = f"{LOCAL_MODELS}/flan-t5-large"
    if not os.path.isdir(text_encoder_path):
        text_encoder_path = None  # 让 TangoFluxModel 使用默认 (从 config 中加载)
        print(f"  ⚠️  flan-t5-large 不在 {LOCAL_MODELS}, 使用 TangoFlux 内置 text encoder")

    if text_encoder_path:
        model = TangoFluxModel(config, text_encoder_dir=text_encoder_path)
    else:
        model = TangoFluxModel(config)
    weights = load_file(os.path.join(model_path, "tangoflux.safetensors"))
    model.load_state_dict(weights, strict=False)

    vae = vae.to(device)
    model = model.to(device)
    print(f"  ✅ TangoFlux 手动加载完成")

    for case in cases:
        case_id = case["id"]
        prompt = case["sfx_prompt"]
        duration = min(case.get("sfx_duration", 15), 30)  # TangoFlux max 30s
        save_path = os.path.join(output_dir, case_id, "sfx.wav")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
            print(f"  ⏭️  {case_id} SFX 已存在")
            continue

        print(f"\n🔊 [{case_id}] SFX ({duration}s): {prompt[:60]}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        t0 = time.time()

        with torch.no_grad():
            latents = model.inference_flow(
                prompt, duration=duration, num_inference_steps=50, guidance_scale=4.5,
            )
            wave = vae.decode(latents.transpose(2, 1)).sample.cpu()[0]
        waveform_end = int(duration * vae.config.sampling_rate)
        audio = wave[:, :waveform_end]
        torchaudio.save(save_path, audio, sample_rate=44100)

        elapsed = time.time() - t0
        print(f"  ✅ 保存: {save_path} ({duration}s, 耗时 {elapsed:.1f}s)")

        gc.collect()
        torch.cuda.empty_cache()

    del model, vae
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["tts", "bgm", "sfx"])
    parser.add_argument("--prompts", required=True, help="audio_prompts.json 路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    args = parser.parse_args()

    with open(args.prompts, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"]

    print(f"📋 加载 {len(cases)} 个 case, 模式: {args.mode}")

    if args.mode == "tts":
        run_tts(cases, args.output_dir)
    elif args.mode == "bgm":
        run_bgm(cases, args.output_dir)
    elif args.mode == "sfx":
        run_sfx(cases, args.output_dir)


if __name__ == "__main__":
    main()
PYSCRIPT_EOF

echo "✅ 音频生成脚本: ${AUDIO_GEN_SCRIPT}"

# ===== 多卡并行执行 =====
STEP_START=$(date +%s)

echo ""
echo ">>> 启动音频生成..."

# ===== TTS: 优先使用混元 API, 无需 GPU =====
# 混元 TTS API 质量更稳定, 且不占用 GPU 资源
HUNYUAN_TTS_SCRIPT=${GROUP_CODE}/utils/audio/Hunyuan-tts/hunyuan_tts.py
export HUNYUAN_API_KEY="${HUNYUAN_API_KEY:-}"

# 默认: 配置了 HUNYUAN_API_KEY + HUNYUAN_API_BASE 才走混元 API, 否则回退到本地 CosyVoice
USE_HUNYUAN_TTS=${USE_HUNYUAN_TTS:-$([[ -n "${HUNYUAN_API_KEY}" && -n "${HUNYUAN_API_BASE:-}" ]] && echo true || echo false)}

if [ "${USE_HUNYUAN_TTS}" = true ] && [ -f "${HUNYUAN_TTS_SCRIPT}" ]; then
    echo "  [API] 混元 TTS 启动 (无需 GPU)..."
    python3 ${HUNYUAN_TTS_SCRIPT} \
        --prompts_json ${AUDIO_PROMPTS} \
        --output_dir ${OUTPUT_BASE} \
        --voice wenrouxuemei \
        --sample_rate 24000 \
        --stream \
        > ${OUTPUT_BASE}/log_tts.txt 2>&1 &
    PID_TTS=$!
    echo "  [API] 混元 TTS (PID: ${PID_TTS})"
else
    # 回退: CosyVoice 本地推理 (需要 GPU)
    echo "  [GPU ${GPU_TTS}] CosyVoice TTS 启动 (本地推理)..."
    COSYVOICE_CODE=${LOCAL_BASE}/code/utils/audio/CosyVoice
    MATCHA_PATH=${COSYVOICE_CODE}/third_party/Matcha-TTS
    if [ ! -f "${MATCHA_PATH}/setup.py" ]; then
        MATCHA_PATH=${LOCAL_BASE}/code/utils/audio/Matcha-TTS
        echo "  使用独立 Matcha-TTS: ${MATCHA_PATH}"
    fi
    CUDA_VISIBLE_DEVICES=${GPU_TTS} \
    PYTHONPATH="${COSYVOICE_CODE}:${MATCHA_PATH}:${GROUP_CODE}:$PYTHONPATH" \
    python ${AUDIO_GEN_SCRIPT} \
        --mode tts \
        --prompts ${AUDIO_PROMPTS} \
        --output_dir ${OUTPUT_BASE} \
        > ${OUTPUT_BASE}/log_tts.txt 2>&1 &
    PID_TTS=$!
    echo "  [GPU ${GPU_TTS}] CosyVoice TTS (PID: ${PID_TTS})"
fi

# GPU 6: ACE-Step BGM
ACE_STEP_CODE=${LOCAL_BASE}/code/utils/audio/ACE-Step
echo "  [GPU ${GPU_BGM}] ACE-Step BGM 启动..."
CUDA_VISIBLE_DEVICES=${GPU_BGM} \
PYTHONPATH="${ACE_STEP_CODE}:${GROUP_CODE}:$PYTHONPATH" \
python ${AUDIO_GEN_SCRIPT} \
    --mode bgm \
    --prompts ${AUDIO_PROMPTS} \
    --output_dir ${OUTPUT_BASE} \
    > ${OUTPUT_BASE}/log_bgm.txt 2>&1 &
PID_BGM=$!
echo "  [GPU ${GPU_BGM}] ACE-Step BGM (PID: ${PID_BGM})"

# GPU 7: TangoFlux SFX
TANGOFLUX_CODE=${LOCAL_BASE}/code/utils/audio/TangoFlux
echo "  [GPU ${GPU_SFX}] TangoFlux SFX 启动..."
CUDA_VISIBLE_DEVICES=${GPU_SFX} \
PYTHONPATH="${TANGOFLUX_CODE}:${GROUP_CODE}:$PYTHONPATH" \
python ${AUDIO_GEN_SCRIPT} \
    --mode sfx \
    --prompts ${AUDIO_PROMPTS} \
    --output_dir ${OUTPUT_BASE} \
    > ${OUTPUT_BASE}/log_sfx.txt 2>&1 &
PID_SFX=$!
echo "  [GPU ${GPU_SFX}] TangoFlux SFX (PID: ${PID_SFX})"

echo ""
echo ">>> 等待所有音频生成完成..."

FAILED=0
wait ${PID_TTS} || { echo "  ❌ TTS 失败! 查看 ${OUTPUT_BASE}/log_tts.txt"; FAILED=1; }
echo "  ✅ TTS 完成"

wait ${PID_BGM} || { echo "  ❌ ACE-Step BGM 失败! 查看 ${OUTPUT_BASE}/log_bgm.txt"; FAILED=1; }
echo "  ✅ ACE-Step BGM 完成"

wait ${PID_SFX} || { echo "  ❌ TangoFlux SFX 失败! 查看 ${OUTPUT_BASE}/log_sfx.txt"; FAILED=1; }
echo "  ✅ TangoFlux SFX 完成"

STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))

# ===== 检查结果 =====
echo ""
echo "============================================================"
echo "🎵 Step 2 完成!"
echo "  耗时: ${STEP_TIME}s"
echo "  输出目录: ${OUTPUT_BASE}"
echo ""
echo "  📁 生成的音频:"
for CASE_ID in 1 2 3 4 5; do
    echo "  case_${CASE_ID}:"
    for TYPE in tts bgm sfx; do
        FILE=${OUTPUT_BASE}/case_${CASE_ID}/${TYPE}.wav
        if [ -f "$FILE" ]; then
            SIZE=$(du -h "$FILE" | cut -f1)
            echo "    ✅ ${TYPE}.wav ($SIZE)"
        else
            echo "    ❌ ${TYPE}.wav (缺失)"
        fi
    done
done

if [ ${FAILED} -eq 1 ]; then
    echo ""
    echo "  ⚠️  部分音频生成失败, 请检查日志:"
    echo "     ${OUTPUT_BASE}/log_tts.txt"
    echo "     ${OUTPUT_BASE}/log_bgm.txt"
    echo "     ${OUTPUT_BASE}/log_sfx.txt"
fi

echo ""
echo "  下一步: bash step3_liveavatar_talking.sh"
echo "============================================================"
