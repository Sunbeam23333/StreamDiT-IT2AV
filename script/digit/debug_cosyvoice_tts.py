#!/usr/bin/env python3
"""
CosyVoice2 TTS 调试脚本 - 系统性排查"非人类语言"问题

诊断内容:
  1. 模型文件完整性检查
  2. whisper 版本与 mel spectrogram 兼容性
  3. speech_tokenizer 输出验证
  4. tokenizer 文本编码验证
  5. prompt_wav 采样率验证
  6. 端到端推理 + 中间结果保存

用法:
  CUDA_VISIBLE_DEVICES=5 python debug_cosyvoice_tts.py
"""
import os
import sys
import hashlib
import json
import numpy as np

# ===== 配置 =====
LOCAL_BASE = os.environ.get("IT2AV_LOCAL_ROOT", "/tmp/it2av")
LOCAL_MODELS = f"{LOCAL_BASE}/models/Audio"
LOCAL_CODE = f"{LOCAL_BASE}/code/utils/audio"
GROUP_BASE = os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace"))
MODEL_DIR = f"{LOCAL_MODELS}/CosyVoice2-0.5B"
COSYVOICE_CODE = f"{LOCAL_CODE}/CosyVoice"
OUTPUT_DIR = f"{GROUP_BASE}/data/audio/digit/debug_tts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Step 0: 环境检查 =====
print("=" * 70)
print("🔍 CosyVoice2 TTS 调试诊断")
print("=" * 70)

print("\n📦 Step 0: 环境检查")
print(f"  Python: {sys.version}")

import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")

import torchaudio
print(f"  torchaudio: {torchaudio.__version__}")

import whisper
print(f"  openai-whisper: {whisper.__version__}")

import onnxruntime
print(f"  onnxruntime: {onnxruntime.__version__}")
print(f"  onnxruntime providers: {onnxruntime.get_available_providers()}")

# ===== Step 1: 模型文件完整性检查 =====
print(f"\n📁 Step 1: 模型文件完整性 ({MODEL_DIR})")
expected_files = [
    "cosyvoice2.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v2.onnx",
    "spk2info.pt",
]

# CosyVoice-BlankEN (Qwen2 tokenizer)
expected_subdirs = ["CosyVoice-BlankEN"]

all_ok = True
for f in expected_files:
    path = os.path.join(MODEL_DIR, f)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  ✅ {f} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ {f} - 缺失!")
        all_ok = False

for d in expected_subdirs:
    path = os.path.join(MODEL_DIR, d)
    if os.path.isdir(path):
        files = os.listdir(path)
        print(f"  ✅ {d}/ ({len(files)} files: {', '.join(sorted(files)[:10])})")
    else:
        print(f"  ❌ {d}/ - 缺失! (Qwen2 tokenizer 目录)")
        all_ok = False

# 列出模型目录所有文件
print(f"\n  模型目录完整文件列表:")
for root, dirs, files in os.walk(MODEL_DIR):
    level = root.replace(MODEL_DIR, '').count(os.sep)
    indent = "    " + "  " * level
    folder_name = os.path.basename(root) or MODEL_DIR
    if level == 0:
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"{indent}{f} ({size_mb:.1f} MB)")
    else:
        print(f"{indent}{folder_name}/")
        for f in sorted(files)[:5]:
            fpath = os.path.join(root, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"{indent}  {f} ({size_kb:.1f} KB)")
        if len(files) > 5:
            print(f"{indent}  ... and {len(files) - 5} more files")

if not all_ok:
    print("\n  ⚠️  模型文件不完整! 请重新下载。")

# ===== Step 2: CosyVoice 代码初始化 =====
print(f"\n🔧 Step 2: CosyVoice 代码初始化")

# 添加路径
matcha_path = os.path.join(COSYVOICE_CODE, "third_party", "Matcha-TTS")
if not os.path.exists(matcha_path):
    matcha_path = f"{LOCAL_CODE}/Matcha-TTS"
for p in [COSYVOICE_CODE, matcha_path]:
    if p not in sys.path:
        sys.path.insert(0, p)
print(f"  CosyVoice code: {COSYVOICE_CODE}")
print(f"  Matcha-TTS: {matcha_path}")

from cosyvoice.cli.cosyvoice import AutoModel
print(f"  ✅ cosyvoice 模块导入成功")

# ===== Step 3: 加载模型 =====
print(f"\n📦 Step 3: 加载 CosyVoice2 模型")
cosyvoice = AutoModel(model_dir=MODEL_DIR, fp16=False)
print(f"  ✅ 模型加载完成")
print(f"  sample_rate: {cosyvoice.sample_rate}")
print(f"  model class: {cosyvoice.__class__.__name__}")
print(f"  model.model class: {cosyvoice.model.__class__.__name__}")
print(f"  text_frontend: {cosyvoice.frontend.text_frontend}")
print(f"  tokenizer type: {type(cosyvoice.frontend.tokenizer)}")
print(f"  device: {cosyvoice.model.device}")
print(f"  fp16: {cosyvoice.model.fp16}")

# ===== Step 4: 检查参考音频 =====
print(f"\n🎤 Step 4: 检查参考音频 (zero_shot_prompt.wav)")
ref_audio = os.path.join(COSYVOICE_CODE, "asset", "zero_shot_prompt.wav")
print(f"  路径: {ref_audio}")
print(f"  存在: {os.path.exists(ref_audio)}")

if os.path.exists(ref_audio):
    ref_speech, ref_sr = torchaudio.load(ref_audio)
    print(f"  采样率: {ref_sr}")
    print(f"  通道数: {ref_speech.shape[0]}")
    print(f"  样本数: {ref_speech.shape[1]}")
    print(f"  时长: {ref_speech.shape[1] / ref_sr:.2f}s")
    print(f"  数据范围: [{ref_speech.min():.4f}, {ref_speech.max():.4f}]")
    print(f"  dtype: {ref_speech.dtype}")
    
    # 检查是否是有效音频（不是全零或全噪声）
    rms = torch.sqrt(torch.mean(ref_speech ** 2)).item()
    print(f"  RMS: {rms:.6f}")
    if rms < 0.001:
        print(f"  ⚠️  音频 RMS 极低，可能是静音!")
    elif rms > 0.5:
        print(f"  ⚠️  音频 RMS 异常高!")
    else:
        print(f"  ✅ 音频 RMS 正常")
else:
    print(f"  ❌ 参考音频不存在!")

# ===== Step 5: 检查 speech tokenizer =====
print(f"\n🔤 Step 5: 检查 speech tokenizer (关键!)")
from cosyvoice.utils.file_utils import load_wav

# 5a: 加载音频并提取 mel
speech_16k = load_wav(ref_audio, 16000)
print(f"  speech_16k shape: {speech_16k.shape}, dtype: {speech_16k.dtype}")
print(f"  speech_16k duration: {speech_16k.shape[1] / 16000:.2f}s")

feat = whisper.log_mel_spectrogram(speech_16k, n_mels=128)
print(f"  whisper mel shape: {feat.shape}, dtype: {feat.dtype}")
print(f"  whisper mel range: [{feat.min():.4f}, {feat.max():.4f}]")
print(f"  whisper mel mean: {feat.mean():.4f}")

# 5b: 运行 speech tokenizer
print(f"\n  运行 speech_tokenizer_v2.onnx...")
tokenizer_session = cosyvoice.frontend.speech_tokenizer_session
input_names = [inp.name for inp in tokenizer_session.get_inputs()]
output_names = [out.name for out in tokenizer_session.get_outputs()]
print(f"  input names: {input_names}")
print(f"  output names: {output_names}")

# 检查 onnxruntime 实际使用的 provider
print(f"  session providers: {tokenizer_session.get_providers()}")

speech_tokens = tokenizer_session.run(None, {
    input_names[0]: feat.detach().cpu().numpy(),
    input_names[1]: np.array([feat.shape[2]], dtype=np.int32)
})[0].flatten()

print(f"  speech token 数量: {len(speech_tokens)}")
print(f"  speech token 范围: [{speech_tokens.min()}, {speech_tokens.max()}]")
print(f"  speech token 前20个: {speech_tokens[:20].tolist()}")
print(f"  speech token 唯一值数: {len(np.unique(speech_tokens))}")

# 关键检查: 如果 speech token 全是 0 或者范围很小，说明 tokenizer 有问题
unique_ratio = len(np.unique(speech_tokens)) / max(len(speech_tokens), 1)
print(f"  唯一值比率: {unique_ratio:.4f}")
if unique_ratio < 0.05:
    print(f"  ⚠️  speech token 多样性极低! 很可能 tokenizer 输出有问题!")
elif unique_ratio > 0.8:
    print(f"  ✅ speech token 多样性正常")
else:
    print(f"  ℹ️  speech token 多样性中等")

# ===== Step 6: 检查 text tokenizer =====
print(f"\n📝 Step 6: 检查 text tokenizer")
prompt_text = "希望你以后能够做的比我还好呦。"
tts_text = "晚上好，生活再忙再累，也别忘了好好爱自己。"

# 使用 text_frontend=False（CosyVoice2 官方推荐）
normalized_prompt = cosyvoice.frontend.text_normalize(prompt_text, split=False, text_frontend=False)
normalized_tts_list = cosyvoice.frontend.text_normalize(tts_text, split=True, text_frontend=False)

print(f"  prompt_text 原始: {prompt_text}")
print(f"  prompt_text normalize: {normalized_prompt}")
print(f"  tts_text 原始: {tts_text}")
print(f"  tts_text normalize: {normalized_tts_list}")

# 编码为 token
prompt_token = cosyvoice.frontend.tokenizer.encode(normalized_prompt, allowed_special='all')
print(f"  prompt_text token 数量: {len(prompt_token)}")
print(f"  prompt_text tokens: {prompt_token}")

for i, txt in enumerate(normalized_tts_list):
    tts_token = cosyvoice.frontend.tokenizer.encode(txt, allowed_special='all')
    print(f"  tts_text[{i}] token 数量: {len(tts_token)}")
    print(f"  tts_text[{i}] tokens: {tts_token}")

# ===== Step 7: 完整 frontend_zero_shot 调用 =====
print(f"\n🔗 Step 7: 检查 frontend_zero_shot 完整输出")
model_input = cosyvoice.frontend.frontend_zero_shot(
    tts_text=normalized_tts_list[0] if normalized_tts_list else tts_text,
    prompt_text=normalized_prompt,
    prompt_wav=ref_audio,
    resample_rate=cosyvoice.sample_rate,
    zero_shot_spk_id=''
)

for key, value in model_input.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
        if value.numel() < 100:
            print(f"    值: {value.flatten()[:30].tolist()}")
        else:
            print(f"    范围: [{value.min():.4f}, {value.max():.4f}], mean={value.mean():.4f}")
    else:
        print(f"  {key}: {value}")

# ===== Step 8: 生成测试音频 =====
print(f"\n🎵 Step 8: 生成测试音频")
test_texts = [
    ("test_short.wav", "你好世界。"),
    ("test_medium.wav", "晚上好，生活再忙再累，也别忘了好好爱自己。"),
]

for filename, text in test_texts:
    save_path = os.path.join(OUTPUT_DIR, filename)
    print(f"\n  生成: {text}")
    
    import time
    t0 = time.time()
    all_speech = []
    try:
        for i, chunk in enumerate(cosyvoice.inference_zero_shot(
            text, prompt_text, ref_audio,
            stream=False, text_frontend=False
        )):
            all_speech.append(chunk["tts_speech"])
            print(f"    chunk {i}: shape={chunk['tts_speech'].shape}, "
                  f"range=[{chunk['tts_speech'].min():.4f}, {chunk['tts_speech'].max():.4f}]")
    except Exception as e:
        print(f"    ❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    if all_speech:
        speech = torch.cat(all_speech, dim=-1)
        torchaudio.save(save_path, speech, cosyvoice.sample_rate)
        elapsed = time.time() - t0
        duration = speech.shape[-1] / cosyvoice.sample_rate
        
        rms = torch.sqrt(torch.mean(speech ** 2)).item()
        print(f"    ✅ 保存: {save_path}")
        print(f"    时长: {duration:.2f}s, 耗时: {elapsed:.1f}s")
        print(f"    RMS: {rms:.6f}")
        print(f"    range: [{speech.min():.4f}, {speech.max():.4f}]")
        
        # 简单的频谱分析
        signal = speech.numpy().flatten()
        fft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1.0 / cosyvoice.sample_rate)
        
        # 检查能量分布
        low_energy = np.sum(fft[freqs < 300] ** 2)
        mid_energy = np.sum(fft[(freqs >= 300) & (freqs < 3000)] ** 2)
        high_energy = np.sum(fft[freqs >= 3000] ** 2)
        total_energy = low_energy + mid_energy + high_energy
        
        if total_energy > 0:
            print(f"    能量分布: low(<300Hz)={low_energy/total_energy:.2%}, "
                  f"mid(300-3kHz)={mid_energy/total_energy:.2%}, "
                  f"high(>3kHz)={high_energy/total_energy:.2%}")
            
            # 人声主要在 300-3000Hz
            if mid_energy / total_energy < 0.2:
                print(f"    ⚠️  中频能量极低 ({mid_energy/total_energy:.2%}), 不像人声!")
            else:
                print(f"    ✅ 频谱分布正常（含人声特征）")
    else:
        print(f"    ❌ 未生成任何音频!")

# ===== Step 9: 与官方 demo 完全对齐的测试 =====
print(f"\n🎯 Step 9: 官方 demo 完全对齐测试")
print("  使用和 ModelScope 官方 demo 完全相同的参数:")
print(f"  tts_text: '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'")
print(f"  prompt_text: '希望你以后能够做的比我还好呦。'")
print(f"  prompt_wav: '{ref_audio}'")

official_tts_text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"

save_path = os.path.join(OUTPUT_DIR, "test_official_demo.wav")
all_speech = []
try:
    for i, chunk in enumerate(cosyvoice.inference_zero_shot(
        official_tts_text,
        '希望你以后能够做的比我还好呦。',
        ref_audio
    )):
        all_speech.append(chunk["tts_speech"])
        print(f"    chunk {i}: shape={chunk['tts_speech'].shape}")
except Exception as e:
    print(f"    ❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()

if all_speech:
    speech = torch.cat(all_speech, dim=-1)
    torchaudio.save(save_path, speech, cosyvoice.sample_rate)
    duration = speech.shape[-1] / cosyvoice.sample_rate
    rms = torch.sqrt(torch.mean(speech ** 2)).item()
    print(f"    ✅ 保存: {save_path} ({duration:.2f}s, RMS={rms:.6f})")
else:
    print(f"    ❌ 未生成音频!")

print(f"\n{'=' * 70}")
print(f"📊 诊断完成! 请将上面的完整输出发给我分析。")
print(f"   调试音频目录: {OUTPUT_DIR}")
print(f"{'=' * 70}")
