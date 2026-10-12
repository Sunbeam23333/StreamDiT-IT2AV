#!/usr/bin/env python3
"""
CosyVoice2 深度诊断 - 检查模型权重完整性和推理中间结果

重点排查:
  1. 模型权重是否被截断/损坏 (Git LFS 指针文件 vs 真实权重)
  2. LLM 生成的 speech token 是否合理
  3. Flow 输出的 mel 是否合理  
  4. HiFT 输出的波形是否合理
"""
import os
import sys
import time
import hashlib
import numpy as np

LOCAL_BASE = os.environ.get("IT2AV_LOCAL_ROOT", "/tmp/it2av")
LOCAL_MODELS = f"{LOCAL_BASE}/models/Audio"
LOCAL_CODE = f"{LOCAL_BASE}/code/utils/audio"
GROUP_BASE = os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace"))
MODEL_DIR = f"{LOCAL_MODELS}/CosyVoice2-0.5B"
COSYVOICE_CODE = f"{LOCAL_CODE}/CosyVoice"
OUTPUT_DIR = f"{GROUP_BASE}/data/audio/digit/debug_tts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

import torch
import torchaudio

# 添加 CosyVoice 代码路径
matcha_path = os.path.join(COSYVOICE_CODE, "third_party", "Matcha-TTS")
if not os.path.exists(matcha_path):
    matcha_path = f"{LOCAL_CODE}/Matcha-TTS"
for p in [COSYVOICE_CODE, matcha_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

print("=" * 70)
print("🔬 CosyVoice2 深度诊断 - 权重完整性 & 推理中间结果")
print("=" * 70)

# ===== Step 1: 检查模型权重是否是 Git LFS 指针文件 =====
print("\n📦 Step 1: 模型权重完整性 (Git LFS 指针检测)")
weight_files = {
    "llm.pt": 1929.6,       # 预期 MB
    "flow.pt": 429.7,
    "hift.pt": 79.5,
}

all_weights_ok = True
for fname, expected_mb in weight_files.items():
    fpath = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  ❌ {fname} 缺失!")
        all_weights_ok = False
        continue
    
    actual_mb = os.path.getsize(fpath) / 1024 / 1024
    
    # Git LFS 指针文件通常很小 (<1KB)
    if actual_mb < 1:
        print(f"  ❌ {fname}: {actual_mb:.2f} MB — 可能是 Git LFS 指针文件! (预期 {expected_mb:.1f} MB)")
        # 读取前几行检查
        with open(fpath, 'rb') as f:
            header = f.read(100)
        if b'version https://git-lfs' in header:
            print(f"      确认: 这是 Git LFS 指针文件! 需要 git lfs pull")
            all_weights_ok = False
        continue
    
    # 检查大小是否大致匹配 (允许 5% 误差)
    size_ratio = actual_mb / expected_mb
    if 0.95 <= size_ratio <= 1.05:
        print(f"  ✅ {fname}: {actual_mb:.1f} MB (预期 {expected_mb:.1f} MB)")
    else:
        print(f"  ⚠️  {fname}: {actual_mb:.1f} MB (预期 {expected_mb:.1f} MB, 偏差 {(size_ratio-1)*100:.1f}%)")
    
    # 快速 MD5 检查 (只读前后各 1MB)
    with open(fpath, 'rb') as f:
        head = f.read(1024 * 1024)
        f.seek(max(0, os.path.getsize(fpath) - 1024 * 1024))
        tail = f.read(1024 * 1024)
    md5 = hashlib.md5(head + tail).hexdigest()
    print(f"      head+tail MD5: {md5}")

# ===== Step 2: 加载权重并检查参数统计 =====
print("\n📊 Step 2: 权重参数统计")

for fname in weight_files:
    fpath = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(fpath):
        continue
    
    try:
        state_dict = torch.load(fpath, map_location='cpu', weights_only=True)
        n_params = len(state_dict)
        total_elements = sum(v.numel() for v in state_dict.values())
        
        # 统计 NaN 和 Inf
        nan_count = 0
        inf_count = 0
        zero_layers = 0
        for k, v in state_dict.items():
            if torch.isnan(v).any():
                nan_count += 1
            if torch.isinf(v).any():
                inf_count += 1
            if v.abs().max() == 0:
                zero_layers += 1
        
        print(f"  {fname}: {n_params} 参数, {total_elements/1e6:.1f}M elements")
        if nan_count > 0:
            print(f"    ❌ 发现 {nan_count} 个含 NaN 的参数!")
        if inf_count > 0:
            print(f"    ❌ 发现 {inf_count} 个含 Inf 的参数!")
        if zero_layers > 0:
            print(f"    ⚠️  {zero_layers} 个全零参数")
        if nan_count == 0 and inf_count == 0:
            print(f"    ✅ 无 NaN/Inf")
        
        # 抽样检查几个关键层的值域
        sample_keys = list(state_dict.keys())[:3] + list(state_dict.keys())[-3:]
        for k in sample_keys:
            v = state_dict[k]
            print(f"    {k}: shape={list(v.shape)}, range=[{v.min():.4f}, {v.max():.4f}], mean={v.mean():.6f}")
        
        del state_dict
    except Exception as e:
        print(f"  ❌ {fname} 无法加载: {e}")
        all_weights_ok = False

if not all_weights_ok:
    print("\n  ⚠️  模型权重有问题! 建议重新下载/同步。")

# ===== Step 3: 加载模型并检查 LLM 生成的 speech token =====
print("\n🧠 Step 3: LLM 推理中间结果")

from cosyvoice.cli.cosyvoice import AutoModel
cosyvoice = AutoModel(model_dir=MODEL_DIR, fp16=False)
print(f"  模型加载完成")

# Hook LLM 生成过程
ref_audio = os.path.join(COSYVOICE_CODE, "asset", "zero_shot_prompt.wav")
prompt_text = "希望你以后能够做的比我还好呦。"
tts_text = "晚上好，生活再忙再累，也别忘了好好爱自己。"

# 准备 model_input
prompt_text_norm = cosyvoice.frontend.text_normalize(prompt_text, split=False, text_frontend=False)
tts_text_norm_list = cosyvoice.frontend.text_normalize(tts_text, split=True, text_frontend=False)
tts_text_norm = tts_text_norm_list[0] if tts_text_norm_list else tts_text

model_input = cosyvoice.frontend.frontend_zero_shot(
    tts_text_norm, prompt_text_norm, ref_audio,
    resample_rate=cosyvoice.sample_rate, zero_shot_spk_id=''
)

print(f"  text shape: {model_input['text'].shape}")
print(f"  prompt_text shape: {model_input['prompt_text'].shape}")
print(f"  llm_prompt_speech_token shape: {model_input['llm_prompt_speech_token'].shape}")
print(f"  prompt_speech_feat shape: {model_input['prompt_speech_feat'].shape}")

# ===== Step 4: 手动运行 LLM 并检查输出 token =====
print("\n🔮 Step 4: LLM 生成 speech token 分析")
import threading
import uuid as uuid_lib

this_uuid = str(uuid_lib.uuid1())
model = cosyvoice.model

with model.lock:
    model.tts_speech_token_dict[this_uuid] = []
    model.llm_end_dict[this_uuid] = False
    model.hift_cache_dict[this_uuid] = None

# 直接运行 LLM
t0 = time.time()
model.llm_job(
    model_input['text'],
    model_input['prompt_text'],
    model_input['llm_prompt_speech_token'],
    model_input['llm_embedding'],
    this_uuid
)
llm_time = time.time() - t0

llm_tokens = model.tts_speech_token_dict[this_uuid]
print(f"  LLM 生成 token 数: {len(llm_tokens)}")
print(f"  LLM 耗时: {llm_time:.2f}s")

if len(llm_tokens) > 0:
    tokens_arr = np.array(llm_tokens)
    print(f"  token 范围: [{tokens_arr.min()}, {tokens_arr.max()}]")
    print(f"  token 前30个: {tokens_arr[:30].tolist()}")
    print(f"  token 后30个: {tokens_arr[-30:].tolist()}")
    print(f"  唯一值: {len(np.unique(tokens_arr))}/{len(tokens_arr)} ({len(np.unique(tokens_arr))/len(tokens_arr):.2%})")
    
    # 检查是否有大量连续重复
    diffs = np.diff(tokens_arr)
    repeat_count = np.sum(diffs == 0)
    print(f"  连续重复 token 数: {repeat_count}/{len(tokens_arr)-1} ({repeat_count/(len(tokens_arr)-1):.2%})")
    
    if repeat_count / (len(tokens_arr) - 1) > 0.5:
        print(f"  ⚠️  超过50%的 token 是连续重复的! LLM 可能出现了退化/重复模式!")
    else:
        print(f"  ✅ LLM token 分布正常")
    
    # 检查 token 值是否在 CosyVoice2 FSQ codebook 范围内 (通常 0-6560)
    max_valid_token = 6561  # CosyVoice2 FSQ codebook size
    out_of_range = np.sum((tokens_arr < 0) | (tokens_arr >= max_valid_token))
    if out_of_range > 0:
        print(f"  ⚠️  {out_of_range} 个 token 超出 FSQ codebook 范围 [0, {max_valid_token})!")
    else:
        print(f"  ✅ 所有 token 在 FSQ codebook 范围内")
else:
    print(f"  ❌ LLM 未生成任何 token!")

# ===== Step 5: Flow + HiFT 手动运行 =====
print("\n🌊 Step 5: Flow + HiFT 推理中间结果")

if len(llm_tokens) > 0:
    this_tts_speech_token = torch.tensor(llm_tokens).unsqueeze(dim=0)
    print(f"  speech token tensor: shape={this_tts_speech_token.shape}")
    
    # 手动运行 token2wav (非流式)
    t0 = time.time()
    try:
        this_tts_speech = model.token2wav(
            token=this_tts_speech_token,
            prompt_token=model_input['flow_prompt_speech_token'],
            prompt_feat=model_input['prompt_speech_feat'],
            embedding=model_input['flow_embedding'],
            token_offset=0,
            uuid=this_uuid,
            finalize=True,
            speed=1.0
        )
        flow_hift_time = time.time() - t0
        
        print(f"  输出 speech: shape={this_tts_speech.shape}")
        print(f"  Flow+HiFT 耗时: {flow_hift_time:.2f}s")
        print(f"  speech 范围: [{this_tts_speech.min():.4f}, {this_tts_speech.max():.4f}]")
        print(f"  speech RMS: {torch.sqrt(torch.mean(this_tts_speech**2)).item():.6f}")
        
        duration = this_tts_speech.shape[-1] / cosyvoice.sample_rate
        print(f"  时长: {duration:.2f}s")
        
        # 保存
        save_path = os.path.join(OUTPUT_DIR, "debug_manual_pipeline.wav")
        if this_tts_speech.dim() == 1:
            this_tts_speech = this_tts_speech.unsqueeze(0)
        torchaudio.save(save_path, this_tts_speech.cpu(), cosyvoice.sample_rate)
        print(f"  ✅ 保存: {save_path}")
        
        # 频谱分析
        signal = this_tts_speech.cpu().numpy().flatten()
        fft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1.0 / cosyvoice.sample_rate)
        
        # 检查 F0 (基频) - 人声通常在 85-255Hz
        low_freq_mask = (freqs >= 85) & (freqs <= 255)
        if np.sum(fft[low_freq_mask] ** 2) > 0:
            f0_peak = freqs[low_freq_mask][np.argmax(fft[low_freq_mask])]
            print(f"  F0 峰值频率: {f0_peak:.1f} Hz")
            if 85 <= f0_peak <= 255:
                print(f"  ✅ F0 在正常人声范围 (女声 ~200Hz, 男声 ~130Hz)")
            else:
                print(f"  ⚠️  F0 不在典型人声范围")
        
        # 更详细的频谱段分析
        bands = [(0, 85, "次低频"), (85, 255, "F0基频"), (255, 500, "低谐波"),
                 (500, 2000, "中频共振峰"), (2000, 4000, "高频清晰度"), (4000, 8000, "超高频")]
        total_energy = np.sum(fft ** 2)
        for lo, hi, name in bands:
            mask = (freqs >= lo) & (freqs < hi)
            energy = np.sum(fft[mask] ** 2) / total_energy * 100
            print(f"    {name} ({lo}-{hi}Hz): {energy:.1f}%")
            
    except Exception as e:
        print(f"  ❌ Flow/HiFT 推理失败: {e}")
        import traceback
        traceback.print_exc()

# 清理
with model.lock:
    model.tts_speech_token_dict.pop(this_uuid, None)
    model.llm_end_dict.pop(this_uuid, None)
    model.hift_cache_dict.pop(this_uuid, None)

# ===== Step 6: 对比正常推理 =====
print("\n🎵 Step 6: 标准推理对比")
print(f"  使用 cosyvoice.inference_zero_shot() 的标准接口")

all_speech = []
t0 = time.time()
for i, chunk in enumerate(cosyvoice.inference_zero_shot(
    tts_text, prompt_text, ref_audio,
    stream=False, text_frontend=False
)):
    all_speech.append(chunk["tts_speech"])

if all_speech:
    speech = torch.cat(all_speech, dim=-1)
    duration = speech.shape[-1] / cosyvoice.sample_rate
    rms = torch.sqrt(torch.mean(speech ** 2)).item()
    elapsed = time.time() - t0
    print(f"  ✅ 生成完成: {duration:.2f}s, RMS={rms:.6f}, 耗时={elapsed:.1f}s")
    
    save_path = os.path.join(OUTPUT_DIR, "debug_standard_pipeline.wav")
    torchaudio.save(save_path, speech, cosyvoice.sample_rate)
    print(f"  ✅ 保存: {save_path}")
    
    # 最终建议
    print(f"\n{'=' * 70}")
    print(f"📋 请听以下两个文件并对比:")
    print(f"  1. {OUTPUT_DIR}/debug_manual_pipeline.wav  (手动 pipeline)")
    print(f"  2. {OUTPUT_DIR}/debug_standard_pipeline.wav (标准 pipeline)")
    print(f"  两者应该完全一致。如果都不是人声,")
    print(f"  则问题在模型权重本身或 CosyVoice2-0.5B 这个模型的能力。")
    print(f"{'=' * 70}")
else:
    print(f"  ❌ 未生成音频!")

print("\n✅ 深度诊断完成!")
