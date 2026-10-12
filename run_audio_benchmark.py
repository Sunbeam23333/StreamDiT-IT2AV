#!/usr/bin/env python3
"""
Audio Generation Benchmark - 统一音频推理脚本.

支持模型:
  - TangoFlux          : 文生音效 (Sound Effects)
  - ACE-Step           : 文生BGM/音乐 (Background Music)
  - SongGeneration     : 歌曲生成 (Song with Vocals)
  - CosyVoice          : 文本转语音 (Text-to-Speech)

用法:
    # 单模型推理
    python run_audio_benchmark.py --model tangoflux --prompts_file prompts.json --output_dir ./results
    python run_audio_benchmark.py --model ace_step --prompts_file prompts.json --output_dir ./results
    python run_audio_benchmark.py --model songgen --prompts_file prompts.json --output_dir ./results
    python run_audio_benchmark.py --model cosyvoice --prompts_file prompts.json --output_dir ./results

    # 全部模型
    python run_audio_benchmark.py --model all --prompts_file prompts.json --output_dir ./results

    # 指定场景
    python run_audio_benchmark.py --model tangoflux --scene_ids scene_01_pure_music scene_04_anime

注意: 模型需先通过 fast_sync.py 同步到 /tmp 本地盘再推理
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import torch

# ============================================================================
# 路径配置
# ============================================================================
GROUP_BASE = os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace"))
GROUP_MODELS_AUDIO = f"{GROUP_BASE}/models/Audio"
GROUP_CODE = os.environ.get("IT2AV_CODE_ROOT", os.path.dirname(os.path.abspath(__file__)))
GROUP_CODE_AUDIO = f"{GROUP_CODE}/utils/audio"
LOCAL_BASE = os.environ.get("IT2AV_LOCAL_ROOT", "/tmp/it2av")
LOCAL_MODELS_AUDIO = f"{LOCAL_BASE}/models/Audio"
LOCAL_CODE = f"{LOCAL_BASE}/code"
LOCAL_CODE_AUDIO = f"{LOCAL_CODE}/utils/audio"
SYNC_SCRIPT = f"{GROUP_CODE}/fast_sync.py"

# 模型目录名
MODEL_DIRS = {
    "tangoflux": "TangoFlux",
    "flan_t5_large": "flan-t5-large",
    "ace_step": "ACE-Step-v1-3.5B",
    "songgen": "SongGeneration-v2-large",
    "songgen_runtime": "SongGeneration-Runtime",
    "cosyvoice3": "Fun-CosyVoice3-0.5B",
    "cosyvoice2": "CosyVoice2-0.5B",
    "cosyvoice_ttsfrd": "CosyVoice-ttsfrd",
}

# 代码目录名
CODE_DIRS = {
    "tangoflux": "TangoFlux",
    "ace_step": "ACE-Step",
    "songgen": "SongGeneration",
    "cosyvoice": "CosyVoice",
}


def sync_to_local(src: str, dst: str, name: str) -> str:
    """用 fast_sync.py 同步文件到本地盘."""
    sync_marker = os.path.join(dst, ".sync_done")
    if os.path.exists(sync_marker):
        logging.info("[%s] Already cached at %s, skipping sync.", name, dst)
        return dst

    if not os.path.isdir(src):
        logging.error("[%s] Source not found: %s", name, src)
        return src  # 回退到 group 盘

    logging.info("[%s] Syncing %s -> %s ...", name, src, dst)
    os.makedirs(dst, exist_ok=True)
    workers = os.environ.get("SYNC_WORKERS", "32")
    cmd = [sys.executable, SYNC_SCRIPT, src, dst, "--workers", workers, "--skip-existing"]
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        logging.warning("[%s] Sync failed (rc=%d), falling back to group disk.", name, ret.returncode)
        return src
    Path(sync_marker).touch()
    logging.info("[%s] Sync complete.", name)
    return dst


def sync_model(model_key: str) -> str:
    """同步模型到本地盘, 返回实际路径."""
    dir_name = MODEL_DIRS[model_key]
    src = os.path.join(GROUP_MODELS_AUDIO, dir_name)
    dst = os.path.join(LOCAL_MODELS_AUDIO, dir_name)
    return sync_to_local(src, dst, dir_name)


def sync_code(code_key: str) -> str:
    """同步代码到本地盘, 返回实际路径."""
    dir_name = CODE_DIRS[code_key]
    src = os.path.join(GROUP_CODE_AUDIO, dir_name)
    dst = os.path.join(LOCAL_CODE_AUDIO, dir_name)
    return sync_to_local(src, dst, dir_name)


def ensure_cosyvoice_ttsfrd_resource(cosyvoice_code: str, ttsfrd_path: str) -> None:
    """确保 CosyVoice 能在预期路径找到 `CosyVoice-ttsfrd/resource`。"""
    resource_src = os.path.join(ttsfrd_path, "resource")
    resource_dst = os.path.join(cosyvoice_code, "pretrained_models", "CosyVoice-ttsfrd", "resource")

    if not os.path.isdir(resource_src):
        logging.warning("[CosyVoice] ttsfrd resource not found: %s", resource_src)
        return

    if os.path.islink(resource_dst) and os.path.realpath(resource_dst) == os.path.realpath(resource_src):
        logging.info("[CosyVoice] ttsfrd resource already linked: %s", resource_dst)
        return

    os.makedirs(os.path.dirname(resource_dst), exist_ok=True)
    if os.path.lexists(resource_dst):
        if os.path.islink(resource_dst):
            os.unlink(resource_dst)
        elif os.path.isdir(resource_dst) and not os.listdir(resource_dst):
            os.rmdir(resource_dst)
        else:
            logging.info("[CosyVoice] Keep existing non-empty frontend resource dir: %s", resource_dst)
            return

    os.symlink(resource_src, resource_dst)
    logging.info("[CosyVoice] Linked frontend resource: %s -> %s", resource_dst, resource_src)


def save_wav_pcm16(audio: torch.Tensor, save_path: str, sample_rate: int) -> None:
    """将音频统一写成 PCM16 WAV，避免 float WAV 触发后续汇总脚本报错。"""
    audio_tensor = audio.detach().cpu() if isinstance(audio, torch.Tensor) else torch.as_tensor(audio)
    audio_tensor = audio_tensor.float()

    if audio_tensor.ndim == 0:
        raise ValueError(f"Invalid audio shape: {tuple(audio_tensor.shape)}")
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.ndim != 2:
        audio_tensor = audio_tensor.reshape(audio_tensor.shape[0], -1)

    if audio_tensor.ndim == 2 and audio_tensor.shape[0] > audio_tensor.shape[1] and audio_tensor.shape[1] <= 8:
        audio_tensor = audio_tensor.transpose(0, 1)

    audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
    pcm16 = (audio_tensor * 32767.0).round().to(torch.int16)
    pcm16 = pcm16.transpose(0, 1).contiguous().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with wave.open(save_path, "wb") as wf:
        wf.setnchannels(int(audio_tensor.shape[0]))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def resolve_cosyvoice_speaker(available_speakers: list[str], requested_speaker: str | None,
                              language: str) -> str | None:
    """选择 CosyVoice SFT speaker；指定 speaker 时做合法性校验。"""
    if not available_speakers:
        return None

    if requested_speaker is None:
        return None

    if requested_speaker not in available_speakers:
        raise ValueError(
            f"[CosyVoice] Unknown speaker: {requested_speaker}. Available: {available_speakers}"
        )
    return requested_speaker


def load_prompts(prompts_file: str, scene_ids: list[str] | None = None) -> list[dict]:
    """加载 prompts JSON 文件."""
    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data["scenarios"]
    if scene_ids:
        scenarios = [s for s in scenarios if s["id"] in scene_ids]

    logging.info("Loaded %d scenarios from %s", len(scenarios), prompts_file)
    return scenarios


# ============================================================================
# TangoFlux 音效生成
# ============================================================================
def run_tangoflux(scenarios: list[dict], output_dir: str, device: str = "cuda:0"):
    """TangoFlux 文生音效推理."""
    import torchaudio

    model_path = sync_model("tangoflux")
    text_encoder_path = sync_model("flan_t5_large")  # TangoFlux 依赖 google/flan-t5-large 作为 text encoder

    logging.info("=" * 60)
    logging.info("[TangoFlux] Loading model from %s", model_path)
    logging.info("[TangoFlux] Text encoder from %s", text_encoder_path)

    # TangoFlux 使用 snapshot_download，我们需要让它使用本地路径
    # 修改环境变量指向本地缓存
    os.environ["HF_HOME"] = os.path.join(LOCAL_BASE, "hf_cache")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # 创建符号链接让 snapshot_download 使用本地模型
    hf_cache_model = os.path.join(os.environ["HF_HOME"], "hub", "models--declare-lab--TangoFlux")
    if not os.path.exists(hf_cache_model):
        os.makedirs(os.path.dirname(hf_cache_model), exist_ok=True)
        # 直接从本地加载
        pass

    # 添加 TangoFlux 代码路径
    tangoflux_code = sync_code("tangoflux")
    if tangoflux_code not in sys.path:
        sys.path.insert(0, tangoflux_code)

    from tangoflux import TangoFluxInference

    # 构造一个使用本地路径的推理实例
    from diffusers import AutoencoderOobleck
    from safetensors.torch import load_file

    vae = AutoencoderOobleck()
    vae_weights = load_file(os.path.join(model_path, "vae.safetensors"))
    vae.load_state_dict(vae_weights)

    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)

    from tangoflux.model import TangoFlux as TangoFluxModel

    model = TangoFluxModel(config, text_encoder_dir=text_encoder_path)
    weights = load_file(os.path.join(model_path, "tangoflux.safetensors"))
    model.load_state_dict(weights, strict=False)

    vae = vae.to(device)
    model = model.to(device)

    sfx_dir = os.path.join(output_dir, "tangoflux_sfx")
    os.makedirs(sfx_dir, exist_ok=True)

    logging.info("[TangoFlux] Model loaded. Starting generation...")

    total_start = time.time()
    gen_stats = []
    for idx, scene in enumerate(scenarios):
        sfx_cfg = scene["tangoflux_sfx"]
        prompt = sfx_cfg["prompt"]
        duration = min(sfx_cfg.get("duration", 30), 30)  # TangoFlux max 30s
        steps = sfx_cfg.get("steps", 50)

        logging.info("[TangoFlux] [%d/%d] %s (duration=%ds, steps=%d)",
                     idx + 1, len(scenarios), scene["id"], duration, steps)
        logging.info("  Prompt: %s", prompt[:80])

        t0 = time.time()
        with torch.no_grad():
            latents = model.inference_flow(
                prompt, duration=duration, num_inference_steps=steps, guidance_scale=4.5,
            )
            wave = vae.decode(latents.transpose(2, 1)).sample.cpu()[0]

        waveform_end = int(duration * vae.config.sampling_rate)
        wave = wave[:, :waveform_end]

        save_path = os.path.join(sfx_dir, f"{scene['id']}_sfx.wav")
        torchaudio.save(save_path, wave, sample_rate=44100)
        elapsed = time.time() - t0
        rtf = duration / elapsed if elapsed > 0 else float("inf")
        logging.info("  Saved: %s (%.1fs inference, %ds audio, RTF=%.1fx)", save_path, elapsed, duration, rtf)
        gen_stats.append({"scene": scene["id"], "duration_s": duration, "inference_s": round(elapsed, 2), "rtf": round(rtf, 2)})

        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    avg_rtf = sum(s["rtf"] for s in gen_stats) / len(gen_stats) if gen_stats else 0
    logging.info("[TangoFlux] Done! %d files, total %.1fs, avg RTF=%.1fx", len(scenarios), total_elapsed, avg_rtf)

    # 释放模型
    del model, vae
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# ACE-Step BGM 生成
# ============================================================================
def run_ace_step(scenarios: list[dict], output_dir: str, device_id: int = 0):
    """ACE-Step 文生BGM推理."""
    model_path = sync_model("ace_step")
    ace_code = sync_code("ace_step")

    if ace_code not in sys.path:
        sys.path.insert(0, ace_code)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    logging.info("=" * 60)
    logging.info("[ACE-Step] Loading pipeline from %s", model_path)

    from acestep.pipeline_ace_step import ACEStepPipeline

    pipeline = ACEStepPipeline(
        checkpoint_dir=model_path,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=False,
        overlapped_decode=False,
    )

    bgm_dir = os.path.join(output_dir, "ace_step_bgm")
    os.makedirs(bgm_dir, exist_ok=True)

    logging.info("[ACE-Step] Pipeline loaded. Starting generation...")

    total_start = time.time()
    gen_stats = []
    for idx, scene in enumerate(scenarios):
        bgm_cfg = scene["ace_step_bgm"]
        prompt = bgm_cfg["prompt"]
        lyrics = bgm_cfg.get("lyrics", "[inst-long]")
        duration = min(bgm_cfg.get("duration", 240), 240)  # ACE-Step max 240s

        logging.info("[ACE-Step] [%d/%d] %s (duration=%ds)", idx + 1, len(scenarios), scene["id"], duration)
        logging.info("  Prompt: %s", prompt[:80])

        t0 = time.time()
        save_path = os.path.join(bgm_dir, f"{scene['id']}_bgm.wav")

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
        rtf = duration / elapsed if elapsed > 0 else float("inf")
        logging.info("  Saved: %s (%.1fs inference, %ds audio, RTF=%.1fx)", save_path, elapsed, duration, rtf)
        gen_stats.append({"scene": scene["id"], "duration_s": duration, "inference_s": round(elapsed, 2), "rtf": round(rtf, 2)})

        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    avg_rtf = sum(s["rtf"] for s in gen_stats) / len(gen_stats) if gen_stats else 0
    logging.info("[ACE-Step] Done! %d files, total %.1fs, avg RTF=%.1fx", len(scenarios), total_elapsed, avg_rtf)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# SongGeneration 歌曲生成
# ============================================================================
def _ensure_songgen_deps():
    """确保 SongGeneration 的全部 Python 依赖已安装."""
    # (pip 包名, import 验证名) — 只列 requirements.txt 之外的隐性依赖
    _deps = [
        ("descript-audiotools>=0.7.2", "audiotools"),  # dac 的依赖
        ("argbind>=0.3.7", "argbind"),                 # dac 的依赖
        ("k-diffusion", "k_diffusion"),                # stable_audio_tools 的依赖
    ]
    missing = []
    for pip_name, import_name in _deps:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        logging.info("[SongGeneration] Installing missing dependencies: %s", missing)
        pip_cmd = [sys.executable, "-m", "pip", "install", *missing, "-q"]
        # 需要代理访问 PyPI 时通过 IT2AV_PROXY_URL / HTTP(S)_PROXY 传入
        env = os.environ.copy()
        _proxy = os.environ.get("IT2AV_PROXY_URL", "")
        if _proxy and not env.get("HTTP_PROXY") and not env.get("HTTPS_PROXY"):
            env["HTTP_PROXY"] = _proxy
            env["HTTPS_PROXY"] = _proxy
            env.setdefault("NO_PROXY", "localhost,127.0.0.1")
        subprocess.check_call(pip_cmd, env=env, timeout=300)


def run_songgen(scenarios: list[dict], output_dir: str, device: str = "cuda:0"):
    """SongGeneration 歌曲推理 - 使用命令行调用."""
    _ensure_songgen_deps()
    model_path = sync_model("songgen")
    runtime_path = sync_model("songgen_runtime")
    songgen_code = sync_code("songgen")

    logging.info("=" * 60)
    logging.info("[SongGeneration] Model: %s", model_path)
    logging.info("[SongGeneration] Runtime: %s", runtime_path)

    song_dir = os.path.join(output_dir, "songgen_song")
    os.makedirs(song_dir, exist_ok=True)
    os.makedirs(os.path.join(song_dir, "audios"), exist_ok=True)
    os.makedirs(os.path.join(song_dir, "jsonl"), exist_ok=True)

    # 生成 JSONL 输入文件
    jsonl_path = os.path.join(song_dir, "benchmark_input.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, scene in enumerate(scenarios):
            song_cfg = scene["songgen_song"]
            item = {
                "idx": scene["id"],
                "gt_lyric": song_cfg["gt_lyric"],
                "descriptions": song_cfg.get("descriptions", ""),
            }
            if "auto_prompt_audio_type" in song_cfg:
                item["auto_prompt_audio_type"] = song_cfg["auto_prompt_audio_type"]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info("[SongGeneration] Generated JSONL: %s (%d items)", jsonl_path, len(scenarios))

    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{songgen_code}/codeclm/tokenizer/:"
        f"{songgen_code}:"
        f"{songgen_code}/codeclm/tokenizer/Flow1dVAE/:"
        f"{songgen_code}/codeclm/tokenizer/:"
        + env.get("PYTHONPATH", "")
    )
    env["TRANSFORMERS_CACHE"] = f"{runtime_path}/hub"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    # 确保 ckpt / third_party / tools 指向 runtime 目录下的真实文件
    # 注意: SongGeneration 代码仓库中这些目录可能含 Git LFS 指针文件 (非真实数据),
    #       必须用 runtime 中的实际文件替换, 否则 torch.load 会失败
    for link_name in ["ckpt", "third_party", "tools"]:
        link_src = os.path.join(runtime_path, link_name)
        link_dst = os.path.join(songgen_code, link_name)
        if not os.path.exists(link_src):
            continue
        # 如果目标已存在但不是指向正确源的 symlink, 需要替换
        if os.path.exists(link_dst) or os.path.islink(link_dst):
            # 已经是指向正确位置的 symlink → 跳过
            if os.path.islink(link_dst) and os.path.realpath(link_dst) == os.path.realpath(link_src):
                continue
            # 否则移除旧目录 / 旧 symlink (可能是 LFS 指针文件)
            try:
                if os.path.islink(link_dst) or os.path.isfile(link_dst):
                    os.remove(link_dst)
                else:
                    import shutil
                    shutil.rmtree(link_dst)
                logging.info("  Removed stale %s (replacing with runtime symlink)", link_dst)
            except OSError as e:
                logging.warning("  Failed to remove %s: %s", link_dst, e)
                continue
        try:
            os.symlink(link_src, link_dst)
            logging.info("  Created symlink: %s -> %s", link_dst, link_src)
        except OSError as e:
            logging.warning("  Symlink failed: %s", e)

    # 检查 tools/new_auto_prompt.pt 是否为 Git LFS 指针文件
    # LFS 指针文件很小 (< 1KB) 且以 "version https://git-lfs" 开头
    auto_prompt_pt = os.path.join(songgen_code, "tools", "new_auto_prompt.pt")
    if os.path.isfile(auto_prompt_pt) and os.path.getsize(auto_prompt_pt) < 1024:
        with open(auto_prompt_pt, "rb") as f:
            header = f.read(40)
        if header.startswith(b"version https://git-lfs"):
            logging.info("[SongGeneration] tools/new_auto_prompt.pt is a Git LFS pointer (%d bytes), "
                         "downloading real file from GitHub...", os.path.getsize(auto_prompt_pt))
            # 先尝试在 group 盘源仓库执行 git lfs pull
            group_songgen = os.path.join(GROUP_CODE_AUDIO, "SongGeneration")
            group_auto_prompt = os.path.join(group_songgen, "tools", "new_auto_prompt.pt")
            lfs_pulled = False
            if os.path.isdir(os.path.join(group_songgen, ".git")):
                try:
                    subprocess.run(
                        ["git", "lfs", "pull", "--include", "tools/new_auto_prompt.pt"],
                        cwd=group_songgen, check=True, timeout=120,
                        capture_output=True,
                    )
                    if os.path.isfile(group_auto_prompt) and os.path.getsize(group_auto_prompt) > 1024:
                        import shutil
                        shutil.copy2(group_auto_prompt, auto_prompt_pt)
                        logging.info("[SongGeneration] git lfs pull succeeded, copied from group: %s (%d bytes)",
                                     auto_prompt_pt, os.path.getsize(auto_prompt_pt))
                        lfs_pulled = True
                except Exception as e:
                    logging.warning("[SongGeneration] git lfs pull failed: %s", e)

            # 上游 GitHub 仓库已下线 (2026-10), 优先使用本地已有的真实文件
            local_copy = os.environ.get("SONGGEN_AUTO_PROMPT", "")
            if not lfs_pulled and local_copy and os.path.isfile(local_copy) and os.path.getsize(local_copy) > 1024:
                import shutil
                shutil.copy2(local_copy, auto_prompt_pt)
                logging.info("[SongGeneration] Copied auto_prompt from SONGGEN_AUTO_PROMPT=%s", local_copy)
                lfs_pulled = True

            if not lfs_pulled:
                github_url = os.environ.get(
                    "SONGGEN_AUTO_PROMPT_URL",
                    "https://media.githubusercontent.com/media/tencent-ailab/SongGeneration/main/tools/new_auto_prompt.pt",
                )
                logging.info("[SongGeneration] Downloading auto_prompt: %s", github_url)
                try:
                    import urllib.request
                    _proxy_url = os.environ.get("IT2AV_PROXY_URL", "")
                    _proxy_env_set = False
                    if _proxy_url and not os.environ.get("HTTP_PROXY") and not os.environ.get("HTTPS_PROXY"):
                        os.environ["HTTP_PROXY"] = _proxy_url
                        os.environ["HTTPS_PROXY"] = _proxy_url
                        os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
                        _proxy_env_set = True
                    try:
                        tmp_path = auto_prompt_pt + ".downloading"
                        urllib.request.urlretrieve(github_url, tmp_path)
                        dl_size = os.path.getsize(tmp_path)
                        # 验证下载的不是 LFS 指针
                        with open(tmp_path, "rb") as f:
                            dl_header = f.read(40)
                        if dl_header.startswith(b"version https://git-lfs"):
                            os.remove(tmp_path)
                            raise RuntimeError("Downloaded file is still a Git LFS pointer")
                        os.replace(tmp_path, auto_prompt_pt)
                        logging.info("[SongGeneration] Downloaded real auto_prompt: %s (%d bytes)",
                                     auto_prompt_pt, dl_size)
                    finally:
                        if _proxy_env_set:
                            os.environ.pop("HTTP_PROXY", None)
                            os.environ.pop("HTTPS_PROXY", None)
                        if os.path.exists(auto_prompt_pt + ".downloading"):
                            os.remove(auto_prompt_pt + ".downloading")
                except Exception as e:
                    logging.error("[SongGeneration] Failed to download tools/new_auto_prompt.pt: %s", e)
                    logging.error("[SongGeneration] Please manually run: "
                                  "cd %s && git lfs pull --include='tools/new_auto_prompt.pt'", group_songgen)

    # 调用 generate.py
    cmd = [
        sys.executable, os.path.join(songgen_code, "generate.py"),
        "--ckpt_path", model_path,
        "--input_jsonl", jsonl_path,
        "--save_dir", song_dir,
        "--generate_type", "mixed",
        "--use_flash_attn",
    ]

    logging.info("[SongGeneration] Running: %s", " ".join(cmd))
    t0 = time.time()
    ret = subprocess.run(cmd, env=env, cwd=songgen_code, check=False)
    elapsed = time.time() - t0

    if ret.returncode == 0:
        logging.info("[SongGeneration] Done! Output: %s (%.1fs)", song_dir, elapsed)
    else:
        logging.error("[SongGeneration] Failed (rc=%d) after %.1fs", ret.returncode, elapsed)


# ============================================================================
# CosyVoice 语音合成
# ============================================================================
def run_cosyvoice(scenarios: list[dict], output_dir: str, language: str = "zh",
                  tts_engine: str = "cosyvoice2", speaker: str | None = None,
                  ref_audio: str | None = None, ref_text: str | None = None,
                  speed: float = 1.0):
    """CosyVoice 文本转语音推理.

    Args:
        tts_engine: "cosyvoice2" (默认, 更稳定) 或 "cosyvoice3".
            CosyVoice3 存在已知的质量问题 (GitHub Issue #1703):
            生成乱码、重复、非人类语言等。建议使用 CosyVoice2。
        speaker: 指定后优先走 SFT speaker 模式；未指定时走 zero-shot 参考音频模式。
        ref_audio/ref_text: zero-shot / cross-lingual 模式使用的参考音频和参考文本。
        speed: CosyVoice 语速，仅在 SFT 模式下生效。
    """
    # 选择模型: CosyVoice2 更稳定, CosyVoice3 存在已知问题
    model_key = tts_engine if tts_engine in ("cosyvoice2", "cosyvoice3") else "cosyvoice2"
    model_path = sync_model(model_key)
    cosyvoice_code = sync_code("cosyvoice")

    # 同步并挂载 ttsfrd 前端资源到 CosyVoice 代码预期位置
    ttsfrd_path = sync_model("cosyvoice_ttsfrd")
    ensure_cosyvoice_ttsfrd_resource(cosyvoice_code, ttsfrd_path)

    if cosyvoice_code not in sys.path:
        sys.path.insert(0, cosyvoice_code)

    # CosyVoice 需要 third_party/Matcha-TTS 在 path 中
    # Matcha-TTS 已预先同步到 group 盘，通过 fast_sync 同步到本地盘
    matcha_group_src = f"{GROUP_CODE_AUDIO}/Matcha-TTS"
    matcha_path = os.path.join(cosyvoice_code, "third_party", "Matcha-TTS")
    # 检查 CosyVoice/third_party/Matcha-TTS 是否就绪
    matcha_ready = os.path.exists(matcha_path) and os.path.isfile(os.path.join(matcha_path, "setup.py"))
    if not matcha_ready:
        # 从 group 盘同步 Matcha-TTS 到本地 CosyVoice/third_party/Matcha-TTS
        logging.info("[CosyVoice] Matcha-TTS not ready at %s, syncing from group disk...", matcha_path)
        os.makedirs(os.path.dirname(matcha_path), exist_ok=True)
        # 如果是空目录(git submodule 未初始化)先删掉再同步
        if os.path.isdir(matcha_path) and not os.listdir(matcha_path):
            os.rmdir(matcha_path)
        sync_to_local(matcha_group_src, matcha_path, "Matcha-TTS")
    if os.path.exists(matcha_path) and matcha_path not in sys.path:
        sys.path.insert(0, matcha_path)

    # CosyVoice 依赖 pyworld (dataset/processor.py import),
    # hyperpyyaml 加载 YAML 配置时会间接触发该 import
    try:
        import pyworld  # noqa: F401
    except ImportError:
        logging.info("[CosyVoice] Installing missing dependency: pyworld")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyworld", "-q"])

    logging.info("=" * 60)
    logging.info("[CosyVoice] Loading %s model from %s", model_key.upper(), model_path)

    from cosyvoice.cli.cosyvoice import AutoModel

    # fp16=False 是关键: CosyVoice 2/3 在 fp16 下可能导致音频出现
    # 前后较长空白、重复、乱读等问题 (参见 GitHub issues)
    cosyvoice = AutoModel(model_dir=model_path, fp16=False)

    tts_dir = os.path.join(output_dir, "cosyvoice_tts")
    os.makedirs(tts_dir, exist_ok=True)

    is_v3 = model_key == "cosyvoice3"
    use_text_frontend = is_v3
    available_speakers = cosyvoice.list_available_spks() if hasattr(cosyvoice, "list_available_spks") else []
    selected_speaker = resolve_cosyvoice_speaker(available_speakers, speaker, language) if speaker else None
    use_sft = selected_speaker is not None
    logging.info("[CosyVoice] %s model loaded. Starting generation...", model_key.upper())
    logging.info("[CosyVoice] Model class: %s, sample_rate=%s", cosyvoice.__class__.__name__, cosyvoice.sample_rate)
    logging.info("[CosyVoice] Frontend backend: %s", getattr(cosyvoice.frontend, "text_frontend", "unknown"))
    if available_speakers:
        logging.info("[CosyVoice] Available SFT speakers: %s", ", ".join(available_speakers))
    if use_sft:
        logging.info("[CosyVoice] Using SFT speaker: %s (speed=%.2f)", selected_speaker, speed)
    else:
        logging.info("[CosyVoice] Using zero-shot reference voice")
    try:
        import onnxruntime
        logging.info("[CosyVoice] onnxruntime=%s, providers=%s",
                     onnxruntime.__version__, onnxruntime.get_available_providers())
    except Exception as exc:
        logging.warning("[CosyVoice] Failed to inspect onnxruntime: %s", exc)
    if not use_text_frontend:
        logging.info("[CosyVoice] CosyVoice2 zero-shot uses text_frontend=False (official recommendation)")

    # 获取参考音频路径
    ref_audio = ref_audio or os.path.join(cosyvoice_code, "asset", "zero_shot_prompt.wav")

    # prompt_text 格式:
    #   CosyVoice3: 需要 "You are a helpful assistant.<|endofprompt|>" 前缀
    #   CosyVoice2: 不需要前缀, 直接用参考音频对应文本
    base_ref_text = ref_text or "希望你以后能够做的比我还好呦。"
    if is_v3:
        prompt_text = f"You are a helpful assistant.<|endofprompt|>{base_ref_text}"
    else:
        prompt_text = base_ref_text

    total_start = time.time()
    for idx, scene in enumerate(scenarios):
        tts_cfg = scene["cosyvoice_tts"]

        # 选择语言
        if language == "zh":
            text = tts_cfg.get("text_zh", tts_cfg.get("text_en", ""))
        else:
            text = tts_cfg.get("text_en", tts_cfg.get("text_zh", ""))

        if not text:
            logging.warning("[CosyVoice] [%d/%d] %s - No text, skipping.", idx + 1, len(scenarios), scene["id"])
            continue

        logging.info("[CosyVoice] [%d/%d] %s", idx + 1, len(scenarios), scene["id"])
        logging.info("  Text: %s", text[:80])

        t0 = time.time()

        try:
            if use_sft:
                for i, j in enumerate(cosyvoice.inference_sft(
                    text, selected_speaker, stream=False, speed=speed,
                    text_frontend=use_text_frontend
                )):
                    save_path = os.path.join(tts_dir, f"{scene['id']}_tts_{language}_{i}.wav")
                    save_wav_pcm16(j["tts_speech"], save_path, cosyvoice.sample_rate)
                    elapsed = time.time() - t0
                    logging.info("  Saved (SFT:%s): %s (%.1fs, PCM16)", selected_speaker, save_path, elapsed)
            else:
                for i, j in enumerate(cosyvoice.inference_zero_shot(
                    text, prompt_text, ref_audio, stream=False,
                    text_frontend=use_text_frontend
                )):
                    save_path = os.path.join(tts_dir, f"{scene['id']}_tts_{language}_{i}.wav")
                    save_wav_pcm16(j["tts_speech"], save_path, cosyvoice.sample_rate)
                    elapsed = time.time() - t0
                    logging.info("  Saved: %s (%.1fs, PCM16)", save_path, elapsed)
        except Exception as e:
            logging.error("  TTS failed: %s", e)
            # zero-shot 模式下回退到 cross_lingual；SFT 模式保留原报错方便定位 speaker/模型问题
            if use_sft:
                logging.error("  TTS SFT mode failed without fallback.")
            else:
                try:
                    cross_prefix = "You are a helpful assistant.<|endofprompt|>" if is_v3 else ""
                    for i, j in enumerate(cosyvoice.inference_cross_lingual(
                        f"{cross_prefix}{text}",
                        ref_audio, stream=False,
                        text_frontend=use_text_frontend
                    )):
                        save_path = os.path.join(tts_dir, f"{scene['id']}_tts_{language}_{i}.wav")
                        save_wav_pcm16(j["tts_speech"], save_path, cosyvoice.sample_rate)
                        elapsed = time.time() - t0
                        logging.info("  Saved (fallback): %s (%.1fs, PCM16)", save_path, elapsed)
                except Exception as e2:
                    logging.error("  TTS fallback also failed: %s", e2)

        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    logging.info("[CosyVoice] Done! total %.1fs", total_elapsed)

    del cosyvoice
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# 主入口
# ============================================================================
RUNNERS = {
    "tangoflux": run_tangoflux,
    "ace_step": run_ace_step,
    "songgen": run_songgen,
    "cosyvoice": run_cosyvoice,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio Generation Benchmark (TangoFlux / ACE-Step / SongGeneration / CosyVoice)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        choices=list(RUNNERS.keys()) + ["all"],
        help="Which model to run (or 'all')",
    )
    parser.add_argument(
        "--prompts_file", required=True,
        help="Path to prompts JSON file",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory (default: GROUP_BASE/results/audio_benchmark)",
    )
    parser.add_argument(
        "--scene_ids", nargs="*", default=None,
        help="Only run specific scene IDs",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="CUDA device (for TangoFlux, CosyVoice)",
    )
    parser.add_argument(
        "--device_id", type=int, default=0,
        help="GPU device ID (for ACE-Step)",
    )
    parser.add_argument(
        "--language", default="zh", choices=["zh", "en"],
        help="TTS language (for CosyVoice)",
    )
    parser.add_argument(
        "--tts_engine", default="cosyvoice2",
        choices=["cosyvoice2", "cosyvoice3"],
        help="TTS engine: cosyvoice2 (default, stable) or cosyvoice3 (known quality issues)",
    )
    parser.add_argument(
        "--cosyvoice_speaker", default=None,
        help="CosyVoice SFT speaker name，例如 中文女 / 中文男；指定后优先走 SFT 模式",
    )
    parser.add_argument(
        "--cosyvoice_ref_audio", default=None,
        help="CosyVoice zero-shot/cross-lingual 参考音频路径；未指定则使用仓库内置 zero_shot_prompt.wav",
    )
    parser.add_argument(
        "--cosyvoice_ref_text", default=None,
        help="CosyVoice zero-shot 参考文本；未指定则使用仓库内置示例文本",
    )
    parser.add_argument(
        "--cosyvoice_speed", type=float, default=1.0,
        help="CosyVoice SFT 语速，默认 1.0",
    )
    parser.add_argument(
        "--duration_override", type=float, default=None,
        help="Override audio duration for all scenarios (TangoFlux max=30, ACE-Step max=240). "
             "SongGeneration/CosyVoice not affected (duration by lyrics/text length).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = args.output_dir or os.path.join(GROUP_BASE, "results", "audio_benchmark")
    os.makedirs(output_dir, exist_ok=True)

    scenarios = load_prompts(args.prompts_file, args.scene_ids)

    if not scenarios:
        logging.error("No scenarios found! Check prompts file and scene_ids.")
        sys.exit(1)

    # Duration override
    dur_info = ""
    if args.duration_override is not None:
        dur_info = f" (override={args.duration_override}s)"
        for s in scenarios:
            if "tangoflux_sfx" in s:
                s["tangoflux_sfx"]["duration"] = min(args.duration_override, 30.0)
            if "ace_step_bgm" in s:
                s["ace_step_bgm"]["duration"] = min(args.duration_override, 240.0)

    logging.info("=" * 60)
    logging.info("Audio Generation Benchmark")
    logging.info("  Model(s):     %s", args.model)
    logging.info("  Scenarios:    %d", len(scenarios))
    logging.info("  Output:       %s", output_dir)
    logging.info("  Device:       %s (id=%d)", args.device, args.device_id)
    logging.info("  TTS Language: %s", args.language)
    logging.info("  TTS Engine:   %s", args.tts_engine)
    logging.info("  Duration:     from prompts%s", dur_info)
    logging.info("=" * 60)

    total_start = time.time()

    if args.model == "all":
        models_to_run = list(RUNNERS.keys())
    else:
        models_to_run = [args.model]

    # Benchmark report
    benchmark_report = {
        "config": {
            "model": args.model,
            "num_scenarios": len(scenarios),
            "device": args.device,
            "language": args.language,
            "tts_engine": args.tts_engine,
            "cosyvoice_speaker": args.cosyvoice_speaker,
            "cosyvoice_ref_audio": args.cosyvoice_ref_audio,
            "cosyvoice_ref_text": args.cosyvoice_ref_text,
            "cosyvoice_speed": args.cosyvoice_speed,
            "duration_override": args.duration_override,
        },
        "results": {},
    }

    for model_name in models_to_run:
        logging.info("")
        logging.info(">>> Running %s...", model_name)
        model_start = time.time()

        try:
            runner = RUNNERS[model_name]
            if model_name == "tangoflux":
                runner(scenarios, output_dir, device=args.device)
            elif model_name == "ace_step":
                runner(scenarios, output_dir, device_id=args.device_id)
            elif model_name == "songgen":
                runner(scenarios, output_dir, device=args.device)
            elif model_name == "cosyvoice":
                runner(
                    scenarios,
                    output_dir,
                    language=args.language,
                    tts_engine=args.tts_engine,
                    speaker=args.cosyvoice_speaker,
                    ref_audio=args.cosyvoice_ref_audio,
                    ref_text=args.cosyvoice_ref_text,
                    speed=args.cosyvoice_speed,
                )
        except Exception as e:
            logging.error(">>> %s FAILED: %s", model_name, e)
            import traceback
            traceback.print_exc()

        model_elapsed = time.time() - model_start
        logging.info(">>> %s finished in %.1fs", model_name, model_elapsed)

        benchmark_report["results"][model_name] = {
            "total_time_s": round(model_elapsed, 2),
            "num_scenarios": len(scenarios),
        }

    total_elapsed = time.time() - total_start

    benchmark_report["total_time_s"] = round(total_elapsed, 2)

    # Save benchmark report
    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

    logging.info("")
    logging.info("=" * 60)
    logging.info("Audio Benchmark Complete!")
    logging.info("  Total time: %.1fs", total_elapsed)
    logging.info("  Output:     %s", output_dir)
    logging.info("  Report:     %s", report_path)
    logging.info("  Subdirs:")
    logging.info("    tangoflux_sfx/  - 音效 (max 30s)")
    logging.info("    ace_step_bgm/   - BGM音乐 (max 240s)")
    logging.info("    songgen_song/   - 歌曲(人声+伴奏, max 4m30s)")
    logging.info("    cosyvoice_tts/  - 语音旁白 (由文本决定)")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
