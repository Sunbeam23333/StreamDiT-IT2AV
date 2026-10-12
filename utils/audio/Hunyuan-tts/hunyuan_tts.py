#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混元 Hunyuan-TTS-2.5 语音合成 API 脚本
端点: 由 HUNYUAN_API_BASE 指定 (混元内部 OpenAPI 网关, 需自行申请)
认证: Bearer token
模型: hunyuan-tts-2.5
输出: PCM 格式 (24kHz, 单声道), 自动转换为 WAV

用法:
    # 单句生成
    python hunyuan_tts.py --text "你好世界" --output /path/to/output.wav

    # 批量生成 (从 audio_prompts.json 读取)
    python hunyuan_tts.py --prompts_json /path/to/audio_prompts.json --output_dir /path/to/output/

    # 指定音色和语速
    python hunyuan_tts.py --text "你好" --voice nana --speed 1.2 --output /path/to/output.wav

    # 流式模式 (适合长文本, 减少等待)
    python hunyuan_tts.py --text "很长的文本..." --stream --output /path/to/output.wav

audio_prompts.json 格式 (与 step2_generate_audio.sh 兼容):
    {
      "cases": [
        {"id": "case_1", "tts_text": "你好世界"},
        {"id": "case_2", "tts_text": "晚上好"}
      ]
    }
"""

import argparse
import base64
import json
import os
import struct
import sys
import time
import wave
import requests
from urllib.parse import urlparse

# ============= 配置 =============
HUNYUAN_API_KEY = os.environ.get("HUNYUAN_API_KEY", "")
HUNYUAN_BASE_URL = os.environ.get("HUNYUAN_API_BASE", "")
HUNYUAN_TTS_ENDPOINT = "/openapi/v1/audio/speech"

# 默认模型
DEFAULT_MODEL = "hunyuan-tts-2.5"

# 默认音色 (精品音色列表中的一个)
DEFAULT_VOICE = "wenrouxuemei"

# PCM 参数 (混元 TTS 默认)
DEFAULT_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2  # 16-bit

# ===== 代理配置 =====
# 内网 API 调用: 不走代理
# 外网 COS 下载: 可选代理 (PROXY_HOST / PROXY_PORT / PROXY_USER / PROXY_PASS)
PROXY_USER = os.environ.get("PROXY_USER", "")
PROXY_PASS = os.environ.get("PROXY_PASS", "")
PROXY_HOST = os.environ.get("PROXY_HOST", "")
PROXY_PORT = os.environ.get("PROXY_PORT", "")
PROXY_URL = (
    f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}" if PROXY_USER
    else (f"http://{PROXY_HOST}:{PROXY_PORT}" if PROXY_HOST else "")
)
INTERNAL_DOMAINS = tuple(
    d.strip() for d in os.environ.get("IT2AV_INTERNAL_DOMAINS", "").split(",") if d.strip()
)

# Session for 内网 API (不走代理)
session = requests.Session()
session.proxies = {"http": None, "https": None}
session.trust_env = False

# Session for 外网下载 (走代理)
download_session = requests.Session()
download_session.proxies = {"http": PROXY_URL, "https": PROXY_URL}
download_session.trust_env = False


def _is_internal_url(url: str) -> bool:
    """判断是否为内网 URL (IT2AV_INTERNAL_DOMAINS / myqcloud.com / tencentcos.cn 等)"""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return (
        any(host.endswith(d) for d in INTERNAL_DOMAINS)
        or host.endswith(".myqcloud.com")
        or "tencentcos" in host
    )


def pcm_to_wav(pcm_data: bytes, save_path: str,
               sample_rate: int = DEFAULT_SAMPLE_RATE,
               channels: int = PCM_CHANNELS,
               sample_width: int = PCM_SAMPLE_WIDTH) -> str:
    """将 PCM 原始数据转换为 WAV 文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with wave.open(save_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return save_path


def call_hunyuan_tts_nonstream(
    text: str,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    speed: float = None,
    volume: float = None,
    timeout: int = 120,
) -> dict:
    """
    调用混元 TTS 非流式接口

    非流式模式: 返回 COS URL, 下载 PCM 文件后转 WAV.

    Args:
        text: 输入文本
        voice: 音色名 (如 jilingshitong, nana 等)
        model: 模型名称
        sample_rate: 输出采样率
        speed: 语速 (可选)
        volume: 音量 (可选)
        timeout: 超时时间 (秒)

    Returns:
        {"success": True, "pcm_data": bytes, "usage": {...}}
        或 {"success": False, "error": "..."}
    """
    url = HUNYUAN_BASE_URL + HUNYUAN_TTS_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HUNYUAN_API_KEY}",
    }
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "pcm",
        "stream_format": "audio",
        "sample_rate": sample_rate,
    }
    if speed is not None:
        payload["speed"] = speed
    if volume is not None:
        payload["volume"] = volume

    print(f"  🔧 Model: {model}, Voice: {voice}")
    print(f"  🔧 Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    if speed:
        print(f"  🔧 Speed: {speed}")

    t0 = time.time()
    try:
        resp = session.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"请求超时 ({timeout}s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}

    elapsed = time.time() - t0
    print(f"  📡 HTTP {resp.status_code} ({elapsed:.1f}s)")

    if resp.status_code != 200:
        return {
            "success": False,
            "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
            "status": resp.status_code,
        }

    data = resp.json()

    # 检查错误
    if "error" in data and data["error"]:
        return {
            "success": False,
            "error": f"{data['error'].get('code', 'unknown')}: {data['error'].get('message', '')}",
        }

    # 非流式: 从 audio.url 下载 PCM
    audio_url = data.get("audio", {}).get("url", "")
    if not audio_url:
        return {"success": False, "error": "响应中无 audio.url"}

    print(f"  📥 下载 PCM 音频...")

    # 直接用裸 requests.get() 下载 COS 音频 (不走 session, 避免 trust_env/proxy 设置干扰 HTTPS)
    for attempt in range(1, 4):
        try:
            dl_resp = requests.get(audio_url, timeout=60)
            if dl_resp.status_code == 200:
                pcm_data = dl_resp.content
                duration = len(pcm_data) / (sample_rate * PCM_CHANNELS * PCM_SAMPLE_WIDTH)
                print(f"  ✅ PCM 下载完成: {len(pcm_data)/1024:.1f} KB, {duration:.1f}s")
                return {
                    "success": True,
                    "pcm_data": pcm_data,
                    "duration": duration,
                    "usage": data.get("usage", {}),
                    "elapsed": elapsed,
                }
            else:
                print(f"  ❌ 下载失败: HTTP {dl_resp.status_code}")
        except Exception as e:
            print(f"  ❌ 下载失败 (尝试 {attempt}/3): {e}")
            if attempt < 3:
                time.sleep(attempt * 2)

    return {"success": False, "error": "PCM 下载最终失败, 建议使用 --stream 流式模式"}


def call_hunyuan_tts_stream(
    text: str,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    speed: float = None,
    volume: float = None,
    timeout: int = 120,
) -> dict:
    """
    调用混元 TTS 流式 (SSE) 接口

    流式模式: 逐块接收 base64 编码的 PCM 数据, 适合长文本.

    Returns:
        {"success": True, "pcm_data": bytes, "usage": {...}}
        或 {"success": False, "error": "..."}
    """
    url = HUNYUAN_BASE_URL + HUNYUAN_TTS_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HUNYUAN_API_KEY}",
    }
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "pcm",
        "stream_format": "sse",
        "sample_rate": sample_rate,
    }
    if speed is not None:
        payload["speed"] = speed
    if volume is not None:
        payload["volume"] = volume

    print(f"  🔧 Model: {model}, Voice: {voice} (流式)")
    print(f"  🔧 Text: {text[:80]}{'...' if len(text) > 80 else ''}")

    t0 = time.time()
    try:
        resp = session.post(url, headers=headers, json=payload,
                            timeout=timeout, stream=True)
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"请求超时 ({timeout}s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}

    if resp.status_code != 200:
        return {
            "success": False,
            "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
            "status": resp.status_code,
        }

    # 解析 SSE 流
    pcm_chunks = []
    usage = {}
    chunk_count = 0

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            if event_type == "speech.audio.delta":
                audio_b64 = data.get("audio", {}).get("data", "")
                if audio_b64:
                    pcm_chunks.append(base64.b64decode(audio_b64))
                    chunk_count += 1

            elif event_type == "speech.audio.done":
                usage = data.get("usage", {})

            # 检查错误
            if "error" in data and data["error"]:
                return {
                    "success": False,
                    "error": f"{data['error'].get('code', 'unknown')}: {data['error'].get('message', '')}",
                }

    elapsed = time.time() - t0

    if not pcm_chunks:
        return {"success": False, "error": "流式响应未返回任何音频数据"}

    pcm_data = b"".join(pcm_chunks)
    duration = len(pcm_data) / (sample_rate * PCM_CHANNELS * PCM_SAMPLE_WIDTH)
    print(f"  📡 流式完成: {chunk_count} chunks, {len(pcm_data)/1024:.1f} KB, "
          f"{duration:.1f}s ({elapsed:.1f}s)")

    return {
        "success": True,
        "pcm_data": pcm_data,
        "duration": duration,
        "usage": usage,
        "elapsed": elapsed,
    }


def generate_tts(
    text: str,
    save_path: str,
    voice: str = DEFAULT_VOICE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    speed: float = None,
    volume: float = None,
    stream: bool = False,
    max_retries: int = 2,
) -> bool:
    """
    生成单条 TTS 并保存为 WAV

    Args:
        text: 输入文本
        save_path: 输出 WAV 路径
        voice: 音色名
        sample_rate: 采样率
        speed: 语速
        volume: 音量
        stream: 是否使用流式接口
        max_retries: 最大重试次数

    Returns:
        True 成功, False 失败
    """
    call_fn = call_hunyuan_tts_stream if stream else call_hunyuan_tts_nonstream

    for attempt in range(1, max_retries + 1):
        result = call_fn(
            text=text,
            voice=voice,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
        )

        if result["success"]:
            pcm_to_wav(result["pcm_data"], save_path, sample_rate=sample_rate)
            duration = result.get("duration", 0)
            usage = result.get("usage", {})
            print(f"  ✅ 已保存: {save_path} ({duration:.1f}s, "
                  f"tokens: {usage.get('total_tokens', '?')})")
            return True
        else:
            print(f"  ❌ 生成失败 (尝试 {attempt}/{max_retries}): {result['error']}")
            if attempt < max_retries:
                wait = attempt * 3
                print(f"  ⏳ 等待 {wait}s 后重试...")
                time.sleep(wait)

    return False


def generate_batch(prompts_json: str, output_dir: str,
                   voice: str = DEFAULT_VOICE,
                   sample_rate: int = DEFAULT_SAMPLE_RATE,
                   speed: float = None,
                   stream: bool = False) -> list:
    """
    批量生成 TTS

    支持两种 JSON 格式:
    1. audio_prompts.json (step2 格式): {"cases": [{"id": "case_1", "tts_text": "..."}]}
    2. 简单列表: [{"id": "case_1", "text": "..."}]
    """
    with open(prompts_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容 step2 的 audio_prompts.json 格式
    if isinstance(data, dict) and "cases" in data:
        cases = data["cases"]
    elif isinstance(data, list):
        cases = data
    else:
        print(f"❌ 不支持的 JSON 格式")
        return []

    os.makedirs(output_dir, exist_ok=True)
    results = []

    total_start = time.time()
    for idx, item in enumerate(cases):
        case_id = item.get("id", f"case_{idx+1}")
        text = item.get("tts_text") or item.get("text", "")
        if not text:
            print(f"  ⚠️  [{case_id}] 无文本, 跳过")
            results.append({"id": case_id, "status": "skipped", "reason": "no text"})
            continue

        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(cases)}] {case_id}")
        print(f"{'='*70}")

        save_path = os.path.join(output_dir, case_id, "tts.wav")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
            print(f"  ⏭️  已存在: {save_path}")
            results.append({"id": case_id, "path": save_path, "status": "skipped"})
            continue

        success = generate_tts(
            text=text,
            save_path=save_path,
            voice=voice,
            sample_rate=sample_rate,
            speed=speed,
            stream=stream,
        )
        results.append({
            "id": case_id,
            "path": save_path if success else None,
            "status": "success" if success else "failed",
        })

        # API 限流保护
        time.sleep(0.5)

    total_elapsed = time.time() - total_start
    success_count = sum(1 for r in results if r["status"] == "success")
    skip_count = sum(1 for r in results if r["status"] == "skipped")
    fail_count = sum(1 for r in results if r["status"] == "failed")

    print(f"\n{'='*70}")
    print(f"✅ 批量 TTS 完成: {success_count} 成功, {skip_count} 跳过, "
          f"{fail_count} 失败, 耗时 {total_elapsed:.1f}s")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="混元 Hunyuan-TTS-2.5 语音合成")
    parser.add_argument("--text", type=str, help="输入文本 (单句模式)")
    parser.add_argument("--prompts_json", type=str, help="批量 prompts JSON 文件")
    parser.add_argument("--output", type=str, help="输出 WAV 路径 (单句模式)")
    parser.add_argument("--output_dir", type=str, help="输出目录 (批量模式)")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                        help=f"音色名 (默认: {DEFAULT_VOICE})")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"采样率 (默认: {DEFAULT_SAMPLE_RATE})")
    parser.add_argument("--speed", type=float, default=None, help="语速 (可选)")
    parser.add_argument("--volume", type=float, default=None, help="音量 (可选)")
    parser.add_argument("--stream", action="store_true", help="使用流式接口")
    args = parser.parse_args()

    print("🚀 混元 Hunyuan-TTS-2.5 语音合成")
    print(f"   域名: {HUNYUAN_BASE_URL}")
    print(f"   端点: {HUNYUAN_TTS_ENDPOINT}")
    print(f"   模型: {DEFAULT_MODEL}")
    print(f"   音色: {args.voice}")
    print(f"   采样率: {args.sample_rate}")
    print(f"   模式: {'流式' if args.stream else '非流式'}")

    if args.prompts_json:
        # 批量模式
        output_dir = args.output_dir or os.path.dirname(args.prompts_json)
        generate_batch(
            args.prompts_json, output_dir,
            voice=args.voice,
            sample_rate=args.sample_rate,
            speed=args.speed,
            stream=args.stream,
        )
    elif args.text:
        # 单句模式
        output = args.output or "hunyuan_tts_output.wav"
        print(f"\n{'='*70}")
        success = generate_tts(
            text=args.text,
            save_path=output,
            voice=args.voice,
            sample_rate=args.sample_rate,
            speed=args.speed,
            volume=args.volume,
            stream=args.stream,
        )
        if not success:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
