#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone TTS (Text-to-Speech) script using Hunyuan TTS API.

Usage:
    # Basic usage - generate PCM audio
    python tools/tts_generate.py "你好，欢迎来到混元世界"

    # Specify output file
    python tools/tts_generate.py "你好" -o output.wav

    # Change voice and speed
    python tools/tts_generate.py "你好" --voice qinhenvsheng --speed 1.2

    # Output as WAV format (auto-convert from PCM)
    python tools/tts_generate.py "你好" -o output.wav --format wav

    # Output as MP3 format
    python tools/tts_generate.py "你好" -o output.mp3 --format mp3

    # List available voices
    python tools/tts_generate.py --list-voices
"""

import sys
import os
import json
import time
import base64
import argparse
import struct
import subprocess
import requests
from pathlib import Path
from datetime import datetime

# ========== Configuration ==========
API_KEY = os.environ.get("HUNYUAN_API_KEY", "")
API_BASE = os.environ.get("HUNYUAN_API_BASE", "")
TTS_URL = f"{API_BASE}/openapi/v1/audio/speech"

# Default TTS parameters (aligned with project config in config/models.py)
DEFAULT_MODEL = "hunyuan-tts-2.5"
DEFAULT_VOICE = "qinhenvsheng"
DEFAULT_SPEED = 1.25
DEFAULT_RESPONSE_FORMAT = "pcm"
DEFAULT_STREAM_FORMAT = "audio"

# PCM audio parameters (from project's audio_producer.py)
PCM_SAMPLE_RATE = 24000   # 24kHz
PCM_CHANNELS = 1          # Mono
PCM_SAMPLE_WIDTH = 2      # 16-bit (2 bytes per sample)

# Available voices
AVAILABLE_VOICES = {
    "qinhenvsheng": "沁和女声 - 温柔知性女声",
    "jilingshitong": "机灵石头 - 活泼男声",
    "cuicuinvsheng": "翠翠女声 - 甜美女声",
    "wanquannansheng": "万全男声 - 成熟男声",
}

OUTPUT_DIR = Path("data/output/tts")


def make_tts_request(text: str, model: str, voice: str, speed: float,
                     response_format: str, stream_format: str) -> dict:
    """
    Call Hunyuan TTS API.

    Args:
        text: Text to convert to speech
        model: TTS model name
        voice: Voice ID
        speed: Speech speed (0.5 - 2.0)
        response_format: Audio format (pcm, mp3, etc.)
        stream_format: Stream format

    Returns:
        API response dict
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "stream_format": stream_format,
        "speed": speed,
    }

    print(f"📡 Calling TTS API...")
    print(f"   Model: {model}")
    print(f"   Voice: {voice}")
    print(f"   Speed: {speed}")
    print(f"   Text: {text[:80]}{'...' if len(text) > 80 else ''}")

    response = requests.post(TTS_URL, json=payload, headers=headers, timeout=120)

    if response.status_code != 200:
        print(f"❌ API error: HTTP {response.status_code}")
        print(f"   Response: {response.text[:500]}")
        sys.exit(1)

    result = response.json()
    return result


def save_audio_from_response(response: dict, output_path: Path) -> Path:
    """
    Save audio data from TTS API response.

    Handles three response formats:
    1. response['audio']['url'] - URL to download
    2. response['url'] - direct URL
    3. response['audio'] (base64 string) - base64 encoded audio

    Args:
        response: API response dict
        output_path: Output file path

    Returns:
        Actual saved file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if 'audio' in response and isinstance(response['audio'], dict) and 'url' in response['audio']:
        url = response['audio']['url']
        print(f"📥 Downloading audio from URL...")
        dl = requests.get(url, timeout=120)
        dl.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(dl.content)

    elif 'url' in response:
        url = response['url']
        print(f"📥 Downloading audio from URL...")
        dl = requests.get(url, timeout=120)
        dl.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(dl.content)

    elif 'audio' in response and isinstance(response['audio'], str):
        print(f"📥 Decoding base64 audio...")
        audio_data = base64.b64decode(response['audio'])
        with open(output_path, 'wb') as f:
            f.write(audio_data)

    else:
        print(f"❌ Unexpected response format. Keys: {list(response.keys())}")
        print(f"   Response: {json.dumps(response, ensure_ascii=False)[:500]}")
        sys.exit(1)

    return output_path


def pcm_to_wav(pcm_path: Path, wav_path: Path,
               sample_rate: int = PCM_SAMPLE_RATE,
               channels: int = PCM_CHANNELS,
               sample_width: int = PCM_SAMPLE_WIDTH) -> Path:
    """
    Convert raw PCM file to WAV format by adding WAV header.

    Args:
        pcm_path: Input PCM file
        wav_path: Output WAV file
        sample_rate: Sample rate in Hz
        channels: Number of channels
        sample_width: Bytes per sample

    Returns:
        Output WAV path
    """
    pcm_data = pcm_path.read_bytes()
    data_size = len(pcm_data)

    # WAV header (44 bytes)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with open(wav_path, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))  # File size - 8
        f.write(b'WAVE')

        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))               # Chunk size
        f.write(struct.pack('<H', 1))                 # PCM format
        f.write(struct.pack('<H', channels))          # Channels
        f.write(struct.pack('<I', sample_rate))       # Sample rate
        byte_rate = sample_rate * channels * sample_width
        f.write(struct.pack('<I', byte_rate))         # Byte rate
        block_align = channels * sample_width
        f.write(struct.pack('<H', block_align))       # Block align
        f.write(struct.pack('<H', sample_width * 8))  # Bits per sample

        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(pcm_data)

    return wav_path


def convert_to_mp3(wav_path: Path, mp3_path: Path) -> Path:
    """
    Convert WAV to MP3 using ffmpeg.

    Args:
        wav_path: Input WAV file
        mp3_path: Output MP3 file

    Returns:
        Output MP3 path
    """
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-codec:a", "libmp3lame",
        "-b:a", "192k",
        str(mp3_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️  ffmpeg error: {result.stderr[:300]}")
        print(f"   WAV file kept at: {wav_path}")
        return wav_path
    return mp3_path


def measure_pcm_duration(pcm_path: Path) -> float:
    """Calculate PCM audio duration in seconds."""
    file_size = pcm_path.stat().st_size
    bytes_per_second = PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_SAMPLE_WIDTH
    return file_size / bytes_per_second if bytes_per_second > 0 else 0.0


def list_voices():
    """Print available voice options."""
    print("\n🎤 Available voices:")
    print("-" * 50)
    for voice_id, description in AVAILABLE_VOICES.items():
        marker = " ⭐" if voice_id == DEFAULT_VOICE else ""
        print(f"  {voice_id:<20s} {description}{marker}")
    print("-" * 50)
    print(f"  Default: {DEFAULT_VOICE}")
    print(f"\n  Note: More voices may be available. Check Hunyuan TTS docs.")


def main():
    parser = argparse.ArgumentParser(
        description="Hunyuan TTS - Convert text to speech audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/tts_generate.py "你好，欢迎来到混元世界"
  python tools/tts_generate.py "Hello world" -o hello.wav --format wav
  python tools/tts_generate.py "测试语音" --voice qinhenvsheng --speed 1.0
  python tools/tts_generate.py --list-voices
        """
    )

    parser.add_argument(
        "text",
        nargs="?",
        type=str,
        help="Text to convert to speech"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: data/output/tts/tts_<timestamp>.<format>)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pcm", "wav", "mp3"],
        default="wav",
        help="Output audio format (default: wav)"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Voice ID (default: {DEFAULT_VOICE})"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=f"Speech speed 0.5~2.0 (default: {DEFAULT_SPEED})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"TTS model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voice options and exit"
    )

    args = parser.parse_args()

    # List voices mode
    if args.list_voices:
        list_voices()
        return

    # Validate text input
    if not args.text:
        parser.print_help()
        print("\n❌ Please provide text to convert.")
        sys.exit(1)

    text = args.text
    output_format = args.format

    # Determine output path
    if args.output:
        output_path = Path(args.output)
        # Infer format from extension if not explicitly set
        ext = output_path.suffix.lower()
        if ext in ('.wav', '.mp3', '.pcm') and args.format == "wav":
            output_format = ext.lstrip('.')
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"tts_{timestamp}.{output_format}"

    print("=" * 60)
    print("🔊 Hunyuan TTS - Text to Speech")
    print("=" * 60)
    print(f"   Text: {text}")
    print(f"   Voice: {args.voice}")
    print(f"   Speed: {args.speed}")
    print(f"   Format: {output_format}")
    print(f"   Output: {output_path}")
    print("=" * 60)

    # Step 1: Call TTS API
    start_time = time.time()
    response = make_tts_request(
        text=text,
        model=args.model,
        voice=args.voice,
        speed=args.speed,
        response_format=DEFAULT_RESPONSE_FORMAT,
        stream_format=DEFAULT_STREAM_FORMAT,
    )

    # Step 2: Save raw PCM audio
    pcm_path = output_path.with_suffix('.pcm')
    save_audio_from_response(response, pcm_path)

    duration = measure_pcm_duration(pcm_path)
    file_size_kb = pcm_path.stat().st_size / 1024
    elapsed = time.time() - start_time

    print(f"\n✅ PCM audio saved: {pcm_path}")
    print(f"   Size: {file_size_kb:.1f} KB")
    print(f"   Duration: {duration:.2f}s")
    print(f"   API time: {elapsed:.2f}s")

    # Step 3: Convert format if needed
    final_path = pcm_path
    if output_format == "wav":
        wav_path = output_path.with_suffix('.wav')
        final_path = pcm_to_wav(pcm_path, wav_path)
        print(f"✅ Converted to WAV: {final_path}")
        # Clean up PCM if WAV created successfully
        if final_path != pcm_path and final_path.exists():
            pcm_path.unlink()

    elif output_format == "mp3":
        wav_path = output_path.with_suffix('.wav')
        pcm_to_wav(pcm_path, wav_path)
        mp3_path = output_path.with_suffix('.mp3')
        final_path = convert_to_mp3(wav_path, mp3_path)
        print(f"✅ Converted to MP3: {final_path}")
        # Clean up intermediate files
        if final_path.suffix == '.mp3':
            if wav_path.exists():
                wav_path.unlink()
            if pcm_path.exists():
                pcm_path.unlink()

    print(f"\n🎉 Done! Output: {final_path}")
    print(f"   Duration: {duration:.2f}s | Size: {final_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
