#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混元 HY-Image-v3.0 文生图 (T2I) API 脚本
端点: 由 HUNYUAN_API_BASE 指定 (混元内部 OpenAPI 网关, 需自行申请)
认证: Bearer token
模型: hunyuan-image-v3.0-v1.0.4 (推荐) / hunyuan-image-v3.0-v1.0.1

用法:
    # 单张生成
    python test_hy_t2i.py --prompt "一个美女在温暖的卧室" --output /path/to/output.png

    # 批量生成 (从 JSON 文件读取)
    python test_hy_t2i.py --prompts_json /path/to/prompts.json --output_dir /path/to/output/

    # 指定尺寸 (默认 1024x1024)
    python test_hy_t2i.py --prompt "..." --size 768x1280 --output /path/to/output.png

JSON 格式:
    [
      {"id": "case_1", "prompt": "...", "size": "768x1280"},
      {"id": "case_2", "prompt": "...", "size": "1024x1024"}
    ]
"""

import argparse
import json
import os
import sys
import time
import requests
from pathlib import Path
from urllib.parse import urlparse

# ============= 配置 =============
HUNYUAN_API_KEY = os.environ.get("HUNYUAN_API_KEY", "")
HUNYUAN_BASE_URL = os.environ.get("HUNYUAN_API_BASE", "")
HUNYUAN_T2I_ENDPOINT = "/openapi/v1/images/ar/generations"

# 默认模型 (推荐 v1.0.4, 尺寸约束更灵活)
DEFAULT_MODEL = "hunyuan-image-v3.0-v1.0.4"

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


def call_hunyuan_t2i(
    prompt: str,
    size: str = "1024x1024",
    model: str = DEFAULT_MODEL,
    seed: int = 0,
    revise: bool = True,
    enable_thinking: bool = False,
    timeout: int = 180,
) -> dict:
    """
    调用混元 HY-Image-v3.0 文生图 API

    Args:
        prompt: 文本描述 (最长 8192 字符)
        size: 图片尺寸, 格式 "宽x高", 如 "1024x1024", "768x1280"
        model: 模型版本
        seed: 随机种子 (0=随机)
        revise: 是否开启 prompt 改写 (增加 ~30s 耗时，但效果更好)
        enable_thinking: 是否开启 thinking 模式 (增加耗时但效果更好)
        timeout: 超时时间 (秒)

    Returns:
        {"success": True, "data": {...}, "url": "...", "revised_prompt": "..."}
        或 {"success": False, "error": "..."}
    """
    url = HUNYUAN_BASE_URL + HUNYUAN_T2I_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HUNYUAN_API_KEY}",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "revise": {"value": revise},
        "enable_thinking": {"value": enable_thinking},
    }
    if seed > 0:
        payload["seed"] = seed

    print(f"  🔧 Model: {model}")
    print(f"  🔧 Size: {size}")
    print(f"  🔧 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"  🔧 Revise: {revise}, Thinking: {enable_thinking}")

    t0 = time.time()
    try:
        resp = session.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"请求超时 ({timeout}s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}

    elapsed = time.time() - t0
    print(f"  📡 HTTP {resp.status_code} ({elapsed:.1f}s)")

    if resp.status_code == 200:
        data = resp.json()
        result = {"success": True, "data": data, "elapsed": elapsed}

        # 提取 URL 和 revised_prompt
        if "data" in data and len(data["data"]) > 0:
            result["url"] = data["data"][0].get("url", "")
            result["revised_prompt"] = data["data"][0].get("revised_prompt", "")

        return result
    else:
        return {
            "success": False,
            "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
            "status": resp.status_code,
        }


def _is_external_url(url: str) -> bool:
    """判断是否为外网 URL (不在 IT2AV_INTERNAL_DOMAINS 中)"""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return not any(host.endswith(d) for d in INTERNAL_DOMAINS)


def download_image(url: str, save_path: str, max_retries: int = 3) -> bool:
    """下载图片到本地, 外网 URL 走代理, 内网 URL 直连, 带重试"""
    # 根据 URL 选择 session
    if _is_external_url(url):
        dl_session = download_session
        print(f"  🌐 外网 URL, 使用代理下载: {PROXY_HOST}:{PROXY_PORT}")
    else:
        dl_session = session
        print(f"  🏠 内网 URL, 直连下载")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  📥 下载尝试 {attempt}/{max_retries}...")
            resp = dl_session.get(url, timeout=120)
            if resp.status_code == 200:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(resp.content)
                size_kb = len(resp.content) / 1024
                print(f"  ✅ 已保存: {save_path} ({size_kb:.1f} KB)")
                return True
            else:
                print(f"  ❌ 下载失败: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  ❌ 下载失败 (尝试 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                wait = attempt * 5
                print(f"  ⏳ 等待 {wait}s 后重试...")
                time.sleep(wait)

    print(f"  ❌ 下载最终失败, 已重试 {max_retries} 次")
    return False


def generate_single(prompt: str, output_path: str, size: str = "1024x1024",
                    seed: int = 0, revise: bool = True) -> bool:
    """生成单张图片"""
    result = call_hunyuan_t2i(prompt, size=size, seed=seed, revise=revise)

    if result["success"] and result.get("url"):
        if result.get("revised_prompt"):
            print(f"  📝 改写 prompt: {result['revised_prompt'][:150]}...")
        return download_image(result["url"], output_path)
    else:
        print(f"  ❌ 生成失败: {result.get('error', 'Unknown error')}")
        return False


def generate_batch(prompts_json: str, output_dir: str, default_size: str = "1024x1024",
                   revise: bool = True) -> list:
    """批量生成图片"""
    with open(prompts_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    results = []

    total_start = time.time()
    for idx, item in enumerate(prompts):
        prompt = item["prompt"]
        case_id = item.get("id", f"case_{idx}")
        size = item.get("size", default_size)

        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(prompts)}] {case_id}")
        print(f"{'='*70}")

        output_path = os.path.join(output_dir, f"{case_id}.png")

        if os.path.exists(output_path):
            print(f"  ⏭️  已存在: {output_path}")
            results.append({"id": case_id, "path": output_path, "status": "skipped"})
            continue

        success = generate_single(prompt, output_path, size=size, revise=revise)
        results.append({
            "id": case_id,
            "path": output_path if success else None,
            "status": "success" if success else "failed",
        })

        # API 限流保护
        time.sleep(1)

    total_elapsed = time.time() - total_start
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n{'='*70}")
    print(f"✅ 批量生成完成: {success_count}/{len(prompts)} 成功, 耗时 {total_elapsed:.1f}s")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="混元 HY-Image-v3.0 文生图")
    parser.add_argument("--prompt", type=str, help="文本描述 (单张模式)")
    parser.add_argument("--prompts_json", type=str, help="批量 prompts JSON 文件")
    parser.add_argument("--output", type=str, help="输出图片路径 (单张模式)")
    parser.add_argument("--output_dir", type=str, help="输出目录 (批量模式)")
    parser.add_argument("--size", type=str, default="1024x1024", help="图片尺寸 (默认 1024x1024)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子 (0=随机)")
    parser.add_argument("--no-revise", action="store_true", help="关闭 prompt 改写")
    parser.add_argument("--thinking", action="store_true", help="开启 thinking 模式")
    args = parser.parse_args()

    print("🚀 混元 HY-Image-v3.0 文生图 (T2I)")
    print(f"   域名: {HUNYUAN_BASE_URL}")
    print(f"   端点: {HUNYUAN_T2I_ENDPOINT}")
    print(f"   模型: {DEFAULT_MODEL}")

    if args.prompts_json:
        # 批量模式
        output_dir = args.output_dir or os.path.dirname(args.prompts_json)
        generate_batch(args.prompts_json, output_dir, default_size=args.size,
                       revise=not args.no_revise)
    elif args.prompt:
        # 单张模式
        output = args.output or "hunyuan_t2i_output.png"
        print(f"\n{'='*70}")
        generate_single(args.prompt, output, size=args.size, seed=args.seed,
                        revise=not args.no_revise)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
