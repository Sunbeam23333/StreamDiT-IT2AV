#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混元 HY-Image-v3.0-I2I 图生图 API 本地调试脚本
端点: 由 HUNYUAN_API_BASE 指定 (混元内部 OpenAPI 网关, 需自行申请)
认证: Bearer token
"""

import os
import sys
import json
import time
import base64
import struct
import zlib
import requests

# ============= 配置 =============
HUNYUAN_API_KEY = os.environ.get("HUNYUAN_API_KEY", "")
HUNYUAN_BASE_URL = os.environ.get("HUNYUAN_API_BASE", "")
HUNYUAN_I2I_ENDPOINT = "/openapi/v1/images/ar/generations"

# 绕过代理
session = requests.Session()
session.proxies = {"http": None, "https": None}
session.trust_env = False

OUTPUT_DIR = os.environ.get("HY_IMAGE_OUTPUT_DIR", "./outputs/hy_image")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_test_png(w=256, h=256):
    """创建一张简单的测试PNG图片（渐变色）"""
    try:
        from PIL import Image
        img = Image.new('RGB', (w, h))
        for x in range(w):
            for y in range(h):
                img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
        import io
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except ImportError:
        # Minimal PNG fallback
        def chunk(ctype, data):
            c = ctype + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        raw = b''
        for y in range(h):
            raw += b'\x00'
            for x in range(w):
                raw += bytes([x % 256, y % 256, (x + y) % 256])
        return (b'\x89PNG\r\n\x1a\n' +
                chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
                chunk(b'IDAT', zlib.compress(raw)) +
                chunk(b'IEND', b''))


def call_hunyuan_i2i(prompt, images_b64=None, image_urls=None, size=None, seed=None):
    """调用混元 HY-Image-v3.0-I2I API"""
    url = HUNYUAN_BASE_URL + HUNYUAN_I2I_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HUNYUAN_API_KEY}",
    }
    payload = {
        "model": "HY-Image-v3.0-I2I-ToB-v1.0.0",
        "prompt": prompt,
    }
    if images_b64:
        payload["images"] = images_b64[:3]
    if image_urls:
        payload["image_urls"] = image_urls[:3]
    if size:
        payload["size"] = size
    if seed and seed > 0:
        payload["seed"] = seed

    print(f"🔧 URL: {url}")
    print(f"🔧 Model: {payload['model']}")
    print(f"🔧 Prompt: {prompt[:80]}")
    if images_b64:
        print(f"🔧 Images(base64): {len(images_b64)}张")
    if image_urls:
        print(f"🔧 Image URLs: {len(image_urls)}张")

    resp = session.post(url, headers=headers, json=payload, timeout=120)
    print(f"📡 HTTP {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ 成功！")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:500])
        return {"success": True, "data": data}
    else:
        print(f"❌ 失败: {resp.text[:500]}")
        return {"success": False, "error": resp.text[:500], "status": resp.status_code}


def download_image(url, save_path):
    """下载图片"""
    try:
        resp = session.get(url, timeout=60)
        if resp.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(resp.content)
            print(f"📥 已保存: {save_path} ({len(resp.content)} bytes)")
            return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
    return False


def main():
    print("🚀 混元 HY-Image-v3.0-I2I 图生图 API 调试")
    print(f"   域名: {HUNYUAN_BASE_URL}")
    print(f"   端点: {HUNYUAN_I2I_ENDPOINT}")

    # 创建测试图片
    print("\n📷 创建测试图片...")
    test_png = create_test_png(256, 256)
    b64_str = base64.b64encode(test_png).decode()
    print(f"   图片大小: {len(test_png)} bytes, Base64长度: {len(b64_str)}")

    # 保存测试图片
    test_img_path = os.path.join(OUTPUT_DIR, "test_input.png")
    with open(test_img_path, "wb") as f:
        f.write(test_png)
    print(f"   保存: {test_img_path}")

    # ====== 测试1: 单图 base64 ======
    print("\n" + "=" * 60)
    print("测试1: 单张base64图片 + prompt")
    print("=" * 60)
    
    result1 = call_hunyuan_i2i(
        prompt="把这张图片变成一幅水彩风格的美丽风景画",
        images_b64=[b64_str],
        size="1024x1024"
    )

    if result1["success"] and "data" in result1["data"]:
        for i, item in enumerate(result1["data"]["data"]):
            if item.get("url"):
                save_path = os.path.join(OUTPUT_DIR, f"hunyuan_test1_{i}.png")
                download_image(item["url"], save_path)
            if item.get("revised_prompt"):
                print(f"📝 改写: {item['revised_prompt'][:200]}")

    print("\n" + "=" * 60)
    print("🏁 调试完成！")
    return result1


if __name__ == "__main__":
    main()
