#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Nano-Banana-Pro (gemini-3-pro-image) 模型的文生图和图生图功能
"""

import os
import json
import base64
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime

# ================= 配置区域 =================
# 使用环境变量或直接配置 token
ENV_VENUS_OPENAPI_SECRET_ID = os.environ.get(
    'ENV_VENUS_OPENAPI_SECRET_ID', ''
)
PROXY_TOKEN = ENV_VENUS_OPENAPI_SECRET_ID + "@" + os.environ.get("VENUS_APP_GROUP_ID", "")
API_URL = os.environ.get("VENUS_API_BASE", "").rstrip("/") + "/llmproxy/chat/completions"

# 模型配置
MODEL_NAME = "gemini-3-pro-image"  # Nano-Banana-Pro 模型

# 输出目录配置
OUTPUT_DIR = os.environ.get("BANANA_OUTPUT_DIR", "./outputs/nanobanana")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= 工具函数 =================
def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    将图片编码为 base64 字符串
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        base64 编码的字符串，失败返回 None
    """
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_str = base64.b64encode(image_data).decode('utf-8')
            return base64_str
    except Exception as e:
        print(f"❌ 图片编码失败: {e}")
        return None


def call_nanobanana_api(messages: list, max_tokens: int = 2000) -> dict:
    """
    调用 Nano-Banana-Pro API
    
    Args:
        messages: 消息列表
        max_tokens: 最大输出 token 数
        
    Returns:
        API 响应结果
    """
    payload = {
        'model': MODEL_NAME,
        'messages': messages,
        'max_tokens': max_tokens
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {PROXY_TOKEN}'
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False, 
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def save_generated_image(image_data: str, output_path: str) -> bool:
    """
    保存生成的图片（如果 API 返回 base64 图片数据）
    
    Args:
        image_data: base64 编码的图片数据（可能包含 data:image 前缀）
        output_path: 输出文件路径
        
    Returns:
        是否保存成功
    """
    try:
        # 移除可能的 data:image 前缀
        if image_data.startswith('data:'):
            image_data = image_data.split(',', 1)[1]
        
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(image_data))
        
        file_size = os.path.getsize(output_path) / 1024
        print(f"✅ 图片已保存到: {output_path}")
        print(f"   文件大小: {file_size:.2f} KB")
        return True
    except Exception as e:
        print(f"❌ 图片保存失败: {e}")
        return False


def extract_and_save_images(response_data: dict, output_prefix: str = "generated") -> list:
    """
    从 API 响应中提取并保存所有图片
    
    Args:
        response_data: API 响应数据
        output_prefix: 输出文件名前缀
        
    Returns:
        保存的图片路径列表
    """
    saved_images = []
    
    try:
        message = response_data['choices'][0]['message']
        content = message.get('content', [])
        
        # 检查是否是列表格式（Venus multimodal format）
        if isinstance(content, list):
            for idx, item in enumerate(content):
                if item.get('type') == 'venus_multimodal_url':
                    # 提取图片数据
                    multimodal_data = item.get('venus_multimodal_url', {})
                    image_url = multimodal_data.get('url', '')
                    mime_type = multimodal_data.get('mimeType', 'image/png')
                    
                    if image_url.startswith('data:'):
                        # 根据 mime_type 确定文件扩展名
                        ext = 'png' if 'png' in mime_type else 'jpg'
                        
                        # 生成文件名（带时间戳）
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{output_prefix}_{timestamp}_{idx}.{ext}"
                        output_path = os.path.join(OUTPUT_DIR, filename)
                        
                        # 保存图片
                        if save_generated_image(image_url, output_path):
                            saved_images.append(output_path)
        
        # 如果是纯文本内容（非图片）
        elif isinstance(content, str):
            print(f"💬 模型返回文本内容（非图片）:\n{content[:500]}...")
            
    except Exception as e:
        print(f"⚠️  提取图片失败: {e}")
    
    return saved_images


# ================= 测试功能 =================
def test_text_to_image(prompt: str, output_filename: str = "text_to_image.jpg"):
    """
    测试文生图功能
    
    Args:
        prompt: 文本描述
        output_filename: 输出图片文件名
    """
    print("=" * 70)
    print("📝 测试 1: 文生图 (Text-to-Image)")
    print("=" * 70)
    print(f"Prompt: {prompt}\n")
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': f"请根据以下描述生成一张图片:\n{prompt}"
                }
            ]
        }
    ]
    
    print("⏳ 正在调用 API...")
    result = call_nanobanana_api(messages, max_tokens=4000)
    
    if result["success"]:
        print("✅ API 调用成功!")
        print(f"\n完整响应:\n{json.dumps(result['data'], indent=2, ensure_ascii=False)}\n")
        
        # 提取并保存图片
        saved_images = extract_and_save_images(result['data'], "text_to_image")
        
        if saved_images:
            print(f"\n🎉 成功生成 {len(saved_images)} 张图片!")
            for img_path in saved_images:
                print(f"   📁 {img_path}")
        else:
            print("\n⚠️  未检测到图片数据")
    else:
        print(f"❌ API 调用失败: {result['error']}")


def test_image_to_image(input_image_path: str, prompt: str, output_filename: str = "image_to_image.jpg"):
    """
    测试图生图功能
    
    Args:
        input_image_path: 输入图片路径
        prompt: 修改描述
        output_filename: 输出图片文件名
    """
    print("\n" + "=" * 70)
    print("🖼️  测试 2: 图生图 (Image-to-Image)")
    print("=" * 70)
    print(f"输入图片: {input_image_path}")
    print(f"Prompt: {prompt}\n")
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # 检查图片是否存在
    if not os.path.exists(input_image_path):
        print(f"❌ 图片文件不存在: {input_image_path}")
        return
    
    # 编码图片为 base64
    base64_image = encode_image_to_base64(input_image_path)
    if not base64_image:
        return
    
    # 构建消息（multimodal format）
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{base64_image}'
                    }
                },
                {
                    'type': 'text',
                    'text': f"请根据这张图片和以下描述生成新的图片:\n{prompt}"
                }
            ]
        }
    ]
    
    print("⏳ 正在调用 API...")
    result = call_nanobanana_api(messages, max_tokens=4000)
    
    if result["success"]:
        print("✅ API 调用成功!")
        print(f"\n完整响应:\n{json.dumps(result['data'], indent=2, ensure_ascii=False)}\n")
        
        # 提取并保存图片
        saved_images = extract_and_save_images(result['data'], "image_to_image")
        
        if saved_images:
            print(f"\n🎉 成功生成 {len(saved_images)} 张图片!")
            for img_path in saved_images:
                print(f"   📁 {img_path}")
        else:
            print("\n⚠️  未检测到图片数据")
    else:
        print(f"❌ API 调用失败: {result['error']}")


def test_image_understanding(image_path: str):
    """
    测试图像理解功能（作为对比测试）
    
    Args:
        image_path: 图片路径
    """
    print("\n" + "=" * 70)
    print("👁️  测试 3: 图像理解 (Image Understanding)")
    print("=" * 70)
    print(f"输入图片: {image_path}\n")
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return
    
    # 编码图片为 base64
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return
    
    # 构建消息
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{base64_image}'
                    }
                },
                {
                    'type': 'text',
                    'text': "请详细描述这张图片的内容，包括主要元素、颜色、风格等。"
                }
            ]
        }
    ]
    
    print("⏳ 正在调用 API...")
    result = call_nanobanana_api(messages, max_tokens=1000)
    
    if result["success"]:
        print("✅ API 调用成功!")
        try:
            content = result['data']['choices'][0]['message']['content']
            print(f"\n📝 图像描述:\n{content}\n")
        except Exception as e:
            print(f"⚠️  解析响应失败: {e}")
            print(f"\n完整响应:\n{json.dumps(result['data'], indent=2, ensure_ascii=False)}\n")
    else:
        print(f"❌ API 调用失败: {result['error']}")


# ================= 主函数 =================
def main():
    print("🚀 Nano-Banana-Pro (gemini-3-pro-image) 功能测试")
    print(f"🔑 Token: {PROXY_TOKEN[:20]}...{PROXY_TOKEN[-10:]}")
    print(f"🤖 模型: {MODEL_NAME}")
    print(f"🌐 API URL: {API_URL}")
    print(f"📁 输出目录: {OUTPUT_DIR}\n")
    
    # 测试 1: 文生图
    text_prompt = "一只可爱的橙色猫咪在阳光下睡觉，背景是温馨的家居环境，高清摄影风格"
    test_text_to_image(text_prompt, "text_to_image_cat.jpg")
    
    # 测试 2: 图生图（需要提供一张测试图片）
    # 这里需要你提供一张测试图片的路径
    test_image_path = "test_image.jpg"  # 请替换为实际的图片路径
    
    if os.path.exists(test_image_path):
        image_prompt = "将这张图片转换为水彩画风格，保持主体不变"
        test_image_to_image(test_image_path, image_prompt, "image_to_image_watercolor.jpg")
    else:
        print(f"\n⚠️  跳过图生图测试: 找不到测试图片 {test_image_path}")
        print(f"    如需测试，请将图片放置到此路径或修改 test_image_path 变量\n")
    
    # 测试 3: 图像理解（作为对比）
    if os.path.exists(test_image_path):
        test_image_understanding(test_image_path)
    
    print("\n" + "=" * 70)
    print("✅ 测试完成!")
    print("=" * 70)
    print(f"\n📁 生成的图片保存在: {OUTPUT_DIR}")
    print("\n💡 提示:")
    print("   - 如果 API 不支持图像生成，可能会返回文本描述或错误信息")
    print("   - 请根据实际返回格式调整代码中的图片数据提取逻辑")
    print("   - 建议查看 Venus 平台文档确认 gemini-3-pro-image 的具体功能")
    print("   - 请参考你所使用的 API 网关文档确认返回格式\n")


if __name__ == "__main__":
    main()
