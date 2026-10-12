#!/usr/bin/env python3
"""
Text-to-Image generation using HuggingFace Diffusers (单卡推理).

支持模型: FLUX.1-schnell, FLUX.1-dev, SD3.5-Large, PixArt-Sigma
输出分辨率: 适配 LightX2V Wan2.2 MoE i2v/flf2v (720x1280 或 1280x720)

用法:
    # FLUX.1 schnell (最快, 4步)
    python run_t2i_diffusers.py --model flux_schnell --prompt "A cat on a beach" --output output.png

    # FLUX.1 dev (最高质量, 28步)
    python run_t2i_diffusers.py --model flux_dev --prompt "A cat on a beach" --output output.png

    # SD3.5 Large
    python run_t2i_diffusers.py --model sd35 --prompt "A cat on a beach" --output output.png

    # PixArt-Sigma (轻量)
    python run_t2i_diffusers.py --model pixart_sigma --prompt "A cat on a beach" --output output.png

    # 批量生成 (用prompts文件, 每行一个prompt)
    python run_t2i_diffusers.py --model flux_schnell --prompt_file prompts.txt --output_dir ./t2i_outputs/

    # 生成首帧+末帧 (用于 flf2v)
    python run_t2i_diffusers.py --model flux_schnell --mode flf2v \\
        --prompt "A cat sitting on a sunny beach" \\
        --last_prompt "The cat walking into the ocean waves" \\
        --output_dir ./flf2v_frames/

注意: 模型需先通过 fast_sync.py 同步到 /tmp 本地盘再推理
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# ============================================================================
# 路径配置
# ============================================================================
GROUP_BASE = os.path.expanduser(os.environ.get("IT2AV_WORKSPACE", "~/it2av_workspace"))
GROUP_MODELS = f"{GROUP_BASE}/models/Image"
LOCAL_BASE = os.environ.get("IT2AV_LOCAL_ROOT", "/tmp/it2av")
LOCAL_MODELS = f"{LOCAL_BASE}/models/Image"
SYNC_SCRIPT = f"{GROUP_BASE}/code/fast_sync.py"

# ============================================================================
# 模型配置
# ============================================================================
MODEL_CONFIGS = {
    "flux_schnell": {
        "repo_name": "FLUX.1-schnell",
        "pipeline_cls": "FluxPipeline",
        "dtype": torch.bfloat16,
        "default_steps": 4,
        "guidance_scale": 0.0,  # schnell 不使用 classifier-free guidance
        "max_sequence_length": 256,
        "default_height": 720,
        "default_width": 1280,
    },
    "flux_dev": {
        "repo_name": "FLUX.1-dev",
        "pipeline_cls": "FluxPipeline",
        "dtype": torch.bfloat16,
        "default_steps": 28,
        "guidance_scale": 3.5,
        "max_sequence_length": 512,
        "default_height": 720,
        "default_width": 1280,
    },
    "sd35": {
        "repo_name": "stable-diffusion-3.5-large",
        "pipeline_cls": "StableDiffusion3Pipeline",
        "dtype": torch.bfloat16,
        "default_steps": 28,
        "guidance_scale": 4.5,
        "max_sequence_length": 512,
        "default_height": 720,
        "default_width": 1280,
    },
    "pixart_sigma": {
        "repo_name": "PixArt-Sigma-XL-2-1024-MS",
        "pipeline_cls": "PixArtSigmaPipeline",
        "dtype": torch.float16,
        "default_steps": 20,
        "guidance_scale": 4.5,
        "max_sequence_length": 300,
        "default_height": 720,
        "default_width": 1280,
    },
}

# LightX2V Wan2.2 MoE 支持的分辨率 buckets
RESOLUTION_PRESETS = {
    "720p_landscape": (720, 1280),   # Wan2.2 MoE i2v/flf2v 默认
    "720p_portrait": (1280, 720),
    "720p_square": (960, 960),
    "480p_landscape": (480, 832),    # Wan2.1 i2v 默认
    "480p_portrait": (832, 480),
    "1024_square": (1024, 1024),     # FLUX/SD3 标准
}


def sync_model_to_local(model_name: str) -> str:
    """用 fast_sync.py 将模型从 group 盘同步到本地 /tmp 盘."""
    src = os.path.join(GROUP_MODELS, model_name)
    dst = os.path.join(LOCAL_MODELS, model_name)
    sync_marker = os.path.join(dst, ".sync_done")

    if os.path.exists(sync_marker):
        logging.info("Model already cached at %s, skipping sync.", dst)
        return dst

    if not os.path.isdir(src):
        logging.error("Source model not found: %s", src)
        logging.error("Please run download_t2i_models.py first!")
        sys.exit(1)

    logging.info("Syncing model %s -> %s ...", src, dst)
    os.makedirs(dst, exist_ok=True)

    cmd = [
        sys.executable, SYNC_SCRIPT,
        src, dst,
        "--workers", str(os.environ.get("SYNC_WORKERS", "32")),
        "--skip-existing",
    ]
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        logging.error("Sync failed with return code %d", ret.returncode)
        sys.exit(1)

    # 标记同步完成
    Path(sync_marker).touch()
    logging.info("Model sync complete: %s", dst)
    return dst


def load_pipeline(model_key: str, device: str = "cuda:0"):
    """加载指定的文生图 Pipeline."""
    config = MODEL_CONFIGS[model_key]
    model_dir = sync_model_to_local(config["repo_name"])

    logging.info("Loading pipeline: %s from %s", config["pipeline_cls"], model_dir)

    if config["pipeline_cls"] == "FluxPipeline":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            model_dir,
            torch_dtype=config["dtype"],
        )
    elif config["pipeline_cls"] == "StableDiffusion3Pipeline":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_dir,
            torch_dtype=config["dtype"],
        )
    elif config["pipeline_cls"] == "PixArtSigmaPipeline":
        from diffusers import PixArtSigmaPipeline
        pipe = PixArtSigmaPipeline.from_pretrained(
            model_dir,
            torch_dtype=config["dtype"],
        )
    else:
        raise ValueError(f"Unknown pipeline class: {config['pipeline_cls']}")

    # H20 95GB 显存充足，直接加载到 GPU
    pipe = pipe.to(device)

    # 启用内存优化
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing()

    logging.info("Pipeline loaded. Peak memory: %.2f GB",
                 torch.cuda.max_memory_allocated(device) / 1e9)
    return pipe, config


def generate_image(
    pipe,
    config: dict,
    prompt: str,
    height: int,
    width: int,
    seed: int = 42,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
) -> Image.Image:
    """生成单张图片."""
    steps = num_inference_steps or config["default_steps"]
    gs = guidance_scale if guidance_scale is not None else config["guidance_scale"]

    gen_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": gs,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
    }

    # FLUX 和 SD3 支持 max_sequence_length
    if "max_sequence_length" in config:
        gen_kwargs["max_sequence_length"] = config["max_sequence_length"]

    t0 = time.time()
    with torch.inference_mode():
        output = pipe(**gen_kwargs)
    elapsed = time.time() - t0

    image = output.images[0]
    logging.info("Generated %dx%d image in %.2fs (steps=%d, guidance=%.1f, seed=%d)",
                 width, height, elapsed, steps, gs, seed)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-to-Image Generation with Diffusers (Single GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use")
    parser.add_argument("--mode", default="i2v", choices=["i2v", "flf2v"],
                        help="i2v: single image; flf2v: first+last frame pair")

    # Prompt 配置
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--last_prompt", type=str,
                        help="Prompt for last frame (flf2v mode)")
    parser.add_argument("--prompt_file", type=str,
                        help="File with prompts (one per line, or JSON)")

    # 分辨率
    parser.add_argument("--height", type=int, default=None,
                        help="Image height (default: 720 for LightX2V compatibility)")
    parser.add_argument("--width", type=int, default=None,
                        help="Image width (default: 1280 for LightX2V compatibility)")
    parser.add_argument("--resolution", type=str, default=None,
                        choices=list(RESOLUTION_PRESETS.keys()),
                        help="Use preset resolution")

    # 生成参数
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of inference steps (default depends on model)")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Guidance scale (default depends on model)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_images", type=int, default=1,
                        help="Number of images per prompt")

    # 输出
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (single prompt mode)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (batch mode or flf2v mode)")

    # 设备
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 确定分辨率
    if args.resolution:
        height, width = RESOLUTION_PRESETS[args.resolution]
    elif args.height and args.width:
        height, width = args.height, args.width
    else:
        config = MODEL_CONFIGS[args.model]
        height, width = config["default_height"], config["default_width"]

    logging.info("=" * 60)
    logging.info("Text-to-Image Generation (Diffusers)")
    logging.info("  Model:      %s", args.model)
    logging.info("  Mode:       %s", args.mode)
    logging.info("  Resolution: %dx%d", width, height)
    logging.info("  Device:     %s", args.device)
    logging.info("=" * 60)

    # 加载模型
    pipe, config = load_pipeline(args.model, args.device)

    # 收集 prompts
    prompts = []
    last_prompts = []

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    prompts.append(line)
        logging.info("Loaded %d prompts from %s", len(prompts), args.prompt_file)
    elif args.prompt:
        prompts = [args.prompt]
    else:
        logging.error("Please provide --prompt or --prompt_file")
        sys.exit(1)

    # flf2v 模式的 last frame prompts
    if args.mode == "flf2v":
        if args.last_prompt:
            last_prompts = [args.last_prompt] * len(prompts)
        else:
            # 默认策略: 对每个 prompt 添加时间变化描述来生成末帧
            last_prompts = [
                f"Later in time: {p}. The scene has evolved and changed."
                for p in prompts
            ]

    # 确定输出路径
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    elif args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    else:
        args.output_dir = f"./t2i_outputs/{args.model}"
        os.makedirs(args.output_dir, exist_ok=True)

    total_start = time.time()

    for idx, prompt in enumerate(prompts):
        for img_idx in range(args.num_images):
            seed = args.seed + idx * args.num_images + img_idx

            # 生成首帧
            image = generate_image(
                pipe, config, prompt,
                height=height, width=width,
                seed=seed,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
            )

            # 保存首帧
            if args.mode == "flf2v":
                if args.output_dir:
                    first_path = os.path.join(args.output_dir, f"case_{idx+1:03d}_first_frame.png")
                    image.save(first_path)
                    logging.info("Saved first frame: %s", first_path)
                else:
                    first_path = args.output.replace(".png", "_first_frame.png")
                    image.save(first_path)

                # 生成末帧
                last_prompt = last_prompts[idx]
                last_seed = seed + 10000  # 不同 seed 确保视觉差异

                last_image = generate_image(
                    pipe, config, last_prompt,
                    height=height, width=width,
                    seed=last_seed,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                )

                if args.output_dir:
                    last_path = os.path.join(args.output_dir, f"case_{idx+1:03d}_last_frame.png")
                    last_image.save(last_path)
                    logging.info("Saved last frame: %s", last_path)
                else:
                    last_path = args.output.replace(".png", "_last_frame.png")
                    last_image.save(last_path)

            else:
                # 单张图片模式
                if len(prompts) == 1 and args.num_images == 1 and args.output:
                    save_path = args.output
                else:
                    save_path = os.path.join(
                        args.output_dir,
                        f"case_{idx+1:03d}_seed{seed}.png"
                    )
                image.save(save_path)
                logging.info("Saved: %s", save_path)

            # 清理显存
            gc.collect()
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    total_images = len(prompts) * args.num_images * (2 if args.mode == "flf2v" else 1)

    logging.info("")
    logging.info("=" * 60)
    logging.info("Generation complete!")
    logging.info("  Total images: %d", total_images)
    logging.info("  Total time:   %.1fs", total_elapsed)
    logging.info("  Avg time:     %.1fs/image", total_elapsed / total_images if total_images > 0 else 0)
    logging.info("  Peak memory:  %.2f GB", torch.cuda.max_memory_allocated(args.device) / 1e9)
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
