#!/usr/bin/env python3
"""
Text-to-Image generation using xDiT (多GPU并行推理).

支持模型: FLUX.1-schnell, FLUX.1-dev, SD3.5-Large, PixArt-Sigma, PixArt-Alpha, HunyuanDiT
输出分辨率: 适配 LightX2V Wan2.2 MoE i2v/flf2v (720x1280)

用法 (通过 torchrun 启动):
    # 8卡 FLUX.1 schnell
    torchrun --nproc_per_node=8 run_t2i_xdit.py \\
        --model /tmp/.../FLUX.1-schnell \\
        --height 720 --width 1280 \\
        --prompt "A cat on a beach" \\
        --num_inference_steps 4 --guidance_scale 0.0 \\
        --pipefusion_parallel_degree 2 --ulysses_degree 2 --ring_degree 2 \\
        --no_use_resolution_binning

    # 8卡 SD3.5 Large
    torchrun --nproc_per_node=8 run_t2i_xdit.py \\
        --model /tmp/.../stable-diffusion-3.5-large \\
        --model_type sd35 \\
        --height 720 --width 1280 \\
        --prompt "A cat on a beach" \\
        --num_inference_steps 28 --guidance_scale 4.5 \\
        --pipefusion_parallel_degree 2 --ulysses_degree 2 --ring_degree 2

注意: 此脚本应通过 run_t2i_benchmark.sh 启动，模型已预先同步到 /tmp 本地盘
"""
import gc
import logging
import os
import time

import torch
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] rank %(process)d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def detect_model_type(model_path: str) -> str:
    """根据模型目录名自动检测模型类型."""
    model_name = os.path.basename(model_path).lower()
    if "flux" in model_name:
        return "flux"
    elif "stable-diffusion-3" in model_name or "sd3" in model_name:
        return "sd35"
    elif "pixart" in model_name and "sigma" in model_name:
        return "pixart_sigma"
    elif "pixart" in model_name and "alpha" in model_name:
        return "pixart_alpha"
    elif "hunyuan" in model_name:
        return "hunyuandit"
    else:
        logging.warning("Cannot auto-detect model type from '%s', defaulting to flux", model_name)
        return "flux"


def load_pipeline(engine_config, input_config, model_type: str, args):
    """根据模型类型加载对应的 xDiT Pipeline."""
    local_rank = get_world_group().local_rank

    if model_type == "flux":
        from transformers import T5EncoderModel
        from xfuser.model_executor.pipelines import xFuserFluxPipeline

        engine_config.runtime_config.dtype = torch.bfloat16

        text_encoder_2 = T5EncoderModel.from_pretrained(
            engine_config.model_config.model,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
        )

        if args.use_fp8_t5_encoder:
            from optimum.quanto import freeze, qfloat8, quantize
            logging.info("rank %d quantizing T5 encoder to FP8", local_rank)
            quantize(text_encoder_2, weights=qfloat8)
            freeze(text_encoder_2)

        cache_args = {
            "use_teacache": args.use_teacache,
            "use_fbcache": args.use_fbcache,
            "rel_l1_thresh": 0.12,
            "return_hidden_states_first": False,
            "num_steps": input_config.num_inference_steps,
        }

        pipe = xFuserFluxPipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            cache_args=cache_args,
            torch_dtype=torch.bfloat16,
            text_encoder_2=text_encoder_2,
        )
        return pipe, "flux"

    elif model_type == "sd35":
        from transformers import T5EncoderModel
        from xfuser.model_executor.pipelines import xFuserStableDiffusion3Pipeline

        text_encoder_3 = T5EncoderModel.from_pretrained(
            engine_config.model_config.model,
            subfolder="text_encoder_3",
            torch_dtype=torch.float16,
        )

        if args.use_fp8_t5_encoder:
            from optimum.quanto import freeze, qfloat8, quantize
            logging.info("rank %d quantizing T5 encoder to FP8", local_rank)
            quantize(text_encoder_3, weights=qfloat8)
            freeze(text_encoder_3)

        pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=torch.float16,
            text_encoder_3=text_encoder_3,
        )
        return pipe, "sd35"

    elif model_type in ("pixart_sigma", "pixart_alpha"):
        from transformers import T5EncoderModel

        if model_type == "pixart_sigma":
            from xfuser.model_executor.pipelines import xFuserPixArtSigmaPipeline as PipeClass
        else:
            from xfuser.model_executor.pipelines import xFuserPixArtAlphaPipeline as PipeClass

        text_encoder = T5EncoderModel.from_pretrained(
            engine_config.model_config.model,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )

        if args.use_fp8_t5_encoder:
            from optimum.quanto import freeze, qfloat8, quantize
            logging.info("rank %d quantizing T5 encoder to FP8", local_rank)
            quantize(text_encoder, weights=qfloat8)
            freeze(text_encoder)

        pipe = PipeClass.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=torch.float16,
            text_encoder=text_encoder,
        )
        return pipe, model_type

    elif model_type == "hunyuandit":
        from xfuser.model_executor.pipelines import xFuserHunyuanDiTPipeline

        pipe = xFuserHunyuanDiTPipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=torch.float16,
        )
        return pipe, "hunyuandit"

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main():
    parser = FlexibleArgumentParser(description="xDiT Text-to-Image Generation")
    # 额外参数
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["flux", "sd35", "pixart_sigma", "pixart_alpha", "hunyuandit"],
                        help="Model type (auto-detected if not specified)")
    parser.add_argument("--output_dir", type=str, default="./t2i_outputs",
                        help="Output directory for generated images")
    parser.add_argument("--output_prefix", type=str, default="t2i",
                        help="Prefix for output filenames")
    parser.add_argument("--mode", type=str, default="i2v", choices=["i2v", "flf2v"],
                        help="i2v: single image; flf2v: first+last frame pair")
    parser.add_argument("--last_prompt", type=str, default=None,
                        help="Prompt for last frame in flf2v mode")
    parser.add_argument("--last_seed_offset", type=int, default=10000,
                        help="Seed offset for last frame generation")

    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()

    local_rank = get_world_group().local_rank
    world_size = get_world_group().world_size

    # 自动检测模型类型
    model_type = args.model_type or detect_model_type(engine_config.model_config.model)
    logging.info("rank %d: model_type=%s, model=%s", local_rank, model_type, engine_config.model_config.model)

    # 加载 pipeline
    pipe, model_type = load_pipeline(engine_config, input_config, model_type, args)

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info("rank %d: sequential CPU offload enabled", local_rank)
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    logging.info("rank %d: model loaded, memory: %.2f GB", local_rank, parameter_peak_memory / 1e9)

    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    # 构建生成参数
    gen_kwargs = {
        "height": input_config.height,
        "width": input_config.width,
        "prompt": input_config.prompt,
        "num_inference_steps": input_config.num_inference_steps,
        "output_type": input_config.output_type,
        "guidance_scale": input_config.guidance_scale,
        "generator": torch.Generator(device="cuda").manual_seed(input_config.seed),
    }

    # 模型特定参数
    if model_type == "flux":
        gen_kwargs["max_sequence_length"] = getattr(input_config, "max_sequence_length", 256)
    elif model_type == "sd35":
        gen_kwargs["max_sequence_length"] = getattr(input_config, "max_sequence_length", 512)
    elif model_type in ("pixart_sigma", "pixart_alpha"):
        gen_kwargs["use_resolution_binning"] = input_config.use_resolution_binning

    # 生成首帧
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(**gen_kwargs)
    elapsed = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)

    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups

        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i

                if args.mode == "flf2v":
                    suffix = "first_frame"
                else:
                    suffix = f"img_{image_rank}"

                image_name = f"{args.output_prefix}_{model_type}_{parallel_info}_{suffix}_seed{input_config.seed}.png"
                save_path = os.path.join(args.output_dir, image_name)
                image.save(save_path)
                logging.info("rank %d: saved %s", local_rank, save_path)

    if get_world_group().rank == world_size - 1:
        logging.info(
            "First image: time=%.2fs, param_memory=%.2f GB, peak_memory=%.2f GB",
            elapsed, parameter_peak_memory / 1e9, peak_memory / 1e9,
        )

    # flf2v 模式: 生成末帧
    if args.mode == "flf2v":
        last_prompt = args.last_prompt or f"Later in time: {input_config.prompt}. The scene has evolved and changed."
        last_seed = input_config.seed + args.last_seed_offset

        gen_kwargs["prompt"] = last_prompt
        gen_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(last_seed)

        gc.collect()
        torch.cuda.empty_cache()

        pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

        start_time2 = time.time()
        output2 = pipe(**gen_kwargs)
        elapsed2 = time.time() - start_time2

        if input_config.output_type == "pil":
            if pipe.is_dp_last_group():
                for i, image in enumerate(output2.images):
                    image_rank = dp_group_index * dp_batch_size + i
                    image_name = f"{args.output_prefix}_{model_type}_{parallel_info}_last_frame_seed{last_seed}.png"
                    save_path = os.path.join(args.output_dir, image_name)
                    image.save(save_path)
                    logging.info("rank %d: saved last frame %s", local_rank, save_path)

        if get_world_group().rank == world_size - 1:
            logging.info("Last frame: time=%.2fs", elapsed2)

    # 清理
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
