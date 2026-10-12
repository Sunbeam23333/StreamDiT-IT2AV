from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

if __package__:
    from ..cpu_pipeline import run_cpu_pipeline
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    run_cpu_pipeline = importlib.import_module("agent_video_pipeline.cpu_pipeline").run_cpu_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU/API smoke test for storyboard -> image -> video -> review -> concat")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--scenario", default="generic")
    parser.add_argument("--quality-tier", default="quality", choices=["speed", "balanced", "quality"])
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--route", default="auto")
    parser.add_argument("--planning-model", default="gemini-3-pro")
    parser.add_argument("--image-model", default="gemini-3-pro-image")
    parser.add_argument("--video-model", default="veo-3.1-fast")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    project_dir = Path(args.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    print("STEP1 pipeline start", flush=True)
    result = run_cpu_pipeline(
        user_prompt=args.prompt,
        duration_s=args.duration,
        project_id=project_dir.name,
        project_dir=str(project_dir),
        preferred_route=args.route,
        aspect_ratio="9:16",
        scenario_hint=args.scenario,
        quality_tier=args.quality_tier,
        dry_run=False,
        save_plan=True,
        planning_model=args.planning_model,
        image_model=args.image_model,
        video_model=args.video_model,
    )
    print("STEP2 pipeline done", flush=True)

    summary_path = project_dir / "smoke_result.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUCCESS {result.get('success')} {result.get('output_video')}", flush=True)
    print(f"SUMMARY {summary_path}", flush=True)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
