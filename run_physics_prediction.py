#!/usr/bin/env python3
"""CLI entrypoint to run physics prediction on VisPhyBench-style datasets."""

import argparse
import concurrent.futures as cf
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Ensure we can import the local src/ package when invoked from anywhere.
sys.path.append(os.path.dirname(__file__))

from src.physics_prediction import PhysicsPredictionPipeline, SUPPORTED_MODELS
from visphybench_layout import resolve_visphybench_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run physics prediction on VisPhyBench-style datasets")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent / "data"),
        help=(
            "Dataset path. Can point to:\n"
            "- VisPhyBench root (contains sub/ and test/), e.g. data/\n"
            "- split root (contains videos/ and detection_json/), e.g. data/sub/\n"
            "- videos directory that directly contains *.mp4, e.g. data/sub/videos/"
        ),
    )
    parser.add_argument(
        "--split",
        choices=["sub", "test"],
        default="sub",
        help="Split name when --data-dir points to the dataset root (default: sub).",
    )
    parser.add_argument(
        "--video",
        action="append",
        default=None,
        help="Only process the specified video name(s) (without extension). Can be used multiple times.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional: when --video is not provided, only process the first N videos (sorted by filename).",
    )
    parser.add_argument(
        "--engine",
        choices=["threejs", "p5js", "svg", "manim"],
        default="threejs",
        help="Rendering engine (default: threejs).",
    )
    parser.add_argument(
        "--scene-dim",
        choices=["2d", "3d"],
        default="2d",
        help="Scene dimensionality: 2d (default) or 3d (threejs only).",
    )
    available_models = ", ".join(sorted(SUPPORTED_MODELS.keys()))
    parser.add_argument(
        "--model",
        default="gpt-5",
        help=f"LLM model name (default: gpt-5). Available: {available_models}",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional: override the default output directory.",
    )
    parser.add_argument(
        "--detection-json-dir",
        "--llm-detections-dir",
        dest="detection_json_dir",
        default=None,
        help=(
            "Optional: directory containing first-frame annotations "
            "named '<video_name>_frame_01.json'. If omitted, the script "
            "will try to auto-discover 'detection_json/' next to the videos directory."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Parallel workers (default: 8). Large values may hit rate limits or exhaust CPU/memory.",
    )
    parser.add_argument(
        "--dashscope-api-key",
        default=None,
        help="Optional: override DashScope API key (also exported as QWEN_API_KEY).",
    )
    parser.add_argument(
        "--dashscope-base-url",
        default=None,
        help=(
            "Optional: override DashScope OpenAI-compatible base_url (also exported as QWEN_API_BASE). "
            'Example: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"'
        ),
    )
    return parser.parse_args()


def build_pipeline(
    model_name: Optional[str],
    output_dir: Optional[str],
    detections_dir: Optional[str],
    engine: str,
    scene_dim: str,
) -> PhysicsPredictionPipeline:
    kwargs = {}
    if model_name:
        kwargs["model_name"] = model_name
    if output_dir:
        kwargs["output_dir"] = output_dir
    if detections_dir:
        kwargs["detection_dir"] = detections_dir
    if engine:
        kwargs["engine"] = engine
    if scene_dim:
        kwargs["scene_dim"] = scene_dim
    return PhysicsPredictionPipeline(**kwargs)


def _default_output_dir(project_root: Path, model_name: str, engine: str) -> str:
    # Keep consistent with PhysicsPredictionPipeline default naming, but ensure we choose it once
    # so parallel workers share the same output directory.
    import re
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^a-zA-Z0-9_-]+", "", (model_name or "unknown").replace(" ", "_").lower()) or "unknown"
    safe_engine = re.sub(r"[^a-zA-Z0-9_-]+", "", (engine or "threejs")) or "unknown"
    return str(project_root / "output" / f"physics_prediction_{timestamp}_{safe_model}_{safe_engine}")


def _collect_targets(data_dir: Path, video_names: Optional[List[str]], max_videos: Optional[int]) -> List[str]:
    if video_names:
        return list(video_names)
    video_files = sorted([f.name for f in data_dir.glob("*.mp4") if f.is_file()])
    target = [os.path.splitext(f)[0] for f in video_files]
    if max_videos is not None:
        try:
            limit = int(max_videos)
        except Exception:
            limit = 0
        if limit > 0:
            target = target[:limit]
    return target


def _collect_frame_dirs(videos_dir: Path) -> List[str]:
    candidates = [
        videos_dir / "extracted_frames",
        videos_dir / "frames",
        videos_dir.parent / "extracted_frames",
        videos_dir.parent / "frames",
    ]
    return [str(p) for p in candidates if p.exists() and p.is_dir()]


def _process_one_video(
    video_name: str,
    data_dir: str,
    frame_dirs: List[str],
    *,
    model_name: Optional[str],
    output_dir: str,
    detections_dir: Optional[str],
    engine: str,
    scene_dim: str,
) -> Dict:
    try:
        pipeline = build_pipeline(model_name, output_dir, detections_dir, engine, scene_dim)
        result = pipeline._process_video_entry(video_name, data_dir, frame_dirs)  # noqa: SLF001
        result["success"] = bool(result.get("success", True))
        return result
    except Exception as exc:
        return {
            "success": False,
            "video_name": video_name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def print_summary(summary: dict) -> None:
    total = summary.get("total", 0)
    successful = summary.get("successful", 0)
    failed = summary.get("failed", 0)
    results: List[dict] = summary.get("results", [])

    print("=" * 80, flush=True)
    print("📊 处理结果", flush=True)
    print("=" * 80, flush=True)
    print(f"总计: {total} | 成功: {successful} | 失败: {failed}", flush=True)

    if not results:
        if summary.get("error"):
            print(f"⚠️ {summary['error']}", flush=True)
        return

    if successful:
        print("\n✅ 成功条目:", flush=True)
        for item in results:
            if item.get("success"):
                name = item.get("video_name") or item.get("image_name") or "unknown"
                print(f"  - {name}", flush=True)

    if failed:
        print("\n❌ 失败条目:", flush=True)
        for item in results:
            if not item.get("success"):
                name = item.get("video_name") or item.get("image_name") or "unknown"
                error = item.get("error", "未提供错误信息")
                print(f"  - {name}: {error}", flush=True)


def main() -> None:
    # Ensure credentials in .env are loaded.
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    load_dotenv()  # fallback to current working directory if needed

    args = parse_args()

    # DashScope (Qwen) OpenAI-compatible settings:
    # - Allow using DASHSCOPE_API_KEY / DASHSCOPE_API_BASE in .env
    # - Export QWEN_API_KEY / QWEN_API_BASE for the pipeline.
    dashscope_key = (args.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY", "")).strip()
    if dashscope_key:
        os.environ["QWEN_API_KEY"] = dashscope_key
    dashscope_base = (args.dashscope_base_url or os.environ.get("DASHSCOPE_API_BASE", "")).strip()
    if dashscope_base:
        os.environ["QWEN_API_BASE"] = dashscope_base
    # Default to DashScope international OpenAI-compatible endpoint (do not override explicit settings).
    if not os.environ.get("QWEN_API_BASE", "").strip():
        os.environ["QWEN_API_BASE"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    target_videos: Optional[List[str]] = args.video if args.video else None

    resolved = resolve_visphybench_paths(Path(args.data_dir), split=args.split)
    videos_dir = resolved.videos_dir.resolve()
    if not videos_dir.exists() or not videos_dir.is_dir():
        raise SystemExit(f"--data-dir does not resolve to a valid videos directory: {videos_dir}")

    detections_dir: Optional[str] = None
    if args.detection_json_dir:
        detections_dir = str(Path(args.detection_json_dir).expanduser().resolve())
    elif resolved.detection_json_dir and resolved.detection_json_dir.exists():
        detections_dir = str(resolved.detection_json_dir.resolve())

    jobs = max(1, int(args.jobs or 1))
    output_dir = args.output_dir
    if not output_dir:
        default_model = args.model or "gpt-5"
        output_dir = _default_output_dir(Path(__file__).resolve().parent, default_model, args.engine)
    os.makedirs(output_dir, exist_ok=True)

    frame_dirs = _collect_frame_dirs(videos_dir)
    targets = _collect_targets(videos_dir, target_videos, args.max_videos)

    if jobs == 1:
        pipeline = build_pipeline(args.model, output_dir, detections_dir, args.engine, args.scene_dim)
        summary = pipeline.process_phyworld_dataset(
            data_dir=str(videos_dir),
            video_names=targets,
            max_videos=None,
        )
    else:
        results: List[Dict] = []
        total = len(targets)
        print(f"🚀 并行模式启动: jobs={jobs} total_videos={total} output_dir={output_dir}", flush=True)
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            future_map = {
                ex.submit(
                    _process_one_video,
                    name,
                    str(videos_dir),
                    frame_dirs,
                    model_name=args.model,
                    output_dir=output_dir,
                    detections_dir=detections_dir,
                    engine=args.engine,
                    scene_dim=args.scene_dim,
                ): name
                for name in targets
            }
            done = 0
            for fut in cf.as_completed(future_map):
                done += 1
                name = future_map[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {"success": False, "video_name": name, "error": f"{type(exc).__name__}: {exc}"}
                results.append(result)
                status = "✅" if result.get("success") else "❌"
                print(f"[{done}/{total}] {status} {name}", flush=True)

        # Keep results in input order
        idx_by_name = {name: i for i, name in enumerate(targets)}
        results.sort(key=lambda r: idx_by_name.get(r.get("video_name") or r.get("image_name") or "", 10**9))

        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful
        summary = {
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "results": results,
            "data_dir": str(videos_dir),
            "output_dir": output_dir,
            "jobs": jobs,
        }

    print_summary(summary)

    total = summary.get("total", 0)
    successful = summary.get("successful", 0)
    failed = summary.get("failed", 0)

    if total == 0:
        msg = summary.get("error", "未找到可处理的目标")
        print(f"⚠️ 没有处理任何视频: {msg}", flush=True)
    elif successful == total:
        print(f"✅ 全部完成，共 {total} 个视频", flush=True)
    elif successful == 0:
        print(f"❌ 全部失败，共 {failed} 个视频", flush=True)
    else:
        print(f"⚠️ 部分成功: 成功 {successful} / 失败 {failed}", flush=True)


if __name__ == "__main__":
    main()
