#!/usr/bin/env python3
"""Run handcrafted baselines on phyworld-style datasets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.baselines import BASELINE_REGISTRY, get_baseline
from src.baselines.base import BaselineGenerator, ensure_frame_paths


BASELINE_CHOICES = sorted(BASELINE_REGISTRY.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute baseline generators for physics prediction dataset.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="数据根目录（包含 MP4 与 extracted_frames 子目录）",
    )
    parser.add_argument(
        "--video",
        action="append",
        default=None,
        help="仅运行指定视频（不含扩展名），可多次使用",
    )
    parser.add_argument(
        "--baseline",
        choices=BASELINE_CHOICES + ["all"],
        default="all",
        help="指定运行的 baseline 类型，默认为 all",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="可选：自定义输出目录，默认写入 output/baseline_<name>_<timestamp>",
    )
    return parser.parse_args()


def build_output_dir(root: Path, run_label: str, forced: Optional[str]) -> Path:
    if forced:
        return Path(forced).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "output" / f"{run_label}_{timestamp}"


def select_videos(data_dir: Path, targets: Optional[List[str]]) -> List[str]:
    if targets:
        return targets
    video_files = sorted([f for f in data_dir.glob("*.mp4") if f.is_file()])
    if not video_files:
        raise FileNotFoundError(f"数据目录 {data_dir} 下未找到 MP4 视频")
    return [video.stem for video in video_files]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()

    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    if args.baseline == "all":
        baseline_names = list(BASELINE_CHOICES)
        run_label = "baseline_all"
    else:
        baseline_names = [args.baseline]
        run_label = f"baseline_{args.baseline}"

    output_dir = build_output_dir(PROJECT_ROOT, run_label, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🚀 Baselines: {', '.join(baseline_names)}")
    print(f"📁 数据目录: {data_dir}")
    print(f"📦 输出目录: {output_dir}")
    print("=" * 80)

    videos = select_videos(data_dir, args.video)
    frame_dirs = [data_dir / "extracted_frames", data_dir / "frames"]

    total_success = 0
    total_fail = 0

    for baseline_name in baseline_names:
        baseline_cls = get_baseline(baseline_name)
        generator: BaselineGenerator = baseline_cls(output_dir)

        print("\n" + "=" * 80)
        print(f"▶ Baseline: {generator.name}")
        print("=" * 80)

        successes = 0
        failures = 0

        for idx, name in enumerate(videos, start=1):
            print(f"\n[{idx}/{len(videos)}] ▶ {name}")
            frame_one, frame_ten = ensure_frame_paths(data_dir, name, frame_dirs)
            video_path = data_dir / f"{name}.mp4"
            if frame_one is None or frame_ten is None:
                print("  ⚠️ 未找到关键帧，跳过该样本")
                failures += 1
                total_fail += 1
                continue

            result = generator.run(
                dataset_dir=data_dir,
                video_name=name,
                frame_one=frame_one,
                frame_ten=frame_ten,
                video_path=video_path if video_path.exists() else None,
            )
            if result.success:
                successes += 1
                total_success += 1
                status_icon = "✅"
            else:
                failures += 1
                total_fail += 1
                status_icon = "⚠️"

            manifest_path = generator.log_dir / f"{name}_{generator.name}.json"
            payload = result.to_dict()
            payload["video_name"] = name
            payload["baseline"] = generator.name
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"  {status_icon} {result.message}")
            if result.html_path:
                print(f"  📄 HTML: {result.html_path}")
            if result.video_path:
                print(f"  🎞️ 视频: {result.video_path}")

        print("\n-- Baseline 统计 --")
        print(f"成功: {successes} / 失败: {failures}")

    print("\n" + "=" * 80)
    print(f"全部完成: 成功 {total_success} / 失败 {total_fail}")
    print("=" * 80)


if __name__ == "__main__":
    main()
