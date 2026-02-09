from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from threading import Lock

import math


HEAVY_METRIC_MODULES = {
    "clip": ("utils.evaluate_video_metrics", ["--metrics", "clip"]),
    "lpips": ("utils.evaluate_video_metrics", ["--metrics", "lpips"]),
    "dino": ("utils.evaluate_video_metrics", ["--metrics", "dino"]),
    "fsim": ("utils.evaluate_video_metrics", ["--metrics", "fsim"]),
    "vsi": ("utils.evaluate_video_metrics", ["--metrics", "vsi"]),
    "dists": ("utils.evaluate_video_metrics", ["--metrics", "dists"]),
    "physics": ("utils.evaluate_video_metrics", ["--metrics", "physics"]),
    "raft": ("utils.evaluate_video_metrics", ["--metrics", "raft", "--enable-raft"]),
    "llm": ("utils.evaluate_video_metrics", ["--metrics", "llm"]),
}

_RUN_LOCK = Lock()


def run_heavy_metric(
    metric: str,
    reference_video: Path,
    generated_video: Path,
    sample_every: int,
    caption: Optional[str] = None,
    max_frames: Optional[int] = None,
    extra_cli_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # NOTE: Do not globally disable metrics after a single failure.
    # We want to keep evaluating subsequent samples and simply record per-sample errors.

    module_entry = HEAVY_METRIC_MODULES.get(metric)
    if not module_entry:
        raise RuntimeError(f"未知的重指标: {metric}")
    module, extra_args = module_entry

    cmd: List[str] = [
        sys.executable,
        "-m",
        module,
        "--reference",
        str(reference_video),
        "--generated",
        str(generated_video),
        "--sample-every",
        str(max(1, sample_every)),
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    cmd.extend(extra_args)
    if caption and metric == "clip":
        cmd.extend(["--caption", caption])
    if extra_cli_args:
        cmd.extend([str(arg) for arg in extra_cli_args])

    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(project_root)
    )

    try:
        # Avoid running multiple heavy subprocesses concurrently within a single
        # evaluation process; this reduces GPU OOM risk when caller sets jobs>1.
        with _RUN_LOCK:
            completed = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
    except FileNotFoundError as exc:
        raise RuntimeError(f"无法执行 {module}: {exc}")

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "未知错误"
        raise RuntimeError(stderr)

    output = completed.stdout.strip()
    if not output:
        raise RuntimeError("脚本未返回任何输出")

    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"解析 JSON 失败: {exc}: {output[:200]}")


def merge_metrics(base: Dict[str, Any], extra: Dict[str, Any]) -> None:
    for key, value in extra.items():
        if key == "per_frame":
            per_frame = base.setdefault("per_frame", {})
            if isinstance(value, dict):
                per_frame.update(value)
        else:
            base[key] = value


def metric_keys_missing(metrics: Dict[str, Any], keys: Sequence[str]) -> bool:
    if not keys:
        return False
    for key in keys:
        if key not in metrics:
            return True
        value = metrics[key]
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
    return False
