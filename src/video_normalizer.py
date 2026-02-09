#!/usr/bin/env python3
"""
Video normalization utilities: enforce consistent fps/duration/resolution across engines.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional


@dataclass(frozen=True)
class VideoTarget:
    width: int
    height: int
    fps: float
    duration_s: float


def _parse_fraction(text: str) -> float:
    raw = (text or "").strip()
    if not raw:
        return 0.0
    try:
        return float(Fraction(raw))
    except Exception:
        try:
            return float(raw)
        except Exception:
            return 0.0


def probe_video(path: str) -> dict:
    """
    Return a small dict with keys: width, height, fps, duration.
    Missing values are returned as 0/0.0.
    """
    payload = {"width": 0, "height": 0, "fps": 0.0, "duration": 0.0}
    if not path or not os.path.exists(path):
        return payload
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "default=nw=1",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception:
        return payload
    for line in (result.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "width":
            try:
                payload["width"] = int(float(value))
            except Exception:
                pass
        elif key == "height":
            try:
                payload["height"] = int(float(value))
            except Exception:
                pass
        elif key == "r_frame_rate":
            payload["fps"] = _parse_fraction(value)
        elif key == "duration":
            try:
                payload["duration"] = float(value)
            except Exception:
                pass
    return payload


def needs_normalize(path: str, target: VideoTarget, *, tolerance_s: float = 0.15) -> bool:
    meta = probe_video(path)
    if meta.get("width") != int(target.width) or meta.get("height") != int(target.height):
        return True
    fps = float(meta.get("fps") or 0.0)
    if abs(fps - float(target.fps)) > 1e-3:
        return True
    duration = float(meta.get("duration") or 0.0)
    if abs(duration - float(target.duration_s)) > float(tolerance_s):
        return True
    return False


def normalize_video_inplace(
    path: str,
    target: VideoTarget,
    *,
    log_path: Optional[str] = None,
    white_background: bool = True,
) -> bool:
    """
    Normalize an MP4 (in-place) to target fps/duration/resolution.
    Returns True on success or if no normalization needed.
    """
    if not needs_normalize(path, target):
        return True

    if not path or not os.path.exists(path):
        return False

    out_dir = os.path.dirname(path) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="normalized_", suffix=".mp4", dir=out_dir)
    os.close(tmp_fd)

    bg = "white" if white_background else "black"
    vf = (
        f"tpad=stop_mode=clone:stop_duration=100,"
        f"fps={target.fps:.4f},"
        f"scale={target.width}:{target.height}:force_original_aspect_ratio=decrease,"
        f"pad={target.width}:{target.height}:(ow-iw)/2:(oh-ih)/2:color={bg},"
        f"format=yuv420p"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        "-an",
        "-vf",
        vf,
        "-t",
        f"{target.duration_s:.4f}",
        "-r",
        f"{target.fps:.4f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        tmp_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if log_path:
            try:
                with open(log_path, "w", encoding="utf-8") as handle:
                    handle.write(result.stdout or "")
                    if result.stderr:
                        handle.write("\n--- STDERR ---\n")
                        handle.write(result.stderr)
            except Exception:
                pass
    except Exception as exc:
        if log_path:
            try:
                with open(log_path, "w", encoding="utf-8") as handle:
                    handle.write(f"ffmpeg normalization failed: {exc}\n")
            except Exception:
                pass
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return False

    try:
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return False

    return not needs_normalize(path, target)

