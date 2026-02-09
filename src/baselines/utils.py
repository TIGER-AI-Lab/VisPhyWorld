#!/usr/bin/env python3
"""Utility helpers shared across baseline implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np


def load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compute_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    moments = cv2.moments(mask.astype(np.uint8))
    if abs(moments["m00"]) < 1e-6:
        return None
    x = moments["m10"] / moments["m00"]
    y = moments["m01"] / moments["m00"]
    return float(x), float(y)


def dominant_motion_centroids(frame_one: np.ndarray, frame_ten: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Estimate a single dominant object's centroid at two timepoints."""
    blurred_a = cv2.GaussianBlur(frame_one, (5, 5), 0)
    blurred_b = cv2.GaussianBlur(frame_ten, (5, 5), 0)
    diff = cv2.absdiff(blurred_a, blurred_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(cleaned)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    centroid_diff = compute_centroid(mask)
    if centroid_diff is None:
        return None, None
    frame_one_mask = cv2.cvtColor(frame_one, cv2.COLOR_RGB2GRAY)
    frame_ten_mask = cv2.cvtColor(frame_ten, cv2.COLOR_RGB2GRAY)
    _, mask_one = cv2.threshold(frame_one_mask, 5, 255, cv2.THRESH_BINARY)
    _, mask_ten = cv2.threshold(frame_ten_mask, 5, 255, cv2.THRESH_BINARY)
    centroid_one = compute_centroid(mask_one)
    centroid_two = compute_centroid(mask_ten)
    return centroid_one, centroid_two


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _select_salient_indices(total: int, sample_count: int, motion_scores: np.ndarray) -> List[int]:
    """Choose representative frame indices using motion saliency plus coverage."""
    if total <= 0:
        return []
    if total <= sample_count:
        return list(range(total))

    sample_count = max(sample_count, 2)
    indices: set[int] = {0, total - 1}
    min_gap = max(1, total // (sample_count * 3))
    ranked = list(range(total))
    ranked.sort(key=lambda i: float(motion_scores[i]), reverse=True)
    for idx in ranked:
        if len(indices) >= sample_count:
            break
        if idx in indices:
            continue
        if any(abs(idx - existing) < min_gap for existing in indices):
            continue
        indices.add(idx)
    if len(indices) < sample_count:
        uniform = {int(round(x)) for x in np.linspace(0, total - 1, sample_count)}
        indices.update(uniform)
    if len(indices) > sample_count:
        removable = sorted(indices - {0, total - 1}, key=lambda i: float(motion_scores[i]))
        while len(indices) > sample_count and removable:
            indices.remove(removable.pop(0))
    return sorted(indices)


def sample_video_frames(
    video_path: Path,
    sample_count: int = 6,
    strategy: str = "hybrid",
) -> List[Tuple[np.ndarray, int]]:
    """Return representative RGB frames and indices from a video."""
    sample_count = max(sample_count, 2)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    motion_scores: List[float] = []
    frames_total = 0
    prev_gray: Optional[np.ndarray] = None
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            motion_scores.append(0.0)
        else:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(float(np.mean(diff)))
        prev_gray = gray
        frames_total += 1
    cap.release()

    if frames_total == 0:
        return []

    if strategy == "uniform":
        selected = sorted(
            {int(round(x)) for x in np.linspace(0, frames_total - 1, min(sample_count, frames_total))}
        )
    else:
        scores_array = np.asarray(motion_scores, dtype=np.float32)
        selected = _select_salient_indices(frames_total, sample_count, scores_array)

    frames: List[Tuple[np.ndarray, int]] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    for idx in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append((rgb, idx))
    cap.release()
    return frames
