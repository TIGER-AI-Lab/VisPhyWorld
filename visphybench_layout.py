#!/usr/bin/env python3
"""Helpers for resolving the on-disk VisPhyBench dataset layout.

The Hugging Face dataset `TIGER-Lab/VisPhyBench-Data` is expected to be placed
under this repo's `data/` directory with the following structure:

    data/
      sub/
        videos/*.mp4
        detection_json/*.json
        difficulty_table.json
        metadata.jsonl
      test/
        videos/*.mp4
        detection_json/*.json
        difficulty_table.json
        metadata.jsonl

This module provides small, dependency-free utilities to map a user-provided
path (dataset root, split root, or videos directory) to the canonical
directories used by the pipeline and evaluation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_KNOWN_SPLITS = ("sub", "test")


@dataclass(frozen=True)
class VisPhyBenchPaths:
    """Resolved paths for one split of VisPhyBench."""

    split: Optional[str]
    videos_dir: Path
    detection_json_dir: Optional[Path] = None
    difficulty_table: Optional[Path] = None
    metadata_jsonl: Optional[Path] = None


def _as_dir(path: Path) -> Optional[Path]:
    return path if path.exists() and path.is_dir() else None


def _as_file(path: Path) -> Optional[Path]:
    return path if path.exists() and path.is_file() else None


def resolve_visphybench_paths(data_dir: Path, *, split: str = "sub") -> VisPhyBenchPaths:
    """Resolve VisPhyBench split directories from a user-provided path.

    Args:
        data_dir: May point to:
            - dataset root that contains `sub/` and/or `test/`
            - split root that contains `videos/` and `detection_json/`
            - `videos/` directory itself
            - a legacy directory that directly contains `*.mp4`
        split: Split name used when `data_dir` is the dataset root.

    Returns:
        VisPhyBenchPaths with `videos_dir` always set.
    """

    split_norm = (split or "sub").strip().lower()
    if split_norm not in _KNOWN_SPLITS:
        split_norm = "sub"

    p = Path(data_dir).expanduser()

    # Case 1: dataset root (contains <split>/videos)
    candidate_videos = _as_dir(p / split_norm / "videos")
    if candidate_videos is not None:
        split_root = p / split_norm
        return VisPhyBenchPaths(
            split=split_norm,
            videos_dir=candidate_videos,
            detection_json_dir=_as_dir(split_root / "detection_json"),
            difficulty_table=_as_file(split_root / "difficulty_table.json"),
            metadata_jsonl=_as_file(split_root / "metadata.jsonl"),
        )

    # Case 2: split root (contains videos/)
    candidate_videos = _as_dir(p / "videos")
    if candidate_videos is not None:
        split_name = p.name.strip().lower()
        split_value = split_name if split_name in _KNOWN_SPLITS else None
        return VisPhyBenchPaths(
            split=split_value,
            videos_dir=candidate_videos,
            detection_json_dir=_as_dir(p / "detection_json"),
            difficulty_table=_as_file(p / "difficulty_table.json"),
            metadata_jsonl=_as_file(p / "metadata.jsonl"),
        )

    # Case 3: videos dir directly
    if p.name.strip().lower() == "videos":
        split_root = p.parent
        split_name = split_root.name.strip().lower()
        split_value = split_name if split_name in _KNOWN_SPLITS else None
        return VisPhyBenchPaths(
            split=split_value,
            videos_dir=p,
            detection_json_dir=_as_dir(split_root / "detection_json"),
            difficulty_table=_as_file(split_root / "difficulty_table.json"),
            metadata_jsonl=_as_file(split_root / "metadata.jsonl"),
        )

    # Fallback: treat as a legacy directory that contains MP4s.
    # Detection JSON might either live next to the MP4s or in a sibling
    # `detection_json/` folder (best-effort).
    return VisPhyBenchPaths(
        split=None,
        videos_dir=p,
        detection_json_dir=_as_dir(p / "detection_json") or _as_dir(p.parent / "detection_json"),
        difficulty_table=_as_file(p / "difficulty_table.json") or _as_file(p.parent / "difficulty_table.json"),
        metadata_jsonl=_as_file(p / "metadata.jsonl") or _as_file(p.parent / "metadata.jsonl"),
    )

