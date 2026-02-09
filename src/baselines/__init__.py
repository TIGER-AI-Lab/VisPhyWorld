#!/usr/bin/env python3
"""Baseline registry and factory utilities."""

from __future__ import annotations

from typing import Dict, Type

from .base import BaselineGenerator
from .stable_video_diffusion import (
    StableVideoDiffusionImg2VidBaseline,
    StableVideoDiffusionImg2VidXTBaseline,
)
from .vlm_direct_video import VLMDirectVideoBaseline, default_config_from_env, VLMDirectVideoConfig

BASELINE_REGISTRY: Dict[str, Type[BaselineGenerator]] = {
    "stable-video-img2vid": StableVideoDiffusionImg2VidBaseline,
    "stable-video-img2vid-xt": StableVideoDiffusionImg2VidXTBaseline,
    "vlm-direct-video": VLMDirectVideoBaseline,
}


def get_baseline(name: str) -> Type[BaselineGenerator]:
    key = name.lower()
    if key not in BASELINE_REGISTRY:
        available = ", ".join(sorted(BASELINE_REGISTRY))
        raise KeyError(f"Unknown baseline '{name}'. Available: {available}")
    return BASELINE_REGISTRY[key]


__all__ = [
    "BaselineGenerator",
    "StableVideoDiffusionImg2VidBaseline",
    "StableVideoDiffusionImg2VidXTBaseline",
    "VLMDirectVideoBaseline",
    "VLMDirectVideoConfig",
    "default_config_from_env",
    "BASELINE_REGISTRY",
    "get_baseline",
]
