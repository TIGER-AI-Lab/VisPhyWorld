from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LLMSummary:
    image_name: str
    llm_calls_path: Path
    llm_calls: List[Dict[str, Any]]
    text_sections: Dict[str, str]
    html_content: str
    assets: Dict[str, Optional[Path]]
    video_metrics: Dict[str, Dict[str, Any]]
    text_metrics: Dict[str, float]
    frame_count: int
    success: bool
    errors: List[str]
    gt_analysis: Optional[str]
    gt_analysis_path: Optional[Path]
    model_analysis: Optional[str]
    model_analysis_path: Optional[Path]
    text_metrics_path: Optional[Path] = None
    model_name: str = "unknown"


@dataclass
class GTGenerationConfig:
    enabled: bool = True
    # Default GT generator model for scene analysis captions.
    # We assume an OpenAI-compatible endpoint exposing `gpt-5.1`.
    model: str = "gpt-5.1"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[float] = None
