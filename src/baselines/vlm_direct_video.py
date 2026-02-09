#!/usr/bin/env python3
"""One-shot VLM baseline: directly generate an animation video (via renderer) in a single model call.

This baseline intentionally disables the agentic retry loop used by the main pipeline.
It makes exactly one LLM call to produce runnable code (HTML for threejs) and renders it to MP4.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2

from .base import BaselineGenerator, BaselineResult
from ..physics_prediction import PhysicsPredictionPipeline


@dataclass(frozen=True)
class VLMDirectVideoConfig:
    model_name: str
    engine: str = "threejs"
    enable_detection_context: bool = True


class VLMDirectVideoBaseline(BaselineGenerator):
    name = "vlm-direct-video"

    def __init__(self, output_root: Path) -> None:
        super().__init__(output_root)
        self.config = default_config_from_env()

        # Use a dedicated pipeline instance for prompt building + sanitization.
        # Keep its internal retries disabled: this baseline must be one-shot.
        self._pipeline = PhysicsPredictionPipeline(
            model_name=self.config.model_name,
            output_dir=str(output_root / "artifacts" / "_vlm_direct_tmp"),
            engine=(self.config.engine or "threejs"),
        )
        self._pipeline.max_retries = 0

    def run(
        self,
        dataset_dir: Path,
        video_name: str,
        frame_one: Path,
        frame_ten: Path,
        video_path: Optional[Path],
    ) -> BaselineResult:
        del dataset_dir  # not used (kept for compatibility)

        if not frame_one.exists() or not frame_ten.exists():
            return BaselineResult(
                success=False,
                message="Missing extracted frames (frame_01/frame_10).",
                html_path=None,
                video_path=None,
                manifest={"model": self.config.model_name, "engine": self.config.engine},
            )

        img1 = cv2.imread(str(frame_one))
        img10 = cv2.imread(str(frame_ten))
        if img1 is None or img10 is None:
            return BaselineResult(
                success=False,
                message="Failed to read frame images.",
                html_path=None,
                video_path=None,
                manifest={"model": self.config.model_name, "engine": self.config.engine},
            )

        fps, duration_ms = self._reference_video_meta(video_path)
        self._pipeline._current_recording_target_fps = fps  # noqa: SLF001
        self._pipeline._current_recording_target_duration_ms = duration_ms  # noqa: SLF001

        source_paths: Dict[str, object] = {"output_original": str(frame_one)}
        detection_context: Optional[str] = None
        if self.config.enable_detection_context:
            detection_context = self._pipeline._load_detection_context(video_name, source_paths)  # noqa: SLF001

        frames = [
            {
                "image": img1,
                "label": "frame_01",
                "description": "initial reference frame (t0)",
                "source": str(frame_one),
            },
            {
                "image": img10,
                "label": "frame_10",
                "description": "later frame of the same scene (guides motion toward this state)",
                "source": str(frame_ten),
            },
        ]

        prediction = self._pipeline.generate_analysis_and_code(
            frames,
            video_name,
            error_feedback="",
            detection_context=detection_context,
        )
        document = (prediction.get("document") or "").strip()
        if not document:
            error_msg = prediction.get("error") or "VLM did not return a runnable HTML document."
            return BaselineResult(
                success=False,
                message=error_msg,
                html_path=None,
                video_path=None,
                manifest={
                    "model": self.config.model_name,
                    "engine": self.config.engine,
                    "error": error_msg,
                },
            )

        html_path = self.pred_dir / f"{video_name}_{self.name}.html"
        html_path.write_text(document, encoding="utf-8")

        video_filename = f"{video_name}_{self.name}.mp4"
        video_out, render_error = self._render_html_document(document, video_filename)
        if not video_out:
            return BaselineResult(
                success=False,
                message=f"Render failed: {render_error}",
                html_path=str(html_path),
                video_path=None,
                manifest={"model": self.config.model_name, "engine": self.config.engine},
            )

        manifest = {
            "baseline": self.name,
            "video_name": video_name,
            "html_path": str(html_path),
            "video_path": str(video_out),
            "model": self.config.model_name,
            "engine": self.config.engine,
            "target_fps": f"{fps:.4f}",
            "target_duration_ms": str(duration_ms),
        }
        # Mirror the convention used by run_baselines.py: artifacts/logs/<baseline>/<video>_<baseline>.json
        self._write_manifest(self.log_dir / f"{video_name}_{self.name}.json", manifest)

        return BaselineResult(
            success=True,
            message="Generated one-shot VLM video.",
            html_path=str(html_path),
            video_path=str(video_out),
            manifest={k: str(v) for k, v in manifest.items()},
        )


def default_config_from_env() -> VLMDirectVideoConfig:
    model_name = (os.environ.get("VLM_DIRECT_MODEL") or "").strip()
    if not model_name:
        raise ValueError("Missing VLM_DIRECT_MODEL environment variable for VLMDirectVideoBaseline.")
    engine = (os.environ.get("VLM_DIRECT_ENGINE") or "threejs").strip() or "threejs"
    enable_detection = (os.environ.get("VLM_DIRECT_ENABLE_DETECTION") or "1").strip()
    enable_detection_bool = enable_detection not in {"0", "false", "False", "no", "NO"}
    return VLMDirectVideoConfig(
        model_name=model_name,
        engine=engine,
        enable_detection_context=enable_detection_bool,
    )
