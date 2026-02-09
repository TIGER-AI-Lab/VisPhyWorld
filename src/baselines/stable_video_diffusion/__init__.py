#!/usr/bin/env python3
"""Stable Video Diffusion baseline (img2vid).

These baselines take the same inputs as other VisExpert baselines:
the dataset root, video name, frame_01 / frame_10 paths and (optional) MP4.
Only frame_01 is used as the conditioning image for SVD.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..base import BaselineGenerator, BaselineResult


def _detect_device() -> Tuple[str, str]:
    """Return (device, dtype_str) based on torch availability."""
    try:
        import torch
    except Exception:
        return "cpu", "float32"
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "float32"


class _StableVideoDiffusionBase(BaselineGenerator):
    """Shared logic for Stable Video Diffusion img2vid variants."""

    model_id: str = ""

    def __init__(self, output_root: Path) -> None:
        super().__init__(output_root)
        self._pipe = None
        self._device, self._dtype_str = _detect_device()
        # Default canvas sizes: align with VisExpert (512x512) unless overridden.
        self.height = 512
        self.width = 512
        self.num_inference_steps = 25 if self._device == "cuda" else 8
        self.seed = 42
        self.motion_bucket_id = 127
        self.noise_aug_strength = 0.02

        def _env_int(name: str, default: int) -> int:
            try:
                raw = (os.environ.get(name) or "").strip()
                return int(raw) if raw else default
            except Exception:
                return default

        def _env_float(name: str, default: float) -> float:
            try:
                raw = (os.environ.get(name) or "").strip()
                return float(raw) if raw else default
            except Exception:
                return default

        self.width = max(64, _env_int("VISEXPERT_SVD_WIDTH", self.width))
        self.height = max(64, _env_int("VISEXPERT_SVD_HEIGHT", self.height))
        self.num_inference_steps = max(1, _env_int("VISEXPERT_SVD_STEPS", self.num_inference_steps))
        self.seed = _env_int("VISEXPERT_SVD_SEED", self.seed)
        self.motion_bucket_id = _env_int("VISEXPERT_SVD_MOTION_BUCKET", self.motion_bucket_id)
        self.noise_aug_strength = _env_float("VISEXPERT_SVD_NOISE_AUG", self.noise_aug_strength)

    # ------------------------------------------------------------------
    # Pipeline setup and preprocessing
    # ------------------------------------------------------------------
    def _load_pipeline(self) -> None:
        if self._pipe is not None:
            return

        try:
            import torch
            from diffusers import StableVideoDiffusionPipeline
        except Exception as exc:
            raise RuntimeError(
                f"无法导入 Stable Video Diffusion 依赖，请确认已安装 diffusers / torch: {exc}"
            ) from exc

        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

        load_kwargs: Dict[str, object] = {
            "torch_dtype": torch_dtype,
        }
        # Many SVD checkpoints provide an fp16 variant; try it when on GPU.
        if self._device == "cuda":
            load_kwargs["variant"] = "fp16"

        try:
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"加载 Stable Video Diffusion 模型失败 ({self.model_id})，"
                "请确认已在 Hugging Face 接受协议并完成 `huggingface-cli login`："
                f"{exc}"
            ) from exc

        if self._device == "cuda":
            pipe = pipe.to("cuda")
            try:
                # Try to be a bit more memory friendly on limited VRAM setups
                pipe.enable_model_cpu_offload()
            except Exception:
                pass

        self._pipe = pipe

    def _prepare_image(self, frame_path: Path) -> Image.Image:
        image = Image.open(frame_path).convert("RGB")
        # Resize to the configured canvas while keeping basic aspect ratio.
        image = image.resize((self.width, self.height), Image.BICUBIC)
        return image

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def run(
        self,
        dataset_dir: Path,
        video_name: str,
        frame_one: Path,
        frame_ten: Path,
        video_path: Optional[Path],
    ) -> BaselineResult:
        del dataset_dir  # unused but kept for signature compatibility
        try:
            self._load_pipeline()
        except Exception as exc:
            return BaselineResult(
                success=False,
                message=str(exc),
                html_path=None,
                video_path=None,
                manifest={
                    "model_id": self.model_id,
                    "video_name": video_name,
                    "stage": "load_pipeline",
                },
            )

        # Try to roughly match reference video length and FPS if available.
        fps, duration_ms = self._reference_video_meta(video_path)
        if fps <= 1e-3:
            fps = 7.0
        fps = float(max(3.0, min(30.0, fps)))
        if duration_ms <= 0:
            duration_ms = 2000
        duration_s = max(duration_ms / 1000.0, 0.5)
        num_frames = int(round(fps * duration_s))
        # Keep within a reasonable range for compute and model limits.
        num_frames = max(8, min(24, num_frames))

        out_video_path = self.video_dir / f"{video_name}_{self.name}.mp4"
        tmp_video_path = out_video_path.with_suffix(".tmp.mp4")

        try:
            import torch

            image = self._prepare_image(frame_one)
            generator = torch.Generator(device=self._device).manual_seed(int(self.seed))

            # Run SVD
            result = self._pipe(
                image=image,
                height=self.height,
                width=self.width,
                num_frames=num_frames,
                num_inference_steps=self.num_inference_steps,
                fps=int(round(fps)),
                motion_bucket_id=int(self.motion_bucket_id),
                noise_aug_strength=float(self.noise_aug_strength),
                decode_chunk_size=8,
                generator=generator,
            )

            frames = result.frames[0]  # type: ignore[assignment]
            if not frames:
                raise RuntimeError("Stable Video Diffusion 未返回任何帧。")

            # Save frames to MP4 via OpenCV to avoid extra deps.
            size = frames[0].size  # (width, height)
            width, height = int(size[0]), int(size[1])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(tmp_video_path),
                fourcc,
                float(fps),
                (width, height),
            )
            for frame in frames:
                rgb = np.array(frame.convert("RGB"))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()

            # Re-encode with FFmpeg + H.264 for maximum player compatibility.
            try:
                import subprocess
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(tmp_video_path),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        str(out_video_path),
                    ],
                    check=True,
                )
                try:
                    if tmp_video_path.exists():
                        tmp_video_path.unlink()
                except Exception:
                    pass
            except Exception:
                # Fallback: if FFmpeg is unavailable, keep the original file.
                if not out_video_path.exists() and tmp_video_path.exists():
                    tmp_video_path.rename(out_video_path)

            manifest: Dict[str, object] = {
                "model_id": self.model_id,
                "video_name": video_name,
                "frame_one": str(frame_one),
                "frame_ten": str(frame_ten),
                "reference_video": str(video_path) if video_path else None,
                "num_frames": len(frames),
                "target_fps": fps,
                "canvas_height": self.height,
                "canvas_width": self.width,
                "device": self._device,
                "dtype": self._dtype_str,
                "num_inference_steps": self.num_inference_steps,
                "seed": self.seed,
                "motion_bucket_id": self.motion_bucket_id,
                "noise_aug_strength": self.noise_aug_strength,
            }

            return BaselineResult(
                success=True,
                message=f"Stable Video Diffusion 成功生成 {len(frames)} 帧视频 @ {fps:.1f} FPS",
                html_path=None,
                video_path=str(out_video_path),
                manifest=manifest,
            )

        except Exception as exc:
            return BaselineResult(
                success=False,
                message=f"Stable Video Diffusion 生成失败: {exc}",
                html_path=None,
                video_path=None,
                manifest={
                    "model_id": self.model_id,
                    "video_name": video_name,
                    "frame_one": str(frame_one),
                    "frame_ten": str(frame_ten),
                    "reference_video": str(video_path) if video_path else None,
                    "device": self._device,
                    "dtype": self._dtype_str,
                    "stage": "inference",
                },
            )


class StableVideoDiffusionImg2VidBaseline(_StableVideoDiffusionBase):
    """Baseline using stabilityai/stable-video-diffusion-img2vid."""

    name: str = "stable-video-img2vid"
    model_id: str = "stabilityai/stable-video-diffusion-img2vid"


class StableVideoDiffusionImg2VidXTBaseline(_StableVideoDiffusionBase):
    """Baseline using stabilityai/stable-video-diffusion-img2vid-xt."""

    name: str = "stable-video-img2vid-xt"
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"


__all__ = [
    "StableVideoDiffusionImg2VidBaseline",
    "StableVideoDiffusionImg2VidXTBaseline",
]
