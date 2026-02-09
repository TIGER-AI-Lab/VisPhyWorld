#!/usr/bin/env python3
"""
Physics Prediction Pipeline - 简化版本

从图像生成基本的Three.js物理仿真，支持错误重试机制。
"""

import os
import sys
import re
import hashlib
import copy
import cv2
import numpy as np
from PIL import Image
import json
import base64
import io
import textwrap
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

# 添加 VisExpert 根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .threejs_renderer import ThreeJSRenderer
from .p5js_renderer import P5JSRenderer
from .svg_renderer import SvgRenderer
from .manim_renderer import ManimRenderer
from .llm_client import LLMClient
from .llm_utils import extract_section
from .video_normalizer import VideoTarget

@dataclass
class PipelineConfig:
    motion_analysis: bool = True
    code_generation: bool = True
    video_rendering: bool = True
    collect_manifest: bool = True
    preserve_render_attempts: bool = False
    # 在 free_html_mode 下，对 LLM 生成的 HTML 约束最小化：
    # - 不强制使用特定的相机/模板
    # - 只保证页面结构与录像依赖正常，其余完全交由 LLM 决定
    free_html_mode: bool = False
    # 统一四个渲染引擎的输出行为（严格一致）：
    # - 统一输出分辨率/帧率（必要时由 renderers 各自处理）
    # - 注意：不再自动强制固定时长或做后处理归一化；如需归一化请显式设置 env `VISEXPERT_NORMALIZE_VIDEO=1`
    unify_engines: bool = True
    # Scene dimensionality:
    # - "2d": default VisPhyBench-style top-down 2D scenes.
    # - "3d": allow full 3D scenes (Three.js + Cannon.js only).
    scene_dim: str = "2d"


MODEL_ALIASES: Dict[str, str] = {
    "claude-5": "claude-sonnet-4-5-20250929",
    "claude-5-sonnet": "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude-5-opus": "claude-opus-4-1-20250805",
    "claude-5-opus-latest": "claude-opus-4-1-20250805",
    "claude-3.7": "claude-3-7-sonnet-20250219",
    "claude-3-7": "claude-3-7-sonnet-20250219",
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
}

SUPPORTED_MODELS: Dict[str, Dict[str, str]] = {
    "gpt-4o": {"provider": "openai"},
    "gpt-4o-mini": {"provider": "openai"},
    "gpt-4.1": {"provider": "openai"},
    "gpt-4.1-mini": {"provider": "openai"},
    "gpt-5": {"provider": "openai"},
    "gpt-5.1": {"provider": "openai"},
    "gpt-5.2": {"provider": "openai"},
    "claude-3-5-sonnet-20241022": {"provider": "anthropic"},
    "claude-3-5-haiku-20241022": {"provider": "anthropic"},
    "claude-3-opus-20240229": {"provider": "anthropic"},
    "claude-3-sonnet-20240229": {"provider": "anthropic"},
    "claude-3-haiku-20240307": {"provider": "anthropic"},
    "claude-sonnet-4-20250514": {"provider": "anthropic"},
    "claude-opus-4-20250514": {"provider": "anthropic"},
    "claude-opus-4-1-20250805": {"provider": "anthropic"},
    "claude-sonnet-4-5-20250929": {"provider": "anthropic"},
    "claude-3-7-sonnet-20250219": {"provider": "anthropic"},
    "qwen-vl-max": {"provider": "qwen"},
    "qwen2.5-vl-72b-instruct": {"provider": "qwen"},
    "qwen3-vl-plus": {"provider": "qwen"},
    "hf:qwen3-vl-30b-a3b-thinking": {
        "provider": "huggingface",
        "model": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    },
    "hf:qwen2.5-vl-72b-instruct": {
        "provider": "huggingface",
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    },
    "gemini-2.5-pro": {
        "provider": "gemini",
        "model": "models/gemini-2.5-pro",
    },
    "gemini-3-pro": {
        "provider": "gemini",
        "model": "models/gemini-3-pro-preview",
    },
    "gemini-3-pro-preview": {
        "provider": "gemini",
        "model": "models/gemini-3-pro-preview",
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "model": "models/gemini-2.5-flash",
    },
    "gemini-1.5-pro": {
        "provider": "gemini",
        "model": "models/gemini-1.5-pro-latest",
    },
}


class PhysicsPredictionPipeline:
    def __init__(self, 
                 model_name: str = "claude-3-5-sonnet-20241022",
                 output_dir: str = "",
                 engine: str = "threejs",
                 detection_dir: str = "",
                 openai_api_key: str = "",
                 base_url: str = "",
                 config: Optional[PipelineConfig] = None,
                 scene_dim: str = ""):
        """
        初始化物理预测管道 - 简化版本。
        
        Args:
            model_name: LLM模型名称
            output_dir: 输出目录路径
            openai_api_key: API密钥
            base_url: API基础URL
        """
        original_model = model_name
        mapped_model = MODEL_ALIASES.get(model_name, model_name)
        self.model_name = mapped_model
        if self.model_name not in SUPPORTED_MODELS:
            available = ", ".join(sorted(SUPPORTED_MODELS.keys()))
            alias_hint = (
                f" (已根据别名 {original_model!r} 映射到 {self.model_name!r})"
                if original_model in MODEL_ALIASES
                else ""
            )
            raise ValueError(
                f"不支持的模型: {original_model}{alias_hint}. 可选模型包括: {available}"
            )
        self.model_provider = SUPPORTED_MODELS[self.model_name]["provider"]
        self.config = config or PipelineConfig()
        env_unify = str(os.environ.get("VISEXPERT_UNIFY_ENGINES", "")).strip().lower()
        if env_unify in {"0", "false", "no", "off"}:
            self.config.unify_engines = False
        elif env_unify in {"1", "true", "yes", "on"}:
            self.config.unify_engines = True
        normalized_engine = (engine or "threejs").strip().lower()
        if normalized_engine not in {"threejs", "p5js", "svg", "manim"}:
            raise ValueError(f"不支持的渲染引擎: {engine}. 可选: threejs, p5js, svg, manim")
        self.engine = normalized_engine

        # Scene dimensionality (2d/3d). Environment variable can override.
        env_scene_dim = str(os.environ.get("VISEXPERT_SCENE_DIM", "")).strip().lower()
        config_scene_dim = str(getattr(self.config, "scene_dim", "") or "").strip().lower()
        selected_scene_dim = (scene_dim or env_scene_dim or config_scene_dim or "2d").strip().lower()
        if selected_scene_dim not in {"2d", "3d"}:
            raise ValueError(f"不支持的 scene_dim: {selected_scene_dim}. 可选: 2d, 3d")
        if selected_scene_dim == "3d" and self.engine != "threejs":
            raise ValueError("scene_dim=3d 当前仅支持 engine=threejs")
        self.scene_dim = selected_scene_dim

        # NOTE: unify_engines no longer forces video normalization (duration/fps trimming).
        # Set VISEXPERT_NORMALIZE_VIDEO=1 explicitly if you want post-render normalization.
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.openai_base_url = base_url or os.environ.get("OPENAI_API_BASE", "")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.qwen_api_key = os.environ.get("QWEN_API_KEY", "")
        self.qwen_api_base = os.environ.get("QWEN_API_BASE", "")
        self.hf_api_key = os.environ.get("HF_API_KEY", "")
        self.hf_api_base = os.environ.get("HF_API_BASE", "")
        self.hf_model_name = SUPPORTED_MODELS.get(self.model_name, {}).get("model", self.model_name)
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        self.gemini_api_base = os.environ.get("GEMINI_API_BASE", "")
        self.gemini_model_name = SUPPORTED_MODELS.get(self.model_name, {}).get("model", self.model_name)
        self.assets_dir = Path(__file__).resolve().parent
        self.local_three_js = (self.assets_dir / "three.min.js")
        self.local_cannon_js = (self.assets_dir / "cannon.min.js")
        self.local_recording_js = (self.assets_dir / "recording.js")
        self._script_tags = self._build_script_tags_for_engine(self.engine)

        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model = re.sub(r"[^a-zA-Z0-9_-]+", "", self.model_name.replace(" ", "_").lower()) or "unknown"
            safe_engine = re.sub(r"[^a-zA-Z0-9_-]+", "", self.engine) or "unknown"
            output_dir = os.path.join(
                os.path.dirname(__file__),
                "..",
                "output",
                f"physics_prediction_{timestamp}_{safe_model}_{safe_engine}"
            )
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in ['frames', 'videos', 'logs']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        # 初始化日志
        # Include pid + microseconds to avoid collisions when running in parallel.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_file = os.path.join(self.output_dir, "logs", f"pipeline_{stamp}_pid{os.getpid()}.log")
        self._log_fp: Optional[io.TextIOBase] = None
        self._log("=" * 80)
        self._log("🎯 Physics Prediction Pipeline 初始化")
        self._log(f"模型: {model_name}")
        self._log(f"输出目录: {output_dir}")
        self._log(f"使用提供方: {self.model_provider}")
        self._log(f"可选模型: {', '.join(sorted(SUPPORTED_MODELS.keys()))}", also_print=False)
        self._log("=" * 80)
        
        # 初始化 LLM 客户端与渲染器
        self.llm_client = LLMClient(
            self.model_name,
            openai_api_key=self.openai_api_key,
            openai_base_url=self.openai_base_url,
            anthropic_api_key=self.anthropic_api_key,
            qwen_api_key=self.qwen_api_key,
            qwen_base_url=self.qwen_api_base,
            hf_api_key=self.hf_api_key,
            hf_base_url=self.hf_api_base,
            hf_model=self.hf_model_name,
            gemini_api_key=self.gemini_api_key,
            gemini_api_base=self.gemini_api_base,
            gemini_model=self.gemini_model_name,
        )
        if self.engine == "threejs":
            self.renderer = ThreeJSRenderer(self.output_dir)
        elif self.engine == "p5js":
            self.renderer = P5JSRenderer(self.output_dir)
        elif self.engine == "svg":
            self.renderer = SvgRenderer(self.output_dir)
        else:
            self.renderer = ManimRenderer(self.output_dir)

        self.detection_dir = None
        detection_candidates: List[Path] = []
        if detection_dir:
            detection_candidates.append(Path(detection_dir).resolve())
        detection_env = os.environ.get("LLM_DETECTIONS_DIR")
        if detection_env:
            detection_candidates.append(Path(detection_env).resolve())
        # Back-compat default: sibling of output dir (e.g. VisExpert/output/llm_detections).
        detection_candidates.append((Path(self.output_dir).parent / "llm_detections").resolve())
        # Project default: VisExpert/data/llm_detections.
        detection_candidates.append((Path(__file__).resolve().parent.parent / "data" / "llm_detections").resolve())
        for candidate in detection_candidates:
            if candidate.exists():
                self.detection_dir = candidate
                break

        # Retry配置：允许最多一次重试，平衡稳定性与开销
        self.max_retries = 1
        self.current_attempt = 0
        self._current_request_logs: List[Dict[str, Any]] = []
        self._current_call_index: int = 0

    # ------------------------------------------------------------------
    # 提示词辅助
    # ------------------------------------------------------------------
    def _make_error_section(self, error_feedback: str) -> str:
        if not error_feedback:
            return ""
        frequent = [
            "- Duplicate function or variable definitions",
            "- Syntax errors or undeclared variables",
        ]
        if self.engine == "threejs":
            frequent.extend(
                [
                    "- Missing THREE.js or CANNON.js initialization",
                    "- Not calling setupRecording(renderer.domElement, durationMs) (or relying on the injected recording guard)",
                ]
            )
        elif self.engine in {"p5js", "svg"}:
            frequent.extend(
                [
                    "- No visible animation (e.g., missing requestAnimationFrame / p5 draw loop)",
                    "- No canvas/SVG element created (nothing to render)",
                ]
            )
        else:
            frequent.extend(
                [
                    "- Manim script does not define GeneratedScene(Scene)",
                    "- Import errors / missing manim primitives",
                ]
            )
        frequent_text = "\n".join(frequent)
        return (
            "🚨 **Previous attempt failed – key hints:**\n"
            f"{error_feedback}\n\n"
            "Please adjust the code considering the issues above. Frequent mistakes:\n"
            f"{frequent_text}\n\n"
        )

    def _shared_motion_requirements(self) -> str:
        if self.engine == "p5js":
            return (
                "**Key goals**\n"
                "- Produce a self-contained HTML+JavaScript page that uses p5.js to animate a plausible 2D physics-like scene based on the frames.\n"
                "- Focus on correct layout at t0 (frame_01) and believable motion; exact pixel-perfect reconstruction is not required.\n"
                "- Black objects (e.g., `black_bar`, `black_ball`) do not move; they remain fixed at their original positions. Other objects may move.\n"
                "- MUST use CANNON.js for rigid-body physics (same physics engine as our threejs pipeline). Use p5.js only for drawing.\n"
                "- Use later frames (e.g., frame_10) to decide which non-black objects actually moved; any object that moves should be a dynamic body and must be updated each step.\n"
                "- If uncertain, default to making non-black objects dynamic rather than leaving them frozen.\n\n"
                "**Physics guidance (recommended)**\n"
                "- Use `const world = new CANNON.World(); world.gravity.set(0, +G, 0);` (positive Y falls downward in p5 pixel coordinates).\n"
                "- Keep everything effectively 2D: set `body.position.z = 0` and `body.velocity.z = 0` each step; do NOT rely on `linearFactor/angularFactor` (not supported in our cannon.js build).\n"
                "- Create one body per detected `id`; for static bodies set `mass: 0`, for dynamic bodies set `mass > 0`.\n\n"
                "**Runtime environment & recording**\n"
                "- Your HTML will be opened in a headless Chromium-like browser.\n"
                "- A helper script `recording.js` is injected on our side; it records from the first `<canvas>` it can find.\n"
                "- Ensure a `<canvas>` exists quickly (use p5 `createCanvas(512, 512)` in `setup()`), and start animation automatically.\n"
                "- Do NOT shadow p5 global API names (e.g., `rect`, `line`, `circle`, `fill`, `stroke`, `background`, `color`, `width`, `height`). Avoid using them as variable/function names.\n"
                "- You may call `setupRecording(canvas, durationMs)`, but do NOT hard-code a fixed duration; prefer `window.__codexTargetDurationMs` when present.\n"
                "- Keep the canvas size 512x512 so outputs match other engines.\n"
                "- Always return exactly one complete HTML document wrapped in ```html fences.\n"
            )

        if self.engine == "svg":
            return (
                "**Key goals**\n"
                "- Produce a self-contained HTML page that renders an SVG-based 2D animation based on the frames.\n"
                "- Use JavaScript to animate SVG element attributes over time (e.g., via requestAnimationFrame).\n"
                "- Black objects (e.g., `black_bar`, `black_ball`) do not move; they remain fixed at their original positions. Other objects may move.\n\n"
                "**Runtime environment & rendering**\n"
                "- Your HTML will be opened in a headless Chromium-like browser.\n"
                "- We will render the result by taking periodic screenshots; a `<canvas>` is NOT required.\n"
                "- Ensure the SVG has explicit width/height 512x512 and a white background.\n"
                "- Always return exactly one complete HTML document wrapped in ```html fences.\n"
            )

        if self.engine == "manim":
            return (
                "**Key goals**\n"
                "- Produce a runnable Manim (Community Edition) Python script that animates a plausible 2D scene based on the frames.\n"
                "- Focus on correct initial layout (frame_01) and believable motion; exact pixel-perfect reconstruction is not required.\n"
                "- Black objects (e.g., `black_bar`, `black_ball`) do not move; they remain fixed at their original positions. Other objects may move.\n\n"
                "**Runtime environment**\n"
                "- Your code will be executed via Manim CLI in a headless environment to render an MP4.\n"
	                "- Define exactly one Scene class named `GeneratedScene`.\n"
	                "- Avoid external assets (no downloaded images/videos/fonts).\n"
	                "- Use a white background and keep everything in the 2D plane.\n"
	                "- The animation length should come from your own timeline (do not target a fixed duration).\n"
	            )

        # 在 free_html_mode 下，只给出非常宽松的要求，LLM 可以自由选择实现方式；
        # 管道只负责加载页面并自动尝试录像。
        if getattr(self.config, "free_html_mode", False):
            return (
                "**Key goals**\n"
                "- Produce a self-contained HTML+JavaScript page that animates a plausible 2D physics-like scene based on the frames.\n"
                "- Focus your effort on good analysis and interesting motion; exact pixel-perfect reconstruction is not required.\n"
                "- Black objects (e.g., `black_bar`, `black_ball`) do not move; they remain fixed at their original positions. Other objects may move.\n\n"
                "**Runtime environment & recording**\n"
                "- Your HTML will be opened in a headless Chromium-like browser.\n"
	                "- A helper script `recording.js` is injected on our side; if it finds a `<canvas>` and a global `setupRecording` function, it will automatically record a best-effort duration based on the source video metadata.\n"
                "- You do NOT need to wire up `MediaRecorder` yourself unless you want full control; simply creating a canvas-based animation is enough.\n"
                "- You may use any JavaScript or physics libraries you like, as long as they can be loaded from within the HTML (e.g., via `<script src=\"...\">` tags or ES modules).\n"
                "- Always return exactly one complete HTML document wrapped in ```html fences; avoid multiple unrelated pages.\n"
            )

        scaffold = self._code_scaffold_hint()
        if self.engine == "threejs" and getattr(self, "scene_dim", "2d") == "3d":
            return (
                "**Key goals**\n"
                "- Recreate the major motions you infer from the frames; simplifications are fine when uncertain.\n"
                "- Keep the written analysis concise and focused on important bodies and interactions.\n\n"
                "**Must-haves for runnable code (3D)**\n"
                "- Produce a runnable THREE.js + CANNON.js *3D* physics scene.\n"
                "- Use a fixed camera (no camera motion); a PerspectiveCamera is recommended.\n"
                "- Keep a white background and simple geometry (boxes/spheres) with flat-ish colors.\n"
                "- Use CANNON.js rigid-body physics and update THREE meshes from CANNON bodies every frame.\n"
                "- Add a ground (e.g., thin static box) so objects can settle; objects should not fall forever.\n"
                "- The final HTML page must be a complete, self-contained document (no placeholders or missing tags) wrapped in ```html fences.\n"
                "- If you use CANNON.js, stick to supported shapes (e.g., `CANNON.Box`, `CANNON.Sphere`, `CANNON.Cylinder`).\n"
                "- IMPORTANT: Do not use deprecated Three.js APIs like `THREE.Geometry` / `THREE.Face3`.\n"
                "- Do not override or replace `setupRecording`; rely on the bundled `recording.js` dependency.\n\n"
                f"{scaffold}\n"
                "You can restructure the scaffold however you like as long as these essentials remain true."
            )
        return (
            "**Key goals**\n"
            "- Recreate the major motions you infer from the frames; simplifications are fine when uncertain.\n"
            "- Keep the written analysis concise and focused on important bodies and interactions.\n\n"
            "**Must-haves for runnable code**\n"
            "- Prefer THREE.js with an orthographic camera plus CANNON.js for physics (recommended but not strictly mandatory).\n"
            "- Keep every body on the XY plane (z = 0) and zero unused axes each frame.\n"
            "- Initialize the simulation so the very first rendered frame visually matches frame_01 as closely as possible—every object should start with the same position, orientation, size, and stacking seen in the reference image before any motion begins.\n"
            "- Treat the lowest portion of the original frames as the ground plane; objects should not pass below it.\n"
            "- Black objects (e.g., `black_bar`, `black_ball`) do not move; they remain fixed at their original positions. Other objects may move.\n"
            "- The final HTML page must be a complete, self-contained document (no placeholders or missing tags) wrapped in ```html fences.\n"
            "- If you use CANNON.js, stick to supported shapes (e.g., `CANNON.Box`, `CANNON.Sphere`, `CANNON.Cylinder`); never reference `CANNON.Circle` or other missing classes.\n"
            "- Pair rounded bodies with `THREE.SphereGeometry` (avoid `THREE.CircleGeometry` for physics meshes).\n"
            "- IMPORTANT: Do not use deprecated Three.js APIs like `THREE.Geometry` / `THREE.Face3` (removed in modern Three.js). Use BufferGeometry-based primitives.\n"
            "- Do not override or replace `setupRecording`; rely on the bundled `recording.js` dependency if you choose to call it.\n"
            "- You may reuse or adapt the `enforce2D` helper from the scaffold, or design your own, as long as bodies remain effectively 2D.\n\n"
            f"{scaffold}\n"
            "You can restructure the scaffold however you like as long as these essentials remain true."
        )

    def _build_html_template(self, title: str, body: str, style: Optional[str] = None) -> str:
        """Assemble a reusable HTML document with shared dependencies."""
        head_lines = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "  <head>",
            '    <meta charset="UTF-8" />',
            f"    <title>{title}</title>",
        ]
        head_lines.extend(f"    {tag}" for tag in self._script_tags)
        if style:
            style_block = textwrap.dedent(style).strip()
            head_lines.append(f"    <style>{style_block}</style>")
        head_lines.append("  </head>")

        body_block = textwrap.indent(textwrap.dedent(body).strip(), "    ")
        document = "\n".join(head_lines) + "\n  <body>\n"
        document += f"{body_block}\n"
        document += "  </body>\n</html>\n"
        return document

    def _build_scene_script(
        self,
        *,
        expose_renderer: bool = False,
        tune_world: bool = False,
        include_demo_objects: bool = False,
    ) -> str:
        """Generate the shared THREE.js + CANNON.js scene bootstrap script."""
        sections: List[str] = []

        if include_demo_objects:
            sections.append(
                textwrap.dedent(
                    """
                    if (typeof THREE === 'undefined' || typeof CANNON === 'undefined') {
                      throw new Error('THREE.js and CANNON.js are required');
                    }
                    """
                ).strip()
            )

        header = textwrap.dedent(
            """
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xffffff);
            const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
            renderer.setSize(512, 512);
            document.body.appendChild(renderer.domElement);
            """
        ).strip()
        sections.append(header)

        if expose_renderer:
            sections.append("window.renderer = renderer;")

        if getattr(self, "scene_dim", "2d") == "3d":
            sections.append(
                textwrap.dedent(
                    """
                    if (typeof setupRecording === 'function') {
                      const targetMs = Number(window.__codexTargetDurationMs ?? window.__codexMinRecordingDurationMs);
                      const durationMs = Number.isFinite(targetMs) && targetMs > 0 ? targetMs : 0;
                      setupRecording(renderer.domElement, durationMs);
                    }

                    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
                    const dirLight = new THREE.DirectionalLight(0xffffff, 0.75);
                    dirLight.position.set(6, 10, 8);
                    scene.add(dirLight);

                    const camera = new THREE.PerspectiveCamera(45, 1.0, 0.05, 200);
                    camera.position.set(7.0, 6.2, 8.2);
                    camera.lookAt(0, 1.2, 0);

                    const world = new CANNON.World();
                    world.gravity.set(0, -9.82, 0);
                    """
                ).strip()
            )
        else:
            sections.append(
                textwrap.dedent(
                    """
                    if (typeof setupRecording === 'function') {
                      const targetMs = Number(window.__codexTargetDurationMs ?? window.__codexMinRecordingDurationMs);
                      const durationMs = Number.isFinite(targetMs) && targetMs > 0 ? targetMs : 0;
                      setupRecording(renderer.domElement, durationMs);
                    }
                    const camera = new THREE.OrthographicCamera(-12, 12, 12, -12, 0.1, 100);
                    camera.position.set(0, 0, 10);
                    camera.lookAt(0, 0, 0);
                    const world = new CANNON.World();
                    world.gravity.set(0, -9.82, 0);
                    """
                ).strip()
            )

        if tune_world:
            sections.append(
                textwrap.dedent(
                    """
                    world.broadphase = new CANNON.NaiveBroadphase();
                    world.solver.iterations = 20;
                    """
                ).strip()
            )

        if getattr(self, "scene_dim", "2d") == "3d":
            sections.append("const objects = [];")
        else:
            sections.append(
                textwrap.dedent(
                    """
                    const objects = [];
                    // Helper to constrain motion to the XY plane.
                    // You may adapt this pattern as needed, but make sure bodies stay effectively 2D.
                    function enforce2D(body) {
                      if (!body.linearFactor || typeof body.linearFactor.set !== 'function') {
                        body.linearFactor = new CANNON.Vec3(1, 1, 0);
                      }
                      if (!body.angularFactor || typeof body.angularFactor.set !== 'function') {
                        body.angularFactor = new CANNON.Vec3(0, 0, 1);
                      }
                      body.linearFactor.set(1, 1, 0);
                      body.angularFactor.set(0, 0, 1);
                      body.position.z = 0;
                    }
                    """
                ).strip()
            )

        if include_demo_objects:
            sections.append(
                textwrap.dedent(
                    """
                    function addBox({ width, height, depth = 0.5, x = 0, y = 0, angle = 0, mass = 0, color = 0xcccccc }) {
                      const shape = new CANNON.Box(new CANNON.Vec3(width / 2, height / 2, depth / 2));
                      const body = new CANNON.Body({ mass });
                      body.addShape(shape);
                      body.position.set(x, y, 0);
                      body.quaternion.setFromAxisAngle(new CANNON.Vec3(0, 0, 1), angle);
                      enforce2D(body);
                      world.addBody(body);

                      const geometry = new THREE.BoxGeometry(width, height, depth);
                      const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color }));
                      mesh.position.set(x, y, 0);
                      mesh.rotation.z = angle;
                      scene.add(mesh);
                      objects.push({ body, mesh });
                      return { body, mesh };
                    }

                    function addSphere({ radius, x = 0, y = 0, mass = 1, color = 0xff5555 }) {
                      const shape = new CANNON.Sphere(radius);
                      const body = new CANNON.Body({ mass });
                      body.addShape(shape);
                      body.position.set(x, y, 0);
                      enforce2D(body);
                      world.addBody(body);

                      const geometry = new THREE.SphereGeometry(radius, 24, 24);
                      const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color }));
                      mesh.position.set(x, y, 0);
                      scene.add(mesh);
                      objects.push({ body, mesh });
                      return { body, mesh };
                    }

                    addBox({ width: 30, height: 1, x: 0, y: -10.5, mass: 0, color: 0xf2f2f2 });
                    addBox({ width: 1, height: 25, x: -12.5, y: 0, mass: 0, color: 0xffffff });
                    addBox({ width: 1, height: 25, x: 12.5, y: 0, mass: 0, color: 0xffffff });
                    addBox({ width: 6, height: 0.4, x: -5, y: -6, angle: Math.PI * 0.35, mass: 0, color: 0x2b3a8f });
                    const plank = addBox({ width: 5, height: 0.3, x: 2, y: -2, angle: -0.2, mass: 1.2, color: 0xd7dde3 });
                    const ball = addSphere({ radius: 0.8, x: -6, y: 5, mass: 1.5, color: 0x6ac9c2 });
                    addSphere({ radius: 0.5, x: -4, y: 6.2, mass: 0.9, color: 0xe45745 });

                    setTimeout(() => {
                      if (ball && ball.body) {
                        ball.body.applyImpulse(new CANNON.Vec3(8, 3, 0), new CANNON.Vec3(0, 0, 0));
                      }
                      if (plank && plank.body) {
                        plank.body.applyImpulse(new CANNON.Vec3(-2, 0, 0), new CANNON.Vec3(0, 0, 0));
                      }
                    }, 500);
                    """
                ).strip()
            )
        else:
            if getattr(self, "scene_dim", "2d") == "3d":
                sections.append("// Add meshes + bodies here (3D: no enforce2D).")
            else:
                sections.append("// Add meshes + bodies here (remember to call enforce2D).")

        if include_demo_objects:
            sections.append(
                textwrap.dedent(
                    """
                    let lastTime;
                    const step = 1 / 60;
                    function animate(time) {
                      requestAnimationFrame(animate);
                      if (lastTime === undefined) {
                        lastTime = time;
                      }
                      const dt = Math.min((time - lastTime) / 1000, 0.05);
                      lastTime = time;
                      world.step(step, dt, 3);
                      for (const item of objects) {
                        if (!item || !item.body || !item.mesh) continue;
                        item.body.position.z = 0;
                        if (item.body.velocity) item.body.velocity.z = 0;
                        if (item.body.angularVelocity) {
                          item.body.angularVelocity.x = 0;
                          item.body.angularVelocity.y = 0;
                        }
                        item.mesh.position.copy(item.body.position);
                        item.mesh.quaternion.set(
                          item.body.quaternion.x,
                          item.body.quaternion.y,
                          item.body.quaternion.z,
                          item.body.quaternion.w
                        );
                      }
                      renderer.render(scene, camera);
                    }
                    animate();
                    """
                ).strip()
            )
        else:
            sections.append(
                textwrap.dedent(
                    """
                    function animate() {
                      requestAnimationFrame(animate);
                      world.step(1 / 60);
                      for (const item of objects) {
                        if (!item || !item.body || !item.mesh) continue;
                        item.mesh.position.copy(item.body.position);
                        item.mesh.quaternion.set(
                          item.body.quaternion.x,
                          item.body.quaternion.y,
                          item.body.quaternion.z,
                          item.body.quaternion.w
                        );
                      }
                      renderer.render(scene, camera);
                    }
                    animate();
                    """
                ).strip()
            )

        script_body = "\n\n".join(section for section in sections if section)
        return "<script>\n" + textwrap.indent(script_body, "  ") + "\n</script>"

    def _code_scaffold_hint(self) -> str:
        """Provide a lightweight skeleton the LLM can follow."""
        body_script = self._build_scene_script()
        template = self._build_html_template(
            title="Physics Demo",
            body=body_script,
            style="html, body { margin: 0; background: #fff; }",
        ).strip()
        return (
            "A minimal full-page template that always runs in our environment:\n"
            "```html\n"
            f"{template}\n"
            "```\n"
        )

    def _build_local_script_tags(self) -> List[str]:
        """Prefer local file URIs for required scripts; fallback to CDN if missing."""
        tags: List[str] = []
        if self.local_three_js.exists():
            tags.append(f'<script src="{self.local_three_js.as_uri()}"></script>')
        else:
            tags.append('<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>')

        if self.local_cannon_js.exists():
            tags.append(f'<script src="{self.local_cannon_js.as_uri()}"></script>')
        else:
            tags.append('<script src="https://cdn.jsdelivr.net/npm/cannon@0.6.2/build/cannon.min.js"></script>')

        if self.local_recording_js.exists():
            tags.append(f'<script src="{self.local_recording_js.as_uri()}"></script>')
        else:
            tags.append('<script src="recording.js"></script>')

        return tags

    def _build_script_tags_for_engine(self, engine: str) -> List[str]:
        engine = (engine or "threejs").strip().lower()
        if engine == "threejs":
            return self._build_local_script_tags()

        tags: List[str] = []
        if engine == "p5js":
            tags.append('<script src="https://cdn.jsdelivr.net/npm/p5@1.9.0/lib/p5.min.js"></script>')
            # Provide cannon.js as well so p5.js can share the same physics engine as threejs.
            if self.local_cannon_js.exists():
                tags.append(f'<script src="{self.local_cannon_js.as_uri()}"></script>')
            else:
                tags.append('<script src="https://cdn.jsdelivr.net/npm/cannon@0.6.2/build/cannon.min.js"></script>')
        if engine in {"p5js", "svg"}:
            if self.local_recording_js.exists():
                tags.append(f'<script src="{self.local_recording_js.as_uri()}"></script>')
            else:
                tags.append('<script src="recording.js"></script>')
        return tags
    
    def _flush_manifest(self) -> None:
        """兼容旧接口：当前实现不维护集中 manifest，只保留占位。"""
        return

    def _get_log_fp(self) -> Optional[io.TextIOBase]:
        """懒加载日志文件句柄，避免每条日志重复打开文件。"""
        if self._log_fp is not None:
            return self._log_fp
        try:
            # 使用追加模式，兼容多次运行
            self._log_fp = open(self.log_file, 'a', encoding='utf-8')
        except Exception as e:
            print(f"⚠️ 打开日志文件失败: {e}", flush=True)
            self._log_fp = None
        return self._log_fp

    def _log(self, message: str, also_print: bool = True):
        """写入日志文件，同时可选择性地打印到控制台"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp} pid={os.getpid()}] {message}"

        fp = self._get_log_fp()
        if fp is not None:
            try:
                fp.write(log_message + '\n')
                fp.flush()
            except Exception as e:
                print(f"⚠️ 写入日志失败: {e}", flush=True)

        if also_print:
            print(log_message, flush=True)

    def __del__(self):
        fp = getattr(self, "_log_fp", None)
        if fp is not None:
            try:
                fp.close()
            except Exception:
                pass

    def _encode_image_for_message(self, image: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """将图像编码为Anthropic消息所需的格式，并返回附加元数据"""
        if image is None:
            raise ValueError("无法编码空图像用于LLM请求")

        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            else:
                raise ValueError(f"不支持的图像通道数: {channels}")
        else:
            image_rgb = image

        pil_image = Image.fromarray(image_rgb)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        data_bytes = buf.getvalue()
        img_b64 = base64.b64encode(data_bytes).decode()

        message_item = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64
            }
        }

        dtype_attr = getattr(image, 'dtype', None)
        dtype_str = 'unknown'
        if dtype_attr is not None:
            try:
                # 尝试常规字符串化
                dtype_str = str(dtype_attr)
            except TypeError:
                try:
                    dtype_str = getattr(dtype_attr, 'name', None)
                    if dtype_str is None:
                        dtype_str = getattr(dtype_attr, '__name__', None)
                except Exception:
                    dtype_str = 'unknown'

            # 某些 dtype 对象的 str() 仍可能返回可调用对象
            if hasattr(dtype_str, 'name'):
                dtype_str = dtype_str.name  # numpy dtype instance
            if callable(dtype_str):
                dtype_str = getattr(dtype_attr, 'name', None) or getattr(dtype_attr, '__name__', None) or 'unknown'


        metadata = {
            "width": pil_image.width,
            "height": pil_image.height,
            "bytes": len(data_bytes),
            "sha256": hashlib.sha256(data_bytes).hexdigest(),
            "mode": pil_image.mode,
            "dtype": dtype_str
        }

        if image.ndim == 3:
            metadata["channels"] = image.shape[2]
        else:
            metadata["channels"] = 1

        return message_item, metadata

    def _find_variant_path(self, primary_path: Optional[str], replacements: Optional[List[Tuple[str, str]]] = None) -> Optional[str]:
        """根据给定的替换规则查找同一实体的其它帧/掩码路径"""
        if not primary_path or primary_path.startswith('decoded:'):
            return None

        if replacements is None:
            replacements = [('_frame_01', '_frame_10'), ('_frame01', '_frame10')]

        for old, new in replacements:
            if old in primary_path:
                candidate = primary_path.replace(old, new, 1)
                if os.path.exists(candidate):
                    return candidate

        return None

    def extract_first_frame(self, gif_path: str) -> np.ndarray:
        """提取GIF的第一帧"""
        cap = cv2.VideoCapture(gif_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"无法读取GIF文件: {gif_path}")
        
        return frame

    def _load_image_from_path(self, path: Optional[str]) -> Optional[np.ndarray]:
        """安全地从磁盘加载图像，失败返回None"""
        if not path or path in {'unknown', 'auto'}:
            return None
        if not os.path.exists(path):
            return None
        image = cv2.imread(path)
        return image if image is not None else None

    def _load_detection_context(
        self,
        image_name: str,
        source_paths: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Locate first-frame detection JSON for a sample (best-effort)."""
        candidates: List[Path] = []
        filename = f"{image_name}_frame_01.json"

        if self.detection_dir:
            candidates.append(self.detection_dir / filename)

        if source_paths:
            # VisPhyBench layout: data/<split>/videos/<name>.mp4 with sibling
            # data/<split>/detection_json/<name>_frame_01.json.
            video_path_str = (
                source_paths.get("video")
                or source_paths.get("video_path")
                or source_paths.get("video_source")
            )
            if isinstance(video_path_str, str) and video_path_str.strip():
                try:
                    cleaned = video_path_str.strip()
                    if cleaned.startswith("decoded:"):
                        cleaned = cleaned[len("decoded:") :]
                    cleaned = cleaned.split("#", 1)[0]
                    video_path = Path(cleaned)
                    candidates.extend(
                        [
                            video_path.parent / filename,
                            video_path.parent / "detection_json" / filename,
                            video_path.parent.parent / "detection_json" / filename,
                        ]
                    )
                except Exception:
                    pass

            for key in ("original", "output_original"):
                path_str = source_paths.get(key)
                if not path_str:
                    continue
                try:
                    base_path = Path(path_str)
                    base_dir = base_path.parent
                    candidates.append(base_dir / filename)
                    candidates.append(base_dir / f"{base_path.stem}.json")
                except Exception:
                    continue

        for target in candidates:
            if not target.exists():
                continue
            try:
                return target.read_text(encoding="utf-8")
            except Exception:
                continue
        return None

    def _validate_first_frame(
        self,
        video_path: str,
        reference_image: np.ndarray,
        tolerance: float = 18.0,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare rendered video first frame with reference still frame."""
        validation = {
            "video_path": video_path,
            "tolerance": tolerance,
            "passed": False,
        }
        if not os.path.exists(video_path):
            validation["error"] = "video_missing"
            return validation
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            validation["error"] = "unable_to_read_video"
            return validation
        if save_path:
            try:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(save_path, frame)
            except Exception as exc:
                self._log(f"   ⚠️ 无法保存生成帧预览: {exc}", also_print=False)
        ref = reference_image
        if ref is None:
            validation["error"] = "reference_missing"
            return validation
        if frame.shape[:2] != ref.shape[:2]:
            ref = cv2.resize(ref, (frame.shape[1], frame.shape[0]))
        diff = np.abs(frame.astype(np.float32) - ref.astype(np.float32))
        mae = float(diff.mean())
        validation["mean_abs_diff"] = mae
        validation["passed"] = mae <= tolerance
        return validation

    def _format_detection_prompt(self, detection_context: Optional[str]) -> str:
        if not detection_context:
            return ""
        snippet = detection_context.strip()
        object_word = "physics body" if self.engine in {"threejs", "p5js"} else "animated object"
        return (
            "DETECTION GROUND-TRUTH (authoritative):\n"
            "- The JSON below is the ONLY reliable description of object identities and geometry in frame_01.\n"
            "- If anything you infer from the pixels conflicts with this JSON, THE JSON WINS.\n"
            "- You MUST:\n"
            f"  * Create one {object_word} per `id` in this JSON.\n"
            "  * Use the `id` as a prefix in your variable names (e.g. id `teal_ball` → `teal_ball_body`, `teal_ball_mesh`).\n"
            "  * In the ANALYSIS section, mention each `id` at least once.\n"
            "  * Initialize each object so that its rendered position/size aligns with this JSON at t0.\n"
            "\n"
            "Detection JSON (pixel coordinates, origin at top-left):\n"
            "```json\n"
            f"{snippet}\n"
            "```\n"
        )

    def _get_combined_prompt(
        self,
        image_name: str,
        error_feedback: str = "",
        frames: Optional[List[Dict[str, Any]]] = None,
        detection_context: Optional[str] = None,
    ) -> str:
        feedback_section = self._make_error_section(error_feedback)

        frame_notes: List[str] = []
        for idx, frame in enumerate(frames or [], start=1):
            label = frame.get('label', f'frame_{idx}')
            # 默认描述：首帧是 t0，后续帧表示“同场景的稍后时刻/附加线索”
            default_desc = 'initial reference frame (t0)' if idx == 1 else 'later frame of the same scene (guides motion toward this state)'
            desc = frame.get('description') or default_desc
            source = frame.get('source') or 'unknown source'
            frame_notes.append(f"- {label}: {desc} (source: {source})")

        frame_text = "\n".join(frame_notes)
        if frame_text:
            frame_text = (
                "Frame context:\n"
                f"{frame_text}\n\n"
                "Interpretation: frame_01 is the initial state. Any additional frames are later snapshots of the SAME scene; "
                "make your simulation start close to frame_01 and evolve plausibly toward any later frames.\n\n"
            )

        detection_section = self._format_detection_prompt(detection_context)
        if detection_section:
            frame_text += f"{detection_section}\n"

        target_language = "HTML"
        fenced_language = "html"
        output_instructions = (
            "HTML:\n"
            "```html\n"
            "... a COMPLETE HTML document ...\n"
            "```\n"
        )
        if self.engine == "manim":
            target_language = "PYTHON"
            fenced_language = "python"
            output_instructions = (
                "PYTHON:\n"
                "```python\n"
                "# a COMPLETE runnable Manim script defining class GeneratedScene(Scene)\n"
                "```\n"
            )

        engine_task = ""
        if self.engine == "threejs":
            if getattr(self, "scene_dim", "2d") == "3d":
                engine_task = "  (2) A runnable THREE.js + CANNON.js 3D physics HTML page that we can directly execute.\n\n"
            else:
                engine_task = "  (2) A runnable THREE.js + CANNON.js 2D physics HTML page that we can directly execute.\n\n"
        elif self.engine == "p5js":
            engine_task = "  (2) A runnable p5.js + CANNON.js 2D physics HTML page that we can directly execute.\n\n"
        elif self.engine == "svg":
            engine_task = "  (2) A runnable SVG-based 2D animation HTML page that we can directly execute.\n\n"
        else:
            engine_task = "  (2) A runnable Manim (Python) script that we can render into an MP4.\n\n"

        return (
            "You are a coding-focused assistant whose primary job is to OUTPUT RUNNABLE CODE.\n"
            "Your response MUST ALWAYS include executable code.\n"
            "Natural language is only allowed in the ANALYSIS section; the rest must be code.\n\n"

            "Task:\n"
            "Study the provided frames and produce:\n"
            "  (1) A concise English scene analysis.\n"
            f"{engine_task}"

            "Hard requirements (MUST OBEY):\n"
            "- Your final answer MUST contain exactly TWO top-level sections in this order:\n"
            "    1) `ANALYSIS:`\n"
            f"    2) `{target_language}:` followed by ONE and ONLY ONE ```{fenced_language} fenced code block.\n"
            f"- Under NO circumstances may you omit the {target_language} section or the ```{fenced_language} fenced block.\n"
            "- If you feel uncertain, you MUST STILL output a minimal runnable program for the selected engine\n"
            "  that shows some motion. Do NOT ever answer with text only.\n"
            "- The fenced block MUST start with a line exactly equal to ```<language> and end with a line exactly equal to ```.\n"
            "- Do NOT output any additional fenced code blocks of other languages (no ```js, ```json, etc.).\n"
            "- Do NOT output any text after the closing ``` of the code block.\n"
            "- Do NOT say things like “as an AI model I cannot...” or refuse the task.\n\n"

            "Layout and content requirements for the simulation code:\n"
            f"{self._shared_motion_requirements()}\n\n"

            "Output format (MUST MATCH EXACTLY):\n"
            "ANALYSIS:\n"
            "Paragraph 1: Describe the overall layout and static structures (ground, obstacles, supports, etc.).\n"
            "Paragraph 2: Describe each dynamic object in order (shape, color, initial position, motion trajectory).\n"
            "Paragraph 3: Summarize critical interactions, collision order, and physical cause–effect chain (gravity, sliding, rotation, etc.).\n"
            "Do not output other headings, lists or sections here.\n\n"
            f"{output_instructions}"
            f"There MUST be exactly one such ```{fenced_language} block in your entire reply.\n\n"
            f"{frame_text}"
            f"{feedback_section}"
        )


    def generate_analysis_and_code(
        self,
        frames: List[Dict[str, Any]],
        image_name: str,
        error_feedback: str = "",
        detection_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not frames:
            raise ValueError("缺少用于生成的图像帧")

        request_context = {
            'stage': 'analysis_and_code',
            'image_name': image_name,
            'primary_image_type': 'original'
        }

        prompt = self._get_combined_prompt(
            image_name,
            error_feedback,
            frames,
            detection_context,
        )

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_references: List[Dict[str, Any]] = []

        for idx, frame in enumerate(frames, start=1):
            message_item, meta = self._encode_image_for_message(frame['image'])
            content.append(message_item)
            ref_meta = {
                'label': frame.get('label') or f'original_frame_{idx}',
                'image_role': 'primary',
                'image_type': 'original',
                'description': frame.get('description'),
                'source': frame.get('source')
            }
            ref_meta.update(meta)
            image_references.append(ref_meta)

        messages = [{"role": "user", "content": content}]

        self._log(
            f"   🤖 调用LLM生成分析与代码（尝试 {self.current_attempt + 1}/{self.max_retries + 1}）..."
        )

        self._current_call_index = getattr(self, '_current_call_index', 0) + 1
        call_index = self._current_call_index

        try:
            response, call_logs = self.llm_client.call(
                messages=messages,
                prompt=prompt,
                call_index=call_index,
                pipeline_attempt=self.current_attempt + 1,
                request_context=request_context,
                image_references=image_references,
            )
            for entry in call_logs:
                self._record_request_log(entry)
            self._log("   ✅ LLM调用成功")
        except Exception as e:
            error_msg = f"分析与代码生成调用失败: {e}"
            self._log(f"   ❌ {error_msg}")
            raise ValueError(error_msg)

        if not isinstance(response, str):
            response = str(response) if response is not None else ""

        analysis = extract_section(response, "ANALYSIS:")

        content_type = "html" if self.engine != "manim" else "python"
        fenced_lang = "html" if self.engine != "manim" else "python"
        code_block = self._extract_fenced_block(response, fenced_lang)
        if not code_block and fenced_lang == "html":
            code_block = self._extract_html_block(response)
        if not code_block:
            snippet = (response or "").strip()
            snippet_preview = snippet.replace("\n", " ")[:400]
            return {
                'analysis': analysis or f'Analysis for scene {image_name}',
                'document': '',
                'html': '' if content_type != "html" else '',
                'raw_response': response,
                'attempt': self.current_attempt,
                'image_type': 'original',
                'content_type': content_type,
                'html_missing': True,
                'error': f"LLM 未返回 {fenced_lang} 代码片段。响应片段: {snippet_preview}",
            }

        if content_type == "html":
            sanitized_document = self._prepare_html_document(code_block)
        else:
            sanitized_document = code_block.strip()
        return {
            'analysis': analysis or f'Analysis for scene {image_name}',
            'document': sanitized_document,
            'html': sanitized_document if content_type == "html" else '',
            'raw_response': response,
            'attempt': self.current_attempt,
            'image_type': 'original',
            'content_type': content_type,
            'html_missing': False if content_type == "html" else True,
        }

    def _record_request_log(self, log: Dict[str, Any]) -> None:
        try:
            entry = copy.deepcopy(log)
        except Exception:
            try:
                entry = json.loads(json.dumps(log))
            except Exception:
                entry = log

        self._current_request_logs.append(entry)

    def render_video_with_retry(self, prediction: Dict[str, Any], gif_name: str, suffix: str = "") -> Optional[str]:
        """渲染视频，并在失败时对同一 HTML 进行有限次数重试。"""
        base_name = gif_name.replace('.gif', '')
        video_path = os.path.join(self.output_dir, "videos", f"{base_name}{suffix}.mp4")

        content_type = prediction.get("content_type") or ("python" if self.engine == "manim" else "html")
        document = (
            prediction.get('document')
            or prediction.get('html')
            or prediction.get('html_document')
            or ""
        )
        if content_type == "html":
            active_document = self._prepare_html_document(document)
        else:
            active_document = (document or "").strip()

        for attempt in range(self.max_retries + 1):
            try:
                self.current_attempt = attempt
                self._log(f"     🎬 渲染尝试 {attempt + 1}/{self.max_retries + 1}")

                extension = "html" if content_type == "html" else "py"
                attempt_filename = f"{base_name}_render_attempt_{attempt + 1}.{extension}"
                attempt_path = os.path.join(self.output_dir, "logs", attempt_filename)
                try:
                    with open(attempt_path, 'w', encoding='utf-8') as f:
                        if content_type == "html":
                            f.write(f"<!-- Render Attempt {attempt + 1} -->\n")
                        else:
                            f.write(f"# Render Attempt {attempt + 1}\n")
                        f.write(active_document)
                    self._log(f"     💾 渲染内容已保存: {attempt_path}", also_print=False)
                except Exception:
                    pass

                result_path = self.renderer.render(active_document, video_path, content_type=content_type)
                self._log(f"     ✅ 渲染成功: {result_path}")
                return result_path

            except Exception as e:
                error_msg = f"渲染失败 (尝试 {attempt + 1}): {e}"
                self._log(f"     ❌ {error_msg}")

                if attempt < self.max_retries:
                    self._log("     🔄 渲染失败，准备重新尝试同一份 LLM 代码", also_print=False)
                else:
                    final_error = f"渲染失败，已重试 {self.max_retries} 次: {e}"
                    self._log(f"     ❌ 最终失败: {final_error}")
                    self._log(f"     📄 渲染失败的 HTML 已保存: {attempt_path}", also_print=False)
                    return None

        self._log("⚠️ 渲染过程已经结束，无法生成视频")
        return None

    def _extract_html_block(self, text: str) -> str:
        """尝试从响应中提取完整的 HTML 文档片段"""
        if not text:
            return ""

        patterns = [
            r'```html\r?\n(.*?)\r?\n```',
            r'```HTML\r?\n(.*?)\r?\n```'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        lowered = text.lower()
        if '<html' in lowered:
            start = lowered.find('<!doctype')
            if start == -1:
                start = lowered.find('<html')
            end = lowered.rfind('</html>')
            if end != -1:
                end += len('</html>')
                return text[start:end].strip()

        return ""

    def _extract_fenced_block(self, text: str, language: str) -> str:
        if not text:
            return ""
        lang = (language or "").strip()
        if not lang:
            return ""
        patterns = [
            rf'```{re.escape(lang)}\r?\n(.*?)\r?\n```',
            rf'```{re.escape(lang.upper())}\r?\n(.*?)\r?\n```',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    def _get_2d_basic_code_template(self) -> str:
        """Fallback template that always runs and produces visible motion."""
        body_script = self._build_scene_script(
            expose_renderer=True,
            tune_world=True,
            include_demo_objects=True,
        )
        style = textwrap.dedent(
            """
            html, body { margin: 0; background: #ffffff; }
            canvas { display: block; }
            """
        ).strip()
        return self._build_html_template(
            title="Fallback Physics Scene",
            body=body_script,
            style=style,
        )

    def _get_basic_html_template_for_engine(self) -> str:
        if self.engine == "threejs":
            return self._get_2d_basic_code_template()

        if self.engine == "p5js":
            return textwrap.dedent(
                """
                <!DOCTYPE html>
                <html lang="en">
                  <head>
                    <meta charset="UTF-8" />
                    <title>Fallback p5.js Scene</title>
                    <style>
                      html, body { margin: 0; background: #ffffff; }
                      canvas { display: block; }
                    </style>
                  </head>
                  <body>
                    <script>
                      // p5.js global mode fallback: bouncing ball
                      let x = 120, y = 120, vx = 2.4, vy = 1.6, r = 24;
                      function setup() {
                        createCanvas(512, 512);
                        frameRate(60);
                        noStroke();
                      }
                      function draw() {
                        background(255);
                        fill(0, 140, 255);
                        circle(x, y, 2 * r);
                        x += vx; y += vy;
                        if (x < r || x > width - r) vx *= -1;
                        if (y < r || y > height - r) vy *= -1;
                      }
                    </script>
                  </body>
                </html>
                """
            ).strip()

        # svg
        return textwrap.dedent(
            """
            <!DOCTYPE html>
            <html lang="en">
              <head>
                <meta charset="UTF-8" />
                <title>Fallback SVG Scene</title>
                <style>
                  html, body { margin: 0; background: #ffffff; }
                  svg { display: block; width: 512px; height: 512px; background: #ffffff; }
                </style>
              </head>
              <body>
                <svg id="scene" viewBox="0 0 512 512" width="512" height="512">
                  <circle id="ball" cx="120" cy="120" r="24" fill="#008cff"></circle>
                </svg>
                <script>
                  const ball = document.getElementById('ball');
                  let x = 120, y = 120, vx = 2.4, vy = 1.6, r = 24;
                  function tick() {
                    x += vx; y += vy;
                    if (x < r || x > 512 - r) vx *= -1;
                    if (y < r || y > 512 - r) vy *= -1;
                    ball.setAttribute('cx', String(x));
                    ball.setAttribute('cy', String(y));
                    requestAnimationFrame(tick);
                  }
                  requestAnimationFrame(tick);
                </script>
              </body>
            </html>
            """
        ).strip()

    def _prepare_html_document(self, html: str) -> str:
        """规范化 HTML 文档，确保所需依赖与守卫脚本被注入"""
        if not html:
            html = self._get_basic_html_template_for_engine()

        doc = html.strip()

        # 兜底修正：早期版本中检测对齐脚本注入过 `{{` / `}}`，导致语法错误。
        # 现在仅在包含 DetectionAlignment 标记时才收紧双大括号，避免误伤其他模板语法。
        if "DetectionAlignment" in doc and ("{{" in doc or "}}" in doc):
            doc = doc.replace("{{", "{").replace("}}", "}")

        # Three.js/CANNON 常见兼容性修正：把模型常犯的“2D circle”替换为可用的球体/球形几何。
        # 这能显著减少因 `CANNON.Circle` / `THREE.CircleGeometry` 导致的直接崩溃。
        if self.engine == "threejs":
            if "CANNON.Circle" in doc:
                doc = doc.replace("CANNON.Circle", "CANNON.Sphere")
            if "THREE.CircleGeometry" in doc:
                doc = re.sub(
                    r"new\s+THREE\.CircleGeometry\s*\(\s*([^,\)\n]+)[^\)]*\)",
                    r"new THREE.SphereGeometry(\1, 24, 24)",
                    doc,
                )
            # Some model outputs use a pixel frustum [0..W,0..H] but also translate the camera to (W/2,H/2),
            # which shifts everything off-screen (only a clipped sliver may appear).
            # If we see that specific combination, re-center the camera to the origin.
            pixel_frustum = re.search(
                r"new\s+THREE\.OrthographicCamera\s*\(\s*0\s*,\s*W\s*,\s*(?:H\s*,\s*0|0\s*,\s*H)\s*,",
                doc,
            )
            if pixel_frustum and re.search(r"\bcamera\.position\.set\s*\(\s*W\s*/\s*2\s*,\s*H\s*/\s*2\s*,", doc):
                doc = re.sub(
                    r"(camera\.position\.set\s*\()\s*W\s*/\s*2\s*,\s*H\s*/\s*2\s*,\s*([^)\n]+)(\)\s*;)",
                    r"\g<1>0, 0, \2\3",
                    doc,
                    flags=re.IGNORECASE,
                )
                # Handle both lookAt(x,y,z) and lookAt(new THREE.Vector3(x,y,z)) forms.
                doc = re.sub(
                    r"(camera\.lookAt\s*\()\s*W\s*/\s*2\s*,\s*H\s*/\s*2\s*,\s*0\s*(\)\s*;)",
                    r"\g<1>0, 0, 0\2",
                    doc,
                    flags=re.IGNORECASE,
                )
                doc = re.sub(
                    r"(camera\.lookAt\s*\(\s*new\s+THREE\.Vector3\s*\()\s*W\s*/\s*2\s*,\s*H\s*/\s*2\s*,\s*0\s*(\)\s*\)\s*;)",
                    r"\g<1>0, 0, 0\2",
                    doc,
                    flags=re.IGNORECASE,
                )

        lower_doc = doc.lower()

        if '<html' not in lower_doc:
            body_payload = doc
            if '<script' not in lower_doc:
                body_payload = f"<script>\n{doc}\n</script>"
            doc = f"<!DOCTYPE html>\n<html>\n<head>\n</head>\n<body>\n{body_payload}\n</body>\n</html>"
        elif not lower_doc.startswith('<!doctype'):
            doc = "<!DOCTYPE html>\n" + doc

        # Ensure <head> and <body> blocks exist
        if '<head' not in doc.lower():
            doc = re.sub(r'<html([^>]*)>', r'<html\1>\n<head>\n</head>', doc, count=1, flags=re.IGNORECASE)
        if '<body' not in doc.lower():
            if '</head>' in doc:
                doc = doc.replace('</head>', '</head>\n<body>', 1) + "\n</body>"
            else:
                doc = doc.replace('<html', '<html>\n<body>', 1) + "\n</body>"
        if '</body>' not in doc.lower():
            doc = doc.replace('</html>', '\n</body>\n</html>', 1)

        # 统一修正/加固 enforce2D 定义，避免 body.linearFactor / angularFactor 缺失导致运行时报错
        safe_enforce2d = (
            "function enforce2D(body) {\n"
            "  if (!body || typeof CANNON === 'undefined' || !CANNON.Vec3) {\n"
            "    if (body && body.position) body.position.z = 0;\n"
            "    return;\n"
            "  }\n"
            "  if (!body.linearFactor || typeof body.linearFactor.set !== 'function') {\n"
            "    body.linearFactor = new CANNON.Vec3(1, 1, 0);\n"
            "  }\n"
            "  if (!body.angularFactor || typeof body.angularFactor.set !== 'function') {\n"
            "    body.angularFactor = new CANNON.Vec3(0, 0, 1);\n"
            "  }\n"
            "  body.linearFactor.set(1, 1, 0);\n"
            "  body.angularFactor.set(0, 0, 1);\n"
            "  if (body.position && typeof body.position.z !== 'undefined') {\n"
            "    body.position.z = 0;\n"
            "  }\n"
            "}\n"
        )

        def _js_find_matching_brace(text: str, open_brace_index: int) -> Optional[int]:
            """Find the matching '}' for a '{' at open_brace_index in JS-like text.

            This is a best-effort scanner that skips strings and comments to avoid
            truncating blocks (previous regex-based approach could corrupt code).
            """

            if open_brace_index < 0 or open_brace_index >= len(text) or text[open_brace_index] != "{":
                return None

            depth = 0
            index = open_brace_index
            in_single = False
            in_double = False
            in_backtick = False
            in_line_comment = False
            in_block_comment = False
            escape_next = False

            while index < len(text):
                ch = text[index]
                nxt = text[index + 1] if index + 1 < len(text) else ""

                if in_line_comment:
                    if ch == "\n":
                        in_line_comment = False
                    index += 1
                    continue

                if in_block_comment:
                    if ch == "*" and nxt == "/":
                        in_block_comment = False
                        index += 2
                        continue
                    index += 1
                    continue

                if in_single or in_double or in_backtick:
                    if escape_next:
                        escape_next = False
                        index += 1
                        continue
                    if ch == "\\":
                        escape_next = True
                        index += 1
                        continue
                    if in_single and ch == "'":
                        in_single = False
                    elif in_double and ch == '"':
                        in_double = False
                    elif in_backtick and ch == "`":
                        in_backtick = False
                    index += 1
                    continue

                # Enter comments
                if ch == "/" and nxt == "/":
                    in_line_comment = True
                    index += 2
                    continue
                if ch == "/" and nxt == "*":
                    in_block_comment = True
                    index += 2
                    continue

                # Enter strings
                if ch == "'":
                    in_single = True
                    index += 1
                    continue
                if ch == '"':
                    in_double = True
                    index += 1
                    continue
                if ch == "`":
                    in_backtick = True
                    index += 1
                    continue

                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return index
                index += 1

            return None

        def _replace_js_enforce2d(document: str) -> str:
            # Prefer function declaration: function enforce2D(...) { ... }
            match = re.search(r"\bfunction\s+enforce2D\b", document)
            if match:
                start = match.start()
                open_idx = document.find("{", match.end())
                if open_idx != -1:
                    close_idx = _js_find_matching_brace(document, open_idx)
                    if close_idx is not None:
                        return document[:start] + safe_enforce2d + document[close_idx + 1 :]

            # Fallback assignment: enforce2D = function(...) { ... }
            match = re.search(r"\benforce2D\s*=\s*function\b", document)
            if match:
                start = match.start()
                open_idx = document.find("{", match.end())
                if open_idx != -1:
                    close_idx = _js_find_matching_brace(document, open_idx)
                    if close_idx is not None:
                        return document[:start] + "enforce2D = " + safe_enforce2d + document[close_idx + 1 :]

            return document

        if "enforce2D" in doc:
            doc = _replace_js_enforce2d(doc)

        def _guard_cannon_factor_setters(document: str) -> str:
            # Some LLM generations use cannon-es style `linearFactor/angularFactor`, which are
            # not present on our bundled cannon.js build. Guard these calls to avoid hard crashes
            # that produce 1-frame videos.
            if "linearFactor" in document:
                document = re.sub(
                    r"\b([A-Za-z_]\w*)\s*\.\s*linearFactor\s*\.\s*set\s*\(\s*([^)]+?)\s*\)",
                    r"(\1.linearFactor && \1.linearFactor.set ? \1.linearFactor.set(\2) : void 0)",
                    document,
                )
            if "angularFactor" in document:
                document = re.sub(
                    r"\b([A-Za-z_]\w*)\s*\.\s*angularFactor\s*\.\s*set\s*\(\s*([^)]+?)\s*\)",
                    r"(\1.angularFactor && \1.angularFactor.set ? \1.angularFactor.set(\2) : void 0)",
                    document,
                )
            return document

        doc = _guard_cannon_factor_setters(doc)

        def _ensure_cannon_gravity(document: str) -> str:
            # Some models incorrectly call `new CANNON.World({ gravity: new CANNON.Vec3(...) })`,
            # which does NOT work with the bundled cannon.js build. Ensure gravity is set via
            # `world.gravity.set(...)`.
            if "CANNON" not in document or "World" not in document:
                return document

            # If gravity is already set explicitly, leave it.
            if re.search(r"\b\w+\s*\.\s*gravity\s*\.\s*set\s*\(", document):
                return document

            # Safer one-line rewrite: avoid regex spanning across unrelated code and breaking syntax.
            # Most LLM generations put the world constructor on a single line.
            pattern = re.compile(
                r"(?m)^(?P<indent>[ \t]*)"
                r"(?P<decl>const|let|var)\s+"
                r"(?P<var>[A-Za-z_]\w*)"
                r"\s*=\s*new\s+CANNON\.World"
                r"\s*\(\s*(?P<args>[^\n]*?)\s*\)\s*;?[ \t]*$",
            )

            def _repl(match: re.Match[str]) -> str:
                world_var = match.group("var")
                args = (match.group("args") or "").strip()
                indent = match.group("indent") or ""
                decl = match.group("decl")

                gx, gy, gz = "0", "-9.82", "0"
                gmatch = re.search(
                    r"gravity\s*:\s*new\s+CANNON\.Vec3\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)",
                    args,
                )
                if gmatch:
                    gx, gy, gz = gmatch.group(1), gmatch.group(2), gmatch.group(3)

                # Use cannon.js supported API: create World with no options and set gravity explicitly.
                return (
                    f"{indent}{decl} {world_var} = new CANNON.World();\n"
                    f"{indent}{world_var}.gravity.set({gx}, {gy}, {gz});"
                )

            new_doc, count = pattern.subn(_repl, document, count=1)
            return new_doc if count else document

        doc = _ensure_cannon_gravity(doc)

        def ensure_in_head(document: str, snippet: str) -> str:
            lower = document.lower()
            if snippet.lower() in lower:
                return document
            if '</head>' in document:
                return document.replace('</head>', f"{snippet}\n</head>", 1)
            return document.replace('<head>', f"<head>\n{snippet}", 1)

        def ensure_in_body(document: str, snippet: str) -> str:
            if snippet in document:
                return document
            if '</body>' in document:
                return document.replace('</body>', f"{snippet}\n</body>", 1)
            return document + f"\n{snippet}"

        # Provide per-video target parameters to recording.js so even if the model calls
        # setupRecording(canvas, 300) we still record at least the source video length.
        target_fps = getattr(self, "_current_recording_target_fps", None)
        target_ms = getattr(self, "_current_recording_target_duration_ms", None)
        try:
            target_fps_val = float(target_fps) if target_fps else 5.0
        except Exception:
            target_fps_val = 5.0
        try:
            target_ms_val = int(target_ms) if target_ms else 0
        except Exception:
            target_ms_val = 0
        duration_lines = ""
        if target_ms_val > 0:
            duration_lines = (
                f"window.__codexTargetDurationMs = {target_ms_val:d};\n"
                # Back-compat with older recording.js variants.
                f"window.__codexMinRecordingDurationMs = {target_ms_val:d};\n"
            )
        config_script = (
            "<script>\n"
            f"window.__codexTargetFps = {target_fps_val:.6f};\n"
            f"window.__codexUnifyEngines = {str(bool(getattr(self.config, 'unify_engines', False))).lower()};\n"
            f"{duration_lines}"
            "</script>"
        )
        doc = ensure_in_head(doc, config_script)

        # 注入依赖脚本：
        # - 默认模式：three.js + cannon.js + recording.js 都会被注入（如果页面中尚未引用）。
        # - free_html_mode：只强制注入 recording.js，用于录像；其他库由 LLM 自行选择。
        for snippet in self._script_tags:
            if getattr(self.config, "free_html_mode", False) and "recording.js" not in snippet:
                continue
            # 避免重复注入 three.js / cannon.js / recording.js 脚本。
            # 有些模型生成的 HTML 已经包含相对路径的脚本标签，而我们本地注入使用 file:// 绝对路径。
            # 这里既检查完整 src，又检查其文件名（basename），只要任一已存在就视为“已注入”。
            match = re.search(r'src="([^"]+)"', snippet)
            token = match.group(1) if match else snippet
            short_token = token.split("/")[-1]
            alt_tokens = [short_token]
            if self.engine == "p5js":
                # Avoid common duplicates: p5.js vs p5.min.js
                if "p5" in short_token.lower():
                    alt_tokens.extend(["p5.js", "p5.min.js"])
            lowered_doc = doc.lower()
            if token.lower() in lowered_doc or any(t.lower() in lowered_doc for t in alt_tokens):
                continue
            doc = ensure_in_head(doc, snippet)

        # Guard 脚本仅负责在页面加载后确保自动调用 setupRecording，不再修改物体或物理参数。
        guard_script = (
            "<script>\n"
            "(() => {\n"
            "  function codexEnsureRecording() {\n"
            "    try {\n"
            "      if (typeof setupRecording !== 'function') return;\n"
            "      if (window.__codexRecordingActive) return;\n"
            "      const canvas = (window.renderer && window.renderer.domElement) || document.querySelector('canvas');\n"
            "      if (!canvas) return;\n"
            "      const targetMs = Number(window.__codexTargetDurationMs ?? window.__codexMinRecordingDurationMs);\n"
            "      const durationMs = Number.isFinite(targetMs) && targetMs > 0 ? targetMs : 0;\n"
            "      setupRecording(canvas, durationMs);\n"
            "    } catch (error) {\n"
            "      console.warn('CODEx recording guard', error);\n"
            "    }\n"
            "  }\n"
            "  window.addEventListener('load', () => {\n"
            "    codexEnsureRecording();\n"
            "    const delays = [250, 750, 1500, 3000, 5000];\n"
            "    delays.forEach((delay) => setTimeout(codexEnsureRecording, delay));\n"
            "  });\n"
            "})();\n"
            "</script>"
        )

        doc = ensure_in_body(doc, guard_script)

        return doc

    def _write_image_request_log(self, image_name: str) -> None:
        if not self._current_request_logs:
            return

        log_path = os.path.join(
            self.output_dir,
            "logs",
            f"{image_name}_llm_calls.json"
        )

        payload = {
            "image_name": image_name,
            "requests": self._current_request_logs
        }

        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._log(f"   💾 LLM请求日志已保存: {log_path}", also_print=False)
        except Exception as e:
            self._log(f"   ⚠️ 无法保存LLM请求日志: {e}")

    def _finalize_image_run(self, image_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """统一收尾：写入当前图像的 LLM 请求日志并清空缓存。"""
        self._write_image_request_log(image_name)
        self._current_request_logs = []
        return results


    def _process_loaded_images(self,
                                original_image: np.ndarray,
                                image_name: str,
                                source_info: Optional[Dict[str, Any]] = None,
                                extra_frames: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """在已经加载原始图像的前提下执行分析、代码生成与渲染流程"""
        if original_image is None:
            raise ValueError("原始图像数据为空，无法继续处理")

        self._current_request_logs = []
        self._current_call_index = 0

        self._log("📋 步骤3: 基于原始图像生成场景分析与代码...")
        if source_info:
            for key, value in source_info.items():
                self._log(f"   📁 资源[{key}]: {value}", also_print=False)

        # 保存帧到输出目录
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_path = os.path.join(frames_dir, f"{image_name}_frame_01.png")
        cv2.imwrite(frame_path, original_image)
        self._log(f"   💾 原始帧已保存: {frame_path}")

        source_paths: Dict[str, Any] = {
            'output_original': frame_path
        }
        if source_info:
            source_paths.update(source_info)

        detection_context = self._load_detection_context(image_name, source_paths)
        if detection_context:
            self._log("   🔍 找到第一帧检测 JSON，将作为权威布局提供给 LLM", also_print=False)

        # Configure per-video recording target (best-effort):
        # - Prefer using source video duration as the default recording length.
        # - Do not force a global fixed duration.
        target_fps = 5.0
        target_duration_s = 0.0
        record_ms_env = os.environ.get("VISEXPERT_RECORD_MS")
        try:
            base_record_ms = int(float(record_ms_env)) if record_ms_env else 0
        except Exception:
            base_record_ms = 0
        base_record_ms = max(0, base_record_ms)
        if source_info:
            try:
                target_fps = float(source_info.get("video_fps") or target_fps)
            except Exception:
                target_fps = 5.0
            try:
                target_duration_s = float(source_info.get("video_duration_s") or 0.0)
            except Exception:
                target_duration_s = 0.0
        if not (target_fps and target_fps > 0):
            target_fps = 5.0

        video_duration_ms = int(round(target_duration_s * 1000.0)) if target_duration_s and target_duration_s > 0 else 0
        target_record_ms = max(base_record_ms, video_duration_ms)

        self._current_recording_target_fps = target_fps
        self._current_recording_target_duration_ms = target_record_ms
        # Keep non-canvas engines (SVG) aligned by configuring their capture settings as well.
        if self.engine == "svg":
            try:
                self.renderer.fps = float(target_fps)
            except Exception:
                pass
            try:
                self.renderer.duration_ms = int(target_record_ms) if target_record_ms > 0 else int(base_record_ms)
            except Exception:
                pass
        try:
            # Only used when VISEXPERT_NORMALIZE_VIDEO=1.
            duration_for_normalize = (target_record_ms / 1000.0) if target_record_ms > 0 else 10.0
            self.renderer.target = VideoTarget(width=512, height=512, fps=target_fps, duration_s=duration_for_normalize)
        except Exception:
            pass

        original_frames: List[Dict[str, Any]] = [{
            'image': original_image,
            'label': 'original_frame_01',
            'description': '原始画面（frame_01）',
            'source': source_paths.get('original') or source_paths.get('output_original')
        }]

        # 尝试加载第10帧作为补充参考
        original_variant_path = (
            source_paths.get('original_frame_10')
            or self._find_variant_path(source_paths.get('original'))
        )
        alt_original = self._load_image_from_path(original_variant_path)
        if alt_original is not None:
            frame10_path = os.path.join(frames_dir, f"{image_name}_frame_10.png")
            cv2.imwrite(frame10_path, alt_original)
            self._log(f"   💾 第10帧已保存: {frame10_path}")
            original_frames.append({
                'image': alt_original,
                'label': 'original_frame_10',
                'description': 'later frame from same scene (frame_10)',
                'source': original_variant_path or frame10_path
            })

        if extra_frames:
            for idx, extra in enumerate(extra_frames, start=1):
                if extra.get('image') is None:
                    continue
                label = extra.get('label') or f'extra_frame_{idx}'
                desc = extra.get('description') or 'auxiliary frame for motion/context'
                original_frames.append({
                    'image': extra['image'],
                    'label': label,
                    'description': desc,
                    'source': extra.get('source')
                })

        dim_label = "3D" if getattr(self, "scene_dim", "2d") == "3d" else "2D"
        self._log(f"   🤖 分析图像并直接生成{dim_label}代码...")

        results: Dict[str, Any] = {
            'success': False,
            'image_name': image_name,
            'frame_path': frame_path,
            'analysis_and_code': None,
            'prediction_original': None,
            'scene_understanding': '',
            'detection_context': detection_context,
        }

        error_feedback = ""
        last_error: Optional[str] = None
        video_original_path: Optional[str] = None
        analysis_and_code: Optional[Dict[str, Any]] = None

        # 统一的“生成 -> 渲染”重试：每次渲染失败都会带日志摘要回灌给 LLM 重新生成。
        for attempt in range(self.max_retries + 1):
            self.current_attempt = attempt
            self._log(f"   🔁 生成+渲染尝试 {attempt + 1}/{self.max_retries + 1}")

            try:
                analysis_and_code = self.generate_analysis_and_code(
                    original_frames,
                    image_name,
                    error_feedback=error_feedback,
                    detection_context=detection_context,
                )
            except Exception as exc:
                last_error = f"分析与代码生成调用失败: {exc}"
                self._log(f"   ❌ {last_error}")
                if attempt < self.max_retries:
                    error_feedback = str(exc)
                    continue
                break

            results['analysis_and_code'] = analysis_and_code
            results['prediction_original'] = analysis_and_code
            results['scene_understanding'] = analysis_and_code.get('analysis', '')

            content_type = analysis_and_code.get("content_type") or ("python" if self.engine == "manim" else "html")
            document = (
                analysis_and_code.get("document")
                or analysis_and_code.get("html")
                or analysis_and_code.get("html_document")
                or ""
            ).strip()
            if not document:
                last_error = analysis_and_code.get('error') or f"LLM 未返回可用的 {content_type} 代码"
                self._log(f"   ❌ {last_error}")
                if attempt < self.max_retries:
                    error_feedback = last_error
                    continue
                break

            ext = "html" if content_type == "html" else "py"
            attempt_path = os.path.join(self.output_dir, "logs", f"{image_name}_attempt_{attempt + 1}.{ext}")
            try:
                with open(attempt_path, 'w', encoding='utf-8') as f:
                    header = f"<!-- attempt {attempt + 1} engine={self.engine} -->\n" if ext == "html" else f"# attempt {attempt + 1} engine={self.engine}\n"
                    f.write(header)
                    f.write(document)
            except Exception:
                pass

            self._log(f"📋 步骤4: 渲染基于场景指令的{dim_label}物理仿真视频...")
            self._log(f"   🎬 渲染原始图像{dim_label}视频...")
            try:
                base_name = image_name.replace('.gif', '')
                video_path = os.path.join(self.output_dir, "videos", f"{base_name}_original_2d.mp4")
                video_original_path = self.renderer.render(document, video_path, content_type=content_type)
                if video_original_path:
                    break
                last_error = "渲染未生成视频文件"
            except Exception as exc:
                log_path = getattr(exc, "log_path", "") or ""
                tail = ""
                if log_path and os.path.exists(log_path):
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
                            lines = handle.read().splitlines()
                        tail_lines = lines[-80:] if len(lines) > 80 else lines
                        tail = "\n".join(tail_lines)
                    except Exception:
                        tail = ""
                last_error = str(exc)
                if tail:
                    last_error = f"{last_error}\n\n[renderer log tail]\n{tail}"
                self._log(f"   ❌ 原始图像{dim_label}视频渲染失败: {exc}")

            if attempt < self.max_retries:
                error_feedback = last_error or "render_failed"

        if not video_original_path:
            results['success'] = False
            results['error'] = last_error or "生成或渲染失败"
            self._log(f"🎉 图像处理完成: {image_name}")
            return self._finalize_image_run(image_name, results)

        results['success'] = True
        results['video_original_path'] = video_original_path

        generated_preview_path = os.path.join(self.output_dir, "frames", f"{image_name}_generated_frame_01.png")
        validation = self._validate_first_frame(
            video_original_path,
            original_image,
            save_path=generated_preview_path,
        )
        results['first_frame_validation'] = validation
        if not validation.get('passed'):
            mae = validation.get('mean_abs_diff')
            if mae is not None:
                validation_msg = f"mean_abs_diff={mae:.2f}"
            else:
                validation_msg = validation.get('error') or "first_frame_mismatch"
            results['first_frame_error'] = validation_msg
            results['quality_warning'] = validation_msg
        if os.path.exists(generated_preview_path):
            results['generated_frame_preview'] = generated_preview_path

        self._log(f"   ✅ 原始图像2D视频渲染成功: {video_original_path}")
        self._log(f"🎉 图像处理完成: {image_name}")
        return self._finalize_image_run(image_name, results)


    def _process_image_paths(self,
                              image_name: str,
                              original_path: str) -> Dict[str, Any]:
        """从磁盘路径读取单张原始图像后复用统一流程（第10帧作为第二张参考帧）"""
        self._log(f"\n🎯 开始处理图像文件: {image_name}")

        if not os.path.exists(original_path):
            error_msg = f'原始图像文件不存在: {original_path}'
            self._log(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'image_name': image_name
            }

        self._log("📋 步骤1: 读取原始图像...")
        original_image = cv2.imread(original_path)
        if original_image is None:
            error_msg = f"无法读取原始图像: {original_path}"
            self._log(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'image_name': image_name
            }
        self._log(f"   ✅ 原始图像读取成功: {original_image.shape}")

        original_variant = self._find_variant_path(
            original_path,
            [('_frame_01', '_frame_10'), ('_frame01', '_frame10')]
        )

        return self._process_loaded_images(
            original_image,
            image_name,
            source_info={
                'original': original_path,
                'original_frame_10': original_variant or 'unknown'
            },
        )


    def process_existing_images(self, image_name: str, temp_data_path: str = "") -> Dict[str, Any]:
        """处理已有的图像文件（跳过GIF提取步骤）"""
        # 设置temp_data路径
        if not temp_data_path:
            temp_data_path = os.path.join(os.path.dirname(__file__), "..", "temp_data")
        
        # 查找原始图像文件
        original_path = os.path.join(temp_data_path, f"{image_name}_original.png")
        return self._process_image_paths(image_name, original_path)


    def _process_video_entry(self,
                              video_name: str,
                              data_dir: str,
                              frame_dirs: List[str]) -> Dict[str, Any]:
        """内部方法：加载单个视频条目并执行处理"""
        self._log(f"\n🎬 开始处理Phyworld视频: {video_name}")

        video_path = os.path.join(data_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            error_msg = f"视频文件不存在: {video_path}"
            self._log(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'video_name': video_name
            }

        # Probe source video for fps/duration so we can record/normalize to the same length.
        video_fps = 0.0
        video_frames = 0.0
        video_duration_s = 0.0
        try:
            cap_meta = cv2.VideoCapture(video_path)
            video_fps = float(cap_meta.get(cv2.CAP_PROP_FPS) or 0.0)
            video_frames = float(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            cap_meta.release()
            if video_fps > 0 and video_frames > 0:
                video_duration_s = video_frames / video_fps
        except Exception:
            video_fps = 0.0
            video_frames = 0.0
            video_duration_s = 0.0

        # 读取预提取的关键帧
        original_image = None
        frame_source = None
        frame_source_10 = None
        extra_frames: List[Dict[str, Any]] = []
        frame_candidates = [
            f"{video_name}_frame_01.png",
            f"{video_name}_frame_01.jpg",
            f"{video_name}_first_frame.png",
            f"{video_name}_first_frame.jpg"
        ]
        for frame_dir in frame_dirs:
            for candidate in frame_candidates:
                candidate_path = os.path.join(frame_dir, candidate)
                if os.path.exists(candidate_path):
                    original_image = cv2.imread(candidate_path)
                    if original_image is not None:
                        frame_source = candidate_path
                        break
            if original_image is not None:
                break

        if original_image is None:
            self._log("⚠️ 未找到预提取帧，尝试直接从视频解码第一帧")
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            # Best-effort: also decode a later reference frame (frame_10) to help the LLM infer motion.
            # This is important when frame_10 PNGs are not pre-extracted (common in dataset_sub).
            try:
                desired = 9  # 0-based index for frame_10
                if video_frames and video_frames > 0:
                    desired = min(desired, max(0, int(video_frames) - 1))
                if desired > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(desired))
                    ret10, frame10 = cap.read()
                    if ret10 and frame10 is not None:
                        extra_frames.append(
                            {
                                "image": frame10,
                                "label": "original_frame_10",
                                "description": "later frame from same scene (frame_10)",
                                "source": f"decoded:{video_path}#frame_{desired+1:02d}",
                            }
                        )
            except Exception:
                pass
            cap.release()
            if not ret or frame is None:
                error_msg = f"无法解析视频帧: {video_path}"
                self._log(f"❌ {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'video_name': video_name,
                    'video_source': video_path
                }
            original_image = frame
            frame_source = f"decoded:{video_path}"
        else:
            frame_source_10 = self._find_variant_path(
                frame_source,
                [('_frame_01', '_frame_10'), ('_frame01', '_frame10')]
            )

        # 读取匹配的掩码
        mask_source = None
        mask_source_10 = None

        source_info = {
            'video': video_path,
            'original_frame': frame_source or 'unknown',
            'original_frame_10': frame_source_10 or 'unknown',
            'video_fps': video_fps,
            'video_num_frames': video_frames,
            'video_duration_s': video_duration_s,
        }

        results = self._process_loaded_images(
            original_image,
            video_name,
            source_info=source_info,
            extra_frames=extra_frames or None,
        )
        results['video_source'] = video_path
        results['video_name'] = video_name
        results['success'] = results.get('success', True)
        return results


    def process_phyworld_dataset(self,
                                 data_dir: str = "",
                                 video_name: Optional[str] = None,
                                 video_names: Optional[List[str]] = None,
                                 max_videos: Optional[int] = None) -> Dict[str, Any]:
        """Process a directory of MP4 videos (VisPhyBench/phyworld compatible)."""
        if not data_dir:
            project_data = Path(__file__).resolve().parent.parent / "data"
            candidates = [
                project_data / "sub" / "videos",
                project_data / "test" / "videos",
                project_data / "phyworld_videos",
            ]
            for candidate in candidates:
                if candidate.exists() and candidate.is_dir():
                    data_dir = str(candidate)
                    break
            if not data_dir:
                data_dir = str(project_data / "sub" / "videos")

        self._log("\n🚀 开始处理Phyworld数据集")
        self._log(f"📁 数据目录: {data_dir}")

        if not os.path.isdir(data_dir):
            error_msg = f"数据目录不存在: {data_dir}"
            self._log(f"❌ {error_msg}")
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'error': error_msg,
                'results': []
            }

        # 收集帧与掩码目录
        frame_dirs = [
            os.path.join(data_dir, candidate)
            for candidate in ["extracted_frames", "frames"]
            if os.path.isdir(os.path.join(data_dir, candidate))
        ]
 
        if not frame_dirs:
            self._log("⚠️ 未找到预提取帧目录，将直接从视频解码帧")

        target_names: List[str]
        if video_names:
            target_names = list(video_names)
        elif video_name:
            target_names = [video_name]
        else:
            video_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.mp4')])
            if not video_files:
                self._log("⚠️ 数据目录下未找到任何MP4视频")
                return {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'results': []
                }
            target_names = [os.path.splitext(f)[0] for f in video_files]
            if max_videos is not None:
                try:
                    limit = int(max_videos)
                except Exception:
                    limit = 0
                if limit > 0:
                    target_names = target_names[:limit]

        results: List[Dict[str, Any]] = []
        for idx, name in enumerate(target_names, start=1):
            self._log("=" * 80)
            self._log(f"📼 当前进度: {idx}/{len(target_names)} - {name}")
            self._log("=" * 80)
            result = self._process_video_entry(name, data_dir, frame_dirs)
            results.append(result)

            if result.get('success'):
                self._log(f"✅ {name} 处理成功")
            else:
                self._log(f"❌ {name} 处理失败: {result.get('error', '未知错误')}")

        successful = sum(1 for r in results if r.get('success'))
        failed = len(results) - successful

        summary: Dict[str, Any] = {
            'total': len(results),
            'successful': successful,
            'failed': failed,
            'results': results,
            'data_dir': data_dir
        }

        self._log("\n📊 数据集处理完成")
        self._log(f"   ✅ 成功: {successful}")
        self._log(f"   ❌ 失败: {failed}")

        if len(results) == 1:
            summary['result'] = results[0]

        return summary


def main():
    """主函数 - 处理所有视频"""
    pipeline = PhysicsPredictionPipeline()
    summary = pipeline.process_phyworld_dataset()

    successful = summary.get('successful', 0)
    failed = summary.get('failed', 0)
    total = summary.get('total', 0)

    if total == 0:
        error_msg = summary.get('error', '未找到可处理的视频')
        print(f"⚠️ 没有处理任何视频: {error_msg}")
    elif successful > 0:
        print(f"✅ 处理完成! 成功 {successful}/{total} 个视频，失败 {failed} 个")
    else:
        print(f"❌ 处理失败: {total} 个视频全部失败")


if __name__ == "__main__":
    main()
