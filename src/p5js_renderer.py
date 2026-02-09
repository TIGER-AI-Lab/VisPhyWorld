#!/usr/bin/env python3
"""
p5.js renderer wrapper.

p5.js renders to a <canvas>, so it can use the same shared CanvasHtmlRenderer
backend as Three.js (Puppeteer + MediaRecorder + FFmpeg).
"""

from __future__ import annotations

import os

from .canvas_html_renderer import CanvasHtmlRenderer


class P5JSRenderer(CanvasHtmlRenderer):
    """p5.js-specific wrapper around the shared CanvasHtmlRenderer backend."""

    def __init__(self, output_dir: str = "./output"):
        # Reuse the existing puppeteer/main.js assets directory.
        assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "threejs")
        super().__init__(
            output_dir=output_dir,
            assets_dir=assets_dir,
            label="p5.js",
        )

