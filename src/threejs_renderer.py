#!/usr/bin/env python3
"""
Three.js renderer wrapper.

Historically this module contained the whole Puppeteer+recording+FFmpeg backend.
That generic backend now lives in `canvas_html_renderer.py` as `CanvasHtmlRenderer`.
This class is a thin wrapper that keeps backward compatibility.
"""

from __future__ import annotations

import os
from pathlib import Path

from .canvas_html_renderer import CanvasHtmlRenderer, RenderingError


class ThreeJSRenderer(CanvasHtmlRenderer):
    """Three.js-specific wrapper around the shared CanvasHtmlRenderer backend."""

    def __init__(self, output_dir: str = "./output"):
        assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "threejs")
        super().__init__(
            output_dir=output_dir,
            assets_dir=assets_dir,
            label="Three.js",
        )
        self._setup_threejs_assets()

    def _setup_threejs_assets(self) -> None:
        """
        Ensure legacy assets exist (index.html/main.js/recording.js).

        Note: The pipeline typically writes a full HTML document and injects
        `VisExpert/src/recording.js` via file://, so this is mainly for backward
        compatibility and `content_type='javascript'`.
        """

        # Keep the legacy main.js/index.html shipped in repo when present.
        # If missing, we create minimal fallbacks.
        assets_path = Path(self.assets_dir)
        assets_path.mkdir(parents=True, exist_ok=True)

        index_path = assets_path / "index.html"
        main_path = assets_path / "main.js"
        recording_path = assets_path / "recording.js"

        if not index_path.exists():
            index_path.write_text(
                "<!DOCTYPE html>\n"
                "<html lang=\"en\">\n"
                "<head>\n"
                "  <meta charset=\"UTF-8\" />\n"
                "  <title>Physics Animation</title>\n"
                "  <style>html, body { margin: 0; background: #fff; } canvas { display:block; }</style>\n"
                "  <script src=\"recording.js\"></script>\n"
                "</head>\n"
                "<body>\n"
                "  <script>\n"
                "    // Placeholder page. The pipeline usually overwrites index.html.\n"
                "    console.log('ThreeJSRenderer fallback index.html loaded');\n"
                "  </script>\n"
                "</body>\n"
                "</html>\n",
                encoding="utf-8",
            )

        if not main_path.exists():
            # Minimal puppeteer script: open index.html, allow downloads, wait, then look for output.webm.
            main_path.write_text(
                "const puppeteer = require('puppeteer');\n"
                "const fs = require('fs');\n"
                "const path = require('path');\n"
                "\n"
                "async function recordAnimation(folderPath) {\n"
                "  const downloadPath = path.join(folderPath, 'downloads');\n"
                "  const browser = await puppeteer.launch({\n"
                "    headless: true,\n"
                "    defaultViewport: null,\n"
                "    args: ['--no-sandbox','--disable-setuid-sandbox','--disable-gpu','--enable-unsafe-swiftshader','--allow-file-access-from-files']\n"
                "  });\n"
                "  try {\n"
                "    const page = await browser.newPage();\n"
                "    page.on('pageerror', err => console.log(`[PAGE ERROR] ${err.message}`));\n"
                "    page.on('console', msg => console.log(`[${msg.type().toUpperCase()}] ${msg.text()}`));\n"
                "    const client = await page.createCDPSession();\n"
                "    await client.send('Page.setDownloadBehavior', { behavior: 'allow', downloadPath });\n"
                "    const htmlPath = path.join(folderPath, 'index.html');\n"
                "    console.log(`[NAVIGATE] Loading: file://${path.resolve(htmlPath)}`);\n"
                "    await page.goto(`file://${path.resolve(htmlPath)}`, { waitUntil: 'domcontentloaded', timeout: 30000 });\n"
                "    console.log('[NAVIGATE] Page navigation completed');\n"
                "    await new Promise(r => setTimeout(r, 15000));\n"
                "    const files = fs.readdirSync(downloadPath);\n"
                "    const videoFile = files.find(f => f.endsWith('.webm'));\n"
                "    if (!videoFile) {\n"
                "      console.error('Video could not be recorded! Available files:', files);\n"
                "      process.exit(1);\n"
                "    }\n"
                "    const oldPath = path.join(downloadPath, videoFile);\n"
                "    const newPath = path.join(downloadPath, 'output.webm');\n"
                "    if (videoFile !== 'output.webm') fs.renameSync(oldPath, newPath);\n"
                "    console.log(`Saved to: ${newPath}`);\n"
                "  } finally {\n"
                "    await browser.close();\n"
                "  }\n"
                "}\n"
                "\n"
                "const folderPath = process.argv[2];\n"
                "if (!folderPath) { console.error('Please provide the folder path as a command line argument'); process.exit(2); }\n"
                "recordAnimation(folderPath).catch(err => { console.error(err); process.exit(1); });\n",
                encoding="utf-8",
            )

        # Keep `recording.js` aligned with src/recording.js if present.
        src_recording = Path(__file__).resolve().parent / "recording.js"
        if src_recording.exists():
            try:
                recording_path.write_text(src_recording.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

