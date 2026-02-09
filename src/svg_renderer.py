#!/usr/bin/env python3
"""
SVG renderer: render an HTML(SVG) animation into an MP4 by taking periodic
screenshots via Puppeteer and encoding them with FFmpeg.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional

from .video_normalizer import VideoTarget, normalize_video_inplace


class RenderingError(Exception):
    def __init__(self, message: str, log_path: str = ""):
        super().__init__(message)
        self.log_path = log_path


class SvgRenderer:
    def __init__(
        self,
        output_dir: str = "./output",
        *,
        fps: float = 5.0,
        duration_ms: int = 10_000,
        viewport_width: int = 512,
        viewport_height: int = 512,
        target: Optional[VideoTarget] = None,
    ) -> None:
        self.output_dir = output_dir
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.fps = float(fps)
        self.duration_ms = int(duration_ms)
        self.viewport_width = int(viewport_width)
        self.viewport_height = int(viewport_height)
        self.target = target or VideoTarget(width=512, height=512, fps=5.0, duration_s=10.0)
        self._node_env_checked = False

        os.makedirs(output_dir, exist_ok=True)

    def _ensure_node_environment(self) -> None:
        if self._node_env_checked:
            return

        try:
            subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise RenderingError("未检测到 Node.js，请安装 Node.js (建议 v18+) 后重试。") from exc
        except subprocess.CalledProcessError as exc:
            raise RenderingError(f"无法调用 node --version: {exc.stderr or exc.stdout}") from exc

        try:
            subprocess.run(["npm", "--version"], capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise RenderingError("未检测到 npm，请安装完整的 Node.js 环境（包含 npm）。") from exc
        except subprocess.CalledProcessError as exc:
            raise RenderingError(f"无法调用 npm --version: {exc.stderr or exc.stdout}") from exc

        try:
            subprocess.run(
                ["node", "-e", "require.resolve('puppeteer')"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root,
            )
        except subprocess.CalledProcessError as exc:
            raise RenderingError(
                "Node.js 环境缺少 puppeteer 依赖，请在项目根目录执行 `npm install puppeteer` 后重试。"
            ) from exc

        self._node_env_checked = True

    def _run_and_log(self, cmd: list[str], log_path: str, *, cwd: Optional[str] = None) -> int:
        env = os.environ.copy()
        node_modules_path = os.path.join(self.project_root, "node_modules")
        if os.path.isdir(node_modules_path):
            existing = env.get("NODE_PATH", "")
            env["NODE_PATH"] = (
                node_modules_path if not existing else node_modules_path + os.pathsep + existing
            )
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
                env=env,
            )
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write(result.stdout or "")
                if result.stderr:
                    handle.write("\n--- STDERR ---\n")
                    handle.write(result.stderr)
            return 0
        except subprocess.CalledProcessError as exc:
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write(f"Command failed with return code {exc.returncode}\n")
                handle.write(f"STDOUT:\n{exc.stdout}\n")
                handle.write(f"STDERR:\n{exc.stderr}\n")
            return int(exc.returncode)

    def _encode_pngs_to_mp4(self, frames_dir: str, output_path: str, log_path: str) -> bool:
        fps = max(self.fps, 0.1)
        input_pattern = os.path.join(frames_dir, "frame_%04d.png")
        commands = [
            [
                "ffmpeg",
                "-y",
                "-framerate",
                f"{fps:.4f}",
                "-start_number",
                "1",
                "-i",
                input_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-preset",
                "fast",
                "-crf",
                "23",
                output_path,
            ],
            [
                "ffmpeg",
                "-y",
                "-framerate",
                f"{fps:.4f}",
                "-start_number",
                "1",
                "-i",
                input_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-preset",
                "fast",
                "-crf",
                "23",
                output_path,
            ],
        ]

        for cmd in commands:
            returncode = self._run_and_log(cmd, log_path)
            if returncode == 0 and os.path.exists(output_path):
                return True
        return False

    def render(self, document: str, output_path: str, content_type: str = "html") -> str:
        if (content_type or "").lower() != "html":
            raise RenderingError(f"SvgRenderer 仅支持 HTML 输入，收到: {content_type}")

        self._ensure_node_environment()
        log_path = os.path.splitext(output_path)[0] + ".log"

        capture_script = r"""const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function main() {
  const [,, folderPath, framesDir, fpsStr, durationStr, widthStr, heightStr] = process.argv;
  if (!folderPath || !framesDir) {
    console.error('Usage: capture_svg_frames.js <folderPath> <framesDir> <fps> <durationMs> <width> <height>');
    process.exit(2);
  }

  const fps = Math.max(parseFloat(fpsStr || '5'), 0.1);
  const durationMs = Math.max(parseInt(durationStr || '10000', 10), 100);
  const width = Math.max(parseInt(widthStr || '800', 10), 64);
  const height = Math.max(parseInt(heightStr || '800', 10), 64);
  const frameCount = Math.max(Math.round((durationMs / 1000) * fps), 1);
  const frameIntervalMs = 1000 / fps;

  fs.mkdirSync(framesDir, { recursive: true });

  const browser = await puppeteer.launch({
    headless: true,
    defaultViewport: { width, height, deviceScaleFactor: 1 },
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-gpu",
      "--enable-unsafe-swiftshader",
      "--allow-file-access-from-files"
    ]
  });

  try {
    const page = await browser.newPage();
    page.on('pageerror', err => console.log(`[PAGE ERROR] ${err.message}`));
    page.on('console', msg => console.log(`[${msg.type().toUpperCase()}] ${msg.text()}`));

    const htmlPath = path.join(folderPath, 'index.html');
    await page.goto(`file://${path.resolve(htmlPath)}`, { waitUntil: 'load', timeout: 30000 });

    await new Promise(resolve => setTimeout(resolve, 500));

    for (let i = 1; i <= frameCount; i++) {
      const name = `frame_${String(i).padStart(4, '0')}.png`;
      const outPath = path.join(framesDir, name);
      await page.screenshot({ path: outPath, type: 'png' });
      await new Promise(resolve => setTimeout(resolve, frameIntervalMs));
    }
  } finally {
    await browser.close();
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
"""

        start_time = time.time()
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
            html_path = os.path.join(temp_dir, "index.html")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(document)

            script_path = os.path.join(temp_dir, "capture_svg_frames.js")
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(capture_script)

            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            returncode = self._run_and_log(
                [
                    "node",
                    script_path,
                    temp_dir,
                    frames_dir,
                    f"{self.fps:.4f}",
                    str(self.duration_ms),
                    str(self.viewport_width),
                    str(self.viewport_height),
                ],
                log_path,
                cwd=self.project_root,
            )
            if returncode != 0:
                raise RenderingError(f"SVG 渲染失败，返回代码 {returncode}", log_path)

            frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
            if not frames:
                raise RenderingError("未生成任何 PNG 帧，无法编码视频", log_path)

            if not self._encode_pngs_to_mp4(frames_dir, output_path, log_path):
                raise RenderingError(f"FFmpeg 编码失败，详见日志: {log_path}", log_path)

        normalize = str(os.environ.get("VISEXPERT_NORMALIZE_VIDEO", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if normalize:
            post_log = os.path.splitext(output_path)[0] + ".normalize.log"
            if not normalize_video_inplace(output_path, self.target, log_path=post_log, white_background=True):
                raise RenderingError(f"视频规格归一化失败，详见日志: {post_log}", post_log)

        end_time = time.time()
        print(f"SVG 渲染耗时 {end_time - start_time:.2f} 秒")
        print(f"视频已保存到: {output_path}")
        print(f"日志已保存到: {log_path}")
        return output_path
