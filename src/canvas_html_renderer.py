#!/usr/bin/env python3
"""
Generic HTML(canvas) renderer: open an HTML page in headless Chromium via Puppeteer,
record a <canvas> stream to WebM, then convert to MP4 via FFmpeg.

This is the shared backend for engines like Three.js and p5.js.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional

from .video_normalizer import VideoTarget, normalize_video_inplace


class RenderingError(Exception):
    """Rendering error with an optional log path."""

    def __init__(self, message: str, log_path: str = ""):
        super().__init__(message)
        self.log_path = log_path


class CanvasHtmlRenderer:
    DEFAULT_TARGET = VideoTarget(width=512, height=512, fps=5.0, duration_s=10.0)

    def __init__(
        self,
        output_dir: str = "./output",
        *,
        assets_dir: Optional[str] = None,
        label: str = "Canvas HTML",
        target: Optional[VideoTarget] = None,
    ) -> None:
        self.output_dir = output_dir
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.assets_dir = assets_dir or os.path.join(self.project_root, "assets", "threejs")
        self.label = label
        self.target = target or self.DEFAULT_TARGET
        self._node_env_checked = False

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.assets_dir, exist_ok=True)

    def _node_env(self) -> dict:
        env = os.environ.copy()
        node_modules_path = os.path.join(self.project_root, "node_modules")
        if os.path.isdir(node_modules_path):
            existing = env.get("NODE_PATH", "")
            env["NODE_PATH"] = (
                node_modules_path if not existing else node_modules_path + os.pathsep + existing
            )
        return env

    def _ensure_node_environment(self) -> None:
        """Ensure Node.js and Puppeteer are available."""
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
                env=self._node_env(),
            )
        except subprocess.CalledProcessError as exc:
            raise RenderingError(
                "Node.js 环境缺少 puppeteer 依赖，请在项目根目录执行 `npm install puppeteer` 后重试。"
            ) from exc

        self._node_env_checked = True

    def run_and_log(self, cmd, log_path: str, *, cwd: Optional[str] = None) -> int:
        """Run command and write stdout/stderr to log_path."""
        env = self._node_env()
        header = f"\n\n===== CMD: {' '.join(map(str, cmd))} =====\n"
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
                env=env,
            )

            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(header)
                handle.write(result.stdout or "")
                if result.stderr:
                    handle.write("\n--- STDERR ---\n")
                    handle.write(result.stderr)

            print(result.stdout)
            return 0
        except subprocess.CalledProcessError as exc:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(header)
                handle.write(f"Command failed with return code {exc.returncode}\n")
                handle.write(f"STDOUT:\n{exc.stdout}\n")
                handle.write(f"STDERR:\n{exc.stderr}\n")

            print(f"STDOUT: {exc.stdout}")
            print(f"STDERR: {exc.stderr}")
            return int(exc.returncode)

    def _wait_for_file_stable(self, path: str, retries: int = 10, delay: float = 0.5) -> None:
        """Wait for file size to stabilize to avoid FFmpeg reading partial WebM."""
        last_size = -1
        stable_count = 0
        for _ in range(retries):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = -1
            if size > 0 and size == last_size:
                stable_count += 1
                if stable_count >= 2:
                    return
            else:
                stable_count = 0
                last_size = size
            time.sleep(delay)

    def _convert_webm_to_mp4(self, webm_path: str, output_path: str, log_path: str) -> bool:
        """Convert WebM to MP4 (x264) with a fallback attempt."""
        commands = [
            [
                "ffmpeg",
                "-y",
                "-i",
                webm_path,
                "-c:v",
                "libx264",
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
                "-fflags",
                "+genpts+discardcorrupt",
                "-err_detect",
                "ignore_err",
                "-analyzeduration",
                "100M",
                "-probesize",
                "100M",
                "-i",
                webm_path,
                "-c:v",
                "libx264",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-preset",
                "fast",
                "-crf",
                "23",
                output_path,
            ],
        ]

        for idx, cmd in enumerate(commands, start=1):
            returncode = self.run_and_log(cmd, log_path, cwd=None)
            if returncode == 0 and os.path.exists(output_path):
                return True
            print(f"[FFmpeg Attempt {idx}] return code {returncode}")
        return False

    def render(self, document: str, output_path: str, content_type: str = "html") -> str:
        """
        Render an animation to MP4.

        Args:
            document: HTML page (content_type=html) or JS snippet (content_type=javascript).
            output_path: target mp4 path
        """
        log_path = os.path.splitext(output_path)[0] + ".log"
        try:
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write(f"CanvasHtmlRenderer log for {self.label}\n")
                handle.write(f"output_path={output_path}\n")
                handle.write(f"content_type={content_type}\n")
                handle.write(f"ts={time.time()}\n")
        except Exception:
            pass
        print(f"开始渲染{self.label}动画...")

        self._ensure_node_environment()

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
            for filename in os.listdir(self.assets_dir):
                shutil.copy(os.path.join(self.assets_dir, filename), temp_dir)
            # Ensure the latest recording.js is used even if the HTML references a relative path.
            # (assets/threejs/recording.js can lag behind src/recording.js during development)
            try:
                src_recording = os.path.join(os.path.dirname(__file__), "recording.js")
                if os.path.exists(src_recording):
                    shutil.copy(src_recording, os.path.join(temp_dir, "recording.js"))
            except Exception:
                pass

            if content_type == "html":
                target_html = os.path.join(temp_dir, "index.html")
                with open(target_html, "w", encoding="utf-8") as handle:
                    handle.write(document)
            else:
                with open(os.path.join(temp_dir, "animation.js"), "w", encoding="utf-8") as handle:
                    handle.write(document)

            downloads_dir = os.path.join(temp_dir, "downloads")
            os.mkdir(downloads_dir)

            print("-" * 100)
            start_time = time.time()
            returncode = self.run_and_log(
                ["node", os.path.join(temp_dir, "main.js"), temp_dir],
                log_path,
                cwd=self.project_root,
            )
            end_time = time.time()
            print("-" * 100)

            if returncode != 0:
                raise RenderingError(f"{self.label}渲染失败，返回代码 {returncode}", log_path)

            webm_path = os.path.join(temp_dir, "downloads", "output.webm")
            print(f"{self.label}渲染耗时 {end_time - start_time:.2f} 秒")

            if not os.path.exists(webm_path):
                raise RenderingError("WebM文件未找到", log_path)

            self._wait_for_file_stable(webm_path)

            if not self._convert_webm_to_mp4(webm_path, output_path, log_path):
                raise RenderingError(f"FFmpeg转换失败，详见日志: {log_path}", log_path)

        # Optional normalization (fps/duration/resolution). By default we keep the original
        # recorded duration/fps to avoid intentional stretching/trimming.
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

        print(f"视频已保存到: {output_path}")
        print(f"日志已保存到: {log_path}")
        return output_path
