#!/usr/bin/env python3
"""
Manim renderer: render a Manim (Python) script into an MP4.

This renderer is optional: it requires `manim` to be installed in the current
Python environment. When unavailable, it raises a RenderingError with a clear
message.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
import re
from typing import Optional

from .video_normalizer import VideoTarget, normalize_video_inplace


class RenderingError(Exception):
    def __init__(self, message: str, log_path: str = ""):
        super().__init__(message)
        self.log_path = log_path


class ManimRenderer:
    def __init__(self, output_dir: str = "./output") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Default to medium quality (typically 720p30) to avoid "PPT-like" choppiness.
        # Can be overridden with MANIM_QUALITY in {"l","m","h","k"} (low/med/high/4k).
        self.quality = (os.environ.get("MANIM_QUALITY") or "m").strip().lower() or "m"
        if self.quality not in {"l", "m", "h", "k"}:
            self.quality = "m"
        self.target = VideoTarget(width=512, height=512, fps=5.0, duration_s=10.0)

    def _sanitize_script(self, script: str) -> str:
        if not script:
            return script

        text = script

        # Remove brittle imports that vary across Manim versions.
        text = re.sub(
            r"^\s*from\s+manim\.utils\.color\s+import\s+Color\s*$\n?",
            "",
            text,
            flags=re.MULTILINE,
        )

        # Rewrite common pattern: Color(rgb=(...)) -> rgb_to_color((...))
        text = re.sub(
            r"Color\s*\(\s*rgb\s*=\s*(\([^\)]*\)|\[[^\]]*\])\s*\)",
            r"rgb_to_color(\1)",
            text,
        )

        # Rate function compatibility across Manim versions.
        # Some models use non-existent shortcuts like rate_functions.ease_in / ease_out / ease_in_out.
        text = re.sub(r"\brate_functions\.ease_in\b", "rate_functions.ease_in_sine", text)
        text = re.sub(r"\brate_functions\.ease_out\b", "rate_functions.ease_out_sine", text)
        text = re.sub(r"\brate_functions\.ease_in_out\b", "rate_functions.ease_in_out_sine", text)

        needs_angle_compat = (
            ".angle" in text
            or ".rotation" in text
            or re.search(r"\.get_angle\s*\(\s*\)", text) is not None
            or re.search(r"\.get_rotation\s*\(\s*\)", text) is not None
        )
        if needs_angle_compat:
            # Replace brittle `.angle` / `.rotation` reads (they may not exist on many Mobjects).
            # IMPORTANT: do NOT rewrite assignments like `obj.angle = ...` (we want those to work).
            text = re.sub(
                r"\b([A-Za-z_]\w*)\.(?:angle|rotation)\b(?!\s*=)",
                r"_codex_angle(\1)",
                text,
            )
            # Deprecated getters route to missing attributes; rewrite to our tracked angle.
            text = re.sub(r"\b([A-Za-z_]\w*)\.get_angle\s*\(\s*\)", r"_codex_angle(\1)", text)
            text = re.sub(r"\b([A-Za-z_]\w*)\.get_rotation\s*\(\s*\)", r"_codex_angle(\1)", text)

            # Wrap Rotate(...) so our angle tracker can advance (best-effort).
            # We do this before injecting the helper to avoid rewriting inside the helper itself.
            text = re.sub(r"\bRotate\s*\(", "_codex_Rotate(", text)

            angle_helper = (
                "\n"
                "# --- Codex compatibility: track mobject angles ---\n"
                "from manim import Rotate as _manim_Rotate\n"
                "\n"
                "_CODEX_ANGLE_STATE = {}\n"
                "\n"
                "def _codex_angle(mobj):\n"
                "    try:\n"
                "        return float(_CODEX_ANGLE_STATE.get(id(mobj), 0.0))\n"
                "    except Exception:\n"
                "        return 0.0\n"
                "\n"
                "def _codex_Rotate(mobj, *args, **kwargs):\n"
                "    angle = None\n"
                "    if 'angle' in kwargs:\n"
                "        angle = kwargs.get('angle')\n"
                "    elif args:\n"
                "        angle = args[0]\n"
                "    try:\n"
                "        if angle is not None:\n"
                "            _CODEX_ANGLE_STATE[id(mobj)] = _codex_angle(mobj) + float(angle)\n"
                "    except Exception:\n"
                "        pass\n"
                "    return _manim_Rotate(mobj, *args, **kwargs)\n"
                "# --- end helper ---\n"
            )

            if "from manim import *" in text:
                text = text.replace("from manim import *", "from manim import *" + angle_helper, 1)
            else:
                text = angle_helper + "\n" + text

        # If the script still references Color(...), inject a safe compatibility helper.
        if "Color(" in text and "def Color" not in text:
            helper = (
                "\n"
                "# --- Codex compatibility: Color(...) helper ---\n"
                "try:\n"
                "    from manim import rgb_to_color, WHITE\n"
                "except Exception:\n"
                "    rgb_to_color = None\n"
                "    WHITE = None\n"
                "\n"
                "def Color(*args, **kwargs):\n"
                "    rgb = kwargs.get('rgb') if kwargs else None\n"
                "    if rgb is None and args:\n"
                "        rgb = args[0]\n"
                "    if rgb_to_color is None:\n"
                "        return WHITE\n"
                "    try:\n"
                "        return rgb_to_color(rgb)\n"
                "    except Exception:\n"
                "        return WHITE\n"
                "# --- end helper ---\n"
            )
            if "from manim import *" in text:
                text = text.replace("from manim import *", "from manim import *" + helper, 1)
            else:
                text = helper + "\n" + text

        return text

    def _run_and_log(self, cmd: list[str], log_path: str, *, cwd: Optional[str] = None) -> int:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
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

    def _ensure_manim(self) -> None:
        try:
            import importlib.util as u

            if not u.find_spec("manim"):
                raise RenderingError(
                    "未检测到 Python 包 `manim`。请在当前环境安装 manim 后重试，例如：\n"
                    "  - `pip install manim`\n"
                    "  - 或使用 conda-forge 安装（推荐）"
                )
        except RenderingError:
            raise
        except Exception as exc:
            raise RenderingError(f"无法检测 manim 是否可用: {exc}") from exc

    def render(self, document: str, output_path: str, content_type: str = "python") -> str:
        if (content_type or "").lower() != "python":
            raise RenderingError(f"ManimRenderer 仅支持 Python 输入，收到: {content_type}")
        self._ensure_manim()

        log_path = os.path.splitext(output_path)[0] + ".log"
        start_time = time.time()

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
            script_path = os.path.join(temp_dir, "scene.py")
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(self._sanitize_script(document))

            media_dir = os.path.join(temp_dir, "media")
            os.makedirs(media_dir, exist_ok=True)

            # Best-effort CLI invocation; we rely on discovering the resulting mp4 under media_dir.
            quality_flag = f"-q{self.quality}"
            cmd = [
                sys.executable,
                "-m",
                "manim",
                quality_flag,
                "--disable_caching",
                "--format=mp4",
                "--media_dir",
                media_dir,
                script_path,
                "GeneratedScene",
            ]
            returncode = self._run_and_log(cmd, log_path, cwd=temp_dir)
            if returncode != 0:
                raise RenderingError(f"Manim 渲染失败，返回代码 {returncode}", log_path)

            candidates = sorted(
                glob.glob(os.path.join(media_dir, "**", "*.mp4"), recursive=True),
                key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
            )
            if not candidates:
                raise RenderingError("Manim 未产出 MP4 文件", log_path)

            produced = candidates[-1]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copyfile(produced, output_path)

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
        print(f"Manim 渲染耗时 {end_time - start_time:.2f} 秒")
        print(f"视频已保存到: {output_path}")
        print(f"日志已保存到: {log_path}")
        return output_path
