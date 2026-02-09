from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.report.models import GTGenerationConfig

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


def extract_section(text: str, section_name: str) -> str:
    if not text:
        return ""

    lines = text.split("\n")
    in_section = False
    buffer: List[str] = []

    def _is_new_section(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        header, sep, _ = stripped.partition(":")
        if not sep:
            return False
        header_key = header.replace("_", "").replace("-", "").replace(" ", "")
        return bool(header_key) and header_key.isupper()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(section_name):
            in_section = True
            remainder = stripped[len(section_name):].strip()
            if remainder:
                buffer.append(remainder)
            continue

        if in_section:
            if _is_new_section(stripped):
                break
            buffer.append(line)

    return "\n".join(buffer).strip()


def extract_html_block(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"```html\r?\n(.*?)\r?\n```",
        r"```HTML\r?\n(.*?)\r?\n```",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    lowered = text.lower()
    if "<html" in lowered:
        start = lowered.find("<!doctype")
        if start == -1:
            start = lowered.find("<html")
        end = lowered.rfind("</html>")
        if end != -1:
            end += len("</html>")
            return text[start:end].strip()

    return ""


def find_variant_path(path: Optional[Path]) -> Optional[Path]:
    if not path:
        return None
    name = path.name
    replacements = [("_frame_01", "_frame_10"), ("_frame01", "_frame10")]
    for old, new in replacements:
        if old in name:
            candidate = path.with_name(name.replace(old, new, 1))
            if candidate.exists():
                return candidate
    return None


def pick_existing(*candidates: Optional[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def read_file_b64(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None


def build_gt_analysis_prompt(image_name: str, frames_meta: List[Dict[str, str]]) -> str:
    frame_lines: List[str] = []
    for item in frames_meta:
        label = item.get("label", "帧")
        desc = item.get("description", "")
        source = item.get("source")
        extras = []
        if desc:
            extras.append(desc)
        if source:
            extras.append(f"source: {source}")
        note = "；".join(extras)
        frame_lines.append(f"- {label}{'：' + note if note else ''}")

    frame_text = "\n".join(frame_lines)

    instructions = [
        "You act as the ANALYSIS module of the Physics Prediction Pipeline.",
        "Given the key frames, produce a professional, detailed, and structured scene analysis in English only.",
        "Output format must follow exactly:",
        "ANALYSIS:",
        "Paragraph 1: Describe the overall layout and static structures (ground, obstacles, supports, etc.).",
        "Paragraph 2: Describe each dynamic object in order (shape, color, initial position, motion trajectory).",
        "Paragraph 3: Summarize critical interactions, collision order, and physical cause-effect chain (gravity, sliding, rotation, etc.).",
        "Do not output CODE sections or any extra headings; respond in clear English paragraphs only.",
    ]

    prompt = "\n".join(instructions)
    prompt += f"\n\nScene ID: {image_name}"
    if frame_text:
        prompt += f"\nFrame notes:\n{frame_text}"
    prompt += "\n\nStrictly follow the format and respond in English."
    return prompt


def _ensure_openai_client(model: str, api_base: Optional[str], api_key: Optional[str]) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai 库缺失: 无法导入 OpenAI 客户端")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("未找到 OPENAI_API_KEY，无法调用 GPT 服务")
    return OpenAI(api_key=key, base_url=api_base) if api_base else OpenAI(api_key=key)


def call_openai_analysis(
    model: str,
    prompt_text: str,
    first_b64: Optional[str],
    tenth_b64: Optional[str],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    client = _ensure_openai_client(model, api_base, api_key)

    content: List[Dict[str, Any]] = []
    if first_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{first_b64}"}})
    if tenth_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tenth_b64}"}})
    content.append({"type": "text", "text": prompt_text})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            timeout=timeout,
        )
    except Exception as exc:
        raise RuntimeError(f"GPT-5.1 请求失败: {exc}")

    if not response or not response.choices:
        raise RuntimeError("GPT-5.1 未返回任何结果")

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("GPT-5.1 返回空白内容")
    return text


def call_openai_text_completion(
    model: str,
    messages: List[Dict[str, Any]],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    client = _ensure_openai_client(model, api_base, api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
        )
    except Exception as exc:
        raise RuntimeError(f"GPT 请求失败: {exc}")

    if not response or not response.choices:
        raise RuntimeError("GPT 未返回任何结果")

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("GPT 返回空白内容")
    return text


def text_needs_translation(text: str) -> bool:
    """Return False because we assume all texts are already in English.

    The pipeline now treats both GT captions and model outputs as English,
    so no language-specific branching or translation is needed.
    """
    return False


def ensure_english_text(text: str) -> str:
    if not text:
        return text
    return text.strip()


def ensure_gt_analysis(
    image_name: str,
    frames_first: Optional[Path],
    frames_tenth: Optional[Path],
    gt_dir: Path,
    gt_config: GTGenerationConfig,
) -> Tuple[Optional[str], Optional[Path]]:
    gt_dir.mkdir(parents=True, exist_ok=True)
    record_path = gt_dir / f"{image_name}_gt_analysis.json"

    if record_path.exists():
        try:
            data = json.loads(record_path.read_text(encoding="utf-8"))
            analysis = data.get("analysis")
            if analysis:
                if not text_needs_translation(analysis):
                    return analysis, record_path
                analysis_en = ensure_english_text(analysis)
                if analysis_en != analysis:
                    data["analysis"] = analysis_en
                    try:
                        record_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                return analysis_en, record_path
        except Exception:
            pass

    legacy_record = gt_dir / f"{image_name}_gt_caption.json"
    if legacy_record.exists() and not record_path.exists():
        try:
            data = json.loads(legacy_record.read_text(encoding="utf-8"))
            analysis = data.get("analysis") or data.get("caption")
            if analysis:
                analysis_en = analysis if not text_needs_translation(analysis) else ensure_english_text(analysis)
                record_path.write_text(
                    json.dumps(
                        {
                            "name": image_name,
                            "analysis": analysis_en,
                            "model": data.get("model"),
                            "timestamp": data.get("timestamp"),
                            "raw_response": data.get("raw_response"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return analysis_en, record_path
        except Exception:
            pass

    # If GT generation is disabled (e.g., --skip-gt), only reuse existing cache files.
    # Do not attempt to call external APIs.
    if not getattr(gt_config, "enabled", False):
        return None, None

    first_b64 = read_file_b64(frames_first)
    tenth_b64 = read_file_b64(frames_tenth)
    if not first_b64 and not tenth_b64:
        raise RuntimeError("缺少生成 GT 所需的帧图像")

    frames_meta: List[Dict[str, str]] = []
    if frames_first:
        frames_meta.append({"label": "第1帧 - 原始", "description": "初始参考帧", "source": str(frames_first)})
    if frames_tenth:
        frames_meta.append({"label": "第10帧 - 原始", "description": "对比参考帧", "source": str(frames_tenth)})

    prompt_text = build_gt_analysis_prompt(image_name, frames_meta)

    analysis_raw = call_openai_analysis(
        model=gt_config.model,
        prompt_text=prompt_text,
        first_b64=first_b64,
        tenth_b64=tenth_b64,
        api_base=gt_config.api_base,
        api_key=gt_config.api_key,
        timeout=gt_config.timeout,
    )

    analysis = extract_section(analysis_raw, "ANALYSIS:") or analysis_raw
    analysis = ensure_english_text(analysis)

    payload = {
        "name": image_name,
        "analysis": analysis,
        "model": gt_config.model,
        "timestamp": datetime.now().isoformat(),
        "raw_response": analysis_raw,
    }
    try:
        record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return analysis, record_path


def extract_response_sections(response_text: str) -> Dict[str, str]:
    if not response_text:
        return {}

    section_aliases = {
        "analysis": ["ANALYSIS:", "ANALYSIS：", "SCENE_ANALYSIS:", "SCENE_ANALYSIS：", "场景分析:", "场景分析："],
        "motion_prediction": [
            "MOTION_PREDICTION:",
            "MOTION_PREDICTION：",
            "PHYSICS_PREDICTION:",
            "PHYSICS_PREDICTION：",
            "PHYSICS_PLAN:",
            "PHYSICS_PLAN：",
            "运动预测:",
            "运动预测：",
            "物理预测:",
            "物理预测：",
        ],
        "code_guidance": ["CODE_GUIDANCE:", "CODE_GUIDANCE：", "CODE_PLAN:", "CODE_PLAN：", "代码指导:", "代码指导："],
        "original_image_analysis": [
            "ORIGINAL_IMAGE_ANALYSIS:",
            "ORIGINAL_IMAGE_ANALYSIS：",
            "ORIGINAL_ANALYSIS:",
            "ORIGINAL_ANALYSIS：",
            "原始图像分析:",
            "原始图像分析：",
        ],
        "original_physics_prediction": [
            "ORIGINAL_PHYSICS_PREDICTION:",
            "ORIGINAL_PHYSICS_PREDICTION：",
            "原始物理预测:",
            "原始物理预测：",
        ],
        "segmentation_image_analysis": [
            "SEGMENTATION_IMAGE_ANALYSIS:",
            "SEGMENTATION_IMAGE_ANALYSIS：",
            "SEGMENTED_IMAGE_ANALYSIS:",
            "SEGMENTED_IMAGE_ANALYSIS：",
            "分割图像分析:",
            "分割图像分析：",
        ],
        "segmentation_physics_prediction": [
            "SEGMENTATION_PHYSICS_PREDICTION:",
            "SEGMENTATION_PHYSICS_PREDICTION：",
            "SEGMENTED_PHYSICS_PREDICTION:",
            "SEGMENTED_PHYSICS_PREDICTION：",
            "分割物理预测:",
            "分割物理预测：",
        ],
        "motion_analysis": [
            "MOTION_ANALYSIS:",
            "MOTION_ANALYSIS：",
            "SCENE_MOTION_ANALYSIS:",
            "SCENE_MOTION_ANALYSIS：",
            "运动分析:",
            "运动分析：",
        ],
        "text_alignment": ["TEXT_ALIGNMENT:", "TEXT_ALIGNMENT：", "TEXT_METRICS:", "TEXT_METRICS：", "文本指标:", "文本指标："],
    }

    sections: Dict[str, str] = {}
    for key, labels in section_aliases.items():
        for label in labels:
            value = extract_section(response_text, label)
            if value:
                sections[key] = value
                break

    if response_text and not sections.get("analysis"):
        sections["analysis"] = response_text.strip()

    return sections


def select_model_analysis_text(sections: Dict[str, str]) -> Optional[Tuple[str, str]]:
    for key in ("analysis", "motion_analysis", "motion_prediction", "original_physics_prediction"):
        text = sections.get(key)
        if text:
            candidate = text.strip()
            if candidate:
                return key, candidate
    return None
