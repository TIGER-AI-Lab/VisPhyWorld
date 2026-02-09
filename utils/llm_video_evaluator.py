#!/usr/bin/env python3
"""
LLM-based video evaluation using Gemini 2.5 Pro.

This helper uploads a pair of videos (reference and generated) to Gemini via
Vertex AI / Generative AI Python SDK, requests a qualitative similarity score,
and returns a normalized result together with the raw textual feedback.

This module intentionally keeps the prompt, logging, and output in English
only to simplify downstream processing and avoid locale-specific artifacts.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_LLM_MODEL_ID = "models/gemini-2.5-pro"
LLM_MODEL_ID = (
    os.environ.get("GEMINI_MODEL_ID")
    or os.environ.get("GEMINI_MODEL")
    or DEFAULT_LLM_MODEL_ID
)
PROMPT_VERSION = "gemini_physics_video_consistency_v2"
MAX_VIDEO_SIZE_BYTES = 750 * 1024 * 1024  # 750MB safety guard
SUPPORTED_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}


class LLMVideoEvaluationError(RuntimeError):
    """Raised when the LLM evaluation fails or returns invalid payload."""


def _read_video_bytes(path: Path) -> Tuple[str, bytes]:
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    suffix = path.suffix.lower()
    mime = SUPPORTED_MIME_TYPES.get(suffix)
    if not mime:
        raise LLMVideoEvaluationError(f"Unsupported video format: {path.name}")
    data = path.read_bytes()
    if len(data) > MAX_VIDEO_SIZE_BYTES:
        raise LLMVideoEvaluationError(f"Video is too large for evaluation: {path.name}")
    return mime, data


def _build_prompt(reference_name: str, generated_name: str) -> str:
    return (
        "You are an expert evaluator of physical simulations and video quality.\n"
        "\n"
        "Compare the provided reference video (Ground Truth) with the generated video.\n"
        "Your goal is to determine if the generated video accurately reconstructs the physical event shown in the reference.\n"
        "\n"
        "Focus on the following dimensions:\n"
        "- Physical Plausibility (Crucial): Do the objects obey 2D physics laws (gravity, rigid-body collisions, friction)?\n"
        "  Are there any hallucinations such as objects passing through each other (ghosting), floating unnaturally,\n"
        "  or failing to move when hit?\n"
        "- Motion Consistency: Does the trajectory, speed, and timing of the movement align with the reference?\n"
        "- Scene Semantics: Are the correct objects (color, shape, count) present in the correct layout?\n"
        "- Visual Fidelity: Overall clarity, ignoring minor rendering style differences if the physics is correct.\n"
        "\n"
        "Return a JSON object with the keys:\n"
        "- score: integer between 1 and 10.\n"
        "  - 10: Perfect physical and visual match.\n"
        "  - 1: Physical laws are violated (e.g., phantom collision, static scene when motion is expected), even if the image looks realistic.\n"
        "- justification: Brief explanation, specifically pointing out any physical violations if present.\n"
        "\n"
        f"Reference label: {reference_name}\n"
        f"Generated label: {generated_name}\n"
        "\n"
        "Respond with JSON only. Do not include Markdown or extra commentary."
    )


def _ensure_client() -> "GenerativeModel":
    try:
        from google import auth  # noqa: F401
        from google.generativeai import GenerativeModel
    except Exception as exc:  # pragma: no cover - dependency guard
        raise LLMVideoEvaluationError(
            "Missing google-generativeai package or auth libraries for Gemini access."
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMVideoEvaluationError("GEMINI_API_KEY environment variable is required.")

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return GenerativeModel(LLM_MODEL_ID)


def evaluate_video_pair(
    reference_path: Path,
    generated_path: Path,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run Gemini-based evaluation for a single pair of videos.

    Returns a dictionary with fields:
        {
            "score": float,
            "justification": str,
            "model": LLM_MODEL_ID,
            "raw_response": dict,
        }
    """

    mime_ref, bytes_ref = _read_video_bytes(reference_path)
    mime_gen, bytes_gen = _read_video_bytes(generated_path)
    prompt = _build_prompt(reference_path.name, generated_path.name)

    safety_settings = None
    try:  # Prefer explicit safety thresholds when available
        from google.generativeai.types import (
            HarmBlockThreshold,
            HarmCategory,
            SafetySetting,
        )

        safety_settings = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUAL_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]
    except Exception:
        safety_settings = None

    request_payload = [
        {"role": "user", "parts": [{"text": prompt}]},
        {
            "role": "user",
            "parts": [
                {
                    "mime_type": mime_ref,
                    "data": base64.b64encode(bytes_ref).decode("utf-8"),
                },
                {
                    "mime_type": mime_gen,
                    "data": base64.b64encode(bytes_gen).decode("utf-8"),
                },
            ],
        },
    ]
    request_kwargs: Dict[str, Any] = {}
    if safety_settings is not None:
        request_kwargs["safety_settings"] = safety_settings

    cache_file: Optional[Path]
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = (
            PROMPT_VERSION
            + "__"
            + LLM_MODEL_ID
            + "__"
            + reference_path.stem
            + "__"
            + generated_path.stem
            + "__"
            + str(os.path.getmtime(reference_path))
            + "__"
            + str(os.path.getmtime(generated_path))
        )
        cache_file = cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                cached_payload = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached_payload, dict) and cached_payload.get("score") is not None:
                    cached_payload.setdefault("model", LLM_MODEL_ID)
                    return cached_payload
            except Exception:
                pass
    else:
        cache_file = None

    try:
        model = _ensure_client()
    except LLMVideoEvaluationError as exc:
        return {"error": str(exc), "model": LLM_MODEL_ID}

    try:
        response = model.generate_content(
            request_payload,
            **request_kwargs,
        )
    except Exception as exc:  # pragma: no cover - API exceptions
        return {"error": f"Gemini API call failed: {exc}", "model": LLM_MODEL_ID}

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return {"error": "Gemini response did not contain candidates.", "model": LLM_MODEL_ID}

    text_payload = None
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = None
        if content is not None:
            parts = getattr(content, "parts", None)
        if parts is None and isinstance(candidate, dict):
            parts = candidate.get("content", {}).get("parts")
        parts = parts or []
        for part in parts:
            text_value = getattr(part, "text", None)
            if text_value is None and isinstance(part, dict):
                text_value = part.get("text")
            if text_value:
                text_payload = text_value
                break
        if text_payload:
            break
    if not text_payload:
        return {"error": "Gemini response did not include text content.", "model": LLM_MODEL_ID}

    text_normalized = text_payload.strip()
    if text_normalized.startswith("```"):
        # Remove markdown fences such as ```json ... ```
        lines = [line for line in text_normalized.splitlines() if not line.strip().startswith("```")]
        text_normalized = "\n".join(lines).strip()
    if "{" in text_normalized and "}" in text_normalized:
        start = text_normalized.find("{")
        end = text_normalized.rfind("}") + 1
        if start >= 0 and end > start:
            text_normalized = text_normalized[start:end]

    try:
        parsed = json.loads(text_normalized)
    except Exception as exc:
        return {"error": f"Gemini response is not valid JSON: {exc}", "model": LLM_MODEL_ID}

    score = parsed.get("score")
    justification = parsed.get("justification", "").strip()
    if not isinstance(score, (int, float)):
        return {"error": f"Gemini returned invalid score payload: {parsed}", "model": LLM_MODEL_ID}
    score_f = float(score)
    # The evaluation prompt specifies an integer score (1-10). Coerce defensively.
    if isinstance(score, float) and not score_f.is_integer():
        score_int = int(round(score_f))
    else:
        score_int = int(score_f)
    score_int = max(1, min(10, score_int))

    result = {
        "score": score_int,
        "justification": justification,
        "model": LLM_MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "raw_response": parsed,
    }

    if cache_file:
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return result


def batch_evaluate_directory(
    reference_dir: Path,
    generated_dir: Path,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all matching video pairs within two directories.

    Returns a mapping: {video_stem: evaluation_result}
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or (output_dir / ".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    ref_videos = {path.stem: path for path in reference_dir.glob("*.mp4")}
    gen_videos = {path.stem: path for path in generated_dir.glob("*.mp4")}
    for stem, ref_path in ref_videos.items():
        gen_path = gen_videos.get(stem)
        if not gen_path:
            continue
        try:
            result = evaluate_video_pair(ref_path, gen_path, cache_dir=cache_dir)
        except Exception as exc:
            result = {
                "error": str(exc),
                "model": LLM_MODEL_ID,
            }
        results[stem] = result
        if "error" not in result:
            payload_path = output_dir / f"{stem}_gemini_eval.json"
            try:
                payload_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
    return results
