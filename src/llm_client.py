#!/usr/bin/env python3
"""LLM client wrapper for physics prediction pipeline."""

from __future__ import annotations

import os
import uuid
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:  # optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

import requests  # type: ignore

load_dotenv()

LogEntry = Dict[str, Any]

MODEL_HINTS: Dict[str, str] = {
    "openai": "示例模型: gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini",
    "anthropic": "示例模型: claude-3-5-sonnet-20241022, claude-4-sonnet, claude-sonnet-4-20250514",
    "qwen": "示例模型: qwen-vl-max, qwen2.5-vl-72b-instruct, qwen3-vl-plus",
    "huggingface": "示例模型: hf:qwen3-vl-30b-a3b-thinking (Qwen/Qwen3-VL-30B-A3B-Thinking)",
    "gemini": "示例模型: gemini-2.5-pro, gemini-1.5-pro",
}

_max_tokens_env = os.getenv("LLM_MAX_COMPLETION_TOKENS", "").strip()
try:
    DEFAULT_MAX_COMPLETION_TOKENS = int(_max_tokens_env) if _max_tokens_env else 16384
except ValueError:
    # 环境变量格式错误时回退到安全默认值
    DEFAULT_MAX_COMPLETION_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.6

DEFAULT_FALLBACK_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Fallback Scene</title>
    <style>
      body { margin: 0; background: #101014; overflow: hidden; }
      canvas { display: block; }
    </style>
  </head>
  <body>
    <script type="module">
      const canvas = document.createElement('canvas');
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      document.body.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      let x = 40;
      let vx = 2.2;
      const radius = 20;

      function loop() {
        ctx.fillStyle = '#101014';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#48c1ff';
        ctx.beginPath();
        ctx.arc(x, canvas.height * 0.4, radius, 0, Math.PI * 2);
        ctx.fill();
        x += vx;
        if (x + radius > canvas.width || x - radius < 0) {
          vx *= -1;
        }
        requestAnimationFrame(loop);
      }
      loop();
    </script>
  </body>
</html>"""


class LLMClient:
    """Dispatches LLM calls to OpenAI, Anthropic, and Qwen multimodal models."""

    def __init__(
        self,
        model_name: str,
        openai_api_key: str = "",
        openai_base_url: str = "",
        anthropic_api_key: str = "",
        qwen_api_key: str = "",
        qwen_base_url: str = "",
        hf_api_key: str = "",
        hf_base_url: str = "",
        hf_model: str = "",
        gemini_api_key: str = "",
        gemini_api_base: str = "",
        gemini_model: str = "",
    ) -> None:
        self.model_name = model_name or ""
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.openai_base_url = openai_base_url or os.environ.get("OPENAI_API_BASE", "")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        # DashScope OpenAI 兼容接口：常见环境变量是 DASHSCOPE_API_KEY / DASHSCOPE_API_BASE。
        # 项目内历史使用 QWEN_API_KEY / QWEN_API_BASE，二者都支持。
        self.qwen_api_key = (
            qwen_api_key
            or os.environ.get("QWEN_API_KEY", "")
            or os.environ.get("DASHSCOPE_API_KEY", "")
        )
        raw_qwen_base = (
            qwen_base_url
            or os.environ.get("QWEN_API_BASE", "")
            or os.environ.get("DASHSCOPE_API_BASE", "")
        )
        self.qwen_base_url = raw_qwen_base.rstrip("/") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self.hf_api_key = hf_api_key or os.environ.get("HF_API_KEY", "")
        hf_base = (hf_base_url or os.environ.get("HF_API_BASE", "")).strip()
        self.hf_base_url = hf_base.rstrip("/") if hf_base else ""
        raw_hf_model = hf_model or os.environ.get("HF_MODEL", "")
        if raw_hf_model:
            self.hf_model = raw_hf_model
        elif self.model_name.lower().startswith("hf:"):
            self.hf_model = self.model_name.split(":", 1)[1]
        else:
            self.hf_model = self.model_name
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        self.gemini_api_base = (gemini_api_base or os.environ.get("GEMINI_API_BASE", "") or "https://generativelanguage.googleapis.com").rstrip("/")
        raw_gemini_model = gemini_model or os.environ.get("GEMINI_MODEL", "")
        if raw_gemini_model:
            self.gemini_model = raw_gemini_model
        elif self.model_name.lower().startswith("gemini"):
            self.gemini_model = self.model_name
        else:
            self.gemini_model = self.model_name

    def call(
        self,
        messages: List[Dict[str, Any]],
        prompt: str,
        call_index: int,
        pipeline_attempt: int,
        request_context: Optional[Dict[str, Any]] = None,
        image_references: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[LogEntry]]:
        request_context = request_context or {}
        image_references = image_references or []

        provider = self._detect_provider()
        if provider == "openai":
            return self._call_openai(messages, prompt, call_index, pipeline_attempt, request_context, image_references)
        if provider == "anthropic":
            return self._call_anthropic(messages, prompt, call_index, pipeline_attempt, request_context, image_references)
        if provider == "qwen":
            return self._call_qwen(messages, prompt, call_index, pipeline_attempt, request_context, image_references)
        if provider == "huggingface":
            return self._call_huggingface(messages, prompt, call_index, pipeline_attempt, request_context, image_references)
        if provider == "gemini":
            return self._call_gemini(messages, prompt, call_index, pipeline_attempt, request_context, image_references)
        return self._call_openai(messages, prompt, call_index, pipeline_attempt, request_context, image_references)

    def _detect_provider(self) -> str:
        name = self.model_name.lower()
        if "gemini" in name:
            return "gemini"
        if name.startswith("hf:") or "huggingface" in name:
            return "huggingface"
        if any(token in name for token in ("claude", "sonnet", "haiku", "opus")):
            return "anthropic"
        if "qwen" in name:
            return "qwen"
        if name.startswith(("gpt", "o1", "o3")) or "gpt" in name:
            return "openai"
        if self.openai_base_url and self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        if self.qwen_api_key:
            return "qwen"
        if self.gemini_api_key:
            return "gemini"
        if self.hf_api_key:
            return "huggingface"
        return "openai"

    def _supports_temperature_override(self) -> bool:
        """部分模型（如 GPT-5 系列）不允许自定义 temperature。"""
        name = self.model_name.lower()
        return "gpt-5" not in name and not name.startswith("o1")

    def _openai_tokens_param(self) -> str:
        return "max_completion_tokens" if self.model_name.lower().startswith("gpt-5") else "max_tokens"

    def _format_openai_messages(self, messages: List[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for message in messages:
            content_items: List[Dict[str, Any]] = []
            for item in message.get("content", []):
                if item.get("type") == "text":
                    content_items.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image":
                    source = item.get("source", {})
                    data = source.get("data")
                    media = source.get("media_type", "image/png")
                    if data:
                        content_items.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media};base64,{data}"},
                            }
                        )
            formatted.append(
                {
                    "role": message.get("role", "user"),
                    "content": content_items or [{"type": "text", "text": prompt}],
                }
            )
        if not formatted:
            formatted.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        return formatted

    def _call_openai(
        self,
        messages: List[Dict[str, Any]],
        prompt: str,
        call_index: int,
        pipeline_attempt: int,
        request_context: Dict[str, Any],
        image_references: List[Dict[str, Any]],
    ) -> Tuple[str, List[LogEntry]]:
        if OpenAI is None:
            raise RuntimeError("openai 库缺失: 请执行 pip install openai")

        api_key = self.openai_api_key
        if not api_key:
            hint = MODEL_HINTS.get("openai", "")
            raise RuntimeError(f"缺少 OPENAI_API_KEY，无法调用 OpenAI 模型。{hint}")

        client = OpenAI(api_key=api_key, base_url=self.openai_base_url or None)

        formatted_messages = self._format_openai_messages(messages, prompt)
        request_id = uuid.uuid4().hex
        # 所有 OpenAI 模型统一使用同一 completion 上限，方便比较与调试
        max_tokens = DEFAULT_MAX_COMPLETION_TOKENS
        log_entry: LogEntry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "attempt": 1,
            "call_index": call_index,
            "pipeline_attempt": pipeline_attempt,
            "model": self.model_name,
            self._openai_tokens_param(): max_tokens,
            "messages": formatted_messages,
            "request_context": request_context,
            "image_references": image_references,
        }

        temperature = DEFAULT_TEMPERATURE if self._supports_temperature_override() else None
        request_payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
        }
        request_payload[self._openai_tokens_param()] = max_tokens
        if temperature is not None:
            request_payload["temperature"] = temperature

        completion = client.chat.completions.create(**request_payload)

        content = getattr(completion.choices[0].message, "content", "")
        if isinstance(content, list):
            response = "\n".join(chunk.get("text", "") for chunk in content if isinstance(chunk, dict))
        else:
            response = content or ""

        log_entry["result"] = "success"
        log_entry["response_text"] = response
        try:
            log_entry["raw_response"] = completion.model_dump_json()
        except Exception:
            log_entry["raw_response"] = str(completion)
        usage = getattr(completion, "usage", None)
        if usage:
            try:
                log_entry["usage"] = usage.model_dump()
            except Exception:
                log_entry["usage"] = usage.dict() if hasattr(usage, "dict") else usage

        log_entry[self._openai_tokens_param()] = max_tokens
        log_entry["temperature"] = temperature if temperature is not None else "default"
        return response, [log_entry]

    def _call_qwen(
        self,
        messages: List[Dict[str, Any]],
        prompt: str,
        call_index: int,
        pipeline_attempt: int,
        request_context: Dict[str, Any],
        image_references: List[Dict[str, Any]],
    ) -> Tuple[str, List[LogEntry]]:
        api_key = self.qwen_api_key
        if not api_key:
            hint = MODEL_HINTS.get("qwen", "")
            raise RuntimeError(f"缺少 QWEN_API_KEY，无法调用通义千问模型。{hint}")

        # 兼容两种 base_url 写法：
        # - https://dashscope-*/compatible-mode      -> /v1/chat/completions
        # - https://dashscope-*/compatible-mode/v1   -> /chat/completions
        base = self.qwen_base_url.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/chat/completions"
        else:
            url = f"{base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        formatted_messages = self._format_openai_messages(messages, prompt)
        max_tokens = DEFAULT_MAX_COMPLETION_TOKENS
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
        }
        if self._supports_temperature_override():
            payload["temperature"] = DEFAULT_TEMPERATURE

        request_id = uuid.uuid4().hex
        log_entry: LogEntry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "attempt": 1,
            "call_index": call_index,
            "pipeline_attempt": pipeline_attempt,
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": formatted_messages,
            "request_context": request_context,
            "image_references": image_references,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            log_entry["http_status"] = response.status_code
            log_entry["elapsed_seconds"] = response.elapsed.total_seconds() if hasattr(response, "elapsed") else None

            if response.status_code != 200:
                log_entry["result"] = "http_error"
                log_entry["raw_response"] = response.text
                hint = MODEL_HINTS.get("qwen", "")
                raise RuntimeError(
                    f"通义千问请求失败 (status={response.status_code})，模型: {self.model_name}。{hint}\n"
                    f"响应片段: {response.text[:200]}"
                )

            data = response.json()
            choices = data.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content")
            if isinstance(content, list):
                response_text = "\n".join(chunk.get("text", "") for chunk in content if isinstance(chunk, dict))
            else:
                response_text = content or ""

            log_entry["result"] = "success"
            log_entry["response_text"] = response_text
            log_entry["raw_response"] = response.text
            usage = data.get("usage")
            if usage:
                log_entry["usage"] = usage
            log_entry["temperature"] = payload.get("temperature", "default")
            return response_text, [log_entry]

        except Exception as exc:
            log_entry["result"] = "exception"
            log_entry["error"] = str(exc)
            hint = MODEL_HINTS.get("qwen", "")
            raise RuntimeError(f"调用通义千问模型 {self.model_name} 失败: {exc}。{hint}")

    def _call_huggingface(
        self,
        messages: List[Dict[str, Any]],
        prompt: str,
        call_index: int,
        pipeline_attempt: int,
        request_context: Dict[str, Any],
        image_references: List[Dict[str, Any]],
    ) -> Tuple[str, List[LogEntry]]:
        api_key = self.hf_api_key
        if not api_key:
            hint = MODEL_HINTS.get("huggingface", "")
            raise RuntimeError(f"缺少 HF_API_KEY，无法调用 Hugging Face 模型。{hint}")

        if not self.hf_base_url:
            raise RuntimeError(
                "未配置 HF_API_BASE，无法调用 Hugging Face OpenAI 兼容端点。"
                " 请设置一个可用的 OpenAI-style Chat Completions 服务地址，例如自建 text-generation-inference `--enable-openai` 端点。"
            )

        if "api-inference.huggingface.co" in self.hf_base_url:
            raise RuntimeError(
                "HF_API_BASE 指向 Hugging Face 托管推理服务，它目前不提供 OpenAI 兼容 `/v1/chat/completions` 接口。"
                " 请改用自建或第三方提供的 OpenAI 兼容端点 (例如 TGI --enable-openai)，并在环境变量 HF_API_BASE 中设置该地址。"
            )

        model_id = self.hf_model or self.model_name
        if self.hf_base_url.endswith("/v1"):
            url = f"{self.hf_base_url}/chat/completions"
        else:
            url = f"{self.hf_base_url}/v1/chat/completions"
        formatted_messages = self._format_openai_messages(messages, prompt)
        max_tokens = 6000
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
        }
        if self._supports_temperature_override():
            payload["temperature"] = 0.6

        request_id = uuid.uuid4().hex
        log_entry: LogEntry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "attempt": 1,
            "call_index": call_index,
            "pipeline_attempt": pipeline_attempt,
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": formatted_messages,
            "request_context": request_context,
            "image_references": image_references,
        }

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            log_entry["http_status"] = response.status_code
            log_entry["elapsed_seconds"] = response.elapsed.total_seconds() if hasattr(response, "elapsed") else None

            if response.status_code != 200:
                log_entry["result"] = "http_error"
                log_entry["raw_response"] = response.text
                hint = MODEL_HINTS.get("huggingface", "")
                raise RuntimeError(
                    f"Hugging Face 请求失败 (status={response.status_code})，模型: {model_id}。{hint}\n"
                    f"响应片段: {response.text[:200]}"
                )

            data = response.json()
            choices = data.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content")
            if isinstance(content, list):
                response_text = "\n".join(chunk.get("text", "") for chunk in content if isinstance(chunk, dict))
            else:
                response_text = content or ""

            log_entry["result"] = "success"
            log_entry["response_text"] = response_text
            log_entry["raw_response"] = response.text
            usage = data.get("usage")
            if usage:
                log_entry["usage"] = usage
            log_entry["temperature"] = payload.get("temperature", "default")
            return response_text, [log_entry]

        except Exception as exc:
            log_entry["result"] = "exception"
            log_entry["error"] = str(exc)
            hint = MODEL_HINTS.get("huggingface", "")
            raise RuntimeError(f"调用 Hugging Face 模型 {model_id} 失败: {exc}。{hint}")

    def _format_gemini_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for message in messages:
            parts: List[Dict[str, Any]] = []
            for item in message.get("content", []):
                if item.get("type") == "text":
                    parts.append({"text": item.get("text", "")})
                elif item.get("type") == "image":
                    source = item.get("source", {})
                    data = source.get("data")
                    media = source.get("media_type", "image/png")
                    if data:
                        parts.append({"inline_data": {"mime_type": media, "data": data}})
            if parts:
                formatted.append(
                    {
                        "role": message.get("role", "user"),
                        "parts": parts,
                    }
                )
        return formatted

    def _call_gemini(
        self,
        messages: List[Dict[str, Any]],
        prompt: str,
        call_index: int,
        pipeline_attempt: int,
        request_context: Dict[str, Any],
        image_references: List[Dict[str, Any]],
    ) -> Tuple[str, List[LogEntry]]:
        api_key = self.gemini_api_key
        if not api_key:
            hint = MODEL_HINTS.get("gemini", "")
            raise RuntimeError(f"缺少 GEMINI_API_KEY/GOOGLE_API_KEY，无法调用 Gemini 模型。{hint}")

        model_id = self.gemini_model or self.model_name
        if not model_id.startswith("models/"):
            model_id = f"models/{model_id}"

        formatted_contents = self._format_gemini_messages(messages)
        if not formatted_contents:
            formatted_contents = [{"role": "user", "parts": [{"text": prompt}]}]

        max_tokens = 11000
        generation_config: Dict[str, Any] = {
            "maxOutputTokens": max_tokens,
            "responseMimeType": "text/plain",
        }
        if self._supports_temperature_override():
            generation_config["temperature"] = 0.6

        payload: Dict[str, Any] = {
            "contents": formatted_contents,
            "generationConfig": generation_config,
        }

        base_url = self.gemini_api_base.rstrip("/")
        url = f"{base_url}/v1beta/{model_id}:generateContent?key={api_key}"

        max_attempts_env = os.getenv("GEMINI_MAX_RETRIES", "").strip()
        try:
            max_attempts = int(max_attempts_env) if max_attempts_env else 6
        except ValueError:
            max_attempts = 6

        retryable = {429, 500, 502, 503, 504, 529}
        logs: List[LogEntry] = []
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            request_id = uuid.uuid4().hex
            log_entry: LogEntry = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "attempt": attempt,
                "call_index": call_index,
                "pipeline_attempt": pipeline_attempt,
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": formatted_contents,
                "request_context": request_context,
                "image_references": image_references,
            }

            try:
                response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=180)
                log_entry["http_status"] = response.status_code
                log_entry["elapsed_seconds"] = response.elapsed.total_seconds() if hasattr(response, "elapsed") else None

                if response.status_code != 200:
                    log_entry["result"] = "http_error"
                    log_entry["raw_response"] = response.text
                    logs.append(log_entry)

                    lower_body = (response.text or "").lower()
                    quota_exceeded = (
                        response.status_code == 429
                        and (
                            "exceeded your current quota" in lower_body
                            or "check your plan and billing details" in lower_body
                            or "insufficient quota" in lower_body
                        )
                    )

                    if response.status_code in retryable and not quota_exceeded and attempt < max_attempts:
                        retry_after = response.headers.get("Retry-After", "").strip()
                        delay: Optional[float] = None
                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except Exception:
                                delay = None
                        if delay is None:
                            # Exponential backoff with small jitter.
                            delay = min(120.0, (2.0 ** (attempt - 1)) * 3.0)
                            delay += random.random() * 0.5
                        time.sleep(delay)
                        continue

                    hint = MODEL_HINTS.get("gemini", "")
                    raise RuntimeError(
                        f"Gemini 请求失败 (status={response.status_code})，模型: {model_id}。{hint}\n"
                        f"响应片段: {response.text[:200]}"
                    )

                data = response.json()
                candidates = data.get("candidates", [])
                text_chunks: List[str] = []
                for candidate in candidates:
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    for part in parts:
                        text = part.get("text")
                        if text:
                            text_chunks.append(text)
                response_text = "\n".join(text_chunks).strip()

                log_entry["result"] = "success"
                log_entry["response_text"] = response_text
                log_entry["raw_response"] = response.text
                log_entry["temperature"] = generation_config.get("temperature", "default")
                usage = data.get("usageMetadata")
                if usage:
                    log_entry["usage"] = usage
                logs.append(log_entry)
                return response_text, logs

            except Exception as exc:
                last_error = exc
                log_entry["result"] = "exception"
                log_entry["error"] = str(exc)
                logs.append(log_entry)
                if attempt < max_attempts:
                    delay = min(120.0, (2.0 ** (attempt - 1)) * 3.0) + random.random() * 0.5
                    time.sleep(delay)
                    continue
                break

        hint = MODEL_HINTS.get("gemini", "")
        raise RuntimeError(f"调用 Gemini 模型 {model_id} 失败: {last_error}。{hint}")

    def _call_anthropic(
        self,
        messages: List[Dict[str, Any]],
        prompt: str,
        call_index: int,
        pipeline_attempt: int,
        request_context: Dict[str, Any],
        image_references: List[Dict[str, Any]],
    ) -> Tuple[str, List[LogEntry]]:
        api_key = self.anthropic_api_key
        logs: List[LogEntry] = []

        # Anthropic 对单次输出 token 有硬上限，如果超过会直接返回 400。
        # 这里对默认上限做一次安全收缩，避免 3.x / 4.x 系列模型因为 max_tokens 过大而报 400。
        max_tokens = min(DEFAULT_MAX_COMPLETION_TOKENS, 8192)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if not api_key:
            fallback_text = (
                "ANALYSIS:\nDefault analysis: missing ANTHROPIC_API_KEY.\n\n"
                "HTML:\n```html\n"
                f"{DEFAULT_FALLBACK_HTML}\n"
                "```"
            )
            logs.append(
                {
                    "request_id": uuid.uuid4().hex,
                    "timestamp": datetime.now().isoformat(),
                    "attempt": 0,
                    "call_index": call_index,
                    "pipeline_attempt": pipeline_attempt,
                    "model": payload.get("model"),
                    "max_tokens": payload.get("max_tokens"),
                    "messages": messages,
                    "request_context": request_context,
                    "image_references": image_references,
                    "result": "fallback_no_api_key",
                    "response_text": fallback_text,
                    "raw_response": fallback_text,
                }
            )
            return fallback_text, logs

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        if self._supports_temperature_override():
            payload["temperature"] = DEFAULT_TEMPERATURE

        max_attempts = 3
        timeouts = [90, 120, 180]

        for attempt in range(max_attempts):
            request_id = uuid.uuid4().hex
            log_entry: LogEntry = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "attempt": attempt + 1,
                "call_index": call_index,
                "pipeline_attempt": pipeline_attempt,
                "model": payload.get("model"),
                "max_tokens": payload.get("max_tokens"),
                "messages": messages,
                "request_context": request_context,
                "image_references": image_references,
            }

            try:
                start = time.time()
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=timeouts[attempt],
                )
                log_entry["elapsed_seconds"] = round(time.time() - start, 3)
                log_entry["http_status"] = response.status_code

                if response.status_code == 200:
                    result = response.json()
                    usage = result.get("usage")
                    if usage:
                        log_entry["usage"] = usage
                    log_entry["response_id"] = result.get("id")
                    log_entry["response_model"] = result.get("model")
                    log_entry["stop_reason"] = result.get("stop_reason")

                    content_blocks = result.get("content", []) or []
                    text_chunks: List[str] = []
                    for block in content_blocks:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_chunks.append(block.get("text", ""))
                    response_text = "\n".join(text_chunks).strip()
                    log_entry["result"] = "success"
                    log_entry["response_text"] = response_text
                    log_entry["raw_response"] = response.text
                    logs.append(log_entry)
                    return response_text, logs

                elif response.status_code in {502, 503, 504, 529}:
                    log_entry["result"] = "retryable_http_error"
                    log_entry["raw_response"] = response.text
                    logs.append(log_entry)
                    continue

                else:
                    log_entry["result"] = "http_error"
                    log_entry["raw_response"] = response.text
                    logs.append(log_entry)
                    response.raise_for_status()

            except requests.Timeout:
                log_entry["result"] = "timeout"
                logs.append(log_entry)
            except Exception as exc:
                log_entry["result"] = "exception"
                log_entry["error"] = str(exc)
                logs.append(log_entry)
                raise

        final_text = (
            "ANALYSIS:\nDefault analysis: Anthropic request failed multiple times.\n\n"
            "HTML:\n```html\n"
            f"{DEFAULT_FALLBACK_HTML}\n"
            "```"
        )
        logs.append(
            {
                "request_id": uuid.uuid4().hex,
                "timestamp": datetime.now().isoformat(),
                "attempt": max_attempts + 1,
                "call_index": call_index,
                "pipeline_attempt": pipeline_attempt,
                "model": payload.get("model"),
                "max_tokens": payload.get("max_tokens"),
                "messages": messages,
                "request_context": request_context,
                "image_references": image_references,
                "result": "fallback_after_failures",
                "response_text": final_text,
                "raw_response": final_text,
            }
        )
        return final_text, logs
