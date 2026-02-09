#!/usr/bin/env python3
"""Utility helpers for parsing LLM responses."""
from __future__ import annotations

from typing import Dict


def extract_section(text: str, section_label: str) -> str:
    """Extract the text following `section_label` until the next label or EOF."""
    if not text or not section_label:
        return ""
    section_label = section_label.strip()
    if not section_label:
        return ""

    lowered = text
    start = lowered.find(section_label)
    if start == -1:
        return ""

    start += len(section_label)
    remainder = lowered[start:]

    lines = remainder.splitlines()
    collected = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ANALYSIS:") or stripped.startswith("CODE:"):
            break
        collected.append(line)

    return "\n".join(collected).strip()


def extract_code_block(text: str) -> str:
    """Return the first JavaScript code block fenced by ```javascript."""
    if not text:
        return ""

    fences = ["```javascript", "```js", "```"]
    start_idx = -1
    fence = None
    for candidate in fences:
        start_idx = text.find(candidate)
        if start_idx != -1:
            fence = candidate
            break
    if start_idx == -1:
        return ""

    start_idx += len(fence)
    end_idx = text.find("```", start_idx)
    if end_idx == -1:
        return ""

    return text[start_idx:end_idx].strip()
