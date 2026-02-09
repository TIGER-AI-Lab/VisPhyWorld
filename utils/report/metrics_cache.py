from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from utils.text_metrics_evaluator import compute_text_metrics


def compute_and_store_text_metrics(
    image_name: str,
    metrics_dir: Path,
    gt_text: str,
    model_text: str,
    use_bertscore: bool,
    gt_source: str = "gpt5.1",
) -> Tuple[Dict[str, float], Optional[Path]]:
    metrics_path = metrics_dir / f"{image_name}_text_metrics.json"
    if metrics_path.exists():
        try:
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
            cached_metrics = cached.get("metrics") or {}
            if cached_metrics:
                # Reuse cache only when it already satisfies the requested metrics.
                # Note: treat None as missing.
                has_rouge = all(k in cached_metrics and cached_metrics.get(k) is not None for k in ("rougeL_f1", "rougeL_precision", "rougeL_recall"))
                has_bertscore = (cached_metrics.get("bertscore_f1") is not None) if use_bertscore else True
                if has_rouge and has_bertscore:
                    return cached_metrics, metrics_path
        except Exception:
            pass

    metrics = compute_text_metrics(gt_text, model_text, use_bertscore=use_bertscore)
    payload = {
        "video_name": image_name,
        "field": "analysis",
        "gt_source": gt_source,
        "metrics": metrics,
    }
    try:
        metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        metrics_path = None
    return metrics, metrics_path


def load_cached_video_metrics(
    metrics_dir: Path,
    image_name: str,
    sample_every: int,
    max_frames: Optional[int] = None,
    raft_sample_indices: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, Any]]:
    path = metrics_dir / f"{image_name}_video_metrics.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("sample_every") != sample_every:
        return None
    stored_max_frames = data.get("max_frames")
    if max_frames is not None or stored_max_frames is not None:
        if stored_max_frames != max_frames:
            return None
    stored_raft_indices = data.get("raft_sample_indices")
    if raft_sample_indices is not None:
        if stored_raft_indices != list(raft_sample_indices):
            return None
    metrics = data.get("metrics")
    if not isinstance(metrics, dict) or "original_vs_generated" not in metrics:
        return None

    return data


def store_video_metrics(
    metrics_dir: Path,
    image_name: str,
    sample_every: int,
    metrics: Dict[str, Any],
    frame_count: int,
    max_frames: Optional[int],
    raft_sample_indices: Optional[Sequence[int]],
) -> None:
    def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in incoming.items():
            if value is None:
                continue
            existing = base.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                base[key] = _deep_merge(existing, value)
            else:
                base[key] = value
        return base

    path = metrics_dir / f"{image_name}_video_metrics.json"
    merged_metrics: Dict[str, Any] = dict(metrics)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if isinstance(existing, dict):
            if existing.get("sample_every") == sample_every:
                stored_max_frames = existing.get("max_frames")
                if max_frames is None and stored_max_frames is None or stored_max_frames == max_frames:
                    stored_raft_indices = existing.get("raft_sample_indices")
                    expected_raft = list(raft_sample_indices) if raft_sample_indices else None
                    if stored_raft_indices == expected_raft:
                        existing_metrics = existing.get("metrics")
                        if isinstance(existing_metrics, dict):
                            merged_metrics = _deep_merge(existing_metrics, merged_metrics)

    payload = {
        "video_name": image_name,
        "sample_every": sample_every,
        "frame_count": frame_count,
        "metrics": merged_metrics,
        "max_frames": max_frames,
        "raft_sample_indices": list(raft_sample_indices) if raft_sample_indices else None,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
