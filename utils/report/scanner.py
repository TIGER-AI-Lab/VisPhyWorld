from __future__ import annotations

import json
import os
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.evaluate_video_metrics import evaluate_pair as evaluate_video_pair
from utils.llm_video_evaluator import (
    evaluate_video_pair as llm_evaluate_video_pair,
    LLMVideoEvaluationError,
    PROMPT_VERSION as GEMINI_EVAL_PROMPT_VERSION,
)
from utils.report.heavy_metrics import HEAVY_METRIC_MODULES, merge_metrics, run_heavy_metric, metric_keys_missing
from utils.report.metrics_cache import (
    compute_and_store_text_metrics,
    load_cached_video_metrics,
    store_video_metrics,
)
from utils.report.models import GTGenerationConfig, LLMSummary
from utils.report.text_utils import (
    ensure_english_text,
    ensure_gt_analysis,
    extract_html_block,
    extract_response_sections,
    find_variant_path,
    pick_existing,
    select_model_analysis_text,
    text_needs_translation,
)


def _resolve_run_subdir(
    root: Path,
    name: str,
    *,
    ensure_exists: bool = False,
    create: bool = False,
) -> Path:
    """
    Prefer the run root directory, otherwise fall back to artifacts/<name>.
    Creates the directory when requested. Raises if ensure_exists and nothing found.
    """

    direct = root / name
    if direct.exists():
        return direct.resolve()

    artifacts_root = root / "artifacts"
    artifacts_candidate = artifacts_root / name
    if artifacts_candidate.exists():
        return artifacts_candidate.resolve()

    if create:
        parent = artifacts_root
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / name
        target.mkdir(parents=True, exist_ok=True)
        return target.resolve()

    if ensure_exists:
        raise FileNotFoundError(
            f"未找到 {name} 目录，检查位置: {direct} 或 {artifacts_candidate}"
        )

    return direct.resolve()


def _append_error(errors: List[str], image_name: str, message: str) -> None:
    errors.append(message)
    print(f"[WARN] {image_name}: {message}", flush=True)


def _infer_model_name(calls: List[Dict[str, Any]]) -> str:
    for call in reversed(calls or []):
        model = call.get("model") or call.get("model_name")
        if model:
            return str(model)
    return "unknown"


def _discover_baseline_dirs(root: Path) -> List[Path]:
    root = root.resolve()
    discovered: List[Path] = []
    direct_logs = root / "logs"
    artifacts_logs = root / "artifacts" / "logs"
    if direct_logs.exists() or artifacts_logs.exists():
        discovered.append(root)
    if not root.exists():
        return discovered
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        logs_dir = child / "logs"
        artifacts_logs_dir = child / "artifacts" / "logs"
        if not logs_dir.exists() and not artifacts_logs_dir.exists():
            continue
        name_lower = child.name.lower()
        if "baseline" not in name_lower:
            search_dir = artifacts_logs_dir if artifacts_logs_dir.exists() else logs_dir
            manifest_sample = next(search_dir.glob("*.json"), None)
            if not manifest_sample:
                continue
            try:
                manifest_data = json.loads(manifest_sample.read_text(encoding="utf-8"))
            except Exception:
                continue
            baseline_flag = manifest_data.get("baseline")
            if not isinstance(baseline_flag, str):
                continue
        discovered.append(child.resolve())
    return discovered


def _collect_baseline_entries(
    baseline_root: Path,
    *,
    dataset_dir: Optional[Path],
    dataset_frames_dir: Optional[Path],
    sample_every: int,
    max_frames: Optional[int],
    auto_align: bool,
    alignment_max_offset: int,
    alignment_window: int,
    alignment_offset_penalty: float,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for run_dir in _discover_baseline_dirs(baseline_root):
        logs_root = run_dir / 'artifacts' / 'logs'
        if not logs_root.exists():
            logs_root = run_dir / 'logs'
        predictions_root = run_dir / 'artifacts' / 'predictions'
        videos_root = run_dir / 'artifacts' / 'videos'
        metrics_root = run_dir / 'artifacts' / 'metrics'
        metrics_root.mkdir(parents=True, exist_ok=True)

        candidate_runs: List[Tuple[str, List[Path], Path, Path, Path]] = []
        if logs_root.exists():
            root_manifests = sorted(p for p in logs_root.glob('*.json') if p.is_file())
            if root_manifests:
                candidate_runs.append((run_dir.name, root_manifests, predictions_root, videos_root, metrics_root))
            for subdir in sorted(p for p in logs_root.iterdir() if p.is_dir()):
                manifest_files = sorted(subdir.glob('*.json'))
                if not manifest_files:
                    continue
                baseline_name = subdir.name
                candidate_runs.append(
                    (
                        baseline_name,
                        manifest_files,
                        predictions_root / baseline_name,
                        videos_root / baseline_name,
                        metrics_root / baseline_name,
                    )
                )

        if not candidate_runs:
            continue

        def _resolve_artifact(value: Optional[str], default_dir: Path) -> Optional[Path]:
            if not isinstance(value, str) or not value:
                return None
            candidate = Path(value)
            if candidate.exists():
                return candidate
            fallback = default_dir / candidate.name
            if fallback.exists():
                return fallback
            return None

        for baseline_label, manifest_files, predictions_dir, videos_dir, metrics_dir in candidate_runs:
            predictions_dir.mkdir(parents=True, exist_ok=True)
            videos_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)

            entries: List[LLMSummary] = []
            aggregate_metrics: Dict[str, Dict[str, List[float]]] = {}

            for manifest_path in manifest_files:
                try:
                    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
                except Exception:
                    continue

                video_name = payload.get('video_name') or manifest_path.stem.split('_')[0]
                baseline_name = payload.get('baseline') or baseline_label
                html_path_value = payload.get('html_path')
                video_path_value = payload.get('video_path')

                html_path = _resolve_artifact(html_path_value, predictions_dir)
                video_path = _resolve_artifact(video_path_value, videos_dir)

                html_content = ''
                if html_path and html_path.exists():
                    try:
                        html_content = html_path.read_text(encoding='utf-8')
                    except Exception:
                        html_content = ''

                dataset_video_path: Optional[Path] = None
                if dataset_dir:
                    candidate_video = dataset_dir / f"{video_name}.mp4"
                    if candidate_video.exists():
                        dataset_video_path = candidate_video

                assets: Dict[str, Optional[Path]] = {}
                if dataset_frames_dir:
                    assets['dataset_first_frame'] = pick_existing(
                        dataset_frames_dir / f"{video_name}_frame_01.png",
                        dataset_frames_dir / f"{video_name}_first_frame.png",
                    )
                    assets['dataset_tenth_frame'] = pick_existing(
                        dataset_frames_dir / f"{video_name}_frame_10.png",
                        dataset_frames_dir / f"{video_name}_tenth_frame.png",
                    )
                assets['dataset_video'] = dataset_video_path
                assets['html_path'] = html_path if html_path and html_path.exists() else None
                assets['generated_original_video'] = video_path if video_path and video_path.exists() else None

                video_metrics: Dict[str, Dict[str, Any]] = {}
                frame_count = 0
                errors: List[str] = []
                success_flag = bool(video_path and video_path.exists())

                if dataset_video_path and video_path and video_path.exists():
                    try:
                        metrics_result = evaluate_video_pair(
                            str(dataset_video_path),
                            str(video_path),
                            sample_every=sample_every,
                            max_frames=max_frames,
                            caption_text=None,
                            use_clip=False,
                            use_lpips=False,
                            use_fsim=False,
                            use_vsi=False,
                            use_dists=False,
                            use_dino=False,
                            use_raft=False,
                            diagnostics_dir=None,
                            video_name=f"{video_name}_baseline_{baseline_name}",
                            variant_label=baseline_name,
                            rel_root=str(run_dir),
                            auto_align=auto_align,
                            alignment_max_offset=alignment_max_offset,
                            alignment_window=alignment_window,
                            alignment_offset_penalty=alignment_offset_penalty,
                            raft_sample_indices=None,
                        )
                        if metrics_result:
                            video_metrics['original_vs_generated'] = metrics_result
                            frame_count = int(metrics_result.get('num_frames', 0))
                            aggregator = aggregate_metrics.setdefault('original_vs_generated', {})
                            metrics_result['video_success'] = 1.0 if (video_path and video_path.exists()) else 0.0
                            for key, value in metrics_result.items():
                                if key in {'per_frame', 'num_frames'}:
                                    continue
                                if isinstance(value, (int, float)) and not math.isnan(value):
                                    aggregator.setdefault(key, []).append(float(value))
                            metrics_path = metrics_dir / f"{video_name}_{baseline_name}_video_metrics.json"
                            with metrics_path.open('w', encoding='utf-8') as handle:
                                json.dump(metrics_result, handle, ensure_ascii=False, indent=2)
                    except Exception as exc:
                        errors.append(f"基线 {baseline_name} 视频指标失败: {exc}")
                else:
                    if not dataset_video_path:
                        errors.append('缺少原始视频用于基线对比')
                    if not video_path or not video_path.exists():
                        errors.append('缺少基线生成视频')

                entry = LLMSummary(
                    image_name=video_name,
                    llm_calls_path=manifest_path,
                    llm_calls=[],
                    text_sections={},
                    html_content=html_content,
                    assets=assets,
                    video_metrics=video_metrics,
                    text_metrics={},
                    frame_count=frame_count,
                    success=success_flag,
                    errors=errors,
                    gt_analysis=None,
                    gt_analysis_path=None,
                    model_analysis=payload.get('message'),
                    model_analysis_path=None,
                    text_metrics_path=None,
                    model_name=f"baseline:{baseline_name}",
                )
                entries.append(entry)
                status_icon = '✅' if success_flag else '⚠️'
                print(f"[baseline:{baseline_name}] {status_icon} {video_name}", flush=True)

            metrics_overview: List[Dict[str, float]] = []
            for metric_name, values in aggregate_metrics.items():
                overview_entry: Dict[str, float] = {'label': metric_name}
                for key, numbers in values.items():
                    if not numbers or key in {'num_frames', 'per_frame'}:
                        continue
                    valid = [x for x in numbers if not math.isnan(x)]
                    if not valid:
                        continue
                    overview_entry[key] = float(statistics.fmean(valid))
                metrics_overview.append(overview_entry)

            summaries.append(
                {
                    'run_dir': run_dir,
                    'baseline_name': baseline_label,
                    'entries': entries,
                    'metrics_overview': metrics_overview,
                    'total': len(entries),
                    'successful': sum(1 for e in entries if e.success),
                    'failed': sum(1 for e in entries if not e.success),
                }
            )

    return summaries


def scan_output_directory(
    result_dir: Path,
    dataset_dir: Optional[Path] = None,
    sample_every: int = 3,
    gt_config: Optional[GTGenerationConfig] = None,
    use_bertscore: bool = True,
    heavy_metrics: bool = True,
    enable_raft_metrics: bool = False,
    auto_align: bool = True,
    alignment_max_offset: int = 30,
    alignment_window: int = 3,
    alignment_offset_penalty: float = 0.05,
    resume_from: Optional[str] = None,
    strict: bool = False,
    enable_llm_evaluation: bool = False,
    include_samples: Optional[Sequence[str]] = None,
    max_frames: Optional[int] = None,
    max_samples: Optional[int] = None,
    raft_sample_indices: Optional[Sequence[int]] = None,
    baseline_root: Optional[Path] = None,
    jobs: int = 1,
) -> Dict[str, Any]:
    alignment_max_offset = max(0, alignment_max_offset)
    alignment_window = max(1, alignment_window)
    alignment_offset_penalty = max(0.0, alignment_offset_penalty)

    root = result_dir.resolve()
    logs_dir = _resolve_run_subdir(root, "logs", ensure_exists=True)
    # Some older runs may not have extracted preview frames (frames/). Those are
    # only used for HTML thumbnails, so we create the directory on-demand instead
    # of failing the whole evaluation.
    frames_dir = _resolve_run_subdir(root, "frames", create=True)
    # Similarly, videos/ might be absent for incomplete runs; keep the report
    # generation resilient and surface "missing generated video" per sample.
    videos_dir = _resolve_run_subdir(root, "videos", create=True)

    gt_dir = _resolve_run_subdir(root, "gt_analysis", create=True)
    model_analysis_dir = _resolve_run_subdir(root, "model_analysis", create=True)
    metrics_dir = _resolve_run_subdir(root, "metrics", create=True)
    if enable_llm_evaluation and not os.environ.get("GEMINI_API_KEY"):
        print("⚠️ GEMINI_API_KEY is not set; skipping Gemini video evaluation.", flush=True)
        enable_llm_evaluation = False
    if enable_raft_metrics:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            _ = getattr(torchvision.models, "optical_flow", None)
        except Exception as exc:
            print(f"⚠️ RAFT 依赖缺失，跳过 RAFT 指标：{exc}", flush=True)
            enable_raft_metrics = False
    llm_cache_root = metrics_dir / "llm_gemini_cache"
    if enable_llm_evaluation:
        llm_cache_root.mkdir(parents=True, exist_ok=True)

    dataset_dir_resolved: Optional[Path] = None
    dataset_frames_dir: Optional[Path] = None
    baseline_root_resolved: Optional[Path] = None

    if dataset_dir:
        candidate = Path(dataset_dir).resolve()
        if candidate.exists():
            dataset_dir_resolved = candidate
            potential_frames = candidate / "extracted_frames"
            dataset_frames_dir = potential_frames if potential_frames.exists() else None

    if baseline_root:
        candidate = Path(baseline_root).resolve()
        if candidate.exists():
            baseline_root_resolved = candidate
        else:
            print(f"⚠️ baseline-root does not contain a valid directory: {candidate}", flush=True)

    if gt_config is None:
        gt_config = GTGenerationConfig(enabled=False)

    entries: List[LLMSummary] = []
    aggregate_metrics: Dict[str, Dict[str, List[float]]] = {}
    resume_target = resume_from.strip() if resume_from else None
    resume_reached = resume_target is None

    include_set = {s.strip() for s in include_samples} if include_samples else None

    log_files = sorted(logs_dir.glob("*_llm_calls.json"))
    if include_set is not None:
        log_files = [log_file for log_file in log_files if log_file.stem.replace("_llm_calls", "") in include_set]

    raft_sample_indices_list: Optional[List[int]] = None
    raft_sample_indices_zero: Optional[List[int]] = None
    if raft_sample_indices:
        raft_sample_indices_list = [int(idx) for idx in raft_sample_indices if idx is not None and int(idx) > 0]
        if raft_sample_indices_list:
            raft_sample_indices_zero = [idx - 1 for idx in raft_sample_indices_list if idx > 0]

    total_logs = len(log_files)

    jobs = max(1, int(jobs or 1))

    def _critical_errors(errors: List[str]) -> List[str]:
        return [
            err
            for err in errors
            if not err.startswith("Video metric computation failed")
            and not err.startswith("Fallback metric computation failed")
            and not err.startswith("Missing reference video")
            and not err.startswith("Missing generated video output")
            and not err.startswith("Text metric computation failed")
            and not err.startswith("Model analysis translation failed")
            and not err.startswith("CLIP")
            and not err.startswith("DINO")
            and not err.startswith("LPIPS")
        ]

    def _update_aggregate_metrics_for_entry(entry: LLMSummary) -> None:
        llm_similarity = entry.video_metrics.get("llm_similarity") if entry.video_metrics else None
        if isinstance(llm_similarity, dict):
            score_val = llm_similarity.get("score")
            if isinstance(score_val, (int, float)) and not math.isnan(float(score_val)):
                aggregator_llm = aggregate_metrics.setdefault("Gemini similarity", {})
                aggregator_llm.setdefault("score", []).append(float(score_val))

        if entry.video_metrics.get("original_vs_generated"):
            final_metrics = entry.video_metrics["original_vs_generated"]
            if isinstance(final_metrics, dict):
                generated_video = entry.assets.get("generated_original_video") if entry.assets else None
                video_exists = bool(generated_video and isinstance(generated_video, Path) and generated_video.exists())
                final_metrics["video_success"] = 1.0 if video_exists else 0.0
                aggregator = aggregate_metrics.setdefault("original_vs_generated", {})
                for key, value in final_metrics.items():
                    if key in {"per_frame", "num_frames"}:
                        continue
                    if isinstance(value, (int, float)) and not math.isnan(float(value)):
                        aggregator.setdefault(key, []).append(float(value))

    def _process_one(log_file: Path) -> LLMSummary:
        with log_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        image_name = payload.get("image_name") or log_file.stem.replace("_llm_calls", "")
        calls: List[Dict[str, Any]] = payload.get("requests", [])
        final_call = calls[-1] if calls else {}
        response_text = final_call.get("response_text", "")
        raw_response = final_call.get("raw_response")
        if not response_text and raw_response:
            try:
                blob = json.loads(raw_response)
                if isinstance(blob, dict):
                    content = blob.get("content") or []
                    if content and isinstance(content, list):
                        response_text = content[0].get("text", "") or ""
            except Exception:
                response_text = raw_response

        text_sections = extract_response_sections(response_text)
        html_content = ""

        assets: Dict[str, Optional[Path]] = {}
        errors: List[str] = []

        generated_frame_path = pick_existing(
            frames_dir / f"{image_name}_generated_frame_01.png",
            frames_dir / f"{image_name}_frame_01.png",
            frames_dir / f"{image_name}_frame.png",
        )
        assets["generated_frame"] = generated_frame_path if generated_frame_path and generated_frame_path.exists() else None
        generated_frame_variant = find_variant_path(generated_frame_path)
        if generated_frame_variant and generated_frame_variant.exists():
            assets["generated_frame_variant"] = generated_frame_variant

        if dataset_frames_dir:
            assets["dataset_first_frame"] = pick_existing(
                dataset_frames_dir / f"{image_name}_frame_01.png",
                dataset_frames_dir / f"{image_name}_first_frame.png",
            )
            assets["dataset_tenth_frame"] = pick_existing(
                dataset_frames_dir / f"{image_name}_frame_10.png",
                dataset_frames_dir / f"{image_name}_tenth_frame.png",
            )

        generated_original_video = videos_dir / f"{image_name}_original_2d.mp4"
        assets["generated_original_video"] = generated_original_video if generated_original_video.exists() else None

        assets["render_log"] = pick_existing(videos_dir / f"{image_name}_original_2d.log")

        html_candidates = sorted(logs_dir.glob(f"{image_name}_*.html"))
        if not html_candidates:
            html_candidates = sorted(videos_dir.glob(f"{image_name}_render_attempt_*.html"))
        assets["html_path"] = html_candidates[-1] if html_candidates else None

        if assets.get("html_path") and assets["html_path"].exists():
            try:
                html_content = assets["html_path"].read_text(encoding="utf-8")
            except Exception as exc:
                _append_error(errors, image_name, f"读取生成HTML失败: {exc}")
                html_content = ""

        if not html_content:
            html_content = extract_html_block(response_text)

        dataset_video_path = None
        if dataset_dir_resolved:
            candidate_video = dataset_dir_resolved / f"{image_name}.mp4"
            if candidate_video.exists():
                dataset_video_path = candidate_video
            else:
                _append_error(errors, image_name, f"未找到原始视频: {candidate_video}")

        assets["dataset_video"] = dataset_video_path

        gt_analysis: Optional[str] = None
        gt_analysis_path: Optional[Path] = None
        try:
            first_frame_for_gt = assets.get("dataset_first_frame") or assets.get("generated_frame")
            tenth_frame_for_gt = assets.get("dataset_tenth_frame") or assets.get("generated_frame_variant")
            # Always attempt to load cached GT analysis if present.
            # `ensure_gt_analysis` will only call external APIs when gt_config.enabled is True.
            gt_analysis, gt_analysis_path = ensure_gt_analysis(
                image_name,
                first_frame_for_gt,
                tenth_frame_for_gt,
                gt_dir,
                gt_config,
            )
        except Exception as exc:
            _append_error(errors, image_name, f"GT 生成失败: {exc}")

        text_metrics: Dict[str, float] = {}
        text_metrics_path: Optional[Path] = None
        selection = select_model_analysis_text(text_sections)
        model_analysis_text = text_sections.get("analysis")
        primary_key: Optional[str] = None
        primary_model_text: Optional[str] = None
        if selection:
            primary_key, primary_model_text = selection
        if not model_analysis_text and primary_model_text:
            model_analysis_text = primary_model_text

        model_analysis_path: Optional[Path] = None
        primary_model_text_en: Optional[str] = None

        cached_analysis_path = model_analysis_dir / f"{image_name}_analysis.txt"
        if cached_analysis_path.exists():
            try:
                cached_text = cached_analysis_path.read_text(encoding="utf-8").strip()
                if cached_text and not text_needs_translation(cached_text):
                    primary_model_text_en = cached_text
                    model_analysis_path = cached_analysis_path
            except Exception:
                pass

        if primary_model_text and primary_model_text_en is None:
            try:
                primary_model_text_en = ensure_english_text(primary_model_text)
            except Exception as exc:
                _append_error(errors, image_name, f"模型分析翻译失败: {exc}")
                primary_model_text_en = primary_model_text

        if primary_model_text_en:
            model_analysis_text = primary_model_text_en
            if primary_key:
                text_sections[primary_key] = primary_model_text_en
            text_sections["analysis"] = primary_model_text_en
            try:
                cached_analysis_path.write_text(primary_model_text_en, encoding="utf-8")
                model_analysis_path = cached_analysis_path
            except Exception:
                model_analysis_path = None
        else:
            if primary_model_text:
                model_analysis_text = primary_model_text

        if gt_analysis and model_analysis_text:
            try:
                gt_source = "gpt5.1" if gt_analysis_path else "manual"
                text_metrics, text_metrics_path = compute_and_store_text_metrics(
                    image_name,
                    metrics_dir,
                    gt_analysis,
                    model_analysis_text,
                    use_bertscore,
                    gt_source=gt_source,
                )
            except Exception as exc:
                _append_error(errors, image_name, f"文本指标计算失败: {exc}")

        def _first_non_empty(*values: Optional[str]) -> Optional[str]:
            for value in values:
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        caption_for_metrics = _first_non_empty(
            gt_analysis,
            model_analysis_text,
            text_sections.get("analysis"),
            text_sections.get("prediction"),
            text_sections.get("final"),
        )

        video_metrics: Dict[str, Dict[str, Any]] = {}
        frame_count = 0
        generated_video_path = assets.get("generated_original_video")

        raft_diag_dir: Optional[Path] = None
        raft_diag_dir_str: Optional[str] = None
        if enable_raft_metrics:
            raft_diag_dir = videos_dir / "raft_visuals" / image_name
            raft_diag_dir.mkdir(parents=True, exist_ok=True)
            raft_diag_dir_str = str(raft_diag_dir)

        cached_video_metrics = load_cached_video_metrics(
            metrics_dir,
            image_name,
            sample_every,
            max_frames,
            raft_sample_indices_list,
        )
        metrics_cached = bool(cached_video_metrics)
        if cached_video_metrics:
            video_metrics = cached_video_metrics.get("metrics", {}) if cached_video_metrics else {}
            frame_count = int(cached_video_metrics.get("frame_count") or 0) if cached_video_metrics else 0

        success_flag = bool(generated_video_path and generated_video_path.exists())

        if not metrics_cached and dataset_video_path and generated_video_path and generated_video_path.exists():
            frame_count = 0
            try:
                metrics_original = evaluate_video_pair(
                    str(dataset_video_path),
                    str(generated_video_path),
                    sample_every=sample_every,
                    max_frames=max_frames,
                    caption_text=caption_for_metrics,
                    use_clip=False,
                    use_lpips=False,
                    use_fsim=False,
                    use_vsi=False,
                    use_dists=False,
                    use_dino=False,
                    use_raft=enable_raft_metrics,
                    diagnostics_dir=raft_diag_dir_str,
                    video_name=image_name,
                    variant_label="report",
                    rel_root=str(result_dir),
                    auto_align=auto_align,
                    alignment_max_offset=alignment_max_offset,
                    alignment_window=alignment_window,
                    alignment_offset_penalty=alignment_offset_penalty,
                    raft_sample_indices=raft_sample_indices_zero,
                )
                if metrics_original:
                    metrics_original.pop("lpips_mean", None)
                    metrics_original.pop("lpips_std", None)
                    metrics_original.pop("fsim_mean", None)
                    metrics_original.pop("fsim_std", None)
                    metrics_original.pop("vsi_mean", None)
                    metrics_original.pop("vsi_std", None)
                    metrics_original.pop("dists_mean", None)
                    metrics_original.pop("dists_std", None)
                    per_frame_stats = metrics_original.get("per_frame")
                    if isinstance(per_frame_stats, dict):
                        for key in ("lpips", "fsim", "vsi", "dists"):
                            per_frame_stats.pop(key, None)
                    video_metrics["original_vs_generated"] = metrics_original
                    frame_count = int(metrics_original.get("num_frames", 0))
                    store_video_metrics(
                        metrics_dir,
                        image_name,
                        sample_every,
                        video_metrics,
                        frame_count,
                        max_frames,
                        raft_sample_indices_list,
                    )
            except RecursionError as exc:
                _append_error(errors, image_name, f"视频指标计算失败: {exc}")
                try:
                    fallback_metrics = evaluate_video_pair(
                        str(dataset_video_path),
                        str(generated_video_path),
                        sample_every=sample_every,
                        max_frames=max_frames,
                        caption_text=caption_for_metrics,
                        use_clip=False,
                        use_lpips=False,
                        use_fsim=False,
                        use_vsi=False,
                        use_dists=False,
                        use_dino=False,
                        use_raft=enable_raft_metrics,
                        diagnostics_dir=raft_diag_dir_str,
                        video_name=image_name,
                        variant_label="report",
                        rel_root=str(result_dir),
                        auto_align=auto_align,
                        alignment_max_offset=alignment_max_offset,
                        alignment_window=alignment_window,
                        alignment_offset_penalty=alignment_offset_penalty,
                        raft_sample_indices=raft_sample_indices_zero,
                    )
                    if fallback_metrics:
                        video_metrics["original_vs_generated_fallback"] = fallback_metrics
                        frame_count = int(fallback_metrics.get("num_frames", 0))
                        store_video_metrics(
                            metrics_dir,
                            image_name,
                            sample_every,
                            video_metrics,
                            frame_count,
                            max_frames,
                            raft_sample_indices_list,
                        )
                except Exception as inner_exc:
                    _append_error(errors, image_name, f"基础指标回退失败: {inner_exc}")
            except Exception as exc:
                _append_error(errors, image_name, f"视频指标计算失败: {exc}")

            metrics_queue: List[str] = []
            if heavy_metrics:
                metrics_queue.extend(["clip", "lpips", "dino", "fsim", "vsi", "dists"])
                if enable_llm_evaluation:
                    metrics_queue.append("llm")
            if enable_raft_metrics:
                metrics_queue.append("raft")
            metrics_queue = [m for m in metrics_queue if m in HEAVY_METRIC_MODULES]

            if metrics_queue and video_metrics.get("original_vs_generated"):
                base_metrics = video_metrics["original_vs_generated"]
                for metric_name in metrics_queue:
                    if metric_name == "raft" and "raft_epe_mean" in base_metrics:
                        continue
                    try:
                        extra_cli_args: List[str] = []
                        if metric_name == "raft" and raft_sample_indices_list:
                            extra_cli_args.extend(["--raft-sample-indices"] + [str(idx) for idx in raft_sample_indices_list])
                        if metric_name == "llm":
                            extra_cli_args.append("--enable-llm")
                            extra_cli_args.extend(["--llm-cache-dir", str(llm_cache_root)])
                        if auto_align:
                            extra_cli_args.extend(
                                [
                                    "--alignment-max-offset",
                                    str(alignment_max_offset),
                                    "--alignment-window",
                                    str(alignment_window),
                                    "--alignment-offset-penalty",
                                    str(alignment_offset_penalty),
                                ]
                            )
                        else:
                            extra_cli_args.append("--no-auto-align")
                        extra = run_heavy_metric(
                            metric_name,
                            dataset_video_path,
                            generated_video_path,
                            sample_every,
                            caption_for_metrics,
                            max_frames=max_frames,
                            extra_cli_args=extra_cli_args,
                        )
                        merge_metrics(base_metrics, extra)
                        store_video_metrics(
                            metrics_dir,
                            image_name,
                            sample_every,
                            video_metrics,
                            frame_count,
                            max_frames,
                            raft_sample_indices_list,
                        )
                    except Exception as exc:
                        _append_error(errors, image_name, f"{metric_name.upper()} 指标计算失败: {exc}")

            llm_result: Optional[Dict[str, Any]] = None
            if enable_llm_evaluation and dataset_video_path and generated_video_path and generated_video_path.exists():
                try:
                    llm_result = llm_evaluate_video_pair(
                        dataset_video_path,
                        generated_video_path,
                        cache_dir=llm_cache_root,
                    )
                except LLMVideoEvaluationError as exc:
                    llm_result = {"error": str(exc)}
                except Exception as exc:
                    llm_result = {"error": str(exc)}
                if llm_result:
                    if isinstance(llm_result, dict) and llm_result.get("error"):
                        _append_error(errors, image_name, f"LLM 视频评估失败: {llm_result.get('error')}")
                    artifact_path = metrics_dir / f"{image_name}_gemini_eval.json"
                    try:
                        artifact_path.write_text(json.dumps(llm_result, ensure_ascii=False, indent=2), encoding="utf-8")
                        try:
                            llm_result["artifact_path"] = str(artifact_path.relative_to(root))
                        except Exception:
                            llm_result["artifact_path"] = str(artifact_path)
                    except Exception:
                        pass
                    video_metrics["llm_similarity"] = llm_result
                    store_video_metrics(
                        metrics_dir,
                        image_name,
                        sample_every,
                        video_metrics,
                        frame_count,
                        max_frames,
                        raft_sample_indices_list,
                    )
            elif enable_llm_evaluation and (not dataset_video_path or not generated_video_path or not generated_video_path.exists()):
                print(f"[INFO] {image_name}: skipped Gemini evaluation (missing reference or generated video)", flush=True)

        # If base metrics are cached, we still may need to (re)compute heavy metrics
        # or Gemini evaluation (e.g., after prompt updates).
        if metrics_cached and dataset_video_path and generated_video_path and generated_video_path.exists():
            base_metrics = video_metrics.get("original_vs_generated") if isinstance(video_metrics, dict) else None
            if isinstance(base_metrics, dict):
                required_keys_by_metric = {
                    "clip": ["clip_cosine_mean", "clip_caption_mean"],
                    "lpips": ["lpips_mean"],
                    "dino": ["dino_similarity_mean"],
                    "fsim": ["fsim_mean"],
                    "vsi": ["vsi_mean"],
                    "dists": ["dists_mean"],
                    "raft": ["raft_epe_mean", "raft_mag_diff_mean", "raft_angle_diff_deg_mean"],
                }
                metrics_queue: List[str] = []
                if heavy_metrics:
                    metrics_queue.extend(["clip", "lpips", "dino", "fsim", "vsi", "dists"])
                    if enable_llm_evaluation:
                        metrics_queue.append("llm")
                if enable_raft_metrics:
                    metrics_queue.append("raft")
                metrics_queue = [m for m in metrics_queue if m in HEAVY_METRIC_MODULES]

                for metric_name in metrics_queue:
                    if metric_name == "llm":
                        continue
                    required = required_keys_by_metric.get(metric_name, [])
                    if required and not metric_keys_missing(base_metrics, required):
                        continue
                    try:
                        extra_cli_args = []
                        if metric_name == "raft" and raft_sample_indices_list:
                            extra_cli_args.extend(["--raft-sample-indices"] + [str(idx) for idx in raft_sample_indices_list])
                        if auto_align:
                            extra_cli_args.extend(
                                [
                                    "--alignment-max-offset",
                                    str(alignment_max_offset),
                                    "--alignment-window",
                                    str(alignment_window),
                                    "--alignment-offset-penalty",
                                    str(alignment_offset_penalty),
                                ]
                            )
                        else:
                            extra_cli_args.append("--no-auto-align")
                        extra = run_heavy_metric(
                            metric_name,
                            dataset_video_path,
                            generated_video_path,
                            sample_every,
                            caption_for_metrics,
                            max_frames=max_frames,
                            extra_cli_args=extra_cli_args,
                        )
                        merge_metrics(base_metrics, extra)
                        store_video_metrics(
                            metrics_dir,
                            image_name,
                            sample_every,
                            video_metrics,
                            frame_count,
                            max_frames,
                            raft_sample_indices_list,
                        )
                    except Exception as exc:
                        _append_error(errors, image_name, f"{metric_name.upper()} 指标计算失败: {exc}")

                if enable_llm_evaluation:
                    existing_llm = video_metrics.get("llm_similarity") if isinstance(video_metrics, dict) else None
                    prompt_ok = (
                        isinstance(existing_llm, dict)
                        and existing_llm.get("score") is not None
                        and existing_llm.get("prompt_version") == GEMINI_EVAL_PROMPT_VERSION
                    )
                    if not prompt_ok:
                        llm_result = None
                        try:
                            llm_result = llm_evaluate_video_pair(
                                dataset_video_path,
                                generated_video_path,
                                cache_dir=llm_cache_root,
                            )
                        except LLMVideoEvaluationError as exc:
                            llm_result = {"error": str(exc)}
                        except Exception as exc:
                            llm_result = {"error": str(exc)}
                        if llm_result:
                            if isinstance(llm_result, dict) and llm_result.get("error"):
                                _append_error(errors, image_name, f"LLM 视频评估失败: {llm_result.get('error')}")
                            artifact_path = metrics_dir / f"{image_name}_gemini_eval.json"
                            try:
                                artifact_path.write_text(json.dumps(llm_result, ensure_ascii=False, indent=2), encoding="utf-8")
                                try:
                                    llm_result["artifact_path"] = str(artifact_path.relative_to(root))
                                except Exception:
                                    llm_result["artifact_path"] = str(artifact_path)
                            except Exception:
                                pass
                            video_metrics["llm_similarity"] = llm_result
                            store_video_metrics(
                                metrics_dir,
                                image_name,
                                sample_every,
                                video_metrics,
                                frame_count,
                                max_frames,
                                raft_sample_indices_list,
                            )

        elif not metrics_cached:
            if not dataset_video_path:
                _append_error(errors, image_name, "Missing reference video for comparison")
            if not generated_video_path or not generated_video_path.exists():
                _append_error(errors, image_name, "Missing generated video output")

        success = bool(generated_video_path and generated_video_path.exists())
        if not html_content:
            _append_error(errors, image_name, "Missing generated HTML segment")

        model_name_inferred = _infer_model_name(calls)

        entry = LLMSummary(
            image_name=image_name,
            llm_calls_path=log_file,
            llm_calls=calls,
            text_sections=text_sections,
            html_content=html_content,
            assets=assets,
            video_metrics=video_metrics,
            text_metrics=text_metrics,
            frame_count=frame_count,
            success=success,
            errors=errors,
            gt_analysis=gt_analysis,
            gt_analysis_path=gt_analysis_path,
            model_analysis=model_analysis_text,
            model_analysis_path=model_analysis_path,
            text_metrics_path=text_metrics_path,
            model_name=model_name_inferred,
        )
        return entry

    # Apply resume/max_samples in both sequential and parallel mode.
    log_files_to_process = log_files
    if resume_target:
        try:
            start = next(i for i, lf in enumerate(log_files_to_process) if lf.stem.replace("_llm_calls", "") == resume_target)
            log_files_to_process = log_files_to_process[start:]
        except StopIteration:
            log_files_to_process = []
    if max_samples is not None and max_samples > 0:
        log_files_to_process = log_files_to_process[: max_samples]
    total_logs = len(log_files_to_process)

    if jobs > 1:
        print(f"[INFO] Parallel scan enabled: jobs={jobs} samples={total_logs}", flush=True)
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            fut_to_name = {ex.submit(_process_one, lf): lf.stem.replace("_llm_calls", "") for lf in log_files_to_process}
            completed = 0
            for fut in as_completed(fut_to_name):
                entry = fut.result()
                entries.append(entry)
                _update_aggregate_metrics_for_entry(entry)
                completed += 1
                status = "✅" if entry.success else "⚠️"
                print(f"[{completed}/{total_logs}] {status} {entry.image_name}", flush=True)
                if strict and entry.errors:
                    critical_errors = _critical_errors(entry.errors)
                    if critical_errors:
                        raise RuntimeError(f"{entry.image_name} has an error: {critical_errors[0]}")
        entries.sort(key=lambda e: e.image_name)
    else:
        for index, log_file in enumerate(log_files_to_process, start=1):
            entry = _process_one(log_file)
            entries.append(entry)
            _update_aggregate_metrics_for_entry(entry)
            status = "✅" if entry.success else "⚠️"
            print(f"[{index}/{total_logs}] {status} {entry.image_name}", flush=True)
            if strict and entry.errors:
                critical_errors = _critical_errors(entry.errors)
                if critical_errors:
                    raise RuntimeError(f"{entry.image_name} has an error: {critical_errors[0]}")

        """

        image_name = payload.get("image_name") or log_file.stem.replace("_llm_calls", "")
        if include_set is not None and image_name not in include_set:
            continue
        if not resume_reached:
            if image_name != resume_target:
                continue
            resume_reached = True
        calls: List[Dict[str, Any]] = payload.get("requests", [])
        final_call = calls[-1] if calls else {}
        response_text = final_call.get("response_text", "")
        raw_response = final_call.get("raw_response")
        if not response_text and raw_response:
            try:
                blob = json.loads(raw_response)
                if isinstance(blob, dict):
                    content = blob.get("content") or []
                    if content and isinstance(content, list):
                        response_text = content[0].get("text", "") or ""
            except Exception:
                response_text = raw_response

        text_sections = extract_response_sections(response_text)
        html_content = ""

        assets: Dict[str, Optional[Path]] = {}
        errors: List[str] = []

        generated_frame_path = pick_existing(
            frames_dir / f"{image_name}_generated_frame_01.png",
            frames_dir / f"{image_name}_frame_01.png",
            frames_dir / f"{image_name}_frame.png",
        )
        assets["generated_frame"] = generated_frame_path if generated_frame_path and generated_frame_path.exists() else None
        generated_frame_variant = find_variant_path(generated_frame_path)
        if generated_frame_variant and generated_frame_variant.exists():
            assets["generated_frame_variant"] = generated_frame_variant

        if dataset_frames_dir:
            assets["dataset_first_frame"] = pick_existing(
                dataset_frames_dir / f"{image_name}_frame_01.png",
                dataset_frames_dir / f"{image_name}_first_frame.png",
            )
            assets["dataset_tenth_frame"] = pick_existing(
                dataset_frames_dir / f"{image_name}_frame_10.png",
                dataset_frames_dir / f"{image_name}_tenth_frame.png",
            )

        generated_original_video = videos_dir / f"{image_name}_original_2d.mp4"
        assets["generated_original_video"] = generated_original_video if generated_original_video.exists() else None

        assets["render_log"] = pick_existing(videos_dir / f"{image_name}_original_2d.log")

        html_candidates = sorted(logs_dir.glob(f"{image_name}_*.html"))
        if not html_candidates:
            html_candidates = sorted(videos_dir.glob(f"{image_name}_render_attempt_*.html"))
        assets["html_path"] = html_candidates[-1] if html_candidates else None

        if assets.get("html_path") and assets["html_path"].exists():
            try:
                html_content = assets["html_path"].read_text(encoding="utf-8")
            except Exception as exc:
                _append_error(errors, image_name, f"读取生成HTML失败: {exc}")
                html_content = ""

        if not html_content:
            html_content = extract_html_block(response_text)

        dataset_video_path = None
        if dataset_dir_resolved:
            candidate_video = dataset_dir_resolved / f"{image_name}.mp4"
            if candidate_video.exists():
                dataset_video_path = candidate_video
            else:
                _append_error(errors, image_name, f"未找到原始视频: {candidate_video}")

        assets["dataset_video"] = dataset_video_path

        gt_analysis: Optional[str] = None
        gt_analysis_path: Optional[Path] = None
        try:
            first_frame_for_gt = assets.get("dataset_first_frame") or assets.get("generated_frame")
            tenth_frame_for_gt = assets.get("dataset_tenth_frame") or assets.get("generated_frame_variant")
            if gt_config.enabled:
                gt_analysis, gt_analysis_path = ensure_gt_analysis(
                    image_name,
                    first_frame_for_gt,
                    tenth_frame_for_gt,
                    gt_dir,
                    gt_config,
                )
        except Exception as exc:
            _append_error(errors, image_name, f"GT 生成失败: {exc}")

        text_metrics: Dict[str, float] = {}
        text_metrics_path: Optional[Path] = None
        selection = select_model_analysis_text(text_sections)
        model_analysis_text = text_sections.get("analysis")
        primary_key: Optional[str] = None
        primary_model_text: Optional[str] = None
        if selection:
            primary_key, primary_model_text = selection
        if not model_analysis_text and primary_model_text:
            model_analysis_text = primary_model_text

        model_analysis_path: Optional[Path] = None
        primary_model_text_en: Optional[str] = None

        cached_analysis_path = model_analysis_dir / f"{image_name}_analysis.txt"
        if cached_analysis_path.exists():
            try:
                cached_text = cached_analysis_path.read_text(encoding="utf-8").strip()
                if cached_text and not text_needs_translation(cached_text):
                    primary_model_text_en = cached_text
                    model_analysis_path = cached_analysis_path
            except Exception:
                pass

        if primary_model_text and primary_model_text_en is None:
            try:
                primary_model_text_en = ensure_english_text(primary_model_text)
            except Exception as exc:
                _append_error(errors, image_name, f"模型分析翻译失败: {exc}")
                primary_model_text_en = primary_model_text

        if primary_model_text_en:
            model_analysis_text = primary_model_text_en
            if primary_key:
                text_sections[primary_key] = primary_model_text_en
            text_sections["analysis"] = primary_model_text_en
            try:
                cached_analysis_path.write_text(primary_model_text_en, encoding="utf-8")
                model_analysis_path = cached_analysis_path
            except Exception:
                model_analysis_path = None
        else:
            if primary_model_text:
                model_analysis_text = primary_model_text

        if gt_analysis and model_analysis_text:
            try:
                gt_source = "gpt5.1" if gt_analysis_path else "manual"
                text_metrics, text_metrics_path = compute_and_store_text_metrics(
                    image_name,
                    metrics_dir,
                    gt_analysis,
                    model_analysis_text,
                    use_bertscore,
                    gt_source=gt_source,
                )
            except Exception as exc:
                _append_error(errors, image_name, f"文本指标计算失败: {exc}")

        def _first_non_empty(*values: Optional[str]) -> Optional[str]:
            for value in values:
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        caption_for_metrics = _first_non_empty(
            gt_analysis,
            model_analysis_text,
            text_sections.get("analysis"),
            text_sections.get("prediction"),
            text_sections.get("final"),
        )

        video_metrics: Dict[str, Dict[str, Any]] = {}
        frame_count = 0
        generated_video_path = assets.get("generated_original_video")

        raft_diag_dir: Optional[Path] = None
        raft_diag_dir_str: Optional[str] = None
        if enable_raft_metrics:
            raft_diag_dir = videos_dir / "raft_visuals" / image_name
            raft_diag_dir.mkdir(parents=True, exist_ok=True)
            raft_diag_dir_str = str(raft_diag_dir)

        cached_video_metrics = load_cached_video_metrics(
            metrics_dir,
            image_name,
            sample_every,
            max_frames,
            raft_sample_indices_list,
        )
        metrics_cached = bool(cached_video_metrics)
        if cached_video_metrics:
            video_metrics = cached_video_metrics.get("metrics", {}) if cached_video_metrics else {}
            frame_count = int(cached_video_metrics.get("frame_count") or 0) if cached_video_metrics else 0

        success_flag = bool(generated_video_path and generated_video_path.exists())

        if not metrics_cached and dataset_video_path and generated_video_path and generated_video_path.exists():
            frame_count = 0
            try:
                metrics_original = evaluate_video_pair(
                    str(dataset_video_path),
                    str(generated_video_path),
                    sample_every=sample_every,
                    max_frames=max_frames,
                    caption_text=caption_for_metrics,
                    use_clip=False,
                    use_lpips=False,
                    use_fsim=False,
                    use_vsi=False,
                    use_dists=False,
                    use_dino=False,
                    use_raft=enable_raft_metrics,
                    diagnostics_dir=raft_diag_dir_str,
                    video_name=image_name,
                    variant_label="report",
                    rel_root=str(result_dir),
                    auto_align=auto_align,
                    alignment_max_offset=alignment_max_offset,
                    alignment_window=alignment_window,
                    alignment_offset_penalty=alignment_offset_penalty,
                    raft_sample_indices=raft_sample_indices_zero,
                )
                if metrics_original:
                    metrics_original.pop("lpips_mean", None)
                    metrics_original.pop("lpips_std", None)
                    metrics_original.pop("fsim_mean", None)
                    metrics_original.pop("fsim_std", None)
                    metrics_original.pop("vsi_mean", None)
                    metrics_original.pop("vsi_std", None)
                    metrics_original.pop("dists_mean", None)
                    metrics_original.pop("dists_std", None)
                    per_frame_stats = metrics_original.get("per_frame")
                    if isinstance(per_frame_stats, dict):
                        for key in ("lpips", "fsim", "vsi", "dists"):
                            per_frame_stats.pop(key, None)
                    video_metrics["original_vs_generated"] = metrics_original
                    frame_count = int(metrics_original.get("num_frames", 0))
                    store_video_metrics(
                        metrics_dir,
                        image_name,
                        sample_every,
                        video_metrics,
                        frame_count,
                        max_frames,
                        raft_sample_indices_list,
                    )
            except RecursionError as exc:
                _append_error(errors, image_name, f"视频指标计算失败: {exc}")
                try:
                    fallback_metrics = evaluate_video_pair(
                        str(dataset_video_path),
                        str(generated_video_path),
                        sample_every=sample_every,
                        max_frames=max_frames,
                        caption_text=caption_for_metrics,
                    use_clip=False,
                    use_lpips=False,
                    use_fsim=False,
                    use_vsi=False,
                    use_dists=False,
                    use_dino=False,
                        use_raft=enable_raft_metrics,
                        diagnostics_dir=raft_diag_dir_str,
                        video_name=image_name,
                        variant_label="report",
                        rel_root=str(result_dir),
                        auto_align=auto_align,
                        alignment_max_offset=alignment_max_offset,
                        alignment_window=alignment_window,
                        alignment_offset_penalty=alignment_offset_penalty,
                        raft_sample_indices=raft_sample_indices_zero,
                    )
                    if fallback_metrics:
                        video_metrics["original_vs_generated_fallback"] = fallback_metrics
                        frame_count = int(fallback_metrics.get("num_frames", 0))
                        store_video_metrics(
                            metrics_dir,
                            image_name,
                            sample_every,
                            video_metrics,
                            frame_count,
                            max_frames,
                            raft_sample_indices_list,
                        )
                except Exception as inner_exc:
                    _append_error(errors, image_name, f"基础指标回退失败: {inner_exc}")
                except Exception as exc:
                    _append_error(errors, image_name, f"视频指标计算失败: {exc}")

            metrics_queue: List[str] = []
            if heavy_metrics:
                metrics_queue.extend(["clip", "lpips", "dino", "fsim", "vsi", "dists"])
                if enable_llm_evaluation:
                    metrics_queue.append("llm")
            if enable_raft_metrics:
                metrics_queue.append("raft")
            metrics_queue = [m for m in metrics_queue if m in HEAVY_METRIC_MODULES]

            if metrics_queue and video_metrics.get("original_vs_generated"):
                base_metrics = video_metrics["original_vs_generated"]
                for metric_name in metrics_queue:
                    if metric_name == "raft" and "raft_epe_mean" in base_metrics:
                        continue
                    try:
                        extra_cli_args: List[str] = []
                        if metric_name == "raft" and raft_sample_indices_list:
                            extra_cli_args.extend(["--raft-sample-indices"] + [str(idx) for idx in raft_sample_indices_list])
                        if metric_name == "llm":
                            extra_cli_args.append("--enable-llm")
                            extra_cli_args.extend(["--llm-cache-dir", str(llm_cache_root)])
                        if auto_align:
                            extra_cli_args.extend(
                                [
                                    "--alignment-max-offset",
                                    str(alignment_max_offset),
                                    "--alignment-window",
                                    str(alignment_window),
                                    "--alignment-offset-penalty",
                                    str(alignment_offset_penalty),
                                ]
                            )
                        else:
                            extra_cli_args.append("--no-auto-align")
                        extra = run_heavy_metric(
                            metric_name,
                            dataset_video_path,
                            generated_video_path,
                            sample_every=sample_every,
                            max_frames=max_frames,
                            caption=caption_for_metrics,
                            extra_cli_args=extra_cli_args if extra_cli_args else None,
                        )
                        merge_metrics(base_metrics, extra)
                        store_video_metrics(
                            metrics_dir,
                            image_name,
                            sample_every,
                            video_metrics,
                            frame_count,
                            max_frames,
                            raft_sample_indices_list,
                        )
                    except Exception as exc:
                        label = metric_name.upper()
                        _append_error(errors, image_name, f"{label} 指标计算失败: {exc}")

            llm_result: Optional[Dict[str, Any]] = None
            if enable_llm_evaluation and dataset_video_path and generated_video_path and generated_video_path.exists():
                existing_llm = video_metrics.get("llm_similarity")
                if isinstance(existing_llm, dict) and existing_llm.get("score") is not None:
                    llm_result = existing_llm
                else:
                    try:
                        llm_result = llm_evaluate_video_pair(
                            dataset_video_path,
                            generated_video_path,
                            cache_dir=llm_cache_root,
                        )
                    except (LLMVideoEvaluationError, FileNotFoundError) as exc:
                        _append_error(errors, image_name, f"LLM 视频评估失败: {exc}")
                        llm_result = {"error": str(exc)}
                    except Exception as exc:
                        _append_error(errors, image_name, f"LLM 视频评估失败: {exc}")
                        llm_result = {"error": str(exc)}
                    if llm_result and llm_result.get("score") is None and llm_result.get("error"):
                        _append_error(errors, image_name, f"LLM 视频评估失败: {llm_result.get('error')}")
                    if llm_result:
                        artifact_path = metrics_dir / f"{image_name}_gemini_eval.json"
                        try:
                            artifact_path.write_text(json.dumps(llm_result, ensure_ascii=False, indent=2), encoding="utf-8")
                            try:
                                llm_result["artifact_path"] = str(artifact_path.relative_to(root))
                            except Exception:
                                llm_result["artifact_path"] = str(artifact_path)
                        except Exception:
                            pass
                        video_metrics["llm_similarity"] = llm_result
                        store_video_metrics(
                            metrics_dir,
                            image_name,
                            sample_every,
                            video_metrics,
                            frame_count,
                            max_frames,
                            raft_sample_indices_list,
                        )
            elif enable_llm_evaluation and (not dataset_video_path or not generated_video_path or not generated_video_path.exists()):
                print(f"[INFO] {image_name}: skipped Gemini evaluation (missing reference or generated video)", flush=True)

            if llm_result and isinstance(llm_result, dict):
                score_val = llm_result.get("score")
                if isinstance(score_val, (int, float)) and not math.isnan(float(score_val)):
                    aggregator_llm = aggregate_metrics.setdefault("Gemini similarity", {})
                    aggregator_llm.setdefault("score", []).append(float(score_val))

            if video_metrics.get("original_vs_generated"):
                final_metrics = video_metrics["original_vs_generated"]
                final_metrics["video_success"] = 1.0 if success_flag else 0.0
                aggregator = aggregate_metrics.setdefault("original_vs_generated", {})
                for key, value in final_metrics.items():
                    if key in {"per_frame", "num_frames"}:
                        continue
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        aggregator.setdefault(key, []).append(float(value))
        elif not metrics_cached:
            if not dataset_video_path:
                _append_error(errors, image_name, "Missing reference video for comparison")
            if not generated_video_path or not generated_video_path.exists():
                _append_error(errors, image_name, "Missing generated video output")

        success = bool(generated_video_path and generated_video_path.exists() and html_content)
        if not html_content:
            _append_error(errors, image_name, "Missing generated HTML segment")

        model_name_inferred = _infer_model_name(calls)

        entry = LLMSummary(
            image_name=image_name,
            llm_calls_path=log_file,
            llm_calls=calls,
            text_sections=text_sections,
            html_content=html_content,
            assets=assets,
            video_metrics=video_metrics,
            text_metrics=text_metrics,
            frame_count=frame_count,
            success=success,
            errors=errors,
            gt_analysis=gt_analysis,
            gt_analysis_path=gt_analysis_path,
            model_analysis=model_analysis_text,
            model_analysis_path=model_analysis_path,
            text_metrics_path=text_metrics_path,
            model_name=model_name_inferred,
        )
        entries.append(entry)
        _update_aggregate_metrics_for_entry(entry)

        status = "✅" if success else "⚠️"
        print(f"[{index}/{total_logs}] {status} {image_name}", flush=True)

        if strict and errors:
            critical_errors = _critical_errors(errors)
            if critical_errors:
                raise RuntimeError(f"{image_name} has an error: {critical_errors[0]}")

        """

    baseline_summaries: List[Dict[str, Any]] = []
    if baseline_root_resolved:
        baseline_summaries = _collect_baseline_entries(
            baseline_root_resolved,
            dataset_dir=dataset_dir_resolved,
            dataset_frames_dir=dataset_frames_dir,
            sample_every=sample_every,
            max_frames=max_frames,
            auto_align=auto_align,
            alignment_max_offset=alignment_max_offset,
            alignment_window=alignment_window,
            alignment_offset_penalty=alignment_offset_penalty,
        )

    metrics_overview: List[Dict[str, float]] = []
    for metric_name, values in aggregate_metrics.items():
        overview_entry: Dict[str, float] = {"label": metric_name}
        for key, numbers in values.items():
            if not numbers or key in {"num_frames", "per_frame"}:
                continue
            valid = [x for x in numbers if not math.isnan(x)]
            if not valid:
                continue
            overview_entry[key] = float(statistics.fmean(valid))
        metrics_overview.append(overview_entry)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "result_dir": root,
        "dataset_dir": dataset_dir_resolved,
        "total": len(entries),
        "successful": sum(1 for e in entries if e.success),
        "failed": sum(1 for e in entries if not e.success),
        "metrics_overview": metrics_overview,
        "entries": entries,
        "baseline_summaries": baseline_summaries,
    }
    if baseline_root_resolved:
        summary["baseline_root"] = baseline_root_resolved
    return summary
