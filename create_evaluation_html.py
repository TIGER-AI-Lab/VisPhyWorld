#!/usr/bin/env python3
"""Simple evaluation report generator for the physics prediction pipeline."""

from __future__ import annotations
import argparse
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Sequence

sys.setrecursionlimit(10000)

from dotenv import load_dotenv

from utils.report.html_renderer import generate_html
from utils.report.models import GTGenerationConfig
from utils.report.scanner import scan_output_directory
from visphybench_layout import resolve_visphybench_paths

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


# Load environment variables from .env if present.
# - First load project-local VisExpert/.env (works regardless of current working directory)
# - Then load .env from current working directory (if any)
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
load_dotenv(override=False)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cleanup_and_organize(result_dir: Path) -> None:
    """Remove legacy artifacts and group common outputs to keep the directory tidy."""

    removable = [
        "metrics_backup",
        "metrics_failed",
        "vbench_eval",
        "vbench_eval_debug",
        "vbench_eval_test",
        "segmentations",
    ]
    for name in removable:
        target = result_dir / name
        if target.is_symlink() or target.exists():
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass

    if not hasattr(os, "symlink"):
        return

    artifacts_dir = result_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    keep_dirs = [
        "frames",
        "videos",
        "metrics",
        "logs",
        "gt_analysis",
        "model_analysis",
    ]

    for name in keep_dirs:
        original = result_dir / name
        if original.is_symlink():
            continue
        destination = artifacts_dir / name
        if not original.exists():
            continue
        if destination.exists():
            continue
        try:
            shutil.move(str(original), destination)
        except Exception as exc:
            print(f"⚠️ Failed to move {name}: {exc}", flush=True)
            continue

    # Keep backward compatibility: create symlinks at the result root pointing
    # to artifacts/<name>, so that evaluation.html can still use the old
    # relative paths frames/, videos/, metrics/, etc.
    for name in keep_dirs:
        link_path = result_dir / name
        target = artifacts_dir / name
        if not target.exists():
            continue
        if link_path.exists():
            if link_path.is_symlink():
                continue
            # Avoid overwriting existing non-symlink directories/files.
            continue
        try:
            rel_target = os.path.relpath(target, result_dir)
            os.symlink(rel_target, link_path)
        except Exception as exc:
            print(f"⚠️ Failed to create symlink {link_path} -> {target}: {exc}", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation HTML for physics prediction outputs")
    parser.add_argument('result_dir', help='Physics prediction output directory, e.g., output/physics_prediction_YYYYMMDD_HHMMSS')
    parser.add_argument('--output', '-o', help='Output HTML path', default=None)
    parser.add_argument(
        '--dataset-dir',
        help=(
            "Dataset path. Can point to:\n"
            "- VisPhyBench root (contains sub/ and test/), e.g. data/\n"
            "- split root (contains videos/), e.g. data/sub/\n"
            "- videos directory containing *.mp4, e.g. data/sub/videos/"
        ),
        default=None,
    )
    parser.add_argument(
        '--split',
        choices=['sub', 'test'],
        default='sub',
        help="Split name when --dataset-dir points to the dataset root (default: sub).",
    )
    parser.add_argument('--baseline-root', help='Root directory that contains baseline runs (logs/ will be scanned)', default=None)
    parser.add_argument('--jobs', type=int, default=1, help='并行 worker 数（默认 1）')
    parser.add_argument('--sample-every', type=int, default=3, help='Frame sampling interval when computing video metrics (default 3)')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames compared per video')
    parser.add_argument('--skip-gt', action='store_true', help='Skip GPT-5.1-based GT caption generation')
    parser.add_argument('--openai-model', default='gpt-5.1', help='OpenAI model used for GT caption generation')
    parser.add_argument('--openai-api-base', help='Custom OpenAI API base URL', default=None)
    parser.add_argument('--openai-api-key', help='Explicit OpenAI API key (defaults to environment variable)', default=None)
    parser.add_argument('--openai-timeout', type=float, help='OpenAI request timeout in seconds', default=None)
    parser.add_argument('--no-bertscore', action='store_true', help='Disable BERTScore (enabled by default)')
    parser.add_argument('--summary-only', action='store_true', help='Only output metric CSVs, do not generate an HTML page')
    parser.add_argument('--enable-heavy-metrics', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--disable-heavy-metrics', action='store_true', help='Disable heavy metrics such as CLIP/LPIPS/DINO')
    parser.add_argument('--enable-raft-metrics', action='store_true', help='Enable RAFT optical-flow metrics (default enabled)')
    parser.add_argument('--disable-raft-metrics', action='store_true', help='Disable RAFT optical-flow metrics')
    parser.add_argument('--raft-sample-indices', nargs='+', type=int, help='Frame indices (1-based) used for RAFT evaluation')
    parser.add_argument('--enable-llm-eval', action='store_true', help='Enable Gemini-based LLM similarity evaluation (default enabled if GEMINI_API_KEY is set)')
    parser.add_argument('--disable-llm-eval', action='store_true', help='Disable Gemini-based LLM similarity evaluation')
    parser.add_argument('--disable-auto-align', action='store_true', help='Disable automatic temporal alignment during evaluation')
    parser.add_argument('--alignment-max-offset', type=int, default=30, help='Maximum offset (in sampled frames) for alignment search')
    parser.add_argument('--alignment-window', type=int, default=3, help='Window size (in sampled frames) for alignment scoring')
    parser.add_argument('--alignment-offset-penalty', type=float, default=0.05, help='Offset penalty coefficient (larger prefers smaller offsets)')
    parser.add_argument('--resume-from', help='Resume from a specific sample name; only process this and following samples', default=None)
    parser.add_argument('--strict', action='store_true', help='Strict mode: abort on the first critical error')
    parser.add_argument('--max-samples', type=int, default=None, help='最多处理的样本数量，用于快速生成小批量报告')
    parser.add_argument('--include-samples', nargs='+', help='仅处理指定的样本名列表')
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {result_dir}")

    if not args.skip_gt:
        normalized_model = (args.openai_model or "").strip().lower()
        if normalized_model != "gpt-5.1":
            raise ValueError(
                f"GT generation requires gpt-5.1, but '{args.openai_model}' was specified. "
                "Use --openai-model gpt-5.1 or add --skip-gt."
            )

    dataset_dir: Optional[Path]
    if args.dataset_dir:
        resolved = resolve_visphybench_paths(Path(args.dataset_dir), split=args.split)
        dataset_dir = resolved.videos_dir.resolve()
        if not dataset_dir.exists():
            print(f"⚠️ Specified dataset directory does not exist: {dataset_dir}")
            dataset_dir = None
    else:
        default_candidate = None
        parents = list(result_dir.parents)
        for candidate_root in (parents[1:] if len(parents) >= 2 else parents):
            visphy_videos = candidate_root / "data" / args.split / "videos"
            if visphy_videos.exists():
                default_candidate = visphy_videos
                break
            legacy = candidate_root / "data" / "phyworld_videos"
            if legacy.exists():
                default_candidate = legacy
                break
        dataset_dir = default_candidate if default_candidate and default_candidate.exists() else None

    baseline_root_path: Optional[Path] = None
    if args.baseline_root:
        candidate = Path(args.baseline_root).resolve()
        if candidate.exists():
            baseline_root_path = candidate
        else:
            print(f'⚠️ Specified baseline directory does not exist: {candidate}')
    else:
        baseline_root_path = None

    heavy_metrics_enabled = True
    if getattr(args, 'disable_heavy_metrics', False):
        heavy_metrics_enabled = False
    if getattr(args, 'enable_heavy_metrics', False):
        heavy_metrics_enabled = True

    # Defaults:
    # - RAFT metrics: enabled by default (can be disabled with --disable-raft-metrics).
    # - Gemini LLM eval: enabled by default (will be auto-disabled if GEMINI_API_KEY is missing).
    raft_metrics_enabled = True if not getattr(args, 'disable_raft_metrics', False) else False
    if getattr(args, 'enable_raft_metrics', False):
        raft_metrics_enabled = True

    llm_eval_enabled = True
    if getattr(args, 'disable_llm_eval', False):
        llm_eval_enabled = False
    if getattr(args, 'enable_llm_eval', False):
        llm_eval_enabled = True

    raft_sample_indices_input: Optional[List[int]] = None
    if args.raft_sample_indices:
        raft_sample_indices_input = [int(idx) for idx in args.raft_sample_indices if idx is not None and int(idx) > 0]
    elif raft_metrics_enabled:
        raft_sample_indices_input = [1, 10, 20, 30, 40, 50]

    auto_align_enabled = not getattr(args, 'disable_auto_align', False)
    alignment_max_offset = max(0, getattr(args, 'alignment_max_offset', 30))
    alignment_window = max(1, getattr(args, 'alignment_window', 3))
    alignment_offset_penalty = max(0.0, getattr(args, 'alignment_offset_penalty', 0.05))

    gt_config = GTGenerationConfig(
        enabled=not args.skip_gt,
        model=args.openai_model,
        api_base=args.openai_api_base,
        api_key=args.openai_api_key,
        timeout=args.openai_timeout,
    )

    if gt_config.enabled:
        api_key = gt_config.api_key or os.getenv('OPENAI_API_KEY')
        if OpenAI is None:
            print("⚠️ openai 库缺失，跳过 GT 文本生成", flush=True)
            gt_config.enabled = False
        elif not api_key:
            print("⚠️ 未找到 OPENAI_API_KEY，跳过 GT 文本生成", flush=True)
            gt_config.enabled = False
        else:
            gt_config.api_key = api_key

    use_bertscore = not args.no_bertscore

    summary = scan_output_directory(
        result_dir,
        dataset_dir=dataset_dir,
        jobs=max(1, int(getattr(args, "jobs", 1) or 1)),
        sample_every=max(1, args.sample_every),
        gt_config=gt_config,
        use_bertscore=use_bertscore,
        heavy_metrics=heavy_metrics_enabled,
        enable_raft_metrics=raft_metrics_enabled,
        enable_llm_evaluation=llm_eval_enabled,
        auto_align=auto_align_enabled,
        alignment_max_offset=alignment_max_offset,
        alignment_window=alignment_window,
        alignment_offset_penalty=alignment_offset_penalty,
        resume_from=args.resume_from,
        strict=args.strict,
        include_samples=args.include_samples,
        max_frames=args.max_frames,
        max_samples=args.max_samples,
        raft_sample_indices=raft_sample_indices_input,
        baseline_root=baseline_root_path,
    )

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = result_dir / "evaluation_full.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_html(summary, result_dir, output_path)

    baseline_summaries = summary.get("baseline_summaries") or []
    baseline_root_for_reports = summary.get("baseline_root") or baseline_root_path
    if baseline_summaries and baseline_root_for_reports:
        combined_entries: List[LLMSummary] = []
        combined_metrics: List[Dict[str, float]] = []
        for baseline_summary in baseline_summaries:
            entries = baseline_summary.get("entries") or []
            if not entries:
                continue
            combined_entries.extend(entries)
            baseline_name = baseline_summary.get("baseline_name", "baseline")
            for metric_entry in baseline_summary.get("metrics_overview", []):
                label = metric_entry.get("label", "metrics")
                merged = {k: v for k, v in metric_entry.items() if k != "label"}
                merged["label"] = f"{baseline_name}:{label}"
                combined_metrics.append(merged)
        if combined_entries:
            baseline_root_path_resolved = Path(baseline_root_for_reports).resolve()
            combined_payload = {
                "timestamp": datetime.now().isoformat(),
                "result_dir": baseline_root_path_resolved,
                "dataset_dir": dataset_dir,
                "total": len(combined_entries),
                "successful": sum(1 for e in combined_entries if e.success),
                "failed": sum(1 for e in combined_entries if not e.success),
                "metrics_overview": combined_metrics,
                "entries": combined_entries,
            }
            combined_output_path = baseline_root_path_resolved / "evaluation_full.html"
            generate_html(combined_payload, baseline_root_path_resolved, combined_output_path)
            print(f"✅ 基线评估已生成: {combined_output_path}", flush=True)
    elif baseline_summaries:
        print("⚠️ 未生成基线汇总报告：缺少有效的 baseline-root", flush=True)
    cleanup_and_organize(result_dir)


if __name__ == '__main__':
    main()
