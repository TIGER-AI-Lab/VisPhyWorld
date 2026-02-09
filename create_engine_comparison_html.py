#!/usr/bin/env python3
"""
Create a cross-engine comparison HTML report from multiple evaluation_full.csv files.

Typical usage (dataset_test 4 engines):
  python /home/jiarong/VisExpert/create_engine_comparison_html.py \
    --run svg=/home/jiarong/VisExpert/output/physics_prediction_20251229_142409_gpt-5_svg_dataset_test \
    --run manim=/home/jiarong/VisExpert/output/physics_prediction_20251229_142435_gpt-5_manim_dataset_test \
    --run threejs=/home/jiarong/VisExpert/output/physics_prediction_20251229_143446_gpt-5_threejs_dataset_test \
    --run p5js=/home/jiarong/VisExpert/output/physics_prediction_20251229_143503_gpt-5_p5js_dataset_test \
    -o /home/jiarong/VisExpert/output/compare_engines_dataset_test.html
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return None if math.isnan(v) else v
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA"}:
        return None
    try:
        v = float(text)
        return None if math.isnan(v) else v
    except Exception:
        return None


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    try:
        return f"{value:.{digits}f}"
    except Exception:
        return str(value)


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.fmean(values))


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool
    digits: int = 4


def _dedupe_metric_specs(specs: Sequence[MetricSpec]) -> List[MetricSpec]:
    seen: set[str] = set()
    out: List[MetricSpec] = []
    for spec in specs:
        if spec.key in seen:
            continue
        seen.add(spec.key)
        out.append(spec)
    return out


KEY_METRICS: List[MetricSpec] = [
    MetricSpec("video_success", "Video exists", True, 0),
    MetricSpec("psnr", "PSNR", True, 2),
    MetricSpec("ssim", "SSIM", True, 4),
    MetricSpec("clip_image", "CLIP(image)", True, 4),
    MetricSpec("dino", "DINO", True, 4),
    MetricSpec("lpips", "LPIPS", False, 4),
    MetricSpec("alignment_error", "Align error", False, 4),
    MetricSpec("llm_gemini_score", "Gemini", True, 2),
]

LEADERBOARD_METRICS: List[MetricSpec] = [
    MetricSpec("lpips", "LPIPS ↓", False, 4),
    MetricSpec("clip_image", "CLIP-Img ↑", True, 4),
    MetricSpec("dino", "DINO ↑", True, 4),
    MetricSpec("clip_caption", "CLIP-Cap ↑", True, 4),
    MetricSpec("bertscore_f1", "BERTScore-F1 ↑", True, 4),
    MetricSpec("raft_epe", "RAFT-EPE ↓", False, 4),
    MetricSpec("llm_gemini_score", "Gemini ↑", True, 2),
]


PER_SAMPLE_METRICS: List[MetricSpec] = _dedupe_metric_specs(
    [
        # Per-sample table keeps the leaderboard metrics first, then appends other
        # commonly-inspected metrics (e.g., PSNR/SSIM).
        *LEADERBOARD_METRICS,
        *KEY_METRICS,
    ]
)


@dataclass
class RunData:
    label: str
    run_dir: Path
    csv_path: Path
    eval_html_path: Optional[Path]
    rows_by_video: Dict[str, Dict[str, str]]
    video_dir: Optional[Path]
    fieldnames: Sequence[str]

    def video_path_for(self, video_name: str) -> Optional[Path]:
        if not self.video_dir:
            return None
        candidate = self.video_dir / f"{video_name}_original_2d.mp4"
        if candidate.exists():
            return candidate
        return None

    def video_exists(self, video_name: str) -> bool:
        return bool(self.video_path_for(video_name))


def _resolve_video_dir(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "videos"
    if direct.exists():
        return direct
    artifacts = run_dir / "artifacts" / "videos"
    if artifacts.exists():
        return artifacts
    return None


def _load_run(label: str, run_dir: Path) -> RunData:
    run_dir = run_dir.resolve()
    csv_path = run_dir / "evaluation_full.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing evaluation CSV for {label}: {csv_path}")

    eval_html = run_dir / "evaluation_full.html"
    eval_html_path = eval_html if eval_html.exists() else None

    rows_by_video: Dict[str, Dict[str, str]] = {}
    fieldnames: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            name = (row.get("video_name") or "").strip()
            if not name:
                continue
            rows_by_video[name] = {k: (v or "") for k, v in row.items() if k}

    return RunData(
        label=label,
        run_dir=run_dir,
        csv_path=csv_path,
        eval_html_path=eval_html_path,
        rows_by_video=rows_by_video,
        video_dir=_resolve_video_dir(run_dir),
        fieldnames=fieldnames,
    )


def _load_run_allow_missing(label: str, run_dir: Path) -> RunData:
    """
    Like `_load_run`, but does not fail when `evaluation_full.csv` is missing.

    This is useful when a run finished rendering videos but the evaluation step
    hasn't been executed yet.
    """
    run_dir = run_dir.resolve()
    csv_path = run_dir / "evaluation_full.csv"
    if not csv_path.exists():
        eval_html = run_dir / "evaluation_full.html"
        return RunData(
            label=label,
            run_dir=run_dir,
            csv_path=csv_path,
            eval_html_path=eval_html if eval_html.exists() else None,
            rows_by_video={},
            video_dir=_resolve_video_dir(run_dir),
            fieldnames=[],
        )
    return _load_run(label, run_dir)


def _best_engine(
    values_by_engine: Dict[str, Optional[float]],
    higher_is_better: bool,
) -> Optional[str]:
    present = {k: v for k, v in values_by_engine.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}
    if not present:
        return None
    items = list(present.items())
    if higher_is_better:
        return max(items, key=lambda kv: kv[1])[0]
    return min(items, key=lambda kv: kv[1])[0]


def _rel(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(str(to_path), str(from_path.parent))


def _render_html(
    runs: List[RunData],
    output_path: Path,
    title: str,
    include_videos: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    engine_labels = [r.label for r in runs]

    video_names: List[str] = sorted({name for r in runs for name in r.rows_by_video.keys()})

    def _metric_digits(key: str) -> int:
        k = key.lower()
        if k == "video_success":
            return 0
        if k.startswith("psnr"):
            return 2
        if k.startswith("ssim"):
            return 4
        if k.startswith("clip"):
            return 4
        if k.startswith("dino"):
            return 4
        if "bertscore" in k:
            return 4
        if "rouge" in k:
            return 4
        if "lpips" in k:
            return 4
        if k.startswith("raft_epe"):
            return 4
        if k.startswith("raft_"):
            return 4
        if k.startswith("alignment_"):
            return 4
        return 4

    def _metric_higher_is_better(key: str) -> bool:
        k = key.lower()
        if k in {"lpips", "alignment_error"}:
            return False
        if k.startswith("raft_") or k.startswith("alignment_"):
            # RAFT errors and alignment errors are generally lower-is-better.
            return False
        return True

    # Build a comprehensive metric set from CSV headers (plus video_success).
    ignore = {
        "",
        "video_name",
        "model_name",
        "timestamp",
        "llm_gemini_model",
        "llm_gemini_justification",
    }
    all_keys: List[str] = ["video_success"]
    seen = {"video_success"}
    for run in runs:
        for key in run.fieldnames:
            if not key or key in ignore:
                continue
            if key.startswith("quality_"):
                continue
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    all_metric_specs: List[MetricSpec] = []
    for key in all_keys:
        if key == "video_success":
            all_metric_specs.append(MetricSpec(key, "Video exists", True, 0))
        else:
            all_metric_specs.append(
                MetricSpec(
                    key=key,
                    label=key,
                    higher_is_better=_metric_higher_is_better(key),
                    digits=_metric_digits(key),
                )
            )

    # Aggregate stats per engine/metric.
    aggregate: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in all_metric_specs:
        per_engine: Dict[str, Dict[str, Any]] = {}
        for run in runs:
            vals: List[float] = []
            for name in video_names:
                if spec.key == "video_success":
                    v = 1.0 if run.video_exists(name) else 0.0
                else:
                    row = run.rows_by_video.get(name) or {}
                    v = _parse_float(row.get(spec.key))
                if v is None:
                    continue
                vals.append(float(v))
            per_engine[run.label] = {
                "n": len(vals),
                "mean": _mean(vals),
                "median": _median(vals),
            }
        aggregate[spec.key] = per_engine

    # Winner counts (by sample) for a small set of metrics.
    winner_counts: Dict[str, Dict[str, int]] = {spec.key: {e: 0 for e in engine_labels} for spec in all_metric_specs}
    for name in video_names:
        for spec in all_metric_specs:
            values: Dict[str, Optional[float]] = {}
            for run in runs:
                if spec.key == "video_success":
                    values[run.label] = 1.0 if run.video_exists(name) else 0.0
                else:
                    row = run.rows_by_video.get(name) or {}
                    values[run.label] = _parse_float(row.get(spec.key))
            best = _best_engine(values, spec.higher_is_better)
            if best is not None:
                winner_counts[spec.key][best] += 1

    def esc(text: str) -> str:
        return html.escape(text or "")

    # Build HTML.
    css = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 18px; background: #f5f7fb; color: #111827; }
.card { background: white; border-radius: 12px; padding: 16px 18px; margin: 14px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
.title { font-size: 20px; font-weight: 700; margin: 0 0 6px 0; }
.subtitle { color: #4b5563; margin: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.muted { color: #6b7280; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #e5e7eb; padding: 8px 10px; font-size: 13px; text-align: center; }
th { background: #f3f4f6; position: sticky; top: 0; z-index: 1; }
td.left, th.left { text-align: left; }
.scroll { overflow-x: auto; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; }
.win { background: #dcfce7; }
.na { color: #9ca3af; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
details > summary { cursor: pointer; user-select: none; }
"""

    # Runs overview cards.
    run_cards: List[str] = []
    for run in runs:
        links: List[str] = []
        if run.csv_path.exists():
            links.append(f"<a href='{esc(_rel(output_path, run.csv_path))}'>CSV</a>")
        else:
            links.append("<span class='na'>CSV missing</span>")
        if run.eval_html_path:
            links.append(f"<a href='{esc(_rel(output_path, run.eval_html_path))}'>Eval HTML</a>")
        run_cards.append(
            "<div class='card'>"
            f"<div class='title'>{esc(run.label)}</div>"
            f"<div class='subtitle mono'>{esc(str(run.run_dir))}</div>"
            f"<div class='muted' style='margin-top:6px'>Links: {' | '.join(links)}</div>"
            "</div>"
        )

    # Aggregate table.
    agg_header = (
        "<tr><th class='left'>Engine</th>"
        + "".join(
            f"<th>{esc(spec.label)}<div class='muted' style='font-weight:400'>mean / median · wins</div></th>"
            for spec in all_metric_specs
        )
        + "</tr>"
    )
    agg_rows: List[str] = []
    for run in runs:
        cells: List[str] = [f"<td class='left'><span class='tag'>{esc(run.label)}</span></td>"]
        for spec in all_metric_specs:
            stats = aggregate.get(spec.key, {}).get(run.label, {})
            mean_v = stats.get("mean")
            med_v = stats.get("median")
            wins = winner_counts.get(spec.key, {}).get(run.label, 0)
            cell = f"{esc(_fmt(mean_v, spec.digits))} / {esc(_fmt(med_v, spec.digits))}<div class='muted'>wins: {wins}</div>"
            if mean_v is None and med_v is None:
                cells.append("<td class='na'>N/A</td>")
            else:
                cells.append(f"<td>{cell}</td>")
        agg_rows.append("<tr>" + "".join(cells) + "</tr>")

    # Leaderboard table (mean values only).
    leaderboard_header = (
        "<tr><th class='left'>Engine</th>"
        + "".join(f"<th>{esc(spec.label)}<div class='muted' style='font-weight:400'>mean · wins</div></th>" for spec in LEADERBOARD_METRICS)
        + "</tr>"
    )
    leaderboard_rows: List[str] = []
    for run in runs:
        cells = [f"<td class='left'><span class='tag'>{esc(run.label)}</span></td>"]
        for spec in LEADERBOARD_METRICS:
            stats = aggregate.get(spec.key, {}).get(run.label, {})
            mean_v = stats.get("mean")
            wins = winner_counts.get(spec.key, {}).get(run.label, 0)
            # Highlight per-metric best engine based on mean value.
            per_engine_means: Dict[str, Optional[float]] = {}
            for other in runs:
                other_stats = aggregate.get(spec.key, {}).get(other.label, {})
                per_engine_means[other.label] = other_stats.get("mean")
            best = _best_engine(per_engine_means, spec.higher_is_better)
            klass = "win" if best == run.label else ""
            if mean_v is None:
                cells.append("<td class='na'>N/A</td>")
            else:
                cells.append(f"<td class='{klass}'>{esc(_fmt(mean_v, spec.digits))}<div class='muted'>wins: {wins}</div></td>")
        leaderboard_rows.append("<tr>" + "".join(cells) + "</tr>")

    # Per-run detailed tables (one table per engine+LLM run).
    def _metric_value(run: RunData, sample: str, spec: MetricSpec) -> Optional[float]:
        if spec.key == "video_success":
            return 1.0 if run.video_exists(sample) else 0.0
        row = run.rows_by_video.get(sample) or {}
        return _parse_float(row.get(spec.key))

    def _run_metric_present_count(run: RunData, spec: MetricSpec) -> Tuple[int, int]:
        """
        Return (count, total) for a metric availability summary.

        - For `video_success`, count means number of existing videos (value==1).
        - For all other metrics, count means number of non-N/A numeric values.
        """
        total = len(video_names)
        count = 0
        for name in video_names:
            v = _metric_value(run, name, spec)
            if spec.key == "video_success":
                if v == 1.0:
                    count += 1
                continue
            if v is None:
                continue
            count += 1
        return count, total

    per_run_blocks: List[str] = []
    for idx, run in enumerate(runs):
        open_attr = " open" if idx == 0 else ""
        csv_missing = not run.csv_path.exists()
        csv_note = "CSV missing" if csv_missing else f"CSV rows: {len(run.rows_by_video)}"
        video_present = sum(1 for name in video_names if run.video_exists(name))

        # Compact availability summary for key metrics.
        summary_keys = [
            "video_success",
            "psnr",
            "clip_image",
            "lpips",
            "raft_epe",
            "llm_gemini_score",
        ]
        summary_specs = [spec for spec in KEY_METRICS if spec.key in set(summary_keys)]
        summary_chips: List[str] = []
        for spec in summary_specs:
            present, total = _run_metric_present_count(run, spec)
            label = spec.label
            summary_chips.append(
                f"<span class='metric-chip'><strong>{esc(label)}:</strong> {present}/{total}</span>"
            )

        run_header_cells = ["<th class='left'>Sample</th>"]
        for spec in PER_SAMPLE_METRICS:
            run_header_cells.append(f"<th>{esc(spec.label)}</th>")
        run_header_cells.append("<th>Links</th>")
        run_header = "<tr>" + "".join(run_header_cells) + "</tr>"

        run_rows: List[str] = []
        for name in video_names:
            row_cells: List[str] = [f"<td class='left mono'>{esc(name)}</td>"]
            for spec in PER_SAMPLE_METRICS:
                v = _metric_value(run, name, spec)
                if v is None:
                    row_cells.append("<td class='na'>N/A</td>")
                else:
                    row_cells.append(f"<td>{esc(_fmt(v, spec.digits))}</td>")

            link_parts: List[str] = []
            if include_videos:
                video_path = run.video_path_for(name)
                if video_path:
                    link_parts.append(f"<a href='{esc(_rel(output_path, video_path))}'>mp4</a>")

            metrics_dir = run.run_dir / "metrics"
            video_metrics = metrics_dir / f"{name}_video_metrics.json"
            text_metrics = metrics_dir / f"{name}_text_metrics.json"
            gemini_eval = metrics_dir / f"{name}_gemini_eval.json"
            if video_metrics.exists():
                link_parts.append(f"<a href='{esc(_rel(output_path, video_metrics))}'>video.json</a>")
            if text_metrics.exists():
                link_parts.append(f"<a href='{esc(_rel(output_path, text_metrics))}'>text.json</a>")
            if gemini_eval.exists():
                link_parts.append(f"<a href='{esc(_rel(output_path, gemini_eval))}'>gemini.json</a>")

            if run.eval_html_path:
                link_parts.append(f"<a href='{esc(_rel(output_path, run.eval_html_path))}'>eval</a>")
            row_cells.append(f"<td>{' | '.join(link_parts) if link_parts else '<span class=na>N/A</span>'}</td>")

            run_rows.append("<tr>" + "".join(row_cells) + "</tr>")

        per_run_blocks.append(
            "<details" + open_attr + ">"
            f"<summary class='muted'><span class='tag'>{esc(run.label)}</span> "
            f"<span class='mono'>{esc(str(run.run_dir))}</span> · "
            f"<span class='mono'>{esc(csv_note)}</span> · "
            f"<span class='mono'>Videos: {video_present}/{len(video_names)}</span>"
            "</summary>"
            "<div style='margin-top:10px'>"
            f"<div class='metric-chip-group'>{''.join(summary_chips)}</div>"
            "<div class='scroll' style='margin-top:10px'>"
            "<table>"
            f"<thead>{run_header}</thead>"
            f"<tbody>{''.join(run_rows)}</tbody>"
            "</table>"
            "</div>"
            "</div>"
            "</details>"
        )

    payload = f"""<!doctype html>
	<html lang="zh">
	<head>
	  <meta charset="utf-8"/>
	  <meta name="viewport" content="width=device-width, initial-scale=1"/>
	  <title>{esc(title)}</title>
	  <style>{css}</style>
	</head>
	<body>
	  <div class="card">
	    <div class="title">{esc(title)}</div>
	    <p class="subtitle">Generated: <span class="mono">{esc(now)}</span> · Samples: <span class="mono">{len(video_names)}</span></p>
	    <p class="muted">Notes: N/A usually means the metric failed (e.g., OOM) or was not available for that sample.</p>
	  </div>

	  <div class="grid">
	    {''.join(run_cards)}
	  </div>

	  <div class="card">
	    <div class="title">Leaderboard</div>
	    <div class="scroll">
	      <table>
	        <thead>{leaderboard_header}</thead>
	        <tbody>
	          {''.join(leaderboard_rows)}
	        </tbody>
	      </table>
	    </div>
	  </div>

		  <div class="card">
		    <div class="title">Per-run Details</div>
        <p class="muted">Each table corresponds to one run (engine + LLM). Cells show per-sample metrics (N/A means missing or failed).</p>
        {''.join(per_run_blocks)}
		  </div>

	  <div class="card">
	    <div class="title">All Metrics (Aggregate)</div>
	    <details>
	      <summary class="muted">Show/Hide table</summary>
	      <div class="scroll" style="margin-top:10px">
	        <table>
	          <thead>{agg_header}</thead>
	          <tbody>
	            {''.join(agg_rows)}
	          </tbody>
	        </table>
	      </div>
	    </details>
	  </div>
	</body>
	</html>
	"""

    output_path.write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple engine runs and generate a single HTML report.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec as label=/abs/path/to/run_dir (repeatable).",
    )
    parser.add_argument("-o", "--output", required=True, help="Output HTML path.")
    parser.add_argument("--title", default="Engine Comparison Report", help="HTML title.")
    parser.add_argument("--no-video-links", action="store_true", help="Do not link per-sample generated mp4 files.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow runs with missing evaluation_full.csv (metrics will show as N/A).",
    )
    args = parser.parse_args()

    runs: List[RunData] = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Invalid --run value (expected label=/path): {item}")
        label, path = item.split("=", 1)
        label = label.strip()
        run_dir = Path(path.strip())
        if not label or not run_dir:
            raise ValueError(f"Invalid --run value: {item}")
        loader = _load_run_allow_missing if args.allow_missing else _load_run
        runs.append(loader(label, run_dir))

    if len(runs) < 2:
        raise ValueError("Need at least 2 runs to compare. Provide multiple --run entries.")

    output_path = Path(args.output).resolve()
    _render_html(
        runs=runs,
        output_path=output_path,
        title=str(args.title),
        include_videos=not args.no_video_links,
    )
    print(f"✅ Engine comparison HTML generated: {output_path}", flush=True)


if __name__ == "__main__":
    main()
