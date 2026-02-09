
from __future__ import annotations

import csv
import html
import math
import statistics
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.report.constants import (
    TEXT_METRIC_LABELS,
    VIDEO_METRIC_FIELDS,
)
from utils.report.models import LLMSummary


def _format_tokens(call: Dict) -> str:
    usage = call.get('usage') or {}
    if not usage:
        return "-"
    inp = usage.get('input_tokens')
    out = usage.get('output_tokens')
    if inp is None and out is None:
        return "-"
    return f"in={inp or 0}, out={out or 0}"


def generate_html(summary_data: Dict, result_dir: Path, output_path: Path) -> None:
    entries: List[LLMSummary] = summary_data.get('entries', [])
    metrics_overview: List[Dict[str, float]] = summary_data.get('metrics_overview', [])
    dataset_dir: Optional[Path] = summary_data.get('dataset_dir')

    timestamp = summary_data.get('timestamp', datetime.now().isoformat())
    total = summary_data.get('total', len(entries))
    successful = summary_data.get('successful', 0)
    failed = summary_data.get('failed', max(0, total - successful))
    csv_rows: List[Dict[str, Any]] = []

    def rel(path: Optional[Path]) -> str:
        if not path:
            return ""
        return os.path.relpath(str(path), str(output_path.parent))

    def escape_text(text: str) -> str:
        return html.escape(text.strip()) if text else ""

    def format_metric(value: Optional[float], pattern: str = "{:.4f}", suffix: str = "") -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float) and math.isnan(value):
            return "N/A"
        try:
            return pattern.format(value) + suffix
        except Exception:
            return str(value)

    def render_image_grid(items: List[Tuple[str, Optional[Path]]], grid_class: str) -> str:
        valid_items = [(title, path) for title, path in items if path]
        if not valid_items:
            return "<p class='empty-note'>No images to display.</p>"
        parts = [f"<div class='{grid_class}'>"]
        for title, path in valid_items:
            parts.append(
                "    <div class='media-item'>"
                f"<h4>{html.escape(title)}</h4>"
                f"<img src='{rel(path)}' alt='{html.escape(title)}'>"
                "</div>"
            )
        parts.append("</div>")
        return "\n".join(parts)

    def render_video_grid(items: List[Tuple[str, Optional[Path]]]) -> str:
        valid_items = [(title, path) for title, path in items if path]
        if not valid_items:
            return "<p class='empty-note'>No videos to display.</p>"
        parts = ["<div class='media-grid'>"]
        for title, path in valid_items:
            parts.append(
                "    <div class='media-item'>"
                f"<h4>{html.escape(title)}</h4>"
                "<video controls>"
                f"<source src='{rel(path)}' type='video/mp4'>"
                "Your browser does not support video playback."
                "</video>"
                "</div>"
            )
        parts.append("</div>")
        return "\n".join(parts)

    def select_primary_video_metrics(metric_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not metric_map:
            return {}
        primary = metric_map.get('original_vs_generated')
        if primary:
            return primary
        for metrics in metric_map.values():
            if metrics:
                return metrics
        return {}

    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except Exception:
            return None

    def render_metric_summary(entry: LLMSummary) -> Tuple[str, Dict[str, Any]]:
        video_metrics = select_primary_video_metrics(entry.video_metrics)
        generated_video = entry.assets.get("generated_original_video") if entry.assets else None
        video_exists = bool(generated_video and isinstance(generated_video, Path) and generated_video.exists())
        video_metrics["video_success"] = 1.0 if video_exists else 0.0
        csv_values: Dict[str, Any] = {}
        chips: List[str] = []

        def add_chip(label: str, formatted: str) -> None:
            chips.append(
                f"<span class='metric-chip'><strong>{html.escape(label)}:</strong> {html.escape(formatted)}</span>"
            )

        llm_payload = entry.video_metrics.get("llm_similarity") if entry.video_metrics else None
        llm_model_name: Optional[str] = None
        llm_justification: Optional[str] = None
        llm_artifact_path: Optional[Path] = None
        if isinstance(llm_payload, dict):
            score_value = _coerce_float(llm_payload.get("score"))
            if score_value is not None:
                video_metrics["llm_similarity_score"] = score_value
            model_value = llm_payload.get("model")
            if isinstance(model_value, str):
                llm_model_name = model_value.strip()
            justification_value = llm_payload.get("justification")
            if isinstance(justification_value, str):
                llm_justification = justification_value.strip()
            artifact_value = llm_payload.get("artifact_path")
            if isinstance(artifact_value, str) and artifact_value:
                artifact_candidate = Path(artifact_value)
                if not artifact_candidate.is_absolute():
                    artifact_candidate = result_dir / artifact_candidate
                llm_artifact_path = artifact_candidate

        for field in VIDEO_METRIC_FIELDS:
            raw_value = _coerce_float(video_metrics.get(field["metric_key"]))
            csv_values[field["csv_key"]] = raw_value
            if field["metric_key"] == "video_success":
                formatted = "Success" if (raw_value is not None and raw_value >= 0.5) else "Failure"
            else:
                formatted = format_metric(raw_value, field["format"], field["suffix"])
            add_chip(field["label"], formatted)

        frame_value: Optional[int]
        raw_frame = entry.frame_count or video_metrics.get('num_frames') or video_metrics.get('frame_count')
        if raw_frame is None:
            frame_value = None
        else:
            try:
                frame_value = int(raw_frame)
            except Exception:
                frame_value = None
        csv_values["frame_count"] = float(frame_value) if frame_value is not None else None
        add_chip("Frames", str(frame_value) if frame_value is not None else "N/A")

        text_metrics = entry.text_metrics or {}
        for key, label, pattern in TEXT_METRIC_LABELS:
            raw = _coerce_float(text_metrics.get(key))
            csv_key = key
            csv_values[csv_key] = raw
            formatted = pattern.format(raw) if raw is not None else "N/A"
            add_chip(label, formatted)

        csv_values["llm_gemini_model"] = llm_model_name
        csv_values["llm_gemini_justification"] = llm_justification

        link_html = ""
        if entry.text_metrics_path:
            link_html = (
                f"<p class='metrics-link'>文本指标文件: "
                f"<a href='{rel(entry.text_metrics_path)}'>{html.escape(entry.text_metrics_path.name)}</a></p>"
            )
        if llm_artifact_path and llm_artifact_path.exists():
            link_html += (
                f"<p class='metrics-link'>Gemini 评估结果: "
                f"<a href='{rel(llm_artifact_path)}'>{html.escape(llm_artifact_path.name)}</a></p>"
            )

        note_fragments: List[str] = []
        if llm_model_name:
            note_fragments.append(f"Gemini model: {html.escape(llm_model_name)}")
        if llm_justification:
            note_fragments.append(f"Gemini verdict: {html.escape(llm_justification)}")
        note_html = ""
        if note_fragments:
            note_html = "<p class='metrics-note'>" + "<br>".join(note_fragments) + "</p>"

        if not chips:
            chips.append("<span class='metric-chip'>暂无指标</span>")

        summary_html = "<div class='metric-chip-group'>" + "".join(chips) + "</div>" + link_html + note_html
        return summary_html, csv_values

    def render_llm_calls(calls: List[Dict]) -> str:
        if not calls:
            return "<p class='empty-note'>无 LLM 调用记录</p>"
        parts = ["<ul>"]
        for call in calls:
            attempt = call.get('attempt', '-')
            status = call.get('result', 'unknown')
            tokens = _format_tokens(call)
            idx = call.get('call_index', '?')
            parts.append(
                f"<li><strong>调用 {idx}/{attempt}</strong> · 状态: {html.escape(str(status))} · Tokens: {html.escape(tokens)}</li>"
            )
        parts.append("</ul>")
        return "\n".join(parts)

    def render_errors(errors: List[str]) -> str:
        if not errors:
            return ""
        items = "<br>".join(html.escape(err) for err in errors)
        return f"<div class='error-message'>{items}</div>"

    csv_field_order = (
        ["video_name", "model_name", "timestamp"]
        + [field["csv_key"] for field in VIDEO_METRIC_FIELDS]
        + ["frame_count"]
        + [metric_key for metric_key, _, _ in TEXT_METRIC_LABELS]
        + ["llm_gemini_model", "llm_gemini_justification"]
    )

    def write_metrics_csv(rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        csv_path = output_path.with_suffix('.csv')
        with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_field_order)
            writer.writeheader()
            for row in rows:
                normalized: Dict[str, Any] = {}
                for key in csv_field_order:
                    value = row.get(key)
                    if value is None or (isinstance(value, str) and value == ""):
                        normalized[key] = "N/A"
                    else:
                        normalized[key] = value
                writer.writerow(normalized)
        print(f"✅ 指标 CSV 已生成: {csv_path}")

    header_html = f"""<!DOCTYPE html>
<html lang=\"zh\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Physics Prediction Results Evaluation</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metrics-summary {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0 30px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.08);
        }}
        .metrics-summary h3 {{
            margin-top: 0;
            color: #333;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .metrics-table th,
        .metrics-table td {{
            border: 1px solid #e0e0e0;
            padding: 8px 12px;
            text-align: center;
            font-size: 0.9em;
        }}
        .metrics-table th {{
            background: #f0f2ff;
            color: #444;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .video-container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .video-title {{
            font-size: 1.8em;
            color: #333;
            margin-bottom: 25px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .metric-chip-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }}
        .metric-chip {{
            background: #eef2ff;
            color: #1f2937;
            padding: 8px 12px;
            border-radius: 18px;
            font-size: 0.9em;
            box-shadow: 0 1px 3px rgba(102, 126, 234, 0.25);
        }}
        .metric-chip strong {{
            margin-right: 6px;
        }}
        .metrics-link {{
            margin: -8px 0 18px 0;
            color: #555;
            font-size: 0.85em;
        }}
        .media-section {{
            margin-bottom: 30px;
        }}
        .media-section h3 {{
            color: #333;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .frame-mask-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .media-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .media-grid.single-column {{
            grid-template-columns: 1fr;
        }}
        .media-item {{
            text-align: center;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
        }}
        .media-item h4 {{
            margin: 0 0 10px 0;
            color: #555;
            font-size: 1em;
        }}
        .media-item img {{
            max-width: 100%;
            height: 200px;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            background: white;
            padding: 5px;
        }}
        .media-item video {{
            max-width: 100%;
            height: 200px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .motion-analysis-section {{
            background: #e8f4fd;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #2196F3;
        }}
        .analysis-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .analysis-section.solo {{
            grid-template-columns: 1fr;
        }}
        .analysis-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }}
        .analysis-card h4 {{
            margin-top: 0;
            color: #333;
        }}
        .analysis-text {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            white-space: pre-wrap;
            font-size: 0.95em;
            color: #333;
            margin-bottom: 15px;
            max-height: 220px;
            overflow-y: auto;
        }}
        .error-message {{
            color: #dc3545;
            background: #f8d7da;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #f5c6cb;
            margin-bottom: 20px;
        }}
        .sub-link {{
            margin-top: 6px;
            color: #666;
            font-size: 0.85em;
        }}
        .success-badge {{
            background: #28a745;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }}
        .failed-badge {{
            background: #dc3545;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }}
        .motion-badge {{
            background: #2196F3;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }}
        .empty-note {{
            color: #888;
        }}
        pre code {{
            white-space: pre;
        }}
        details.code-details {{
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            padding: 12px 16px;
            margin-bottom: 15px;
        }}
        details.code-details summary {{
            cursor: pointer;
            font-weight: 600;
            color: #333;
        }}
        details.code-details pre {{
            margin-top: 12px;
            max-height: 300px;
            overflow-y: auto;
            background: #1e272e;
            color: #d2dae2;
            padding: 12px;
            border-radius: 6px;
            font-size: 0.85em;
            line-height: 1.4;
        }}
        @media (max-width: 768px) {{
            .analysis-section {{
                grid-template-columns: 1fr;
            }}
            .frame-mask-grid {{
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            }}
        }}
    </style>
</head>
<body>
    <div class=\"header\">
        <h1>🎯 Physics Prediction Results</h1>
        <p>物理预测结果评估报告</p>
        <p>生成时间: {html.escape(str(timestamp))}</p>
        <p>输出目录: {html.escape(str(result_dir))}</p>
        {f"<p>数据源目录: {html.escape(str(dataset_dir))}</p>" if dataset_dir else ''}
        <div class=\"stats\">
            <div class=\"stat-card\">\n                <div class=\"stat-number\">{total}</div>\n                <div>总计项目</div>\n            </div>
            <div class=\"stat-card\">\n                <div class=\"stat-number\">{successful}</div>\n                <div>成功处理</div>\n            </div>
            <div class=\"stat-card\">\n                <div class=\"stat-number\">{failed}</div>\n                <div>处理失败</div>\n            </div>
        </div>
    </div>
"""

    metrics_rows: List[str] = []
    for overview in metrics_overview:
        label = html.escape(str(overview.get('label', '未知对比')))
        psnr_text = format_metric(overview.get('psnr_mean'), "{:.2f}", " dB")
        ssim_text = format_metric(overview.get('ssim_mean'), "{:.4f}")
        clip_text = format_metric(overview.get('clip_cosine_mean'), "{:.4f}")
        clip_caption_text = format_metric(overview.get('clip_caption_mean'), "{:.4f}")
        dino_text = format_metric(overview.get('dino_similarity_mean'), "{:.4f}")
        lpips_text = format_metric(overview.get('lpips_mean'), "{:.4f}")
        fsim_text = format_metric(overview.get('fsim_mean'), "{:.4f}")
        vsi_text = format_metric(overview.get('vsi_mean'), "{:.4f}")
        dists_text = format_metric(overview.get('dists_mean'), "{:.4f}")
        metrics_rows.append(
            f"<tr><td>{label}</td>"
            f"<td>{psnr_text}</td>"
            f"<td>{ssim_text}</td>"
            f"<td>{clip_text}</td>"
            f"<td>{clip_caption_text}</td>"
            f"<td>{dino_text}</td>"
            f"<td>{lpips_text}</td>"
            f"<td>{fsim_text}</td>"
            f"<td>{vsi_text}</td>"
            f"<td>{dists_text}</td></tr>"
        )

    metrics_html = ""
    if metrics_rows:
        metrics_html = """
    <div class=\"metrics-summary\">
        <h3>📈 视频相似度概览</h3>
        <table class=\"metrics-table\">
            <thead>
                <tr>
                    <th>对比类型</th>
                    <th>PSNR (dB)</th>
                    <th>SSIM</th>
                    <th>CLIP cosine</th>
                    <th>CLIP text</th>
                    <th>DINO</th>
                    <th>LPIPS</th>
                    <th>FSIM</th>
                    <th>VSI</th>
                    <th>DISTS</th>
                </tr>
            </thead>
            <tbody>
    """ + "\n".join(metrics_rows) + "\n            </tbody>\n        </table>\n    </div>\n"

    body_parts = [header_html]
    if metrics_html:
        body_parts.append(metrics_html)

    for entry in entries:
        success_badge = "<span class='success-badge'>✅ 成功</span>" if entry.success else "<span class='failed-badge'>⚠️ 检查</span>"
        motion_badge = ""
        if entry.text_sections.get('motion_prediction') or entry.text_sections.get('motion_analysis'):
            motion_badge = "<span class='motion-badge'>🔍 Scene Analysis</span>"

        scene_text = entry.text_sections.get('motion_analysis') or entry.text_sections.get('analysis') or ""
        motion_prediction_text = entry.text_sections.get('motion_prediction')
        code_guidance_text = entry.text_sections.get('code_guidance')

        original_analysis_text = entry.text_sections.get('original_image_analysis') or entry.text_sections.get('analysis')

        motion_prediction_block = ""
        if motion_prediction_text:
            motion_prediction_block = (
                "<h3>🧭 运动预测</h3>"
                "<div class='analysis-text'>"
                f"{escape_text(motion_prediction_text)}"
                "</div>"
            )

        code_guidance_block = ""
        if code_guidance_text:
            code_guidance_block = (
                "<h3>🛠️ 代码指导</h3>"
                "<div class='analysis-text'>"
                f"{escape_text(code_guidance_text)}"
                "</div>"
            )

        html_asset_link = ""
        if entry.assets.get('html_path'):
            html_asset_link = (
                f"<p class='sub-link'>HTML 文件: "
                f"<a href='{rel(entry.assets['html_path'])}'>{html.escape(entry.assets['html_path'].name)}</a></p>"
            )

        html_block = ""
        if entry.html_content:
            html_block = (
                "<div class='media-section'>"
                "<h3>🧾 生成 HTML 页面</h3>"
                "<details class='code-details'><summary>点击展开 HTML 内容</summary>"
                f"<pre>{escape_text(entry.html_content)}</pre>"
                "</details>"
                f"{html_asset_link}"
                "</div>"
            )

        frame_items = [
            ("第1帧 - 原始", entry.assets.get('dataset_first_frame')),
            ("第10帧 - 原始", entry.assets.get('dataset_tenth_frame')),
            ("生成帧预览", entry.assets.get('generated_frame')),
        ]

        video_items = [
            ("原始视频", entry.assets.get('dataset_video')),
            ("生成视频", entry.assets.get('generated_original_video')),
        ]

        raft_visualizations: List[Tuple[str, Optional[Path]]] = []
        primary_metrics = entry.video_metrics.get('original_vs_generated') or {}
        raft_payload = primary_metrics.get('raft_visualizations') or []
        variant_map = {
            "original": "原始对比",
            "segmented": "分割对比",
            "single": "单独对比",
        }
        if isinstance(raft_payload, list):
            for idx, item in enumerate(raft_payload, start=1):
                if isinstance(item, dict):
                    path_str = item.get("path")
                    frame_idx = item.get("frame_index")
                    mean_epe = item.get("mean_epe")
                    variant = item.get("variant")
                else:
                    path_str = item
                    frame_idx = None
                    mean_epe = None
                    variant = None
                if not path_str:
                    continue
                vis_path = Path(path_str)
                if not vis_path.is_absolute():
                    vis_path = result_dir / vis_path
                if not vis_path.exists():
                    continue
                label_parts: List[str] = [f"光流对比 #{idx}"]
                if variant:
                    label_parts.append(f"[{variant_map.get(str(variant), str(variant))}]")
                if frame_idx is not None:
                    try:
                        label_parts.append(f"帧 {int(frame_idx) + 1}")
                    except Exception:
                        pass
                if mean_epe is not None:
                    try:
                        label_parts.append(f"EPE {float(mean_epe):.2f}")
                    except Exception:
                        pass
                raft_visualizations.append((" ".join(label_parts), vis_path))

        raft_section = ""
        if raft_visualizations:
            raft_section = (
                "<div class='media-section'>"
                "<h3>🧭 光流向量对比</h3>"
                f"{render_image_grid(raft_visualizations, 'frame-mask-grid')}"
                "</div>"
            )

        errors_block = render_errors(entry.errors)
        render_log_link = ""
        if entry.assets.get('render_log'):
            render_log_link = f"<p>渲染日志: <a href='{rel(entry.assets['render_log'])}'>{html.escape(str(entry.assets['render_log']))}</a></p>"

        llm_log_link = ""

        metrics_summary_html, metrics_csv_values = render_metric_summary(entry)
        model_name = entry.model_name or "unknown"
        csv_row = {
            "video_name": entry.image_name,
            "model_name": model_name,
            "timestamp": str(timestamp),
        }
        for field in VIDEO_METRIC_FIELDS:
            csv_key = field["csv_key"]
            value = metrics_csv_values.get(csv_key)
            if csv_key == "video_success" and value is not None:
                try:
                    csv_row[csv_key] = int(round(float(value)))
                except Exception:
                    csv_row[csv_key] = value
            else:
                csv_row[csv_key] = value

        frame_value = metrics_csv_values.get("frame_count")
        if frame_value is not None:
            try:
                csv_row["frame_count"] = int(frame_value)
            except Exception:
                csv_row["frame_count"] = frame_value
        else:
            csv_row["frame_count"] = None

        csv_row["llm_gemini_model"] = metrics_csv_values.get("llm_gemini_model")
        csv_row["llm_gemini_justification"] = metrics_csv_values.get("llm_gemini_justification")

        for key in [metric_key for metric_key, _, _ in TEXT_METRIC_LABELS]:
            csv_row[key] = metrics_csv_values.get(key)
        csv_rows.append(csv_row)

        gt_block = ""
        if entry.gt_analysis:
            caption = escape_text(entry.gt_analysis)
            gt_link = ""
            if entry.gt_analysis_path:
                gt_link = f"<p class='sub-link'>GT 文件: <a href='{rel(entry.gt_analysis_path)}'>{html.escape(entry.gt_analysis_path.name)}</a></p>"
            gt_block = (
                "<div class='media-section'>"
                "<h3>📜 GT 场景分析</h3>"
                f"<div class='analysis-text'>{caption}</div>"
                f"{gt_link}"
                "</div>"
            )

        model_analysis_block = ""
        if entry.model_analysis:
            model_link = ""
            if entry.model_analysis_path:
                model_link = (
                    f"<p class='sub-link'>分析文件: "
                    f"<a href='{rel(entry.model_analysis_path)}'>{html.escape(entry.model_analysis_path.name)}</a></p>"
                )
            model_analysis_block = (
                "<div class='media-section'>"
                "<h3>🧪 模型场景分析</h3>"
                f"<div class='analysis-text'>{escape_text(entry.model_analysis)}</div>"
                f"{model_link}"
                "</div>"
            )

        entry_html = f"""
    <div class=\"video-container\">
        <h2 class=\"video-title\">[{html.escape(model_name)}] {html.escape(entry.image_name)}{success_badge}{motion_badge}</h2>
        {metrics_summary_html}
        {render_log_link}
        {errors_block}

        {gt_block}
        {model_analysis_block}

        <div class=\"motion-analysis-section\">
            <h3>🔍 场景分析</h3>
            <div class=\"analysis-text\">{escape_text(scene_text) or '暂无分析'}</div>
            {motion_prediction_block}
            {code_guidance_block}
        </div>

        {html_block}

        <div class=\"media-section\">
            <h3>📸 关键帧对比</h3>
            {render_image_grid(frame_items, 'frame-mask-grid')}
        </div>

        <div class=\"media-section\">
            <h3>🎥 视频对比</h3>
            {render_video_grid(video_items)}
        </div>

        {raft_section}

        <div class=\"analysis-section solo\">
            <div class=\"analysis-card\">
                <h3>🤖 原始图像解析</h3>
                <h4>场景分析:</h4>
                <div class=\"analysis-text\">{escape_text(original_analysis_text) or '暂无内容'}</div>
            </div>
        </div>

    </div>
"""

        body_parts.append(entry_html)

    body_parts.append("</body>\n</html>\n")

    write_metrics_csv(csv_rows)
    output_path.write_text("".join(body_parts), encoding='utf-8')
    print(f"✅ HTML 已生成: {output_path}")
