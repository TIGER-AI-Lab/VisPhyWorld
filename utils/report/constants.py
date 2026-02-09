from __future__ import annotations

from typing import Dict, List, Tuple


TEXT_METRIC_LABELS: List[Tuple[str, str, str]] = [
    ("rougeL_f1", "ROUGE-L F1", "{:.4f}"),
    ("rougeL_precision", "ROUGE-L Precision", "{:.4f}"),
    ("rougeL_recall", "ROUGE-L Recall", "{:.4f}"),
    ("bertscore_f1", "BERTScore F1", "{:.4f}"),
]

VIDEO_METRIC_FIELDS: List[Dict[str, str]] = [
    {"label": "PSNR", "metric_key": "psnr_mean", "format": "{:.2f}", "suffix": " dB", "csv_key": "psnr"},
    {"label": "SSIM", "metric_key": "ssim_mean", "format": "{:.4f}", "suffix": "", "csv_key": "ssim"},
    {"label": "CLIP (image)", "metric_key": "clip_cosine_mean", "format": "{:.4f}", "suffix": "", "csv_key": "clip_image"},
    {"label": "CLIP (caption)", "metric_key": "clip_caption_mean", "format": "{:.4f}", "suffix": "", "csv_key": "clip_caption"},
    {"label": "DINO", "metric_key": "dino_similarity_mean", "format": "{:.4f}", "suffix": "", "csv_key": "dino"},
    {"label": "LPIPS", "metric_key": "lpips_mean", "format": "{:.4f}", "suffix": "", "csv_key": "lpips"},
    {"label": "FSIM", "metric_key": "fsim_mean", "format": "{:.4f}", "suffix": "", "csv_key": "fsim"},
    {"label": "VSI", "metric_key": "vsi_mean", "format": "{:.4f}", "suffix": "", "csv_key": "vsi"},
    {"label": "DISTS", "metric_key": "dists_mean", "format": "{:.4f}", "suffix": "", "csv_key": "dists"},
    {"label": "RAFT EPE", "metric_key": "raft_epe_mean", "format": "{:.4f}", "suffix": "", "csv_key": "raft_epe"},
    {"label": "RAFT angle diff", "metric_key": "raft_angle_diff_deg_mean", "format": "{:.2f}", "suffix": " °", "csv_key": "raft_angle_deg"},
    {"label": "RAFT magnitude diff", "metric_key": "raft_mag_diff_mean", "format": "{:.4f}", "suffix": "", "csv_key": "raft_mag_diff"},
    {"label": "RAFT ref flow mag", "metric_key": "raft_ref_flow_mag_mean", "format": "{:.4f}", "suffix": "", "csv_key": "raft_ref_mag"},
    {"label": "RAFT gen flow mag", "metric_key": "raft_gen_flow_mag_mean", "format": "{:.4f}", "suffix": "", "csv_key": "raft_gen_mag"},
    {"label": "Ref offset", "metric_key": "alignment_ref_offset", "format": "{:.0f}", "suffix": "", "csv_key": "alignment_ref_offset"},
    {"label": "Gen offset", "metric_key": "alignment_gen_offset", "format": "{:.0f}", "suffix": "", "csv_key": "alignment_gen_offset"},
    {"label": "Alignment error", "metric_key": "alignment_error", "format": "{:.4f}", "suffix": "", "csv_key": "alignment_error"},
    {"label": "Generation success", "metric_key": "video_success", "format": "{:.0f}", "suffix": "", "csv_key": "video_success"},
    {"label": "Gemini similarity", "metric_key": "llm_similarity_score", "format": "{:.2f}", "suffix": "", "csv_key": "llm_gemini_score"},
]
