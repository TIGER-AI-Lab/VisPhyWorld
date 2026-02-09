#!/usr/bin/env python3
"""
Evaluate LLM-generated analysis/prediction text against ground-truth captions.

Features:
- Load GT captions from a JSONL/JSON mapping or generate GT via GPT-5.1 API (optional, stub).
- Compute ROUGE-L and BERTScore (F1) between GT and model text (analysis/prediction).
- Write metrics to predictions/<name>_text_metrics.json and merge summary.

Usage:
  python utils/text_metrics_evaluator.py \
    --result-dir /path/to/output/physics_prediction_xxx \
    --gt-file /path/to/gt.jsonl --field prediction --use-bert-score
"""

import argparse
import json
import os
import unicodedata
import warnings
from typing import Dict, Optional, Tuple


def _load_mapping_lines(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            name = obj.get('name') or obj.get('id')
            cap = obj.get('caption') or obj.get('text')
            if name and cap:
                mapping[name] = cap
    return mapping


def load_text_mapping(path: str) -> Dict[str, str]:
    """Robustly load a mapping of name->caption from JSON/JSONL or newline-delimited JSON.

    Supports:
    - .jsonl: one JSON object per line
    - .json: either a dict mapping or {items: [{name, caption}]}; if JSON load fails, fallback to line-wise JSON
    - Any file that is newline-delimited JSON objects regardless of extension
    """
    mapping: Dict[str, str] = {}
    if path.endswith('.jsonl'):
        return _load_mapping_lines(path)
    try:
        obj = json.load(open(path, 'r', encoding='utf-8'))
    except Exception:
        # Fallback: try newline-delimited JSON objects
        return _load_mapping_lines(path)

    if isinstance(obj, dict) and 'items' in obj:
        for it in obj['items']:
            name = it.get('name') or it.get('id')
            cap = it.get('caption') or it.get('text')
            if name and cap:
                mapping[name] = cap
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                mapping[k] = v
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                name = it.get('name') or it.get('id')
                cap = it.get('caption') or it.get('text')
                if name and cap:
                    mapping[name] = cap
    return mapping


def try_import_rouge():
    try:
        from rouge_score import rouge_scorer  # type: ignore
        return rouge_scorer
    except Exception:
        return None


def try_import_bertscore():
    try:
        from bert_score import score as bert_score  # type: ignore
        return bert_score
    except Exception:
        return None


def normalize_text(s: str) -> str:
    # Unicode NFKC normalization + whitespace collapse
    s = unicodedata.normalize('NFKC', (s or ''))
    return ' '.join(s.strip().split())


def tokenize_for_rouge(s: str, force_char_level: bool = False) -> list:
    """Simple whitespace tokenization for ROUGE-style metrics.

    We assume both GT and predictions are English text, so no special
    handling for CJK or character-level metrics is required.
    """
    s = normalize_text(s)
    return s.split()


def _rouge_l_scores(gt: str, pred: str, force_char_level: bool = False) -> Tuple[float, float, float]:
    """Compute ROUGE-L precision/recall/F1 with LCS at token level."""
    g = tokenize_for_rouge(gt, force_char_level=force_char_level)
    p = tokenize_for_rouge(pred, force_char_level=force_char_level)
    if not g or not p:
        return 0.0, 0.0, 0.0
    m, n = len(g), len(p)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        gi = g[i - 1]
        for j in range(1, n + 1):
            if gi == p[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    lcs = dp[m][n]
    prec = lcs / max(1, len(p))
    rec = lcs / max(1, len(g))
    if prec + rec == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * prec * rec / (prec + rec)
    return float(prec), float(rec), float(f1)

def compute_text_metrics(gt: str, pred: str, use_bertscore: bool = False) -> Dict[str, float]:
    gt = normalize_text(gt)
    pred = normalize_text(pred)
    out: Dict[str, float] = {}

    rouge_prec = rouge_rec = rouge_f1 = 0.0
    rouge_scorer = try_import_rouge()
    if rouge_scorer is not None:
        rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        r = rs.score(gt, pred)["rougeL"]
        rouge_prec = float(r.precision)
        rouge_rec = float(r.recall)
        rouge_f1 = float(r.fmeasure)
    else:
        rouge_prec, rouge_rec, rouge_f1 = _rouge_l_scores(gt, pred)

    out["rougeL_precision"] = rouge_prec
    out["rougeL_recall"] = rouge_rec
    out["rougeL_f1"] = rouge_f1

    if use_bertscore:
        bert_scorer = try_import_bertscore()
        if bert_scorer is not None:
            # Assume English-only inputs; use an English BERT-based model.
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*pooler\\.dense\\.(?:bias|weight).*",
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=r"You should probably TRAIN this model on a down-stream task.*",
                    )
                    P, R, F1 = bert_scorer([pred], [gt], lang="en", model_type="roberta-base")
                out["bertscore_f1"] = float(F1.mean().item())
            except Exception:
                # 在依赖或模型出错时，不中断整个流程，而是返回空值。
                out["bertscore_f1"] = None  # type: ignore[assignment]
        else:
            # 依赖缺失时同样返回空值，方便上游脚本继续运行。
            out["bertscore_f1"] = None  # type: ignore[assignment]
    return out


def read_model_text(pred_path: str, field: str) -> Optional[str]:
    try:
        obj = json.load(open(pred_path, 'r', encoding='utf-8'))
        if field == 'combined':
            a = obj.get('analysis') or ''
            p = obj.get('prediction') or ''
            text = (a + '\n' + p).strip()
            return text if text else None
        return obj.get(field)
    except Exception:
        return None


def maybe_generate_gt_with_gpt5(name: str, frames_dir: str) -> Optional[str]:
    """Stub: integrate GPT-5.1 API here if you want auto-GT.
    For now, return None so that a provided gt-file is used.
    """
    return None


def read_text_from_pred_dir(base_dir: str, name: str) -> Optional[str]:
    """Read combined text for a sample name from a directory structure like:
    base_dir/<name>/<name>_combined_claude.txt (preferred), or fall back to analysis only.
    """
    subdir = os.path.join(base_dir, name)
    if not os.path.isdir(subdir):
        return None
    combined = os.path.join(subdir, f"{name}_combined_claude.txt")
    if os.path.exists(combined):
        try:
            return open(combined, 'r', encoding='utf-8').read().strip()
        except Exception:
            return None
    analysis = os.path.join(subdir, f"{name}_analysis_claude.txt")
    if os.path.exists(analysis):
        try:
            return open(analysis, 'r', encoding='utf-8').read().strip()
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description="Evaluate text against ground truth")
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--gt-file", help="Path to GT captions (json/jsonl). If absent, try GPT-5.1.")
    ap.add_argument("--pred-caption-file", help="Optional JSON/JSONL of predicted captions: {name, caption} per entry")
    ap.add_argument("--field", default="combined", choices=["analysis","prediction","combined"], help="Which field to evaluate (combined=analysis+prediction)")
    ap.add_argument("--use-bertscore", action="store_true")
    ap.add_argument("--pred-text-dir", help="Optional dir containing <name>/<name>_combined_claude.txt")
    args = ap.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    artifacts_root = os.path.join(result_dir, 'artifacts')
    predictions_dir = os.path.join(result_dir, 'predictions')
    if not os.path.exists(predictions_dir):
        predictions_dir = os.path.join(artifacts_root, 'predictions')
    frames_dir = os.path.join(result_dir, 'frames')
    if not os.path.exists(frames_dir):
        frames_dir = os.path.join(artifacts_root, 'frames')

    gt_map: Dict[str, str] = {}
    if args.gt_file and os.path.exists(args.gt_file):
        gt_map = load_text_mapping(args.gt_file)

    written = 0

    # Mode 1: direct predicted captions file vs GT captions
    if args.pred_caption_file and os.path.exists(args.pred_caption_file):
        pred_map = load_text_mapping(args.pred_caption_file)
        names = sorted(set(gt_map.keys()) & set(pred_map.keys()))
        for name in names:
            gt = gt_map.get(name)
            pred_text = pred_map.get(name)
            if not gt or not pred_text:
                continue
            metrics = compute_text_metrics(gt, pred_text, use_bertscore=args.use_bertscore)
            out = {
                'video_name': name,
                'field': args.field,
                'gt_source': args.gt_file or 'gpt5',
                'metrics': metrics,
            }
            out_path = os.path.join(predictions_dir, f"{name}_text_metrics.json") if args.field == 'combined' else os.path.join(predictions_dir, f"{name}_text_metrics_{args.field}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            written += 1
            print(f"✅ Wrote text metrics (captions): {out_path}")

    # Mode 2: read free-form text files from a directory structure
    elif args.pred_text_dir and os.path.isdir(args.pred_text_dir):
        names = sorted(gt_map.keys()) if gt_map else []
        for name in names:
            gt = gt_map.get(name)
            if not gt:
                continue
            model_pred = read_text_from_pred_dir(args.pred_text_dir, name)
            if not model_pred:
                continue
            # 统一在 compute_text_metrics 内部计算（或忽略）BERTScore。
            metrics = compute_text_metrics(gt, model_pred, use_bertscore=args.use_bertscore)

            out = {
                'video_name': name,
                'field': args.field,
                'gt_source': args.gt_file or 'gpt5',
                'metrics': metrics,
            }
            out_path = os.path.join(predictions_dir, f"{name}_text_metrics.json") if args.field == 'combined' else os.path.join(predictions_dir, f"{name}_text_metrics_{args.field}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            written += 1
            print(f"✅ Wrote text metrics: {out_path}")
    else:
        # Fallback: evaluate from predictions JSONs as before
        for fn in os.listdir(predictions_dir):
            if not fn.endswith('_prediction_original.json'):
                continue
            name = fn.replace('_prediction_original.json','')
            gt = gt_map.get(name)
            if not gt:
                gt = maybe_generate_gt_with_gpt5(name, frames_dir) or None
            if not gt:
                continue
            model_pred = read_model_text(os.path.join(predictions_dir, fn), args.field)
            if not model_pred:
                continue
            metrics = compute_text_metrics(gt, model_pred, use_bertscore=args.use_bertscore)
            out = {
                'video_name': name,
                'field': args.field,
                'gt_source': args.gt_file or 'gpt5',
                'metrics': metrics,
            }
            if args.field == 'combined':
                out_path = os.path.join(predictions_dir, f"{name}_text_metrics.json")
            else:
                out_path = os.path.join(predictions_dir, f"{name}_text_metrics_{args.field}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            written += 1
            print(f"✅ Wrote text metrics: {out_path}")

    print(f"Done. Wrote {written} files.")


if __name__ == '__main__':
    main()
