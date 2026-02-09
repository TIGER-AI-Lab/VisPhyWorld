#!/usr/bin/env python3
"""
Compute video similarity metrics between dataset original videos and generated videos.

Metrics:
- PSNR (per frame, then mean)
- SSIM (per frame, then mean)
- Optional CLIP cosine similarity (per sampled frame, then mean) if torch+open_clip are available

Usage:
  python utils/evaluate_video_metrics.py \
      --result-dir /home/jiarong/VisExpert/physics_prediction/output/physics_prediction_20250812_010311 \
      --comparison-dir /home/jiarong/VisExpert/physics_prediction/video_comparison_5 \
      --sample-every 1

Writes JSON to: <result-dir>/predictions/<name>_video_metrics.json
"""

import argparse
import heapq
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import math
import statistics
import numpy as np

# Ensure project root is on sys.path when invoked as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.llm_video_evaluator import (
    evaluate_video_pair as llm_evaluate_video_pair,
    LLMVideoEvaluationError,
)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    diff = (img1.astype(np.float32) - img2.astype(np.float32)).reshape(-1).astype(np.float64)
    if diff.size == 0:
        return 0.0
    mse = float(np.dot(diff, diff)) / diff.size
    if mse <= 1e-10:
        return 100.0
    PIXEL_MAX = 255.0
    psnr = 10.0 * np.log10((PIXEL_MAX ** 2) / mse)
    return float(psnr)


def compute_ssim_single_channel(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)

    # Gaussian parameters
    K1, K2 = 0.01, 0.03
    L = 255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    kernel_size = (11, 11)
    sigma = 1.5
    mu_x = cv2.GaussianBlur(x, kernel_size, sigma)
    mu_y = cv2.GaussianBlur(y, kernel_size, sigma)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(x * x, kernel_size, sigma) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, kernel_size, sigma) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, kernel_size, sigma) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return float(ssim_map.mean())


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    # If color, compute on Y channel of YCrCb and average with RGB channels for robustness
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
        # Y channel
        y1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        y2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        ssim_y = compute_ssim_single_channel(y1, y2)
        # RGB channels average
        ssim_r = compute_ssim_single_channel(img1[:, :, 2], img2[:, :, 2])
        ssim_g = compute_ssim_single_channel(img1[:, :, 1], img2[:, :, 1])
        ssim_b = compute_ssim_single_channel(img1[:, :, 0], img2[:, :, 0])
        return float((ssim_y + ssim_r + ssim_g + ssim_b) / 4.0)
    else:
        return compute_ssim_single_channel(img1, img2)


class OptionalCLIP:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = None
        self.tokenizer = None
        try:
            import torch  # noqa: F401
            import open_clip  # noqa: F401
        except Exception as exc:
            raise RuntimeError("CLIP 指标计算需要 torch 与 open_clip，请先安装依赖。") from exc
        self._setup()

    def _setup(self):
        import torch
        import open_clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        model.eval()
        model.to(self.device)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def cosine(self, bgr_img_a: np.ndarray, bgr_img_b: np.ndarray) -> float:
        import torch
        # Convert BGR->RGB and to PIL
        rgb_a = cv2.cvtColor(bgr_img_a, cv2.COLOR_BGR2RGB)
        rgb_b = cv2.cvtColor(bgr_img_b, cv2.COLOR_BGR2RGB)
        from PIL import Image
        im_a = Image.fromarray(rgb_a)
        im_b = Image.fromarray(rgb_b)
        with torch.no_grad():
            ta = self.preprocess(im_a).unsqueeze(0).to(self.device)
            tb = self.preprocess(im_b).unsqueeze(0).to(self.device)
            fa = self.model.encode_image(ta)
            fb = self.model.encode_image(tb)
            fa = fa / fa.norm(dim=-1, keepdim=True)
            fb = fb / fb.norm(dim=-1, keepdim=True)
            sim = (fa @ fb.T).squeeze().item()
            return float(sim)

    def caption_score(self, caption: str, bgr_img: np.ndarray) -> float:
        import torch
        # Prepare image
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        im = Image.fromarray(rgb)
        with torch.no_grad():
            ti = self.preprocess(im).unsqueeze(0).to(self.device)
            tokens = self.tokenizer([caption])
            tokens = tokens.to(self.device)
            fi = self.model.encode_image(ti)
            ft = self.model.encode_text(tokens)
            fi = fi / fi.norm(dim=-1, keepdim=True)
            ft = ft / ft.norm(dim=-1, keepdim=True)
            sim = (fi @ ft.T).squeeze().item()
            return float(sim)


class OptionalLPIPS:
    def __init__(self):
        self.model = None
        self.device = None
        try:
            import torch  # noqa: F401
            import lpips  # noqa: F401
        except Exception as exc:
            raise RuntimeError("LPIPS 指标计算需要 torch 与 lpips，请先安装依赖。") from exc
        self._setup()

    def _setup(self):
        import torch
        import lpips
        import contextlib
        import io
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = lpips.LPIPS(net='alex').to(self.device)
        self.model.eval()

    def distance(self, bgr_img_a: np.ndarray, bgr_img_b: np.ndarray) -> float:
        import torch
        # BGR -> RGB, HWC uint8 -> NCHW float32 in [-1, 1]
        rgb_a = cv2.cvtColor(bgr_img_a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_b = cv2.cvtColor(bgr_img_b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ta = torch.from_numpy(rgb_a).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        tb = torch.from_numpy(rgb_b).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        ta = ta.to(self.device)
        tb = tb.to(self.device)
        with torch.no_grad():
            d = self.model(ta, tb)
        return float(d.squeeze().detach().cpu().item())


class OptionalDINO:
    def __init__(self):
        self.model = None
        self.device = None
        self.transform = None
        try:
            import torch  # noqa: F401
            import timm  # noqa: F401
            from torchvision import transforms  # noqa: F401
        except Exception as exc:
            raise RuntimeError("DINO 指标计算需要 torch、timm 与 torchvision，请先安装依赖。") from exc
        self._setup()

    def _setup(self):
        import torch
        import timm
        from torchvision import transforms
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name_candidates = [
            'vit_base_patch16_224.dino',
            'vit_small_patch16_224.dino'
        ]
        model = None
        last_error: Optional[Exception] = None
        for name in model_name_candidates:
            try:
                model = timm.create_model(name, pretrained=True, num_classes=0)
                break
            except Exception as exc:
                last_error = exc
                continue
        if model is None:
            raise RuntimeError("无法加载 DINO 模型，请检查 timm 版本或模型权重缓存是否可用。") from last_error
        model.eval().to(self.device)
        self.model = model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def cosine(self, bgr_img_a: np.ndarray, bgr_img_b: np.ndarray) -> float:
        import torch
        ta = self.transform(cv2.cvtColor(bgr_img_a, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        tb = self.transform(cv2.cvtColor(bgr_img_b, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fa = self.model(ta)
            fb = self.model(tb)
            fa = fa / fa.norm(dim=-1, keepdim=True)
            fb = fb / fb.norm(dim=-1, keepdim=True)
            sim = (fa @ fb.T).squeeze().item()
            return float(sim)


class OptionalPIQMetrics:
    def __init__(self, enable_fsim: bool, enable_vsi: bool, enable_dists: bool):
        self.enable_fsim = enable_fsim
        self.enable_vsi = enable_vsi
        self.enable_dists = enable_dists
        try:
            import torch  # noqa: F401
            import piq  # noqa: F401
        except Exception as exc:
            raise RuntimeError("FSIM/VSI/DISTS 指标计算需要 torch 与 piq，请先安装依赖。") from exc
        self._setup()

    def _setup(self):
        import torch
        import piq

        self.torch = torch
        self.piq = piq
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dists_metric = None
        if self.enable_dists:
            self.dists_metric = piq.DISTS(reduction="none").to(self.device)

    def _to_tensor(self, bgr_img: np.ndarray) -> "torch.Tensor":
        import torch

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def compute(self, bgr_img_a: np.ndarray, bgr_img_b: np.ndarray) -> Dict[str, float]:
        ta = self._to_tensor(bgr_img_a)
        tb = self._to_tensor(bgr_img_b)
        results: Dict[str, float] = {}
        with self.torch.no_grad():
            if self.enable_fsim:
                val = self.piq.fsim(ta, tb, data_range=1.0)
                results["fsim"] = float(val.detach().cpu().item())
            if self.enable_vsi:
                val = self.piq.vsi(ta, tb, data_range=1.0)
                results["vsi"] = float(val.detach().cpu().item())
            if self.enable_dists and self.dists_metric is not None:
                dist = self.dists_metric(ta, tb)
                results["dists"] = float(dist.mean().detach().cpu().item())
        return results


class OptionalRAFT:
    _init_lock = None
    _infer_lock = None
    _shared_model = None
    _shared_device = None

    def __init__(self):
        self.model = None
        self.device = None
        try:
            import torch  # noqa: F401
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights  # noqa: F401
        except Exception as exc:
            raise RuntimeError("RAFT 指标计算需要 torchvision>=0.14 与已安装的预训练权重。") from exc
        self._setup()

    def _setup(self):
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        import threading

        if OptionalRAFT._init_lock is None:
            OptionalRAFT._init_lock = threading.Lock()
        if OptionalRAFT._infer_lock is None:
            OptionalRAFT._infer_lock = threading.Lock()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with OptionalRAFT._init_lock:
            if OptionalRAFT._shared_model is not None:
                self.model = OptionalRAFT._shared_model
                self.device = OptionalRAFT._shared_device
                self.torch = torch
                return

            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=None, progress=False)

            # If another library (e.g., accelerate) temporarily switched module
            # initialisation to meta tensors, we must materialize parameters
            # before loading weights / moving devices.
            try:
                is_meta = any(getattr(p, "is_meta", False) for p in model.parameters())
            except Exception:
                is_meta = False

            if is_meta:
                to_empty = getattr(model, "to_empty", None)
                if not callable(to_empty):
                    raise RuntimeError(
                        "RAFT model initialized on meta tensors but torch.nn.Module.to_empty() is unavailable."
                    )
                model.to_empty(device="cpu")

            # Avoid meta-tensor load warnings by explicitly loading the state dict.
            try:
                state_dict = weights.get_state_dict(progress=False)
                try:
                    model.load_state_dict(state_dict, assign=True)  # torch>=2.0
                except TypeError:
                    model.load_state_dict(state_dict)
            except Exception:
                # Fall back to torchvision's built-in weight loading.
                model = raft_large(weights=weights, progress=False)

            model.eval()
            if self.device == "cuda":
                model.to("cuda")

            OptionalRAFT._shared_model = model
            OptionalRAFT._shared_device = self.device
            self.model = model
            self.torch = torch

    def _prepare_tensor(self, bgr_img: np.ndarray) -> "torch.Tensor":
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _pad_pair(self, tensor_a: "torch.Tensor", tensor_b: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", int, int]:
        h = tensor_a.shape[-2]
        w = tensor_a.shape[-1]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            tensor_a = self.torch.nn.functional.pad(tensor_a, (0, pad_w, 0, pad_h), mode="replicate")
            tensor_b = self.torch.nn.functional.pad(tensor_b, (0, pad_w, 0, pad_h), mode="replicate")
        return tensor_a, tensor_b, h, w

    def compute_flow(self, bgr_img_a: np.ndarray, bgr_img_b: np.ndarray) -> np.ndarray:
        t_a = self._prepare_tensor(bgr_img_a)
        t_b = self._prepare_tensor(bgr_img_b)
        t_a, t_b, h, w = self._pad_pair(t_a, t_b)
        lock = OptionalRAFT._infer_lock
        if lock is None:
            with self.torch.no_grad():
                flow_pyramid = self.model(t_a, t_b)
        else:
            with lock:
                with self.torch.no_grad():
                    flow_pyramid = self.model(t_a, t_b)
        flow = flow_pyramid[-1][0, :, :h, :w].permute(1, 2, 0).detach().cpu().numpy()
        return flow.astype(np.float32, copy=False)


def _save_raft_visualizations(
    samples: List[Dict[str, Any]],
    diagnostics_dir: str,
    video_name: str,
    variant_label: str,
) -> List[Dict[str, Any]]:
    if not samples:
        return []
    global _RAFT_MPL_STATUS  # type: ignore[global-variable-not-assigned]
    try:
        _RAFT_MPL_STATUS
    except Exception:
        _RAFT_MPL_STATUS = None

    if _RAFT_MPL_STATUS is False:
        return []

    try:
        import matplotlib.pyplot as plt  # type: ignore
        _RAFT_MPL_STATUS = True
    except Exception:
        if _RAFT_MPL_STATUS is not False:
            print("⚠️ RAFT 可视化需要 matplotlib，已跳过生成。", flush=True)
        _RAFT_MPL_STATUS = False
        return []

    os.makedirs(diagnostics_dir, exist_ok=True)
    saved: List[Dict[str, Any]] = []
    diag_path = Path(diagnostics_dir)

    for rank, sample in enumerate(samples, start=1):
        frame_ref: np.ndarray = sample["frame_ref"]
        frame_gen: np.ndarray = sample["frame_gen"]
        flow_ref: np.ndarray = sample["flow_ref"]
        flow_gen: np.ndarray = sample["flow_gen"]
        epe_map: np.ndarray = sample["epe_map"]
        frame_index: int = sample["frame_index"]
        epe_value: float = sample["epe"]

        rgb_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)
        rgb_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2RGB)

        h, w = flow_ref.shape[:2]
        step = max(8, min(h, w) // 20)
        if step <= 0:
            step = 8
        y_coords = np.arange(step // 2, h, step, dtype=int)
        x_coords = np.arange(step // 2, w, step, dtype=int)
        if y_coords.size == 0 or x_coords.size == 0:
            continue
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        sample_x = grid_x.astype(int)
        sample_y = grid_y.astype(int)

        ref_u = flow_ref[sample_y, sample_x, 0]
        ref_v = -flow_ref[sample_y, sample_x, 1]
        gen_u = flow_gen[sample_y, sample_x, 0]
        gen_v = -flow_gen[sample_y, sample_x, 1]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.25)

        axes[0].imshow(rgb_ref)
        axes[0].quiver(grid_x, grid_y, ref_u, ref_v, color='cyan', angles='xy', scale_units='xy', scale=1)
        axes[0].set_title(f"参考光流 (帧 {frame_index + 1})")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].imshow(rgb_gen)
        axes[1].quiver(grid_x, grid_y, gen_u, gen_v, color='lime', angles='xy', scale_units='xy', scale=1)
        axes[1].set_title("生成光流")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        axes[2].imshow(rgb_gen)
        heat = axes[2].imshow(epe_map, cmap='inferno', alpha=0.6)
        axes[2].set_title(f"EPE 热力图 (均值 {epe_value:.2f})")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        fig.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout()
        filename = f"{video_name}_{variant_label}_frame{frame_index + 1:04d}_rank{rank}.png"
        output_path = diag_path / filename
        fig.savefig(output_path, bbox_inches="tight", dpi=120)
        plt.close(fig)

        saved.append({
            "path": str(output_path),
            "frame_index": frame_index,
            "mean_epe": epe_value,
            "variant": variant_label,
        })

    return saved


def _frame_feature(frame: np.ndarray, size: int = 48) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _dtw_align_frames(
    ref_frames: List[np.ndarray],
    gen_frames: List[np.ndarray],
    penalty: float = 0.005,
) -> Tuple[List[np.ndarray], List[np.ndarray], float, List[Tuple[int, int]]]:
    n = len(ref_frames)
    m = len(gen_frames)
    if n == 0 or m == 0:
        return ref_frames, gen_frames, 0.0, []

    feat_size = 48
    ref_feats = [_frame_feature(f, feat_size) for f in ref_frames]
    gen_feats = [_frame_feature(f, feat_size) for f in gen_frames]

    cost = np.full((n, m), np.inf, dtype=np.float64)
    prev_i = np.full((n, m), -1, dtype=np.int32)
    prev_j = np.full((n, m), -1, dtype=np.int32)

    for i in range(n):
        for j in range(m):
            diff = ref_feats[i] - gen_feats[j]
            local = float(np.mean(diff * diff))
            if i == 0 and j == 0:
                cost[i, j] = local
                continue

            candidates: List[Tuple[float, int, int, float]] = []
            if i > 0 and j > 0:
                candidates.append((cost[i - 1, j - 1], i - 1, j - 1, 0.0))
            if i > 0:
                candidates.append((cost[i - 1, j], i - 1, j, penalty))
            if j > 0:
                candidates.append((cost[i, j - 1], i, j - 1, penalty))
            if not candidates:
                cost[i, j] = local
                continue
            best = min(candidates, key=lambda x: x[0] + x[3])
            cost[i, j] = local + best[0] + best[3]
            prev_i[i, j] = best[1]
            prev_j[i, j] = best[2]

    i = n - 1
    j = m - 1
    path: List[Tuple[int, int]] = []
    while i >= 0 and j >= 0:
        path.append((i, j))
        pi = prev_i[i, j]
        pj = prev_j[i, j]
        if pi < 0 or pj < 0:
            break
        i, j = pi, pj
    path.reverse()

    if not path:
        path = [(min(n - 1, 0), min(m - 1, 0))]

    aligned_ref = [ref_frames[i] for i, _ in path]
    aligned_gen = [gen_frames[j] for _, j in path]
    total_cost = float(cost[n - 1, m - 1]) if np.isfinite(cost[n - 1, m - 1]) else float("inf")

    return aligned_ref, aligned_gen, total_cost, path


def _downsample_frame(frame: np.ndarray, size: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    vec = small.astype(np.float32).reshape(-1)
    vec -= float(vec.mean())
    norm = float(np.linalg.norm(vec)) + 1e-8
    vec /= norm
    return vec


def _coarse_alignment_candidates(
    ref_frames: List[np.ndarray],
    gen_frames: List[np.ndarray],
    max_offset: int,
    downsample: int = 64,
    max_samples: int = 16,
    top_k: int = 5,
) -> List[int]:
    if max_offset <= 0 or not ref_frames or not gen_frames:
        return [0]

    max_offset = min(max_offset, max(len(ref_frames), len(gen_frames)))
    ref_small = [_downsample_frame(f, downsample) for f in ref_frames]
    gen_small = [_downsample_frame(f, downsample) for f in gen_frames]

    len_ref = len(ref_small)
    len_gen = len(gen_small)

    scores: List[Tuple[float, int]] = []
    for offset in range(-max_offset, max_offset + 1):
        ref_start = max(0, offset)
        gen_start = max(0, -offset)
        overlap = min(len_ref - ref_start, len_gen - gen_start)
        if overlap <= 0:
            continue

        sample_count = min(max_samples, overlap)
        if sample_count <= 0:
            continue
        indices = np.linspace(0, overlap - 1, sample_count, dtype=int)

        corr_vals: List[float] = []
        for idx in indices:
            vec_ref = ref_small[ref_start + idx]
            vec_gen = gen_small[gen_start + idx]
            corr = float(np.dot(vec_ref, vec_gen))
            corr_vals.append(corr)
        if corr_vals:
            scores.append((float(np.mean(corr_vals)), offset))

    if not scores:
        return [0]

    scores.sort(reverse=True, key=lambda item: (item[0], -abs(item[1])))
    top_offsets = [offset for _, offset in scores[:top_k]]

    # Always include zero and clamp within range
    if 0 not in top_offsets:
        top_offsets.append(0)
    top_offsets = [int(max(-max_offset, min(max_offset, off))) for off in top_offsets]

    # add small neighborhood around zero as fallback
    fallback = list(range(-min(2, max_offset), min(2, max_offset) + 1))
    return sorted(set(top_offsets + fallback))


def read_video_frames(path: str, sample_every: int = 1, resize_to: Optional[Tuple[int, int]] = None, max_frames: Optional[int] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx % sample_every) == 0:
            if resize_to is not None:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def evaluate_pair(
    dataset_video: str,
    generated_video: str,
    sample_every: int = 1,
    max_frames: Optional[int] = None,
    use_clip: bool = True,
    use_lpips: bool = True,
    use_fsim: bool = True,
    use_vsi: bool = True,
    use_dists: bool = True,
    caption_text: Optional[str] = None,
    use_dino: bool = True,
    use_raft: bool = False,
    diagnostics_dir: Optional[str] = None,
    video_name: Optional[str] = None,
    variant_label: str = "pair",
    max_raft_visuals: int = 3,
    rel_root: Optional[str] = None,
    auto_align: bool = True,
    alignment_max_offset: int = 30,
    alignment_window: int = 3,
    alignment_offset_penalty: float = 0.05,
    raft_sample_indices: Optional[Sequence[int]] = None,
    use_llm: bool = False,
    llm_cache_dir: Optional[str] = None,
) -> Dict[str, any]:
    # Determine target size from dataset video (reference)
    cap_ref = cv2.VideoCapture(dataset_video)
    if not cap_ref.isOpened():
        raise RuntimeError(f"Failed to open reference video: {dataset_video}")
    w = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_ref.release()

    ref_frames = read_video_frames(dataset_video, sample_every=sample_every, resize_to=None, max_frames=max_frames)
    gen_frames = read_video_frames(generated_video, sample_every=sample_every, resize_to=(w, h), max_frames=max_frames)

    n = 0
    ref_offset = 0
    gen_offset = 0
    alignment_error: Optional[float] = None
    if auto_align and ref_frames and gen_frames:
        def _stack_frames(frames: List[np.ndarray], start: int, window: int) -> np.ndarray:
            slice_frames = frames[start:start + window]
            if len(slice_frames) < window:
                last = slice_frames[-1]
                slice_frames = slice_frames + [last] * (window - len(slice_frames))
            stacked = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in slice_frames], axis=0)
            return stacked.astype(np.float32) / 255.0

        def _stack_similarity(a: np.ndarray, b: np.ndarray) -> float:
            diff = a - b
            mse = float(np.mean(diff * diff))
            mae = float(np.mean(np.abs(diff)))
            flat_a = a.reshape(-1)
            flat_b = b.reshape(-1)
            denom = float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
            if denom > 1e-8:
                cos_sim = float(np.clip(np.dot(flat_a, flat_b) / denom, -1.0, 1.0))
            else:
                cos_sim = 0.0
            angular = 1.0 - cos_sim
            return mse + 0.5 * mae + 0.1 * angular

        max_offset = max(0, alignment_max_offset)
        window = max(1, alignment_window)
        best_error = float("inf")
        raw_error = float("inf")
        best_pair = (0, 0)
        search_offsets = _coarse_alignment_candidates(
            ref_frames,
            gen_frames,
            max_offset=max_offset,
            downsample=64,
            max_samples=16,
            top_k=5,
        )
        for offset in search_offsets:
            ref_start = max(0, offset)
            gen_start = max(0, -offset)
            if ref_start + window > len(ref_frames) or gen_start + window > len(gen_frames):
                continue
            ref_stack = _stack_frames(ref_frames, ref_start, window)
            gen_stack = _stack_frames(gen_frames, gen_start, window)
            error = _stack_similarity(ref_stack, gen_stack)
            penalized = error + abs(offset) * alignment_offset_penalty
            if penalized < best_error:
                best_error = penalized
                best_pair = (ref_start, gen_start)
                raw_error = error
        ref_offset, gen_offset = best_pair
        alignment_error = raw_error if math.isfinite(raw_error) else None
        if ref_offset or gen_offset:
            max_trim = min(len(ref_frames) - ref_offset, len(gen_frames) - gen_offset)
            ref_frames = ref_frames[ref_offset:ref_offset + max_trim]
            gen_frames = gen_frames[gen_offset:gen_offset + max_trim]
        n = min(len(ref_frames), len(gen_frames))
        ref_frames = ref_frames[:n]
        gen_frames = gen_frames[:n]
        ref_flow_frames = list(ref_frames)
        gen_flow_frames = list(gen_frames)
        if auto_align and ref_frames and gen_frames:
            ref_frames, gen_frames, dtw_cost, alignment_path = _dtw_align_frames(ref_flow_frames, gen_flow_frames)
            alignment_path = [(int(i), int(j)) for i, j in alignment_path]
            n = len(ref_frames)
        else:
            alignment_path = [(i, i) for i in range(n)]
            dtw_cost = 0.0
    else:
        n = min(len(ref_frames), len(gen_frames))
        ref_frames = ref_frames[:n]
        gen_frames = gen_frames[:n]
        ref_flow_frames = list(ref_frames)
        gen_flow_frames = list(gen_frames)
        alignment_path = [(int(i), int(i)) for i in range(n)]
        dtw_cost = 0.0

    psnrs: List[float] = []
    ssims: List[float] = []
    clip_sims: List[float] = []
    clip_caption_sims: List[float] = []
    lpips_vals: List[float] = []
    dino_sims: List[float] = []
    fsim_vals: List[float] = []
    vsi_vals: List[float] = []
    dists_vals: List[float] = []
    raft_epe_vals: List[float] = []
    raft_mag_diff_vals: List[float] = []
    raft_angle_vals: List[float] = []
    raft_ref_mag_vals: List[float] = []
    raft_gen_mag_vals: List[float] = []
    raft_visual_candidates: List[Tuple[float, int, Dict[str, Any]]] = []

    raft_opt = OptionalRAFT() if use_raft else None

    clip_opt = OptionalCLIP() if use_clip else None
    lpips_opt = OptionalLPIPS() if use_lpips else None
    dino_opt = OptionalDINO() if use_dino else None
    piq_opt = OptionalPIQMetrics(enable_fsim=use_fsim, enable_vsi=use_vsi, enable_dists=use_dists) if (use_fsim or use_vsi or use_dists) else None
    caption_provided = bool(caption_text and caption_text.strip())

    for i in range(n):
        f1 = ref_frames[i]
        f2 = gen_frames[i]
        psnrs.append(compute_psnr(f1, f2))
        try:
            ssims.append(compute_ssim(f1, f2))
        except cv2.error:
            ssims.append(float('nan'))

        if clip_opt:
            clip_sims.append(clip_opt.cosine(f1, f2))
            if caption_provided:
                clip_caption_sims.append(clip_opt.caption_score(caption_text, f2))
        if lpips_opt:
            lpips_vals.append(lpips_opt.distance(f1, f2))
        if dino_opt:
            dino_sims.append(dino_opt.cosine(f1, f2))
        if piq_opt:
            scores = piq_opt.compute(f1, f2)
            if "fsim" in scores:
                fsim_vals.append(scores["fsim"])
            if "vsi" in scores:
                vsi_vals.append(scores["vsi"])
            if "dists" in scores:
                dists_vals.append(scores["dists"])
        # RAFT metrics handled after loop using alignment path

    clean_psnrs = [float(x) for x in psnrs if not math.isnan(x)]
    clean_ssims = [float(x) for x in ssims if not math.isnan(x)]

    metrics: Dict[str, any] = {
        "num_frames": n,
        "psnr_mean": float(statistics.fmean(clean_psnrs)) if clean_psnrs else float('nan'),
        "psnr_std": float(statistics.pstdev(clean_psnrs)) if clean_psnrs else float('nan'),
        "ssim_mean": float(statistics.fmean(clean_ssims)) if clean_ssims else float('nan'),
        "ssim_std": float(statistics.pstdev(clean_ssims)) if clean_ssims else float('nan'),
        "per_frame": {
            "psnr": [float(x) for x in psnrs[:200]],
            "ssim": [float(x) for x in ssims[:200]],
        }
    }
    if auto_align:
        metrics["alignment_ref_offset"] = float(ref_offset)
        metrics["alignment_gen_offset"] = float(gen_offset)
        metrics["alignment_error"] = float(alignment_error) if alignment_error is not None else float('nan')
        metrics["alignment_path_cost"] = float(dtw_cost) if math.isfinite(dtw_cost) else float('nan')
        metrics["alignment_path_length"] = float(len(alignment_path))
        metrics["alignment_offset_penalty"] = float(alignment_offset_penalty)

    if clip_opt:
        if not clip_sims:
            raise RuntimeError("CLIP 指标计算失败：未生成有效的特征向量。")
        metrics.update({
            "clip_cosine_mean": float(statistics.fmean(clip_sims)),
            "clip_cosine_std": float(statistics.pstdev(clip_sims)) if len(clip_sims) > 1 else 0.0,
        })
        metrics["per_frame"]["clip_cosine"] = [float(x) for x in clip_sims[:200]]
        if caption_provided:
            if not clip_caption_sims:
                raise RuntimeError("CLIP 文本匹配指标计算失败：未生成有效的相似度分数。")
            metrics.update({
                "clip_caption_mean": float(statistics.fmean(clip_caption_sims)),
                "clip_caption_std": float(statistics.pstdev(clip_caption_sims)) if len(clip_caption_sims) > 1 else 0.0,
            })
            metrics["per_frame"]["clip_caption"] = [float(x) for x in clip_caption_sims[:200]]
        else:
            metrics["clip_caption_mean"] = float('nan')
            metrics["clip_caption_std"] = float('nan')
            metrics["per_frame"]["clip_caption"] = []

    if lpips_opt:
        if not lpips_vals:
            raise RuntimeError("LPIPS 指标计算失败：未生成有效的距离值。")
        metrics.update({
            "lpips_mean": float(statistics.fmean(lpips_vals)),
            "lpips_std": float(statistics.pstdev(lpips_vals)) if len(lpips_vals) > 1 else 0.0,
        })
        metrics["per_frame"]["lpips"] = [float(x) for x in lpips_vals[:200]]

    if dino_opt:
        if not dino_sims:
            raise RuntimeError("DINO 指标计算失败：未生成有效的特征向量。")
        metrics.update({
            "dino_similarity_mean": float(statistics.fmean(dino_sims)),
            "dino_similarity_std": float(statistics.pstdev(dino_sims)) if len(dino_sims) > 1 else 0.0,
        })
        metrics["per_frame"]["dino_similarity"] = [float(x) for x in dino_sims[:200]]

    if piq_opt:
        if use_fsim:
            if not fsim_vals:
                raise RuntimeError("FSIM 指标计算失败：未生成有效的相似度分数。")
            metrics.update({
                "fsim_mean": float(statistics.fmean(fsim_vals)),
                "fsim_std": float(statistics.pstdev(fsim_vals)) if len(fsim_vals) > 1 else 0.0,
            })
            metrics["per_frame"]["fsim"] = [float(x) for x in fsim_vals[:200]]
        if use_vsi:
            if not vsi_vals:
                raise RuntimeError("VSI 指标计算失败：未生成有效的相似度分数。")
            metrics.update({
                "vsi_mean": float(statistics.fmean(vsi_vals)),
                "vsi_std": float(statistics.pstdev(vsi_vals)) if len(vsi_vals) > 1 else 0.0,
            })
            metrics["per_frame"]["vsi"] = [float(x) for x in vsi_vals[:200]]
        if use_dists:
            if not dists_vals:
                raise RuntimeError("DISTS 指标计算失败：未生成有效的距离值。")
            metrics.update({
                "dists_mean": float(statistics.fmean(dists_vals)),
                "dists_std": float(statistics.pstdev(dists_vals)) if len(dists_vals) > 1 else 0.0,
            })
            metrics["per_frame"]["dists"] = [float(x) for x in dists_vals[:200]]

    if raft_opt:
        selected_indices: Optional[Set[int]] = None
        if raft_sample_indices is not None:
            normalized: Set[int] = set()
            for idx in raft_sample_indices:
                try:
                    value = int(idx)
                except (TypeError, ValueError):
                    continue
                if value >= 0:
                    normalized.add(value)
            if normalized:
                selected_indices = normalized

        raft_pairs: List[Tuple[int, int, int, int]] = []
        if alignment_path and len(alignment_path) >= 2:
            for (ref_idx_cur, gen_idx_cur), (ref_idx_next, gen_idx_next) in zip(alignment_path, alignment_path[1:]):
                if ref_idx_next == ref_idx_cur + 1 and gen_idx_next == gen_idx_cur + 1:
                    raft_pairs.append((ref_idx_cur, ref_idx_next, gen_idx_cur, gen_idx_next))
        else:
            max_len = min(len(ref_flow_frames), len(gen_flow_frames)) - 1
            if max_len > 0:
                for idx in range(max_len):
                    raft_pairs.append((idx, idx + 1, idx, idx + 1))

        if selected_indices is not None:
            raft_pairs = [pair for pair in raft_pairs if pair[0] in selected_indices]

        for pair_rank, (ref_idx_a, ref_idx_b, gen_idx_a, gen_idx_b) in enumerate(raft_pairs):
            frame_ref_a = ref_flow_frames[ref_idx_a]
            frame_ref_b = ref_flow_frames[ref_idx_b]
            frame_gen_a = gen_flow_frames[gen_idx_a]
            frame_gen_b = gen_flow_frames[gen_idx_b]

            flow_ref = raft_opt.compute_flow(frame_ref_a, frame_ref_b)
            flow_gen = raft_opt.compute_flow(frame_gen_a, frame_gen_b)
            diff = flow_ref - flow_gen
            epe_map = np.linalg.norm(diff, axis=2)
            mag_ref = np.linalg.norm(flow_ref, axis=2)
            mag_gen = np.linalg.norm(flow_gen, axis=2)
            mag_diff = np.abs(mag_ref - mag_gen)
            dot = np.sum(flow_ref * flow_gen, axis=2)
            denom = mag_ref * mag_gen
            angle_map = np.zeros_like(mag_ref, dtype=np.float32)
            mask = denom > 1e-6
            if np.any(mask):
                cos_vals = np.clip(dot[mask] / denom[mask], -1.0, 1.0)
                angle_map[mask] = np.degrees(np.arccos(cos_vals))
            valid_angles = angle_map[mask] if np.any(mask) else np.array([], dtype=np.float32)
            mean_epe = float(np.mean(epe_map))

            raft_epe_vals.append(mean_epe)
            raft_mag_diff_vals.append(float(np.mean(mag_diff)))
            if valid_angles.size:
                raft_angle_vals.append(float(np.mean(valid_angles)))
            else:
                raft_angle_vals.append(float("nan"))
            raft_ref_mag_vals.append(float(np.mean(mag_ref)))
            raft_gen_mag_vals.append(float(np.mean(mag_gen)))

            if max_raft_visuals > 0:
                sample_payload = {
                    "frame_index": int(ref_idx_a),
                    "epe": mean_epe,
                    "frame_ref": frame_ref_a.copy(),
                    "frame_gen": frame_gen_a.copy(),
                    "flow_ref": flow_ref,
                    "flow_gen": flow_gen,
                    "epe_map": epe_map,
                }
                entry = (mean_epe, pair_rank, sample_payload)
                if len(raft_visual_candidates) < max_raft_visuals:
                    heapq.heappush(raft_visual_candidates, entry)
                elif mean_epe > raft_visual_candidates[0][0]:
                    heapq.heapreplace(raft_visual_candidates, entry)

        metrics["raft_pair_count"] = len(raft_epe_vals)
        if raft_epe_vals:
            metrics.update({
                "raft_epe_mean": float(statistics.fmean(raft_epe_vals)),
                "raft_epe_median": float(statistics.median(raft_epe_vals)),
                "raft_epe_p95": float(np.percentile(raft_epe_vals, 95)),
            })
        else:
            metrics.update({
                "raft_epe_mean": float("nan"),
                "raft_epe_median": float("nan"),
                "raft_epe_p95": float("nan"),
            })
        if raft_mag_diff_vals:
            metrics.update({
                "raft_mag_diff_mean": float(statistics.fmean(raft_mag_diff_vals)),
                "raft_mag_diff_median": float(statistics.median(raft_mag_diff_vals)),
            })
        else:
            metrics.update({
                "raft_mag_diff_mean": float("nan"),
                "raft_mag_diff_median": float("nan"),
            })
        valid_angles = [x for x in raft_angle_vals if math.isfinite(x)]
        metrics["raft_angle_diff_deg_mean"] = float(statistics.fmean(valid_angles)) if valid_angles else float("nan")
        metrics["raft_ref_flow_mag_mean"] = float(statistics.fmean(raft_ref_mag_vals)) if raft_ref_mag_vals else float("nan")
        metrics["raft_gen_flow_mag_mean"] = float(statistics.fmean(raft_gen_mag_vals)) if raft_gen_mag_vals else float("nan")
        metrics["per_frame"]["raft_epe"] = [float(x) for x in raft_epe_vals[:200]]
        metrics["per_frame"]["raft_mag_diff"] = [float(x) for x in raft_mag_diff_vals[:200]]
        metrics["per_frame"]["raft_angle_deg"] = [float(x) for x in raft_angle_vals[:200]]
        metrics["per_frame"]["raft_ref_flow_mag"] = [float(x) for x in raft_ref_mag_vals[:200]]
        metrics["per_frame"]["raft_gen_flow_mag"] = [float(x) for x in raft_gen_mag_vals[:200]]
        if selected_indices is not None:
            metrics["raft_sample_indices"] = [int(idx) + 1 for idx in sorted(selected_indices)]
        if diagnostics_dir and raft_visual_candidates:
            selected_samples = [entry[2] for entry in sorted(raft_visual_candidates, key=lambda x: x[0], reverse=True)]
            visual_name = video_name or Path(generated_video).stem
            saved_visuals = _save_raft_visualizations(
                selected_samples,
                diagnostics_dir,
                visual_name,
                variant_label,
            )
            if saved_visuals and rel_root:
                rel_base = Path(rel_root).resolve()
                for item in saved_visuals:
                    abs_path = Path(item["path"]).resolve()
                    try:
                        item["path"] = str(abs_path.relative_to(rel_base))
                    except Exception:
                        item["path"] = str(abs_path)
            if saved_visuals:
                metrics["raft_visualizations"] = saved_visuals

    if use_llm:
        cache_dir_path: Optional[Path] = None
        if llm_cache_dir:
            cache_dir_path = Path(llm_cache_dir).resolve()
            cache_dir_path.mkdir(parents=True, exist_ok=True)
        try:
            llm_result = llm_evaluate_video_pair(
                Path(dataset_video),
                Path(generated_video),
                cache_dir=cache_dir_path,
            )
        except (LLMVideoEvaluationError, FileNotFoundError) as exc:
            llm_result = {"error": str(exc)}
        except Exception as exc:  # pragma: no cover - safety net for unexpected SDK errors
            llm_result = {"error": str(exc)}
        metrics["llm_similarity"] = llm_result
        if isinstance(llm_result, dict):
            score_value = llm_result.get("score")
            if isinstance(score_value, (int, float)) and math.isfinite(float(score_value)):
                metrics["llm_similarity_score"] = float(score_value)

    return metrics


def _single_metric_payload(metrics: Dict[str, Any], metric: str) -> Dict[str, Any]:
    per_frame = metrics.get("per_frame", {})
    payload: Dict[str, Any] = {
        "metric": metric,
        "num_frames": metrics.get("num_frames", 0),
    }
    if metric == "clip":
        payload.update({
            "clip_cosine_mean": metrics.get("clip_cosine_mean"),
            "clip_cosine_std": metrics.get("clip_cosine_std"),
            "clip_caption_mean": metrics.get("clip_caption_mean"),
            "clip_caption_std": metrics.get("clip_caption_std"),
            "per_frame": {
                "clip_cosine": per_frame.get("clip_cosine", [])[:200],
                "clip_caption": per_frame.get("clip_caption", [])[:200],
            },
        })
    elif metric == "lpips":
        payload.update({
            "lpips_mean": metrics.get("lpips_mean"),
            "lpips_std": metrics.get("lpips_std"),
            "per_frame": {
                "lpips": per_frame.get("lpips", [])[:200],
            },
        })
    elif metric == "dino":
        payload.update({
            "dino_similarity_mean": metrics.get("dino_similarity_mean"),
            "dino_similarity_std": metrics.get("dino_similarity_std"),
            "per_frame": {
                "dino_similarity": per_frame.get("dino_similarity", [])[:200],
            },
        })
    elif metric == "fsim":
        payload.update({
            "fsim_mean": metrics.get("fsim_mean"),
            "fsim_std": metrics.get("fsim_std"),
            "per_frame": {
                "fsim": per_frame.get("fsim", [])[:200],
            },
        })
    elif metric == "vsi":
        payload.update({
            "vsi_mean": metrics.get("vsi_mean"),
            "vsi_std": metrics.get("vsi_std"),
            "per_frame": {
                "vsi": per_frame.get("vsi", [])[:200],
            },
        })
    elif metric == "dists":
        payload.update({
            "dists_mean": metrics.get("dists_mean"),
            "dists_std": metrics.get("dists_std"),
            "per_frame": {
                "dists": per_frame.get("dists", [])[:200],
            },
        })
    elif metric == "raft":
        payload.update({
            "raft_pair_count": metrics.get("raft_pair_count"),
            "raft_epe_mean": metrics.get("raft_epe_mean"),
            "raft_epe_median": metrics.get("raft_epe_median"),
            "raft_epe_p95": metrics.get("raft_epe_p95"),
            "raft_mag_diff_mean": metrics.get("raft_mag_diff_mean"),
            "raft_mag_diff_median": metrics.get("raft_mag_diff_median"),
            "raft_angle_diff_deg_mean": metrics.get("raft_angle_diff_deg_mean"),
            "raft_ref_flow_mag_mean": metrics.get("raft_ref_flow_mag_mean"),
            "raft_gen_flow_mag_mean": metrics.get("raft_gen_flow_mag_mean"),
            "raft_visualizations": metrics.get("raft_visualizations", []),
            "per_frame": {
                "raft_epe": metrics.get("per_frame", {}).get("raft_epe", [])[:200],
                "raft_mag_diff": metrics.get("per_frame", {}).get("raft_mag_diff", [])[:200],
                "raft_angle_deg": metrics.get("per_frame", {}).get("raft_angle_deg", [])[:200],
                "raft_ref_flow_mag": metrics.get("per_frame", {}).get("raft_ref_flow_mag", [])[:200],
                "raft_gen_flow_mag": metrics.get("per_frame", {}).get("raft_gen_flow_mag", [])[:200],
            },
        })
    elif metric == "llm":
        llm_payload_raw = metrics.get("llm_similarity")
        llm_payload = llm_payload_raw if isinstance(llm_payload_raw, dict) else {}
        payload.update({
            "llm_similarity": llm_payload,
            "llm_similarity_score": metrics.get("llm_similarity_score"),
            "per_frame": {},
        })
    else:
        payload["per_frame"] = {}
    if "alignment_ref_offset" in metrics:
        payload["alignment_ref_offset"] = metrics.get("alignment_ref_offset")
        payload["alignment_gen_offset"] = metrics.get("alignment_gen_offset")
        payload["alignment_error"] = metrics.get("alignment_error")
    return payload


def _filter_metrics(metrics: Dict[str, Any], keep: Optional[Set[str]]) -> Dict[str, Any]:
    if not keep:
        return metrics

    filtered = dict(metrics)
    per_frame = dict(filtered.get("per_frame", {}))
    filtered["per_frame"] = per_frame

    heavy_keys = {
        "clip": ["clip_cosine_mean", "clip_cosine_std", "clip_caption_mean", "clip_caption_std"],
        "lpips": ["lpips_mean", "lpips_std"],
        "dino": ["dino_similarity_mean", "dino_similarity_std"],
        "fsim": ["fsim_mean", "fsim_std"],
        "vsi": ["vsi_mean", "vsi_std"],
        "dists": ["dists_mean", "dists_std"],
        "llm": ["llm_similarity_score", "llm_similarity"],
        "raft": [
            "raft_pair_count",
            "raft_epe_mean",
            "raft_epe_median",
            "raft_epe_p95",
            "raft_mag_diff_mean",
            "raft_mag_diff_median",
            "raft_angle_diff_deg_mean",
            "raft_ref_flow_mag_mean",
            "raft_gen_flow_mag_mean",
            "raft_visualizations",
        ],
    }
    per_frame_keys = {
        "clip": ["clip_cosine", "clip_caption"],
        "lpips": ["lpips"],
        "dino": ["dino_similarity"],
        "fsim": ["fsim"],
        "vsi": ["vsi"],
        "dists": ["dists"],
        "llm": [],
        "raft": ["raft_epe", "raft_mag_diff", "raft_angle_deg", "raft_ref_flow_mag", "raft_gen_flow_mag"],
    }

    for metric in ["clip", "lpips", "dino", "fsim", "vsi", "dists", "llm", "raft"]:
        if metric not in keep:
            for key in heavy_keys[metric]:
                filtered.pop(key, None)
            for key in per_frame_keys[metric]:
                per_frame.pop(key, None)
            if metric == "llm":
                filtered.pop("llm_similarity", None)

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Evaluate video metrics for generated videos vs dataset originals")
    parser.add_argument("--result-dir", help="Path to a single result directory (physics_prediction_YYYYMMDD_*)")
    parser.add_argument("--comparison-dir", help="Path to video_comparison_5 folder")
    parser.add_argument("--reference", help="Reference video path")
    parser.add_argument("--generated", help="Generated video path")
    parser.add_argument("--caption", help="Optional caption text used for CLIP caption similarity")
    parser.add_argument("--metrics", nargs="+", choices=["clip", "lpips", "dino", "fsim", "vsi", "dists", "raft", "llm", "all"], help="Limit metrics when using --reference/--generated")
    parser.add_argument("--output", help="Optional path to write metrics JSON")
    parser.add_argument("--sample-every", type=int, default=1, help="Sample every N frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames compared")
    parser.add_argument("--no-clip", action="store_true", help="Disable CLIP similarity to avoid heavy deps")
    parser.add_argument("--no-lpips", action="store_true", help="Disable LPIPS metric")
    parser.add_argument("--no-dino", action="store_true", help="Disable DINO similarity")
    parser.add_argument("--no-fsim", action="store_true", help="Disable FSIM metric")
    parser.add_argument("--no-vsi", action="store_true", help="Disable VSI metric")
    parser.add_argument("--no-dists", action="store_true", help="Disable DISTS metric")
    parser.add_argument("--caption-source", type=str, default="prediction", choices=["prediction","analysis","both"], help="Which text to use as caption for CLIP-Score")
    parser.add_argument("--enable-raft", action="store_true", help="Enable RAFT optical flow comparison metrics")
    parser.add_argument("--raft-visual-dir", help="Directory to write RAFT光流可视化（需开启 --enable-raft）")
    parser.add_argument("--raft-sample-indices", nargs="+", type=int, help="指定用于 RAFT 计算的帧索引（1-based）")
    parser.add_argument("--no-auto-align", action="store_true", help="Disable automatic temporal alignment")
    parser.add_argument("--alignment-max-offset", type=int, default=30, help="最大偏移量（采样后帧）用于自动对齐搜索")
    parser.add_argument("--alignment-window", type=int, default=3, help="滑动窗口大小（采样后帧）用于对齐评分")
    parser.add_argument("--alignment-offset-penalty", type=float, default=0.05, help="偏移惩罚系数（越大越倾向小偏移）")
    parser.add_argument("--enable-llm", action="store_true", help="Enable Gemini-based LLM similarity scoring")
    parser.add_argument("--llm-cache-dir", help="Directory used to cache Gemini LLM evaluation results")

    args = parser.parse_args()
    auto_align = not args.no_auto_align
    alignment_max_offset = max(0, args.alignment_max_offset)
    alignment_window = max(1, args.alignment_window)
    alignment_offset_penalty = max(0.0, args.alignment_offset_penalty)
    raft_sample_indices_cli: Optional[List[int]] = None
    if args.raft_sample_indices:
        raft_sample_indices_cli = [int(x) for x in args.raft_sample_indices if x is not None and int(x) > 0]
    raft_sample_indices_zero: Optional[List[int]] = None
    if raft_sample_indices_cli:
        raft_sample_indices_zero = [idx - 1 for idx in raft_sample_indices_cli if idx > 0]

    # Single pair evaluation mode
    if args.reference or args.generated:
        if not (args.reference and args.generated):
            parser.error("--reference and --generated must be used together")

        metrics_filter: Optional[Set[str]] = None
        if args.metrics:
            if "all" in args.metrics:
                metrics_filter = {"clip", "lpips", "dino", "fsim", "vsi", "dists", "raft", "llm"}
            else:
                metrics_filter = set(args.metrics)

        use_clip = (not args.no_clip) and (metrics_filter is None or "clip" in metrics_filter)
        use_lpips = (not args.no_lpips) and (metrics_filter is None or "lpips" in metrics_filter)
        use_dino = (not args.no_dino) and (metrics_filter is None or "dino" in metrics_filter)
        use_fsim = (not args.no_fsim) and (metrics_filter is None or "fsim" in metrics_filter)
        use_vsi = (not args.no_vsi) and (metrics_filter is None or "vsi" in metrics_filter)
        use_dists = (not args.no_dists) and (metrics_filter is None or "dists" in metrics_filter)
        use_raft = (args.enable_raft or (metrics_filter is not None and "raft" in metrics_filter))
        use_llm = args.enable_llm or (metrics_filter is not None and "llm" in metrics_filter)
        raft_sample_indices_zero: Optional[List[int]] = None
        if raft_sample_indices_cli:
            raft_sample_indices_zero = [idx - 1 for idx in raft_sample_indices_cli if idx > 0]

        diagnostics_dir = None
        if use_raft:
            if args.raft_visual_dir:
                diag_path = Path(args.raft_visual_dir).resolve()
            else:
                diag_path = Path(args.generated).resolve().parent / "raft_visuals"
            diag_path.mkdir(parents=True, exist_ok=True)
            diagnostics_dir = str(diag_path)
        video_label = Path(args.generated).stem

        metrics = evaluate_pair(
            args.reference,
            args.generated,
            sample_every=max(1, args.sample_every),
            max_frames=args.max_frames,
            use_clip=use_clip,
            use_lpips=use_lpips,
            use_fsim=use_fsim,
            use_vsi=use_vsi,
            use_dists=use_dists,
            caption_text=args.caption,
            use_dino=use_dino,
            use_raft=use_raft,
            diagnostics_dir=diagnostics_dir,
            video_name=video_label,
            variant_label="single",
            auto_align=auto_align,
            alignment_max_offset=alignment_max_offset,
            alignment_window=alignment_window,
            alignment_offset_penalty=alignment_offset_penalty,
            raft_sample_indices=raft_sample_indices_zero,
            use_llm=use_llm,
            llm_cache_dir=args.llm_cache_dir,
        )

        if metrics_filter and len(metrics_filter) == 1:
            result = _single_metric_payload(metrics, next(iter(metrics_filter)))
        else:
            result = _filter_metrics(metrics, metrics_filter)

        payload = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return

    if not args.result_dir or not args.comparison_dir:
        parser.error("Either provide --reference/--generated or both --result-dir and --comparison-dir")

    result_dir = Path(args.result_dir).resolve()
    comparison_dir = Path(args.comparison_dir).resolve()

    def resolve_artifact(name: str) -> Path:
        artifacts_root = result_dir / "artifacts"
        candidate = artifacts_root / name
        if candidate.exists() or artifacts_root.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        fallback = result_dir / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    predictions_dir = resolve_artifact("predictions")

    # Each subdirectory in comparison dir corresponds to a base name
    bases = [d.name for d in comparison_dir.iterdir() if d.is_dir()]
    bases.sort()

    all_written: List[str] = []

    use_fsim_global = not args.no_fsim
    use_vsi_global = not args.no_vsi
    use_dists_global = not args.no_dists
    use_raft_global = args.enable_raft
    use_llm_global = args.enable_llm
    raft_visuals_root: Optional[Path] = None
    if use_raft_global:
        if args.raft_visual_dir:
            raft_visuals_root = Path(args.raft_visual_dir).resolve()
        else:
            raft_visuals_root = resolve_artifact("raft_visuals")
        raft_visuals_root.mkdir(parents=True, exist_ok=True)

    for base in bases:
        sub = comparison_dir / base
        dataset_video = sub / f"{base}.mp4"
        gen_orig = sub / f"{base}_original_2d.mp4"
        gen_seg = sub / f"{base}_segmented_2d.mp4"

        if not (dataset_video.exists() and gen_orig.exists() and gen_seg.exists()):
            print(f"⚠️  Skip {base}: missing one of the required videos")
            continue

        diagnostics_original: Optional[Path] = None
        diagnostics_segmented: Optional[Path] = None
        if use_raft_global and raft_visuals_root:
            base_root = raft_visuals_root / base
            diagnostics_original = base_root / "original"
            diagnostics_segmented = base_root / "segmented"
            diagnostics_original.mkdir(parents=True, exist_ok=True)
            diagnostics_segmented.mkdir(parents=True, exist_ok=True)

        # Load caption text from predictions JSON if available
        caption_text = None
        try:
            pred_path = predictions_dir / f"{base}_prediction_original.json"
            if pred_path.exists():
                pred_json = json.load(pred_path.open('r', encoding='utf-8'))
                if args.caption_source == 'prediction':
                    caption_text = pred_json.get('prediction')
                elif args.caption_source == 'analysis':
                    caption_text = pred_json.get('analysis')
                else:
                    a = pred_json.get('analysis') or ''
                    p = pred_json.get('prediction') or ''
                    caption_text = (a + '\n' + p).strip()
        except Exception:
            caption_text = None

        print(f"Evaluating {base} (original vs generated-original)")
        m1 = evaluate_pair(
            str(dataset_video),
            str(gen_orig),
            sample_every=args.sample_every,
            max_frames=args.max_frames,
            use_clip=(not args.no_clip),
            use_lpips=(not args.no_lpips),
            use_fsim=use_fsim_global,
            use_vsi=use_vsi_global,
            use_dists=use_dists_global,
            caption_text=caption_text,
            use_dino=(not args.no_dino),
            use_raft=use_raft_global,
            diagnostics_dir=str(diagnostics_original) if diagnostics_original else None,
            video_name=base,
            variant_label="original",
            rel_root=str(result_dir),
            auto_align=auto_align,
            alignment_max_offset=alignment_max_offset,
            alignment_window=alignment_window,
            alignment_offset_penalty=alignment_offset_penalty,
            raft_sample_indices=raft_sample_indices_zero,
            use_llm=use_llm_global,
            llm_cache_dir=args.llm_cache_dir,
        )
        print(f"Evaluating {base} (original vs generated-segmented)")
        m2 = evaluate_pair(
            str(dataset_video),
            str(gen_seg),
            sample_every=args.sample_every,
            max_frames=args.max_frames,
            use_clip=(not args.no_clip),
            use_lpips=(not args.no_lpips),
            use_fsim=use_fsim_global,
            use_vsi=use_vsi_global,
            use_dists=use_dists_global,
            caption_text=caption_text,
            use_dino=(not args.no_dino),
            use_raft=use_raft_global,
            diagnostics_dir=str(diagnostics_segmented) if diagnostics_segmented else None,
            video_name=base,
            variant_label="segmented",
            rel_root=str(result_dir),
            auto_align=auto_align,
            alignment_max_offset=alignment_max_offset,
            alignment_window=alignment_window,
            alignment_offset_penalty=alignment_offset_penalty,
            raft_sample_indices=raft_sample_indices_zero,
            use_llm=use_llm_global,
            llm_cache_dir=args.llm_cache_dir,
        )

        out = {
            "video_name": base,
            "pairs": {
                "original_vs_generated_original": m1,
                "original_vs_generated_segmented": m2,
            }
        }

        out_path = predictions_dir / f"{base}_video_metrics.json"
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        all_written.append(out_path)
        print(f"✅ Wrote metrics: {out_path}")

    print(f"Done. Wrote {len(all_written)} metrics files.")


if __name__ == "__main__":
    main()
