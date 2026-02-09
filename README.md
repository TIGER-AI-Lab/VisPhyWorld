# VisPhyWorld: Probing Physical Reasoning via Code-Driven Video Reconstruction

[🌐 Project Page](https://tiger-ai-lab.github.io/VisPhyWorld/) · [📄 Preprint](TODO) · [🤗 HF Datasets](https://huggingface.co/collections/TIGER-Lab/visphyworld)

VisPhyWorld evaluates physical reasoning by turning a model's prediction into an **executable hypothesis**: given video observations, an LLM generates simulation code that is re-simulated under a fixed physics engine to predict future motion.

<p align="center">
  <img src="docs/static/images/visphyworld2.png" alt="VisPhyWorld overview" width="90%" />
</p>

- **Executable, interpretable reasoning:** models must output runnable simulation code, making failures attributable to explicit state/dynamics rather than opaque text answers.
- **Code-driven re-simulation protocol:** evaluation compares re-simulated motion against ground truth, exposing whether a model truly captures dynamics (not just appearance).
- **Unified, diagnostic metrics:** supports multi-family metrics (reconstruction, motion, holistic/LLM-based, etc.) so gaps become measurable and debuggable.
- **2D + 3D stress testing:** VisPhyBench includes both 2D scenes and perspective-rendered 3D scenes to challenge contact/occlusion and depth-dependent interactions.
- **Structured scene annotations for LLMs:** each sample ships with first-frame detection JSON (objects, colors, positions, bboxes, sizes) to ground code generation.

## 📰 News

- **2026-02-08:** VisPhyWorld and VisPhyBench are now publicly released!

## 📌 Introduction

Evaluating whether Multimodal Large Language Models (MLLMs) genuinely reason about physical dynamics remains challenging.
Most existing benchmarks rely on recognition-style protocols such as Visual Question Answering (VQA) and Violation of Expectation (VoE), which can often be answered without committing to an explicit, testable physical hypothesis. We propose **VisPhyWorld**, an execution-based framework that evaluates physical reasoning by requiring models to generate executable simulator code from visual observations. By producing runnable code, the inferred world representation is directly inspectable, editable, and falsifiable. This separates physical reasoning from rendering. Building on this framework, we introduce **VisPhyBench**, comprising 209 evaluation scenes derived from 108 physical templates and a systematic protocol that evaluates how well models reconstruct appearance and reproduce physically plausible motion. Our pipeline produces valid reconstructed videos in 97.7\% on the benchmark. Experiments show that while state-of-the-art MLLMs achieve strong semantic scene understanding, they struggle to accurately infer physical parameters and to simulate consistent physical dynamics.

<p align="center">
  <img src="docs/static/images/teaser.png" alt="VisPhyWorld teaser" width="90%" />
</p>

## 🛠️ Setup

### Prerequisites

- Miniconda / Anaconda installed (`conda` available in your shell).
- Internet access (first run may download model weights and a Chromium build for Puppeteer).

### 1) Create a conda environment

Recommended: Python 3.10.

```bash
conda create -n visphyworld python=3.10 -y
conda activate visphyworld
```

### 2) Install runtime tools (FFmpeg + Node.js)

The renderers encode videos via the `ffmpeg` CLI, and the Three.js/P5.js/SVG pipelines rely on Puppeteer (Node.js).

```bash
conda install -c conda-forge ffmpeg nodejs=18 -y
```

Verify:

```bash
ffmpeg -version
node -v
npm -v
```

### 3) Install Python dependencies

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

### 4) Install Node dependencies (Puppeteer)

Run this at the repo root (where `package.json` is):

```bash
npm ci
```

### 5) Download the dataset into `data/`

This repo expects the Hugging Face dataset `TIGER-Lab/VisPhyBench-Data` to live under `./data/`.


### 6) Smoke test (run one sample)

From the repo root:

```bash
python run_physics_prediction.py \
  --data-dir data \
  --split sub \
  --video task00000_000 \
  --engine threejs \
  --jobs 1
```


## 🚀 Run Physics Prediction

This repo expects the Hugging Face dataset `TIGER-Lab/VisPhyBench-Data` to be downloaded into `data/`:

Example (default split is `sub`, default model is `gpt-5`):

```bash
python run_physics_prediction.py \
  --data-dir data \
  --split sub \
  --video task00000_000 \
  --engine threejs
```

## 📊 Evaluate a Run

```bash
python create_evaluation_html.py \
  output/physics_prediction_YYYYMMDD_HHMMSS_* \
  --dataset-dir data \
  --split sub \
  --output output/physics_prediction_YYYYMMDD_HHMMSS_*/evaluation.html
```

### Batch evaluate multiple runs

Evaluate every run directory under `output/` (one `evaluation.html` per run):

```bash
for run_dir in output/physics_prediction_*; do
  python create_evaluation_html.py "$run_dir" \
    --dataset-dir data \
    --split sub \
    --output "$run_dir/evaluation.html" \
    --jobs 4 \
    --sample-every 3 \
    --enable-heavy-metrics \
    --strict
done
```

## 📚 Citation

### BibTeX

```bibtex
@misc{visphyworld,
  title        = {VisPhyWorld},
  howpublished = {\\url{https://github.com/TIGER-AI-Lab/VisPhyWorld}},
  url          = {https://github.com/TIGER-AI-Lab/VisPhyWorld},
  note         = {GitHub repository},
}
```
