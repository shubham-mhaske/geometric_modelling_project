# High‑Fidelity Mesh Smoothing for Medical Brain MRI (BraTS)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Course project for **CSCE 645: Geometric Modeling (Texas A&M University)**.

This repository implements a **mask → surface mesh (Marching Cubes) → smoothing → evaluation** pipeline for brain tumor meshes derived from **BraTS** segmentations, with a Streamlit demo and reproducible evaluation scripts.

**Author:** Shubham Vikas Mhaske (Fall 2025)

**Repo:** https://github.com/shubham-mhaske/geometric-modeling-mesh-smoothing

## Deliverables (open in a browser)

- `website/index.html` — landing page
- `website/final_report.html` — final (self‑contained) report
- `academic_presentation.html` — slide deck (Reveal.js)

## What’s inside

### Algorithms evaluated (5)

- **Laplacian** (baseline; fast, but shrinks)
- **Taubin λ|μ** (baseline; volume-aware)
- **Geodesic Heat** *(this work)*
- **Information‑Theoretic** *(this work)*
- **Anisotropic Tensor** *(this work)*

### Metrics tracked

- **Volume change (%)**
- **Smoothness** (curvature variance reduction)
- **Triangle quality** (aspect ratio improvement)
- **Runtime** (ms)

## Key results (BraTS, n=20)

| Algorithm | Mean Volume Δ | Smoothness | Time (ms) | Recommended use |
|---|---:|---:|---:|---|
| **Taubin λ|μ** | **+0.056%** | 89.0% | 25 | volumetrics + good quality |
| Laplacian | −0.92% | **97.4%** | **17** | preview only (shrinks) |
| Geodesic Heat *(this work)* | −0.82% | 97.0% | 27 | strong smoothing, not volume‑safe |
| Info‑Theoretic *(this work)* | +0.042% | 84.4% | 44 | best “volume‑safe + smooth” balance |
| Anisotropic *(this work)* | −0.022% | 59.5% | 126 | maximum volume fidelity |

> The final report focuses on this **n=20 BraTS evaluation**.

## Setup

### 1) Environment

- Python **3.11+** recommended

Install dependencies:

- `pip install -r requirements.txt`
- (optional) editable install: `pip install -e .`

If you prefer Makefile shortcuts:

- `make install`

### 2) Data (BraTS via Synapse)

BraTS data is typically distributed via Synapse and may require an account + acceptance of terms.

This repo includes a downloader for a small sample tarball:

1. Create a local config file:
   - Copy `config.example.json` → `config.json`
   - Set `dataset_synapse_token` to your Synapse personal access token
2. Download:
   - `make download` (or `python scripts/download_data.py`)

Important:

- **Do not commit** `config.json` (it contains credentials). Use `config.example.json` for sharing.
- Large datasets are not stored in git; keep them under `data/` / `labels/` locally.

## Run the Streamlit demo

```bash
make run
```

Then open: `http://localhost:8501`

The app lets you:

- load a label volume / mesh sample (depending on what’s available locally)
- apply smoothing
- visualize before/after (Plotly 3D)
- export results

UI tips and controls: see `docs/QUICKSTART.md`.

## Reproduce evaluation + figures

Common scripts:

- `python scripts/comprehensive_eval.py` — evaluate algorithms over the local dataset
- `python scripts/evaluate_novel_methods.py` — focused runs for novel methods
- `python scripts/generate_final_figures.py` — generate figures used in the report
- `python scripts/generate_all_results_and_viz.py` — end‑to‑end results + visualization bundle

Outputs are written under `outputs/` (JSON + figures).

## ML parameter optimizer (optional)

There’s an optional ML component that predicts smoothing parameters:

- train: `python scripts/train_ml_model.py --samples 200 --epochs 50`
- docs: `docs/ML_GUIDE.md`

If no model is present, the system falls back to heuristics.

## Testing

```bash
make test
```

## Project layout

```
.
├── app.py                        # Streamlit demo
├── academic_presentation.html    # slide deck
├── website/                      # self-contained report site
├── src/
│   ├── algorithms/               # smoothing + metrics + processing
│   ├── ml/                       # ML optimizer
│   └── utils/                    # utilities
├── scripts/                      # download/eval/figure generation
├── outputs/                      # generated results + figures
├── data/                         # local data cache (not meant for git)
└── tests/
```

## Security / credentials

- Put secrets only in **local** `config.json` or `.env`
- Never paste tokens into issues, PRs, or commits

## Citation

```bibtex
@techreport{mhaske2025meshsmoothing,
  title        = {High-Fidelity Mesh Smoothing for Medical Brain MRI Data},
  author       = {Mhaske, Shubham Vikas},
  year         = {2025},
  institution  = {Texas A\&M University},
  type         = {Course Project},
  note         = {CSCE 645: Geometric Modeling}
}
```

## License

MIT (see `LICENSE` if present in the repository root; otherwise the badge reflects intended licensing).

---

**Status:** ✅ Final (December 2025)
