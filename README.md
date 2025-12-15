# Brain Tumor 3D Mesh Smoothing: Volume-Aware Algorithm Evaluation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Evaluation of mesh smoothing algorithms for medical brain tumor meshes (BraTS), emphasizing volume preservation, feature fidelity, and practical usage guidelines

**Course**: CSCE 645 - Geometric Modeling | **Texas A&M University**  
**Author**: Shubham Vikas Mhaske | **Term**: Fall 2025

## 🎯 Overview

This project evaluates **5 mesh smoothing algorithms** (2 classical baselines + 3 feature-aware methods) on **20 BraTS 2023 brain tumor meshes** spanning **5,990–118,970 vertices** (≈20× complexity variation), focusing on the clinically important trade-off between smoothness and volumetric accuracy.

## ✨ Key Findings (n=20 BraTS 2023)

- **Taubin λ-μ** (recommended for volumetrics): **+0.056% ± 0.047%** mean volume change with strong smoothing
- **Laplacian** (preview only): best smoothness but **−0.92%** mean volume shrinkage
- **Semantic-aware smoothing**: large boundary-preservation gains when segmentation labels are available

## 📊 Features

- **5 Smoothing Algorithms**: Taubin, Laplacian, Geodesic Heat, Info-Theoretic, Anisotropic Tensor
- **Evaluation Metrics (primary)**: Volume change, smoothness, aspect ratio improvement, processing time
- **Dataset**: BraTS 2023 (n=20)
- **Interactive Demo**: Streamlit app with real-time 3D visualization
- **Comprehensive Reports**: Academic paper, website, presentation materials

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Download data
python scripts/download_data.py

# Train ML model (optional)
python scripts/train_ml_model.py --samples 200 --epochs 50

# Run app
streamlit run app.py
```

## 📁 Repository Structure

```
project/
├── src/
│   ├── algorithms/          # Smoothing algorithms (5 methods)
│   ├── ml/                  # ML-based parameter optimizer
│   └── utils/              # Mesh processing utilities
├── scripts/
│   ├── comprehensive_eval.py      # 16-sample evaluation script
│   ├── generate_final_figures.py  # Results visualization
│   └── download_data.py          # Dataset downloader
├── docs/
│   ├── README.md              # Comprehensive technical documentation
│   ├── presentations/         # Final oral presentation materials
│   └── archive/              # Historical reports
├── documents/
│   ├── FINAL_PROJECT_REPORT.tex  # LaTeX academic paper
│   └── FINAL_PROJECT_REPORT.pdf  # Compiled paper
├── website/
│   ├── final_report.html     # Interactive HTML report
│   └── figures/              # Generated visualizations
├── app.py                    # Streamlit demo app
└── grad_project_demo.py      # Legacy demo (deprecated)
```

## Usage

1. Load mesh from `data/labels/`
2. Choose smoothing algorithm
3. Enable 🤖 ML optimizer (optional)
4. Apply QEM simplification (optional)
5. Export as STL

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd project
pip install -r requirements.txt

# 2. Download data (BraTS)
python scripts/download_data.py

# 3. Run interactive demo
streamlit run app.py

# 4. Run comprehensive evaluation
python scripts/comprehensive_eval.py

# 5. Generate figures
python scripts/generate_final_figures.py
```

## 🧪 Algorithms Evaluated

This repository contains implementations of:
- **Laplacian smoothing** (baseline)
- **Taubin λ-μ smoothing** (baseline, volume-aware)
- **Geodesic Heat smoothing** (feature-aware)
- **Information-Theoretic smoothing** (feature-aware)
- **Anisotropic Tensor smoothing** (feature-aware)

## 📈 Key Results Summary (n=20 BraTS 2023)

| Algorithm | Volume Δ | Smoothness | Time (ms) | Recommended Use |
|-----------|----------|------------|-----------|-----------------|
| **Taubin λ-μ** | **+0.056%** | 89.0% | 25 | Tumor volumetrics |
| Laplacian | −0.92% | **97.4%** | **17** | Real-time preview only |
| Geodesic Heat | −0.82% | 97.0% | 27 | Publication figures |
| Info-Theoretic | +0.042% | 84.4% | 44 | Feature preservation |
| Anisotropic Tensor | −0.022% | 59.5% | 126 | Extreme volume accuracy |

> Note: The codebase also contains exploratory utilities for other datasets/modalities, but the **final report and headline results** are based on the **n=20 BraTS evaluation** above.

## 📚 Documentation

- **`website/final_report.html`** — Final HTML report (submission-ready)
- **`academic_presentation.html`** — Slide deck for the 12-minute oral presentation
- **`SPEAKER_SCRIPT.md`** — Speaker notes (timed for 12 minutes + Q&A)
- **`docs/presentations/PRESENTATION.md`** — Presentation outline and Q&A prep

## 📖 Citation

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

---

**Status**: ✅ Complete | **Last Updated**: December 2025
