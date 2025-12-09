# Brain Tumor 3D Mesh Smoothing: Novel Algorithm Evaluation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Comprehensive evaluation of mesh smoothing algorithms for MRI/CT medical imaging with novel information-theoretic approaches

**Course**: CSCE 645 - Geometric Modeling | **Texas A&M University**  
**Author**: Shubham Vikas Mhaske | **Term**: Fall 2024

## 🎯 Overview

This project evaluates **5 mesh smoothing algorithms** (2 baseline + 3 novel) on **16 medical samples** (10 MRI brain tumors + 6 CT hemorrhages), discovering critical mesh-size dependencies in traditional methods and validating novel information-theoretic approaches for clinical applications.

## ✨ Key Findings

- **Info-Theoretic Smoothing**: 99.9% volume preservation across modalities (100.0% MRI, 99.8% CT)
- **Taubin Failure Mode**: 22.3% volume loss on small CT meshes (mesh-size dependent)
- **Geodesic Heat**: 68.9% smoothing improvement, competitive with Laplacian
- **Novel Algorithms**: Consistent across modalities (1.1% difference) vs baselines (20.8% difference)

## 📊 Features

- **5 Smoothing Algorithms**: Taubin, Laplacian, Geodesic Heat, Info-Theoretic, Anisotropic Tensor
- **5 Quality Metrics**: Smoothness, Volume Preservation, Mesh Quality, Displacement, Processing Time
- **Dual-Modality Validation**: MRI (n=10, 38,650 avg vertices) + CT (n=6, 13,365 avg vertices)
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

# 2. Download data (BraTS MRI + CT samples)
python scripts/download_data.py

# 3. Run interactive demo
streamlit run app.py

# 4. Run comprehensive evaluation
python scripts/comprehensive_eval.py

# 5. Generate figures
python scripts/generate_final_figures.py
```

## 🧪 Algorithms Evaluated

### Baseline Methods
**Taubin Smoothing**: Two-step volume-preserving (λ=0.5, μ=-0.53)
- MRI: 86.8% smoothing, 98.5% volume preservation
- CT: 72.1% smoothing, **77.7% volume preservation** ⚠️ (22.3% loss on small meshes)

**Laplacian Smoothing**: Simple averaging (λ=0.5)
- MRI: 70.0% smoothing, 99.5% volume preservation
- CT: 67.1% smoothing, 99.6% volume preservation

### Novel Methods (This Work)
**Geodesic Heat Smoothing**: Heat diffusion on surface geodesics
- MRI: **68.9% smoothing**, 99.3% volume preservation
- CT: 56.8% smoothing, 99.7% volume preservation

**Information-Theoretic Smoothing**: Entropy-guided vertex optimization
- MRI: 34.2% smoothing, **100.0% volume preservation** ✨
- CT: 19.7% smoothing, **99.8% volume preservation** ✨

**Anisotropic Tensor Smoothing**: Direction-aware feature preservation
- MRI: 17.2% smoothing, 99.8% volume preservation
- CT: 13.5% smoothing, 99.9% volume preservation

## 📈 Key Results (16 Samples: 10 MRI + 6 CT)

| Algorithm | Smoothing (MRI/CT) | Volume (MRI/CT) | Processing Time |
|-----------|-------------------|-----------------|-----------------|
| **Info-Theoretic** | 34.2% / 19.7% | **100.0%** / **99.8%** | 3.0s / 16.1s |
| **Geodesic Heat** | **68.9%** / 56.8% | 99.3% / 99.7% | 34.6s / 12.2s |
| Taubin | 86.8% / 72.1% | 98.5% / 77.7%* | 41ms / 7ms |
| Laplacian | 70.0% / 67.1% | 99.5% / 99.6% | 15ms / 5ms |
| Anisotropic | 17.2% / 13.5% | 99.8% / 99.9% | 21.8s / 8.5s |

*⚠️ Critical: Taubin shows 22.3% volume loss on small CT meshes (mesh-size dependency)

### Clinical Implications
- **Info-Theoretic**: Ideal for clinical workflows requiring strict volume preservation (RECIST criteria)
- **Geodesic Heat**: Best smoothing quality while maintaining 99%+ volume
- **Taubin**: Fast but unreliable on small/irregular meshes (CT, surgical resections)

## 📚 Documentation

- **[docs/README.md](docs/README.md)** - Comprehensive technical documentation with full results
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - User interface guide for Streamlit app
- **[docs/presentations/PRESENTATION.md](docs/presentations/PRESENTATION.md)** - Final oral presentation (15 min)
- **[documents/FINAL_PROJECT_REPORT.pdf](documents/FINAL_PROJECT_REPORT.pdf)** - Academic paper (LaTeX)
- **[website/final_report.html](website/final_report.html)** - Interactive HTML report with visualizations

## 🔬 Usage Examples

### Run Interactive Demo
```python
streamlit run app.py
# 1. Select algorithm from sidebar
# 2. Upload NIfTI file or use sample data
# 3. Adjust parameters
# 4. View real-time 3D visualization
# 5. Download results as STL
```

### Programmatic API
```python
from src.algorithms.novel_algorithms import (
    information_theoretic_smoothing,
    geodesic_heat_smoothing,
    anisotropic_tensor_smoothing
)
from src.algorithms.smoothing import taubin_smoothing, laplacian_smoothing

# Best volume preservation (clinical use)
smoothed_verts, info = information_theoretic_smoothing(vertices, faces, iterations=10)
print(f"Volume preservation: {info['volume_preservation']:.1%}")

# Best smoothing quality
smoothed_verts, info = geodesic_heat_smoothing(vertices, faces, timestep=0.1)
print(f"Smoothness improvement: {info['smoothness_improvement']:.1%}")

# Fast baseline (beware small meshes)
smoothed_verts, info = taubin_smoothing(vertices, faces, iterations=10)
```

## 🔧 Development

```bash
# Run tests
python tests/test_pipeline.py

# Format code
black src/ scripts/ tests/

# Type checking
mypy src/

# Build documentation
cd documents && pdflatex FINAL_PROJECT_REPORT.tex
```

## 📖 Citation

```bibtex
@techreport{mhaske2024meshsmoothing,
  title={Novel Mesh Smoothing Algorithms for Medical Imaging: 
         A Comprehensive Evaluation on MRI and CT Data},
  author={Mhaske, Shubham Vikas},
  year={2024},
  institution={Texas A&M University},
  type={Course Project},
  note={CSCE 645: Geometric Modeling}
}
```

## 📊 Dataset

- **MRI**: BraTS 2021/2023 Brain Tumor Segmentation (10 samples, 38,650 avg vertices)
- **CT**: Intracranial Hemorrhage Detection (6 samples, 13,365 avg vertices)
- **Source**: Synapse Medical Imaging Platform (syn64952532)
- **License**: Academic research only

## 🏆 Project Outcomes

- ✅ Novel discovery: Taubin mesh-size dependency (never reported)
- ✅ Info-Theoretic: 99.9% volume preservation across modalities
- ✅ Comprehensive 16-sample validation (MRI + CT)
- ✅ Application-specific guidelines for clinical adoption
- ✅ Production-ready Streamlit demo with <1s inference

---

**Status**: ✅ Complete | **Last Updated**: December 2024 | **Contact**: [GitHub](https://github.com/shubham-mhaske)
