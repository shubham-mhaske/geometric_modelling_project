# High-Fidelity Mesh Improvement Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Advanced geometric processing for medical mesh improvement with ML-based parameter optimization

**Course**: CSCE 645 - Geometric Modeling | **Author**: Shubham Vikas Mhaske | Texas A&M University

## Features

- **Smoothing**: Laplacian & Taubin (volume-preserving)
- **Simplification**: QEM mesh reduction
- **ML Optimizer**: Neural network predicts optimal parameters
- **Visualization**: Interactive 3D mesh viewer
- **Batch Processing**: Process multiple files with CSV export
- **Metrics**: Hausdorff distance, volume tracking, quality analysis

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

## Structure

```
project/
├── src/              # Core algorithms & ML
├── scripts/          # Training & utilities
├── tests/            # Automated tests
├── app.py           # Main interface
└── docs/            # Documentation
```

## Usage

1. Load mesh from `data/labels/`
2. Choose smoothing algorithm
3. Enable 🤖 ML optimizer (optional)
4. Apply QEM simplification (optional)
5. Export as STL

## Algorithms

**Laplacian**: `v_new = (1-λ)v + λ·avg(neighbors)` - Fast, simple  
**Taubin**: Two-step (λ=0.5, μ=-0.53) - Volume-preserving  
**QEM**: Quadric error minimization - Shape-preserving reduction  
**ML**: PyTorch neural network - 12D feature extraction → optimal params

## Results

- Original: 50k triangles, noisy
- Taubin (20 iter): Smooth, <0.5% volume change
- QEM (50%): 25k triangles, Hausdorff <2mm
- Processing: 3-5 sec/mesh, ML inference <50ms

## Documentation

- `docs/README.md` - Full technical documentation
- `docs/QUICKSTART.md` - User interface guide
- `docs/ML_GUIDE.md` - ML features & training

## Testing

```bash
python tests/test_pipeline.py
# or
make test
```

## Citation

```bibtex
@software{mhaske2025meshpipeline,
  title={High-Fidelity Mesh Improvement for MRI-Derived Anatomical Models},
  author={Mhaske, Shubham Vikas},
  year={2025},
  institution={Texas A&M University}
}
```

## Dataset

BraTS 2023 Brain Tumor Segmentation - Synapse (syn64952532)

---

**Status**: ✅ Production Ready | November 2025
