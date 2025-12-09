# Brain Tumor 3D Mesh Improvement Pipeline

**CSCE 645: Geometric Modeling - Final Project**  
**Author**: Shubham Vikas Mhaske | Texas A&M University

High-fidelity mesh improvement for MRI-derived brain tumor models with ML-based parameter optimization.

## 🎯 Overview

Transform raw, noisy 3D meshes from Marching Cubes into smooth, high-quality anatomical models suitable for surgical planning.

### Key Features

- **Smoothing Algorithms**: Laplacian & Taubin (volume-preserving)
- **QEM Simplification**: Intelligent triangle reduction
- **ML Parameter Optimizer**: Neural network predicts optimal settings
- **Interactive 3D Visualization**: Plotly-based mesh viewer
- **Batch Processing**: Process multiple files with CSV export
- **Quality Metrics**: Hausdorff distance, volume tracking, aspect ratios

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download sample data (~82MB)
python scripts/download_data.py

# Train ML model (optional, 2-5 min)
python scripts/train_ml_model.py --samples 200 --epochs 50

# Launch app
streamlit run app.py
```

Visit `http://localhost:8501`

## 📁 Structure

```
project/
├── src/
│   ├── algorithms/      # Smoothing, simplification, metrics
│   └── ml/              # Neural network optimizer
├── scripts/             # Training & data utilities
├── tests/               # Automated tests
├── app.py              # Main Streamlit interface
└── docs/               # Documentation
```

## 🔬 Usage

### Basic Workflow
1. Select mesh from `data/labels/`
2. Choose smoothing algorithm (Laplacian/Taubin)
3. Set iterations (5-50) or enable **🤖 ML-Optimized Parameters**
4. Apply QEM simplification (optional)
5. Export processed mesh as STL

### ML-Based Parameter Selection
1. Enable **"🤖 ML-Optimized Parameters"** checkbox
2. AI analyzes mesh geometry (12 features)
3. Predicts: algorithm, iterations, lambda, confidence
4. Apply recommendations automatically

### Batch Processing
1. Enable **"Batch Processing Mode"**
2. Configure settings
3. Process all files in `data/`
4. Download CSV results

### 🧪 Experiment Runner (CLI)
For rigorous ablation studies and generating results for reports:

```bash
# Run full ablation study on all data
python scripts/experiment_runner.py \
  --data-dir ./data/labels \
  --output-dir ./experiments/full_study \
  --log-path ./experiments/full_study/results.csv

# Run without saving mesh artifacts (CSV only)
python scripts/experiment_runner.py \
  --data-dir ./data/labels \
  --log-path ./experiments/results.csv \
  --no-artifacts
```

## 📊 Algorithms

**Laplacian Smoothing**: `v_new = (1-λ)v + λ·avg(neighbors)`
- Fast, simple
- Causes volume shrinkage

**Taubin Smoothing**: Two-step (λ=0.5, μ=-0.53)
- Volume-preserving
- Feature-preserving

**Bilateral Smoothing**: Feature-preserving filter
- Respects edges and high-curvature regions
- Weighs neighbors by spatial distance AND normal similarity

**Guided/Adaptive Smoothing**: Curvature-aware
- Smooths more in flat regions, less at features
- Per-vertex adaptive lambda

**Constrained Smoothing**: Landmark preservation
- Keeps user-specified vertices fixed during smoothing
- Ideal for preserving anatomical landmarks

**QEM Simplification**: Quadric Error Metrics
- Preserves shape while reducing triangles
- Edge collapse with minimal error

**Curvature Metrics**: Mean & Gaussian curvature
- Discrete Laplace-Beltrami operator for mean curvature
- Angle defect for Gaussian curvature
- Curvature preservation error analysis

**ML Optimizer**: PyTorch neural network
- 12D feature extraction (size, edges, quality, shape)
- Multi-task learning (algorithm + iterations + lambda)
- Trained on synthetic/real medical data

## 📈 Results

Comprehensive evaluation across **16 samples (10 MRI + 6 CT)** with 5 metrics:

### MRI Brain Tumors (n=10, mean 38,650 vertices)
| Algorithm | Smoothness | Volume Pres. | Quality | Displacement | Time |
|-----------|------------|--------------|---------|--------------|------|
| **Taubin** | +86.8% | 98.5% | 0.825 | 0.518mm | 41ms |
| **Laplacian** | +70.0% | 99.8% | 0.732 | 0.248mm | 22ms |
| **Geodesic Heat** | +68.9% | 99.3% | 0.803 | 0.387mm | 9,678ms |
| **Info-Theoretic** | +34.2% | **100.0%** | 0.636 | 0.107mm | 15,976ms |
| **Anisotropic** | +16.6% | 99.9% | 0.654 | **0.070mm** | 35,434ms |

### CT Hemorrhage (n=6, mean 13,365 vertices)
| Algorithm | Smoothness | Volume Pres. | Quality | Displacement | Time |
|-----------|------------|--------------|---------|--------------|------|
| **Taubin** | +72.1% | ⚠️ 77.7% | 0.592 | 1.076mm | 13ms |
| **Laplacian** | +45.6% | 94.3% | 0.565 | 0.381mm | 7ms |
| **Geodesic Heat** | +5.3% | 88.1% | 0.596 | 0.501mm | 3,333ms |
| **Info-Theoretic** | +19.7% | **99.8%** | 0.469 | 0.142mm | 5,473ms |
| **Anisotropic** | +5.5% | 98.0% | 0.443 | **0.068mm** | 12,085ms |

**Key Findings:**
- **Info-Theoretic**: 99.9% overall volume preservation (100.0% MRI, 99.8% CT)
- **Geodesic Heat**: 68.9% smoothing on MRI (matches Laplacian 70.0%)
- **⚠️ Taubin Limitation**: 22.3% volume loss on small CT meshes (mesh-size dependency)
- **Novel Algorithms**: 236-476× slower but consistent across modalities
- **Baselines**: Real-time (7-41ms) for interactive visualization

## 🧪 Testing

```bash
# Run all tests
python tests/test_pipeline.py

# Or use Makefile
make test
make run
```

## 📖 Documentation

- `docs/QUICKSTART.md` - User interface guide
- `docs/ML_GUIDE.md` - ML features & training

## 🎓 Academic Context

**Course**: CSCE 645 - Geometric Modeling  
**Dataset**: BraTS 2023 (Brain Tumor Segmentation)  
**Key References**:
- Lorensen & Cline (1987) - Marching Cubes
- Taubin (1995) - Volume-preserving smoothing
- Garland & Heckbert (1997) - QEM simplification

## 📝 Citation

```bibtex
@software{mhaske2025meshpipeline,
  title={High-Fidelity Mesh Improvement for MRI-Derived Anatomical Models},
  author={Mhaske, Shubham Vikas},
  year={2025},
  course={CSCE 645: Geometric Modeling},
  institution={Texas A&M University}
}
```

---

**Status**: ✅ Production Ready | Last Updated: November 23, 2025
