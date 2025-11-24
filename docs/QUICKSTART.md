# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/train_ml_model.py --samples 200 --epochs 50
streamlit run app.py
```

## Interface Guide

### Sidebar Controls

**File Selection**
- Upload NIfTI file OR select from `data/labels/`

**Smoothing Algorithm**
- Laplacian (fast, causes shrinkage)
- Taubin (volume-preserving, better quality)

**Parameters**
- Iterations: 5-50 (more = smoother)
- Lambda: 0-1 (smoothing strength)
- ☑ ML-Optimized: AI predicts best settings

**Simplification**
- ☑ Apply QEM: Reduce triangles
- Target: 0.1-0.9 (keep 10%-90%)

**Analysis Options**
- Side-by-side comparison
- Volume tracking chart
- Hausdorff distance

**Batch Mode**
- Process all files in `data/`
- Export CSV results

### Workflow Examples

**Basic Smoothing**
1. Select file → Taubin → 20 iterations → Process

**ML-Powered**
1. Check "🤖 ML-Optimized Parameters"
2. AI analyzes mesh → Recommends settings
3. Process with predictions

**Comparison**
1. Enable "Side-by-Side Comparison"
2. Process → View before/after

**Export**
1. Process mesh
2. Click "Download Processed Mesh (.stl)"

## Keyboard Shortcuts

Plotly 3D Viewer:
- **Left drag**: Rotate
- **Scroll**: Zoom
- **Right drag**: Pan
- **Double-click**: Reset view

## Tips

- Taubin > Laplacian for medical models
- Try 15-25 iterations for most cases
- Use ML optimizer if unsure about parameters
- Enable Hausdorff to validate accuracy
- Batch mode for dataset-wide analysis

## 🔬 Advanced: Experiment Runner

For generating comprehensive results for reports (Ablation Studies), use the CLI tool:

```bash
# Run on all files in data/labels
python scripts/experiment_runner.py --data-dir ./data/labels --verbose

# Run on specific files
python scripts/experiment_runner.py --files ./data/sample.nii.gz

# Options
--no-artifacts    # Skip saving STL files (CSV only)
--output-dir      # Where to save results (default: experiments/results)
--log-path        # Path to CSV log
```
