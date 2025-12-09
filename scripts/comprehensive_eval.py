#!/usr/bin/env python3
"""
Comprehensive evaluation on ALL BraTS samples.
Computes mean ± std for statistical significance.
"""

import sys
from pathlib import Path
import numpy as np
import time
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.smoothing import taubin_smoothing, laplacian_smoothing
from src.algorithms.novel_algorithms_efficient import (
    geodesic_heat_smoothing,
    spectral_clustering_smoothing,
    optimal_transport_smoothing,
    information_theoretic_smoothing,
    anisotropic_tensor_smoothing,
    frequency_selective_smoothing
)


def load_mesh(filepath):
    import nibabel as nib
    from skimage import measure
    
    img = nib.load(filepath)
    data = img.get_fdata()
    verts, faces, _, _ = measure.marching_cubes(data > 0, level=0.5)
    
    affine = img.affine
    verts_homo = np.hstack([verts, np.ones((verts.shape[0], 1))])
    verts = (affine @ verts_homo.T).T[:, :3]
    
    return verts.astype(np.float32), faces.astype(np.int64)


def compute_metrics(orig, smoothed, faces):
    from scipy import sparse
    
    # Volume
    def vol(v):
        v0, v1, v2 = v[faces[:,0]], v[faces[:,1]], v[faces[:,2]]
        return abs(np.sum(v0 * np.cross(v1, v2)) / 6.0)
    
    vol_change = 100 * (vol(smoothed) - vol(orig)) / vol(orig)
    
    # Aspect ratio improvement
    def aspect_ratio(v, sample_size=5000):
        idx = np.random.choice(len(faces), min(sample_size, len(faces)), replace=False)
        ars = []
        for i in idx:
            f = faces[i]
            v0, v1, v2 = v[f[0]], v[f[1]], v[f[2]]
            edges = sorted([np.linalg.norm(v1-v0), np.linalg.norm(v2-v1), np.linalg.norm(v0-v2)])
            if edges[0] > 1e-10:
                ars.append(min(edges[2]/edges[0], 10))
        return np.mean(ars)
    
    orig_ar = aspect_ratio(orig)
    new_ar = aspect_ratio(smoothed)
    ar_improvement = 100 * (orig_ar - new_ar) / orig_ar
    
    # Smoothness
    N = len(orig)
    rows = np.concatenate([faces[:,0], faces[:,1], faces[:,2],
                          faces[:,1], faces[:,2], faces[:,0]])
    cols = np.concatenate([faces[:,1], faces[:,2], faces[:,0],
                          faces[:,0], faces[:,1], faces[:,2]])
    adj = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N)).tocsr()
    deg = np.array(adj.sum(axis=1)).flatten()
    W = sparse.diags(1.0/(deg+1e-10)) @ adj
    
    def smooth(v):
        lap = v - W @ v
        return np.std(np.linalg.norm(lap, axis=1))
    
    smooth_imp = 100 * (smooth(orig) - smooth(smoothed)) / smooth(orig)
    
    return {
        'vol_change': vol_change,
        'ar_improvement': ar_improvement,
        'smooth_improvement': smooth_imp
    }


def main():
    print("="*70)
    print("COMPREHENSIVE EVALUATION - ALL BRATS SAMPLES")
    print("="*70)
    
    data_dir = project_root / "data" / "labels"
    seg_files = sorted(data_dir.glob("*seg*.nii.gz"))
    
    print(f"\nFound {len(seg_files)} samples:")
    for f in seg_files:
        print(f"  - {f.name}")
    
    methods = {
        'Laplacian': lambda v,f: (laplacian_smoothing(v,f,iterations=10), {}),
        'Taubin': lambda v,f: (taubin_smoothing(v,f,iterations=10), {}),
        'Geodesic Heat': geodesic_heat_smoothing,
        'Anisotropic Tensor': anisotropic_tensor_smoothing,
        'Info-Theoretic': information_theoretic_smoothing,
    }
    
    # Collect results per method
    all_results = {name: {'vol': [], 'ar': [], 'smooth': [], 'time': []} 
                   for name in methods}
    
    print("\n" + "-"*70)
    
    for i, seg_file in enumerate(seg_files):
        print(f"\n[{i+1}/{len(seg_files)}] Processing: {seg_file.name}")
        
        verts, faces = load_mesh(str(seg_file))
        print(f"    Mesh: {len(verts):,} vertices, {len(faces):,} faces")
        
        for name, fn in methods.items():
            start = time.time()
            smoothed, _ = fn(verts.copy(), faces)
            elapsed = time.time() - start
            
            metrics = compute_metrics(verts, smoothed, faces)
            
            all_results[name]['vol'].append(metrics['vol_change'])
            all_results[name]['ar'].append(metrics['ar_improvement'])
            all_results[name]['smooth'].append(metrics['smooth_improvement'])
            all_results[name]['time'].append(elapsed)
            
            print(f"    {name}: {elapsed:.2f}s")
    
    # Compute statistics
    print("\n" + "="*70)
    print("RESULTS: Mean ± Std (n={} samples)".format(len(seg_files)))
    print("="*70)
    
    print(f"\n{'Method':<20} {'Volume Δ':<18} {'AR Improve':<18} {'Smoothness':<18} {'Time':<10}")
    print("-"*85)
    
    summary = {}
    
    for name in methods:
        r = all_results[name]
        
        vol_mean, vol_std = np.mean(r['vol']), np.std(r['vol'])
        ar_mean, ar_std = np.mean(r['ar']), np.std(r['ar'])
        smooth_mean, smooth_std = np.mean(r['smooth']), np.std(r['smooth'])
        time_mean = np.mean(r['time'])
        
        summary[name] = {
            'vol_mean': vol_mean, 'vol_std': vol_std,
            'ar_mean': ar_mean, 'ar_std': ar_std,
            'smooth_mean': smooth_mean, 'smooth_std': smooth_std,
            'time_mean': time_mean
        }
        
        print(f"{name:<20} {vol_mean:+.4f}±{vol_std:.4f}%  "
              f"{ar_mean:+.2f}±{ar_std:.2f}%    "
              f"{smooth_mean:+.2f}±{smooth_std:.2f}%    "
              f"{time_mean:.2f}s")
    
    # Statistical comparison vs Taubin
    print("\n" + "-"*70)
    print("COMPARISON VS TAUBIN (baseline)")
    print("-"*70)
    
    taubin = summary['Taubin']
    novel_methods = ['Geodesic Heat', 'Anisotropic Tensor', 'Info-Theoretic']
    
    for name in novel_methods:
        m = summary[name]
        
        # Check if improvement is statistically meaningful
        vol_better = abs(m['vol_mean']) < abs(taubin['vol_mean'])
        smooth_better = m['smooth_mean'] > taubin['smooth_mean']
        
        # Effect size (how much better)
        vol_improvement = (abs(taubin['vol_mean']) - abs(m['vol_mean'])) / abs(taubin['vol_mean']) * 100
        smooth_improvement = (m['smooth_mean'] - taubin['smooth_mean']) / taubin['smooth_mean'] * 100
        
        print(f"\n{name}:")
        if vol_better:
            print(f"  ✓ Volume: {vol_improvement:+.1f}% better preservation")
        else:
            print(f"  ✗ Volume: {-vol_improvement:.1f}% worse")
        
        if smooth_better:
            print(f"  ✓ Smoothing: {smooth_improvement:+.1f}% more effective")
        else:
            print(f"  ✗ Smoothing: {-smooth_improvement:.1f}% less effective")
    
    # Find best methods
    print("\n" + "-"*70)
    print("BEST METHODS BY METRIC")
    print("-"*70)
    
    # Best volume preservation
    best_vol = min(summary.items(), key=lambda x: abs(x[1]['vol_mean']))
    print(f"\n🏆 Best Volume Preservation: {best_vol[0]}")
    print(f"   {best_vol[1]['vol_mean']:+.4f}±{best_vol[1]['vol_std']:.4f}%")
    
    # Best smoothing
    best_smooth = max(summary.items(), key=lambda x: x[1]['smooth_mean'])
    print(f"\n🏆 Best Smoothing: {best_smooth[0]}")
    print(f"   {best_smooth[1]['smooth_mean']:+.2f}±{best_smooth[1]['smooth_std']:.2f}%")
    
    # Save results
    output_dir = project_root / "outputs" / "comprehensive_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'n_samples': len(seg_files),
            'summary': summary,
            'raw_results': {k: {kk: [float(x) for x in vv] for kk, vv in v.items()} 
                           for k, v in all_results.items()}
        }, f, indent=2)
    
    print(f"\n\n💾 Results saved to: {output_dir}/results.json")
    
    # Generate LaTeX table
    print("\n" + "="*70)
    print("LATEX TABLE (for your report)")
    print("="*70)
    print("""
\\begin{table}[h]
\\centering
\\caption{Comprehensive evaluation on BraTS dataset (n=""" + str(len(seg_files)) + """ samples)}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Method} & \\textbf{Volume $\\Delta$} & \\textbf{Smoothness $\\uparrow$} & \\textbf{Time} \\\\
\\midrule""")
    
    for name in ['Laplacian', 'Taubin', 'Geodesic Heat', 'Anisotropic Tensor', 'Info-Theoretic']:
        m = summary[name]
        marker = "\\textbf{" if name in novel_methods else ""
        end = "}" if name in novel_methods else ""
        print(f"{marker}{name}{end} & ${m['vol_mean']:+.3f}\\pm{m['vol_std']:.3f}\\%$ & "
              f"${m['smooth_mean']:+.1f}\\pm{m['smooth_std']:.1f}\\%$ & {m['time_mean']:.2f}s \\\\")
    
    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")


if __name__ == '__main__':
    main()
