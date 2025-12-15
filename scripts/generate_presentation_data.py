#!/usr/bin/env python3
"""
Generate actual experimental results for presentation.
Runs all 5 algorithms on BraTS samples and outputs comparison data.
"""

import sys
from pathlib import Path
import numpy as np
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.smoothing import taubin_smoothing, laplacian_smoothing
from src.algorithms.novel_algorithms_efficient import (
    geodesic_heat_smoothing,
    information_theoretic_smoothing,
    anisotropic_tensor_smoothing,
)


def load_mesh(filepath):
    """Load mesh from NIfTI segmentation file."""
    import nibabel as nib
    from skimage import measure
    
    img = nib.load(filepath)
    data = img.get_fdata()
    verts, faces, _, _ = measure.marching_cubes(data > 0, level=0.5)
    
    # Apply affine transform to get real-world coordinates
    affine = img.affine
    verts_homo = np.hstack([verts, np.ones((verts.shape[0], 1))])
    verts = (affine @ verts_homo.T).T[:, :3]
    
    return verts.astype(np.float32), faces.astype(np.int64)


def compute_volume(verts, faces):
    """Compute mesh volume using signed tetrahedron method."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return abs(np.sum(v0 * np.cross(v1, v2)) / 6.0)


def compute_aspect_ratio(verts, faces, sample_size=5000):
    """Compute mean aspect ratio of triangles."""
    idx = np.random.choice(len(faces), min(sample_size, len(faces)), replace=False)
    ars = []
    for i in idx:
        f = faces[i]
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        edges = sorted([np.linalg.norm(v1-v0), np.linalg.norm(v2-v1), np.linalg.norm(v0-v2)])
        if edges[0] > 1e-10:
            ars.append(edges[0] / edges[2])  # Ideal = 1.0
    return np.mean(ars)


def compute_smoothness(verts, faces):
    """Compute smoothness as Laplacian magnitude standard deviation."""
    from scipy import sparse
    N = len(verts)
    rows = np.concatenate([faces[:,0], faces[:,1], faces[:,2],
                          faces[:,1], faces[:,2], faces[:,0]])
    cols = np.concatenate([faces[:,1], faces[:,2], faces[:,0],
                          faces[:,0], faces[:,1], faces[:,2]])
    adj = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N)).tocsr()
    deg = np.array(adj.sum(axis=1)).flatten()
    W = sparse.diags(1.0/(deg+1e-10)) @ adj
    
    lap = verts - W @ verts
    return np.std(np.linalg.norm(lap, axis=1))


def compute_curvature_std(verts, faces):
    """Compute curvature standard deviation."""
    from scipy import sparse
    N = len(verts)
    rows = np.concatenate([faces[:,0], faces[:,1], faces[:,2],
                          faces[:,1], faces[:,2], faces[:,0]])
    cols = np.concatenate([faces[:,1], faces[:,2], faces[:,0],
                          faces[:,0], faces[:,1], faces[:,2]])
    adj = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N)).tocsr()
    deg = np.array(adj.sum(axis=1)).flatten()
    W = sparse.diags(1.0/(deg+1e-10)) @ adj
    
    lap = verts - W @ verts
    curv = np.linalg.norm(lap, axis=1)
    return np.std(curv)


def run_algorithm(name, verts, faces, iterations=10):
    """Run a smoothing algorithm and return results."""
    start = time.time()
    
    if name == "Laplacian":
        smoothed = laplacian_smoothing(verts, faces, iterations, lambda_val=0.5)
    elif name == "Taubin":
        smoothed = taubin_smoothing(verts, faces, iterations, lambda_val=0.5, mu_val=-0.53)
    elif name == "Geodesic Heat":
        smoothed, _ = geodesic_heat_smoothing(verts, faces, iterations=iterations)
    elif name == "Info-Theoretic":
        smoothed, _ = information_theoretic_smoothing(verts, faces, iterations=iterations)
    elif name == "Anisotropic":
        smoothed, _ = anisotropic_tensor_smoothing(verts, faces, iterations=iterations)
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    
    elapsed = time.time() - start
    return np.asarray(smoothed), elapsed


def main():
    print("="*70)
    print("GENERATING PRESENTATION DATA")
    print("="*70)
    
    # Get all segmentation files
    data_dir = project_root / "data" / "labels"
    seg_files = sorted(data_dir.glob("BraTS-GLI-*-seg.nii.gz"))[:5]  # Use 5 samples
    
    print(f"\nFound {len(seg_files)} samples")
    
    algorithms = ["Laplacian", "Taubin", "Geodesic Heat", "Info-Theoretic", "Anisotropic"]
    
    all_results = {
        "samples": [],
        "per_algorithm": {alg: [] for alg in algorithms},
        "summary": {}
    }
    
    for seg_file in seg_files:
        sample_name = seg_file.stem.replace("-seg", "")
        print(f"\n{'='*50}")
        print(f"Processing: {sample_name}")
        print(f"{'='*50}")
        
        # Load mesh
        verts, faces = load_mesh(seg_file)
        print(f"  Vertices: {len(verts):,}")
        print(f"  Faces: {len(faces):,}")
        
        # Compute original metrics
        orig_vol = compute_volume(verts, faces)
        orig_ar = compute_aspect_ratio(verts, faces)
        orig_smooth = compute_smoothness(verts, faces)
        orig_curv = compute_curvature_std(verts, faces)
        
        print(f"  Original volume: {orig_vol:,.1f} mm³")
        print(f"  Original AR: {orig_ar:.4f}")
        print(f"  Original smoothness: {orig_smooth:.6f}")
        
        sample_data = {
            "name": sample_name,
            "vertices": int(len(verts)),
            "faces": int(len(faces)),
            "orig_volume": float(orig_vol),
            "orig_ar": float(orig_ar),
            "orig_smoothness": float(orig_smooth),
            "orig_curvature_std": float(orig_curv),
            "algorithms": {}
        }
        
        # Run each algorithm
        for alg in algorithms:
            print(f"\n  Running {alg}...")
            try:
                smoothed, elapsed = run_algorithm(alg, verts, faces)
                
                # Compute metrics
                new_vol = compute_volume(smoothed, faces)
                new_ar = compute_aspect_ratio(smoothed, faces)
                new_smooth = compute_smoothness(smoothed, faces)
                new_curv = compute_curvature_std(smoothed, faces)
                
                vol_change = 100 * (new_vol - orig_vol) / orig_vol
                ar_change = 100 * (new_ar - orig_ar) / orig_ar
                smooth_reduction = 100 * (orig_smooth - new_smooth) / orig_smooth
                curv_change = 100 * (new_curv - orig_curv) / orig_curv
                
                result = {
                    "vol_change_pct": float(vol_change),
                    "ar_change_pct": float(ar_change),
                    "smooth_reduction_pct": float(smooth_reduction),
                    "curv_change_pct": float(curv_change),
                    "time_ms": float(elapsed * 1000),
                    "new_volume": float(new_vol),
                    "new_ar": float(new_ar)
                }
                
                sample_data["algorithms"][alg] = result
                all_results["per_algorithm"][alg].append(result)
                
                print(f"    Volume change: {vol_change:+.4f}%")
                print(f"    AR change: {ar_change:+.2f}%")
                print(f"    Smoothness reduction: {smooth_reduction:.2f}%")
                print(f"    Time: {elapsed*1000:.1f}ms")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                sample_data["algorithms"][alg] = {"error": str(e)}
        
        all_results["samples"].append(sample_data)
    
    # Compute summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for alg in algorithms:
        results = all_results["per_algorithm"][alg]
        valid = [r for r in results if "error" not in r]
        
        if valid:
            summary = {
                "vol_mean": float(np.mean([r["vol_change_pct"] for r in valid])),
                "vol_std": float(np.std([r["vol_change_pct"] for r in valid])),
                "ar_mean": float(np.mean([r["ar_change_pct"] for r in valid])),
                "ar_std": float(np.std([r["ar_change_pct"] for r in valid])),
                "smooth_mean": float(np.mean([r["smooth_reduction_pct"] for r in valid])),
                "smooth_std": float(np.std([r["smooth_reduction_pct"] for r in valid])),
                "time_mean": float(np.mean([r["time_ms"] for r in valid])),
                "time_std": float(np.std([r["time_ms"] for r in valid])),
                "n_samples": len(valid)
            }
            all_results["summary"][alg] = summary
            
            print(f"\n{alg}:")
            print(f"  Volume: {summary['vol_mean']:+.4f}% ± {summary['vol_std']:.4f}%")
            print(f"  AR: {summary['ar_mean']:+.2f}% ± {summary['ar_std']:.2f}%")
            print(f"  Smoothness: {summary['smooth_mean']:.2f}% ± {summary['smooth_std']:.2f}%")
            print(f"  Time: {summary['time_mean']:.1f}ms ± {summary['time_std']:.1f}ms")
    
    # Save results
    output_path = project_root / "outputs" / "presentation_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
