#!/usr/bin/env python3
"""
Generate comprehensive results on ALL available BraTS data
and create high-quality 3D visualizations for presentation.
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


def compute_vertex_normals(verts, faces):
    """Compute per-vertex normals for lighting."""
    # Compute face normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10)
    
    # Accumulate to vertices
    vertex_normals = np.zeros_like(verts)
    for i, f in enumerate(faces):
        vertex_normals[f[0]] += face_normals[i]
        vertex_normals[f[1]] += face_normals[i]
        vertex_normals[f[2]] += face_normals[i]
    
    vertex_normals = vertex_normals / (np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-10)
    return vertex_normals


def create_comparison_visualization(orig_verts, smoothed_verts, faces, title, output_path, algorithm_name):
    """Create side-by-side 3D visualization with proper lighting."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Compute normals for lighting effect
    orig_normals = compute_vertex_normals(orig_verts, faces)
    smooth_normals = compute_vertex_normals(smoothed_verts, faces)
    
    # Light direction (from upper right)
    light_dir = np.array([1, 1, 2])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Compute lighting intensity (Lambertian)
    orig_intensity = np.clip(np.dot(orig_normals, light_dir), 0.2, 1.0)
    smooth_intensity = np.clip(np.dot(smooth_normals, light_dir), 0.2, 1.0)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}]],
        subplot_titles=[f'Original Mesh', f'After {algorithm_name} Smoothing'],
        horizontal_spacing=0.02
    )
    
    # Original mesh
    fig.add_trace(
        go.Mesh3d(
            x=orig_verts[:, 0],
            y=orig_verts[:, 1],
            z=orig_verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=orig_intensity,
            colorscale='Blues',
            showscale=False,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(x=100, y=200, z=300),
            flatshading=False
        ),
        row=1, col=1
    )
    
    # Smoothed mesh
    fig.add_trace(
        go.Mesh3d(
            x=smoothed_verts[:, 0],
            y=smoothed_verts[:, 1],
            z=smoothed_verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=smooth_intensity,
            colorscale='Oranges',
            showscale=False,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(x=100, y=200, z=300),
            flatshading=False
        ),
        row=1, col=2
    )
    
    # Update layout
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.0)
    )
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, family='Arial')
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera,
            aspectmode='data',
            bgcolor='white'
        ),
        scene2=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera,
            aspectmode='data',
            bgcolor='white'
        ),
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=50, b=0),
        width=1400,
        height=600
    )
    
    fig.write_html(output_path)
    print(f"  Saved: {output_path}")
    return fig


def create_multi_algorithm_comparison(orig_verts, smoothed_dict, faces, sample_name, output_path):
    """Create comparison of all algorithms for one sample."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    algorithms = list(smoothed_dict.keys())
    n_algs = len(algorithms)
    
    # Create figure with subplots - original + all algorithms
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}, {'type': 'mesh3d'}],
               [{'type': 'mesh3d'}, {'type': 'mesh3d'}, {'type': 'mesh3d'}]],
        subplot_titles=['Original'] + algorithms,
        horizontal_spacing=0.02,
        vertical_spacing=0.08
    )
    
    # Light direction
    light_dir = np.array([1, 1, 2])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Color scales for each
    colors = ['Greys', 'Blues', 'Greens', 'Oranges', 'Reds', 'Purples']
    
    # Original mesh
    orig_normals = compute_vertex_normals(orig_verts, faces)
    orig_intensity = np.clip(np.dot(orig_normals, light_dir), 0.2, 1.0)
    
    fig.add_trace(
        go.Mesh3d(
            x=orig_verts[:, 0], y=orig_verts[:, 1], z=orig_verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=orig_intensity, colorscale=colors[0], showscale=False,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.5),
            lightposition=dict(x=100, y=200, z=300)
        ),
        row=1, col=1
    )
    
    # Add each algorithm
    positions = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    for idx, (alg, smoothed) in enumerate(smoothed_dict.items()):
        normals = compute_vertex_normals(smoothed, faces)
        intensity = np.clip(np.dot(normals, light_dir), 0.2, 1.0)
        row, col = positions[idx]
        
        fig.add_trace(
            go.Mesh3d(
                x=smoothed[:, 0], y=smoothed[:, 1], z=smoothed[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=intensity, colorscale=colors[idx + 1], showscale=False,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.5),
                lightposition=dict(x=100, y=200, z=300)
            ),
            row=row, col=col
        )
    
    # Camera settings
    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=1.0))
    
    # Update all scenes
    for i in range(1, 7):
        scene_name = 'scene' if i == 1 else f'scene{i}'
        fig.update_layout(**{
            scene_name: dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                camera=camera, aspectmode='data', bgcolor='white'
            )
        })
    
    fig.update_layout(
        title=dict(text=f'Algorithm Comparison: {sample_name}', x=0.5, font=dict(size=18)),
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=80, b=0),
        width=1600, height=900
    )
    
    fig.write_html(output_path)
    print(f"  Saved multi-algorithm comparison: {output_path}")


def main():
    print("="*70)
    print("COMPREHENSIVE EVALUATION - ALL BRATS DATA")
    print("="*70)
    
    # Get all segmentation files
    data_dir = project_root / "data" / "labels"
    seg_files = sorted(list(data_dir.glob("BraTS-GLI-*-seg.nii.gz")) + 
                      list(data_dir.glob("BraTS2021_*_seg.nii.gz")))
    
    print(f"\nFound {len(seg_files)} total samples")
    
    algorithms = ["Laplacian", "Taubin", "Geodesic Heat", "Info-Theoretic", "Anisotropic"]
    
    all_results = {
        "n_samples": len(seg_files),
        "samples": [],
        "per_algorithm": {alg: [] for alg in algorithms},
        "summary": {}
    }
    
    # Create output directories
    viz_dir = project_root / "outputs" / "figures" / "mesh_comparisons"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sample
    for idx, seg_file in enumerate(seg_files):
        sample_name = seg_file.stem.replace("-seg", "").replace("_seg", "")
        print(f"\n[{idx+1}/{len(seg_files)}] Processing: {sample_name}")
        
        try:
            # Load mesh
            verts, faces = load_mesh(seg_file)
            print(f"  Vertices: {len(verts):,}, Faces: {len(faces):,}")
            
            # Skip very small meshes
            if len(verts) < 1000:
                print(f"  Skipping (too small)")
                continue
            
            # Compute original metrics
            orig_vol = compute_volume(verts, faces)
            orig_ar = compute_aspect_ratio(verts, faces)
            orig_smooth = compute_smoothness(verts, faces)
            
            sample_data = {
                "name": sample_name,
                "vertices": int(len(verts)),
                "faces": int(len(faces)),
                "orig_volume": float(orig_vol),
                "algorithms": {}
            }
            
            smoothed_meshes = {}
            
            # Run each algorithm
            for alg in algorithms:
                try:
                    smoothed, elapsed = run_algorithm(alg, verts, faces)
                    smoothed_meshes[alg] = smoothed
                    
                    # Compute metrics
                    new_vol = compute_volume(smoothed, faces)
                    new_ar = compute_aspect_ratio(smoothed, faces)
                    new_smooth = compute_smoothness(smoothed, faces)
                    
                    vol_change = 100 * (new_vol - orig_vol) / orig_vol
                    ar_change = 100 * (new_ar - orig_ar) / orig_ar
                    smooth_reduction = 100 * (orig_smooth - new_smooth) / orig_smooth
                    
                    result = {
                        "vol_change_pct": float(vol_change),
                        "ar_change_pct": float(ar_change),
                        "smooth_reduction_pct": float(smooth_reduction),
                        "time_ms": float(elapsed * 1000),
                    }
                    
                    sample_data["algorithms"][alg] = result
                    all_results["per_algorithm"][alg].append(result)
                    
                    print(f"    {alg}: vol={vol_change:+.4f}%, smooth={smooth_reduction:.1f}%, time={elapsed*1000:.0f}ms")
                    
                except Exception as e:
                    print(f"    {alg}: ERROR - {e}")
            
            all_results["samples"].append(sample_data)
            
            # Create visualizations for first 3 samples
            if idx < 3 and smoothed_meshes:
                print(f"  Creating visualizations...")
                
                # Side-by-side comparison with Taubin (recommended)
                if "Taubin" in smoothed_meshes:
                    create_comparison_visualization(
                        verts, smoothed_meshes["Taubin"], faces,
                        f"BraTS Sample: {sample_name}",
                        viz_dir / f"{sample_name}_taubin_comparison.html",
                        "Taubin λ|μ"
                    )
                
                # Multi-algorithm comparison
                create_multi_algorithm_comparison(
                    verts, smoothed_meshes, faces, sample_name,
                    viz_dir / f"{sample_name}_all_algorithms.html"
                )
                
        except Exception as e:
            print(f"  ERROR loading mesh: {e}")
            continue
    
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
            
            print(f"\n{alg} (n={len(valid)}):")
            print(f"  Volume: {summary['vol_mean']:+.4f}% ± {summary['vol_std']:.4f}%")
            print(f"  AR: {summary['ar_mean']:+.2f}% ± {summary['ar_std']:.2f}%")
            print(f"  Smoothness: {summary['smooth_mean']:.2f}% ± {summary['smooth_std']:.2f}%")
            print(f"  Time: {summary['time_mean']:.1f}ms ± {summary['time_std']:.1f}ms")
    
    # Save results
    output_path = project_root / "outputs" / "all_brats_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    print(f"Visualizations saved to: {viz_dir}")
    
    return all_results


if __name__ == "__main__":
    main()
