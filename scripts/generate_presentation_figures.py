#!/usr/bin/env python3
"""
Generate All Presentation Figures with Actual Data

This script:
1. Loads actual BraTS segmentation data
2. Runs all 5 smoothing algorithms
3. Computes metrics (volume change, smoothness, aspect ratio, time)
4. Generates publication-quality figures
5. Creates 3D mesh comparisons (original vs smoothed)
"""

import os
import sys
import json
import glob
import time
import numpy as np
import nibabel as nib
from skimage import measure
from scipy.ndimage import gaussian_filter, binary_closing
from scipy import sparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.smoothing import laplacian_smoothing, taubin_smoothing
from src.algorithms.novel_algorithms_efficient import (
    geodesic_heat_smoothing,
    anisotropic_tensor_smoothing, 
    information_theoretic_smoothing
)

# Output directories
OUTPUT_DIR = "outputs/figures/presentation"
MESH_DIR = "outputs/figures/mesh_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MESH_DIR, exist_ok=True)


PLOTLY_TEMPLATE = "plotly_white"



def compute_mesh_volume(verts, faces):
    """Compute signed volume of a mesh using divergence theorem."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return np.abs(np.sum(v0 * cross) / 6.0)


def compute_aspect_ratio(verts, faces):
    """Compute mean triangle aspect ratio (higher = better quality)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    
    s = (e0 + e1 + e2) / 2
    area = np.sqrt(np.maximum(s * (s-e0) * (s-e1) * (s-e2), 1e-10))
    
    # Aspect ratio: 4*sqrt(3)*area / (e0^2 + e1^2 + e2^2)
    # For equilateral triangle = 1, lower for degenerate
    ar = (4 * np.sqrt(3) * area) / (e0**2 + e1**2 + e2**2 + 1e-10)
    return np.mean(ar)


def compute_smoothness(verts, faces):
    """Compute curvature variance (lower = smoother)."""
    num_verts = len(verts)
    
    # Build adjacency
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], 
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows))
    
    A = sparse.coo_matrix((data, (rows, cols)), shape=(num_verts, num_verts)).tocsr()
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    D_inv = sparse.diags(1.0 / degrees)
    W = D_inv @ A
    
    # Laplacian = neighbor_avg - vertex
    neighbor_avg = W @ verts
    laplacian = neighbor_avg - verts
    curvature = np.linalg.norm(laplacian, axis=1)
    
    return np.var(curvature)


def load_segmentation(seg_path):
    """Load segmentation mask and create mesh."""
    nii = nib.load(seg_path)
    label_volume = nii.get_fdata().astype(np.int16)
    zooms = nii.header.get_zooms()[:3]
    
    # Create binary mask (all tumor labels)
    binary_mask = (label_volume > 0).astype(np.uint8)
    
    if binary_mask.sum() == 0:
        return None, None, None
    
    # Smooth for cleaner mesh
    binary_mask = binary_closing(binary_mask, iterations=1).astype(np.uint8)
    smooth_mask = gaussian_filter(binary_mask.astype(float), sigma=0.5)
    
    try:
        verts, faces, _, _ = measure.marching_cubes(smooth_mask, level=0.5, spacing=tuple(zooms))
        return verts.astype(np.float32), faces, label_volume
    except:
        return None, None, None


def keep_largest_component(verts, faces):
    """Keep only the largest connected component."""
    try:
        import trimesh  # type: ignore[import-not-found]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            largest = max(components, key=lambda m: len(m.vertices))
            return largest.vertices.astype(np.float32), largest.faces
        return verts, faces
    except:
        return verts, faces


def run_all_algorithms(verts, faces, iterations=10):
    """Run all 5 smoothing algorithms and collect metrics."""
    results = {}
    orig_volume = compute_mesh_volume(verts, faces)
    orig_ar = compute_aspect_ratio(verts, faces)
    orig_smooth = compute_smoothness(verts, faces)
    
    # 1. Laplacian
    t0 = time.time()
    lap_verts = laplacian_smoothing(verts, faces, iterations, lambda_val=0.5)
    lap_time = (time.time() - t0) * 1000
    
    lap_vol = compute_mesh_volume(lap_verts, faces)
    lap_ar = compute_aspect_ratio(lap_verts, faces)
    lap_smooth = compute_smoothness(lap_verts, faces)
    
    results['Laplacian'] = {
        'verts': lap_verts,
        'vol_change_pct': (lap_vol - orig_volume) / orig_volume * 100,
        'ar_change_pct': (lap_ar - orig_ar) / orig_ar * 100,
        'smooth_reduction_pct': (orig_smooth - lap_smooth) / orig_smooth * 100 if orig_smooth > 0 else 0,
        'time_ms': lap_time
    }
    
    # 2. Taubin
    t0 = time.time()
    tau_verts = taubin_smoothing(verts, faces, iterations, lambda_val=0.5, mu_val=-0.53)
    tau_time = (time.time() - t0) * 1000
    
    tau_vol = compute_mesh_volume(tau_verts, faces)
    tau_ar = compute_aspect_ratio(tau_verts, faces)
    tau_smooth = compute_smoothness(tau_verts, faces)
    
    results['Taubin'] = {
        'verts': tau_verts,
        'vol_change_pct': (tau_vol - orig_volume) / orig_volume * 100,
        'ar_change_pct': (tau_ar - orig_ar) / orig_ar * 100,
        'smooth_reduction_pct': (orig_smooth - tau_smooth) / orig_smooth * 100 if orig_smooth > 0 else 0,
        'time_ms': tau_time
    }
    
    # 3. Geodesic Heat
    t0 = time.time()
    geo_verts, _ = geodesic_heat_smoothing(verts, faces, iterations=iterations)
    geo_time = (time.time() - t0) * 1000
    
    geo_vol = compute_mesh_volume(geo_verts, faces)
    geo_ar = compute_aspect_ratio(geo_verts, faces)
    geo_smooth = compute_smoothness(geo_verts, faces)
    
    results['Geodesic Heat'] = {
        'verts': geo_verts,
        'vol_change_pct': (geo_vol - orig_volume) / orig_volume * 100,
        'ar_change_pct': (geo_ar - orig_ar) / orig_ar * 100,
        'smooth_reduction_pct': (orig_smooth - geo_smooth) / orig_smooth * 100 if orig_smooth > 0 else 0,
        'time_ms': geo_time
    }
    
    # 4. Information-Theoretic
    t0 = time.time()
    info_verts, _ = information_theoretic_smoothing(verts, faces, iterations=iterations)
    info_time = (time.time() - t0) * 1000
    
    info_vol = compute_mesh_volume(info_verts, faces)
    info_ar = compute_aspect_ratio(info_verts, faces)
    info_smooth = compute_smoothness(info_verts, faces)
    
    results['Info-Theoretic'] = {
        'verts': info_verts,
        'vol_change_pct': (info_vol - orig_volume) / orig_volume * 100,
        'ar_change_pct': (info_ar - orig_ar) / orig_ar * 100,
        'smooth_reduction_pct': (orig_smooth - info_smooth) / orig_smooth * 100 if orig_smooth > 0 else 0,
        'time_ms': info_time
    }
    
    # 5. Anisotropic Tensor
    t0 = time.time()
    aniso_verts, _ = anisotropic_tensor_smoothing(verts, faces, iterations=iterations)
    aniso_time = (time.time() - t0) * 1000
    
    aniso_vol = compute_mesh_volume(aniso_verts, faces)
    aniso_ar = compute_aspect_ratio(aniso_verts, faces)
    aniso_smooth = compute_smoothness(aniso_verts, faces)
    
    results['Anisotropic'] = {
        'verts': aniso_verts,
        'vol_change_pct': (aniso_vol - orig_volume) / orig_volume * 100,
        'ar_change_pct': (aniso_ar - orig_ar) / orig_ar * 100,
        'smooth_reduction_pct': (orig_smooth - aniso_smooth) / orig_smooth * 100 if orig_smooth > 0 else 0,
        'time_ms': aniso_time
    }
    
    return results, orig_volume


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color like '#RRGGBB' into an rgba(r,g,b,a) string."""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    a = float(np.clip(alpha, 0.0, 1.0))
    return f"rgba({r},{g},{b},{a})"


def _wireframe_trace(verts, faces, max_edges=7000, seed=0, color='rgba(0,0,0,0.12)'):
    """Create a light wireframe overlay for visual clarity in static PNG exports."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    if len(edges) > max_edges:
        rng = np.random.default_rng(seed)
        edges = edges[rng.choice(len(edges), size=max_edges, replace=False)]

    x = []
    y = []
    z = []
    for i, j in edges:
        x.extend([verts[i, 0], verts[j, 0], None])
        y.extend([verts[i, 1], verts[j, 1], None])
        z.extend([verts[i, 2], verts[j, 2], None])

    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color=color, width=1),
        hoverinfo='skip',
        showlegend=False
    )


def create_3d_mesh_figure(verts, faces, title="", color='#ff6b6b', *, zoom=1.0, show_wireframe=True):
    """Create a 3D mesh visualization with a clean white background."""
    lighting = dict(ambient=0.45, diffuse=0.9, specular=0.45, roughness=0.35, fresnel=0.08)
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color,
            opacity=0.98,
            lighting=lighting,
            lightposition=dict(x=1200, y=900, z=1400),
            flatshading=False,
            hoverinfo='skip'
        )
    ])

    if show_wireframe:
        fig.add_trace(_wireframe_trace(verts, faces, seed=0))
    
    center = verts.mean(axis=0)
    max_range = (verts.max(axis=0) - verts.min(axis=0)).max() / 2 * (1.04 / max(zoom, 1e-6))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333'), x=0.5),
        scene=dict(
            xaxis=dict(visible=False, range=[center[0]-max_range, center[0]+max_range]),
            yaxis=dict(visible=False, range=[center[1]-max_range, center[1]+max_range]),
            zaxis=dict(visible=False, range=[center[2]-max_range, center[2]+max_range]),
            bgcolor='white',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.35, y=1.35, z=1.05))
        ),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=40, b=0),
        height=500
    )
    return fig


def create_comparison_figure(orig_verts, smoothed_verts, faces, orig_title, smooth_title, *, zoom=1.0, show_wireframe=True):
    """Create side-by-side 3D comparison with better lighting and optional wireframe."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=[orig_title, smooth_title],
        horizontal_spacing=0.02
    )
    
    lighting = dict(ambient=0.45, diffuse=0.9, specular=0.45, roughness=0.35, fresnel=0.08)
    
    # Original mesh (red/orange)
    fig.add_trace(go.Mesh3d(
        x=orig_verts[:, 0], y=orig_verts[:, 1], z=orig_verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='#ef4444',
        opacity=0.98,
        lighting=lighting,
        lightposition=dict(x=1200, y=900, z=1400),
        flatshading=False,
        hoverinfo='skip'
    ), row=1, col=1)

    if show_wireframe:
        fig.add_trace(_wireframe_trace(orig_verts, faces, seed=1), row=1, col=1)
    
    # Smoothed mesh (green)
    fig.add_trace(go.Mesh3d(
        x=smoothed_verts[:, 0], y=smoothed_verts[:, 1], z=smoothed_verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='#22c55e',
        opacity=0.98,
        lighting=lighting,
        lightposition=dict(x=1200, y=900, z=1400),
        flatshading=False,
        hoverinfo='skip'
    ), row=1, col=2)

    if show_wireframe:
        fig.add_trace(_wireframe_trace(smoothed_verts, faces, seed=2), row=1, col=2)
    
    # Compute shared bounds
    all_verts = np.vstack([orig_verts, smoothed_verts])
    center = all_verts.mean(axis=0)
    max_range = (all_verts.max(axis=0) - all_verts.min(axis=0)).max() / 2 * (1.04 / max(zoom, 1e-6))
    
    scene_common = dict(
        xaxis=dict(visible=False, range=[center[0]-max_range, center[0]+max_range]),
        yaxis=dict(visible=False, range=[center[1]-max_range, center[1]+max_range]),
        zaxis=dict(visible=False, range=[center[2]-max_range, center[2]+max_range]),
        bgcolor='white',
        aspectmode='cube',
        camera=dict(eye=dict(x=1.35, y=1.35, z=1.05))
    )
    
    fig.update_layout(
        scene=scene_common,
        scene2=scene_common,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        font=dict(color='#333')
    )
    return fig


def create_all_algorithms_comparison(orig_verts, results, faces, sample_name, *, zoom=1.0, show_wireframe=True):
    """Create a grid showing original + all 5 algorithms."""
    algorithms = ['Laplacian', 'Taubin', 'Geodesic Heat', 'Info-Theoretic', 'Anisotropic']
    colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899']
    
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'scene'}]*3, [{'type': 'scene'}]*3],
        subplot_titles=['Original'] + algorithms,
        horizontal_spacing=0.02,
        vertical_spacing=0.08
    )
    
    lighting = dict(ambient=0.45, diffuse=0.9, specular=0.45, roughness=0.35, fresnel=0.08)
    
    # Compute shared bounds
    all_verts = [orig_verts] + [results[a]['verts'] for a in algorithms]
    combined = np.vstack(all_verts)
    center = combined.mean(axis=0)
    max_range = (combined.max(axis=0) - combined.min(axis=0)).max() / 2 * (1.04 / max(zoom, 1e-6))
    
    # Original mesh
    fig.add_trace(go.Mesh3d(
        x=orig_verts[:, 0], y=orig_verts[:, 1], z=orig_verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=colors[0], opacity=0.98, lighting=lighting,
        lightposition=dict(x=1200, y=900, z=1400),
        flatshading=False, hoverinfo='skip'
    ), row=1, col=1)

    if show_wireframe:
        fig.add_trace(_wireframe_trace(orig_verts, faces, seed=3), row=1, col=1)
    
    # Algorithm meshes
    positions = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    for i, algo in enumerate(algorithms):
        v = results[algo]['verts']
        row, col = positions[i]
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=colors[i+1], opacity=0.98, lighting=lighting,
            lightposition=dict(x=1200, y=900, z=1400),
            flatshading=False, hoverinfo='skip'
        ), row=row, col=col)

        if show_wireframe:
            fig.add_trace(_wireframe_trace(v, faces, seed=10 + i), row=row, col=col)
    
    scene_common = dict(
        xaxis=dict(visible=False, range=[center[0]-max_range, center[0]+max_range]),
        yaxis=dict(visible=False, range=[center[1]-max_range, center[1]+max_range]),
        zaxis=dict(visible=False, range=[center[2]-max_range, center[2]+max_range]),
        bgcolor='white',
        aspectmode='cube',
        camera=dict(eye=dict(x=1.25, y=1.25, z=0.95))
    )
    
    # Update all scenes
    for i in range(6):
        scene_name = 'scene' if i == 0 else f'scene{i+1}'
        fig.update_layout(**{scene_name: scene_common})
    
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=40, b=0),
        height=700,
        font=dict(color='#333', size=11),
        title=dict(text=f'Algorithm Comparison: {sample_name}', x=0.5, font=dict(size=16))
    )
    return fig


def main():
    print("=" * 60)
    print("GENERATING PRESENTATION FIGURES WITH ACTUAL DATA")
    print("=" * 60)
    
    # Find all segmentation files
    seg_files = []
    
    # BraTS-GLI files
    for seg_file in glob.glob("data/labels/BraTS-GLI-*-seg.nii.gz"):
        seg_files.append(seg_file)
    
    # BraTS2021 files
    for seg_file in glob.glob("data/labels/BraTS2021_*_seg.nii.gz"):
        seg_files.append(seg_file)
    
    print(f"\nFound {len(seg_files)} segmentation files")
    
    if len(seg_files) == 0:
        print("ERROR: No segmentation files found!")
        return
    
    # Process each sample
    all_results = []
    sample_meshes = {}  # Store for 3D viz
    
    for i, seg_path in enumerate(seg_files):
        sample_name = os.path.basename(seg_path).replace('-seg.nii.gz', '').replace('_seg.nii.gz', '')
        print(f"\n[{i+1}/{len(seg_files)}] Processing: {sample_name}")
        
        verts, faces, label_vol = load_segmentation(seg_path)
        
        if verts is None:
            print(f"  SKIPPED: Could not create mesh")
            continue
        
        # Keep largest component
        verts, faces = keep_largest_component(verts, faces)
        
        print(f"  Vertices: {len(verts):,}, Faces: {len(faces):,}")
        
        # Run all algorithms
        results, orig_vol = run_all_algorithms(verts, faces, iterations=10)
        
        # Store results
        sample_result = {
            'name': sample_name,
            'vertices': len(verts),
            'faces': len(faces),
            'orig_volume': float(orig_vol),
            'algorithms': {}
        }
        
        for algo, data in results.items():
            sample_result['algorithms'][algo] = {
                'vol_change_pct': float(data['vol_change_pct']),
                'ar_change_pct': float(data['ar_change_pct']),
                'smooth_reduction_pct': float(data['smooth_reduction_pct']),
                'time_ms': float(data['time_ms'])
            }
            print(f"  {algo}: Vol={data['vol_change_pct']:+.3f}%, Smooth={data['smooth_reduction_pct']:.1f}%, Time={data['time_ms']:.1f}ms")
        
        all_results.append(sample_result)
        
        # Store mesh data for first 3 samples for 3D viz
        if len(sample_meshes) < 3:
            sample_meshes[sample_name] = {
                'orig_verts': verts,
                'faces': faces,
                'results': results
            }
    
    # Compute summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    algorithms = ['Laplacian', 'Taubin', 'Geodesic Heat', 'Info-Theoretic', 'Anisotropic']
    summary = {algo: {'vol': [], 'ar': [], 'smooth': [], 'time': []} for algo in algorithms}
    
    for sample in all_results:
        for algo in algorithms:
            if algo in sample['algorithms']:
                summary[algo]['vol'].append(sample['algorithms'][algo]['vol_change_pct'])
                summary[algo]['ar'].append(sample['algorithms'][algo]['ar_change_pct'])
                summary[algo]['smooth'].append(sample['algorithms'][algo]['smooth_reduction_pct'])
                summary[algo]['time'].append(sample['algorithms'][algo]['time_ms'])
    
    stats = {}
    for algo in algorithms:
        stats[algo] = {
            'vol_mean': np.mean(summary[algo]['vol']),
            'vol_std': np.std(summary[algo]['vol']),
            'ar_mean': np.mean(summary[algo]['ar']),
            'ar_std': np.std(summary[algo]['ar']),
            'smooth_mean': np.mean(summary[algo]['smooth']),
            'smooth_std': np.std(summary[algo]['smooth']),
            'time_mean': np.mean(summary[algo]['time']),
            'time_std': np.std(summary[algo]['time'])
        }
        print(f"\n{algo}:")
        print(f"  Volume: {stats[algo]['vol_mean']:+.3f}% ± {stats[algo]['vol_std']:.3f}%")
        print(f"  Smoothness: {stats[algo]['smooth_mean']:.1f}% ± {stats[algo]['smooth_std']:.1f}%")
        print(f"  AR Change: {stats[algo]['ar_mean']:+.1f}% ± {stats[algo]['ar_std']:.1f}%")
        print(f"  Time: {stats[algo]['time_mean']:.1f}ms ± {stats[algo]['time_std']:.1f}ms")
    
    # Save results to JSON
    output_data = {
        'n_samples': len(all_results),
        'summary': stats,
        'samples': all_results
    }
    
    with open('outputs/presentation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to outputs/presentation_results.json")
    
    # ========================================================================
    # GENERATE FIGURES
    # ========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    
    # Figure 1: Volume Change Bar Chart
    print("\n1. Creating Volume Change Comparison...")
    fig1 = go.Figure()
    
    colors = {'Laplacian': '#ef4444', 'Taubin': '#22c55e', 'Geodesic Heat': '#3b82f6', 
              'Info-Theoretic': '#f59e0b', 'Anisotropic': '#8b5cf6'}
    
    for algo in algorithms:
        fig1.add_trace(go.Bar(
            name=algo,
            x=[algo],
            y=[stats[algo]['vol_mean']],
            error_y=dict(type='data', array=[stats[algo]['vol_std']], visible=True),
            marker=dict(color=colors[algo], opacity=0.82, line=dict(color='rgba(0,0,0,0.25)', width=1)),
            text=[f"{stats[algo]['vol_mean']:+.2f}%"],
            textposition='outside',
            textfont=dict(color='#111', size=12)
        ))
    
    fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig1.update_layout(
        title=dict(text=f'Volume Change by Algorithm (n={len(all_results)} samples)', 
                   font=dict(size=18, color='#333'), x=0.5),
        xaxis_title='Algorithm',
        yaxis_title='Volume Change (%)',
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333', size=12),
        showlegend=False,
        height=500,
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)', linecolor='#333'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)', zerolinecolor='rgba(0,0,0,0.3)', linecolor='#333')
    )
    fig1.write_html(f'{OUTPUT_DIR}/fig1_volume_change.html')
    fig1.write_image(f'{OUTPUT_DIR}/fig1_volume_change.png', scale=2)
    
    # Figure 2: Smoothness Reduction Bar Chart
    print("2. Creating Smoothness Comparison...")
    fig2 = go.Figure()
    
    for algo in algorithms:
        fig2.add_trace(go.Bar(
            name=algo,
            x=[algo],
            y=[stats[algo]['smooth_mean']],
            error_y=dict(type='data', array=[stats[algo]['smooth_std']], visible=True),
            marker=dict(color=colors[algo], opacity=0.82, line=dict(color='rgba(0,0,0,0.25)', width=1)),
            text=[f"{stats[algo]['smooth_mean']:.1f}%"],
            textposition='outside',
            textfont=dict(color='#111', size=12)
        ))
    
    fig2.update_layout(
        title=dict(text=f'Smoothness Improvement by Algorithm (n={len(all_results)} samples)', 
                   font=dict(size=18, color='#333'), x=0.5),
        xaxis_title='Algorithm',
        yaxis_title='Curvature Reduction (%)',
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333', size=12),
        showlegend=False,
        height=500,
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)', linecolor='#333'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)', linecolor='#333')
    )
    fig2.write_html(f'{OUTPUT_DIR}/fig2_smoothness.html')
    fig2.write_image(f'{OUTPUT_DIR}/fig2_smoothness.png', scale=2)
    
    # Figure 3: Processing Time Bar Chart
    print("3. Creating Processing Time Comparison...")
    fig3 = go.Figure()
    
    for algo in algorithms:
        fig3.add_trace(go.Bar(
            name=algo,
            x=[algo],
            y=[stats[algo]['time_mean']],
            error_y=dict(type='data', array=[stats[algo]['time_std']], visible=True),
            marker=dict(color=colors[algo], opacity=0.82, line=dict(color='rgba(0,0,0,0.25)', width=1)),
            text=[f"{stats[algo]['time_mean']:.0f}ms"],
            textposition='outside',
            textfont=dict(color='#111', size=12)
        ))
    
    fig3.update_layout(
        title=dict(text=f'Processing Time by Algorithm (n={len(all_results)} samples)', 
                   font=dict(size=18, color='#333'), x=0.5),
        xaxis_title='Algorithm',
        yaxis_title='Time (ms)',
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333', size=12),
        showlegend=False,
        height=500,
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)', linecolor='#333'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)', linecolor='#333')
    )
    fig3.write_html(f'{OUTPUT_DIR}/fig3_processing_time.html')
    fig3.write_image(f'{OUTPUT_DIR}/fig3_processing_time.png', scale=2)
    
    # Figure 4: Scatter Plot - Volume vs Smoothness Trade-off
    print("4. Creating Trade-off Scatter Plot...")
    fig4 = go.Figure()
    
    for algo in algorithms:
        fig4.add_trace(go.Scatter(
            x=[abs(stats[algo]['vol_mean'])],
            y=[stats[algo]['smooth_mean']],
            mode='markers+text',
            name=algo,
            marker=dict(size=22, color=colors[algo], opacity=0.9, line=dict(color='rgba(0,0,0,0.35)', width=1)),
            text=[algo],
            textposition='top center',
            textfont=dict(size=12, color='#111')
        ))
    
    fig4.update_layout(
        title=dict(text='Trade-off: Volume Change vs Smoothing Quality', 
                   font=dict(size=18, color='#333'), x=0.5),
        xaxis_title='|Volume Change| (%)',
        yaxis_title='Smoothness Improvement (%)',
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333', size=12),
        showlegend=False,
        height=500,
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)', linecolor='#333'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)', linecolor='#333')
    )
    # Add annotation for ideal region
    fig4.add_annotation(
        x=0, y=max([stats[a]['smooth_mean'] for a in algorithms]),
        text="← Ideal Region<br>(Low volume change,<br>High smoothing)",
        showarrow=False, font=dict(size=10, color='#22c55e'),
        align='left'
    )
    fig4.write_html(f'{OUTPUT_DIR}/fig4_tradeoff_scatter.html')
    fig4.write_image(f'{OUTPUT_DIR}/fig4_tradeoff_scatter.png', scale=2)
    
    # Figure 5: Radar Chart
    print("5. Creating Radar Comparison Chart...")
    categories = ['Volume Preservation', 'Smoothing', 'Speed', 'Mesh Quality']
    
    fig5 = go.Figure()
    
    for algo in algorithms:
        # Normalize metrics to 0-100 scale
        vol_score = 100 - min(abs(stats[algo]['vol_mean']) * 10, 100)  # Lower change = higher score
        smooth_score = min(stats[algo]['smooth_mean'], 100)
        speed_score = 100 - min(stats[algo]['time_mean'] / 3, 100)  # Faster = higher score
        ar_score = min(50 + stats[algo]['ar_mean'], 100)  # Center at 50, improve with AR
        
        fig5.add_trace(go.Scatterpolar(
            r=[vol_score, smooth_score, speed_score, ar_score, vol_score],
            theta=categories + [categories[0]],
            name=algo,
            line=dict(color=colors[algo], width=2),
            fill='toself',
            fillcolor=_hex_to_rgba(colors[algo], 0.12),
            opacity=1.0
        ))
    
    fig5.update_layout(
        title=dict(text='Multi-Metric Algorithm Comparison', font=dict(size=18, color='#333'), x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(0,0,0,0.2)'),
            angularaxis=dict(gridcolor='rgba(0,0,0,0.2)'),
            bgcolor='white'
        ),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='white',
        font=dict(color='#333', size=11),
        showlegend=True,
        legend=dict(x=1.1, y=0.5),
        height=550
    )
    fig5.write_html(f'{OUTPUT_DIR}/fig5_radar_comparison.html')
    fig5.write_image(f'{OUTPUT_DIR}/fig5_radar_comparison.png', scale=2)
    
    # Figure 6: Summary Table
    print("6. Creating Summary Table...")
    
    table_data = {
        'Algorithm': algorithms,
        'Vol Change (%)': [f"{stats[a]['vol_mean']:+.3f} ± {stats[a]['vol_std']:.3f}" for a in algorithms],
        'Smoothing (%)': [f"{stats[a]['smooth_mean']:.1f} ± {stats[a]['smooth_std']:.1f}" for a in algorithms],
        'AR Change (%)': [f"{stats[a]['ar_mean']:+.1f} ± {stats[a]['ar_std']:.1f}" for a in algorithms],
        'Time (ms)': [f"{stats[a]['time_mean']:.0f} ± {stats[a]['time_std']:.0f}" for a in algorithms]
    }
    
    fig6 = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Algorithm</b>', '<b>Volume Change</b>', '<b>Smoothing</b>', 
                    '<b>AR Change</b>', '<b>Time</b>'],
            fill_color='#500000',
            font=dict(color='white', size=13),
            align='center',
            height=35
        ),
        cells=dict(
            values=[table_data['Algorithm'], table_data['Vol Change (%)'], 
                    table_data['Smoothing (%)'], table_data['AR Change (%)'], 
                    table_data['Time (ms)']],
            fill_color=[[('#ffffff' if i % 2 == 0 else '#f3f4f6') for i in range(len(algorithms))]] * 5,
            font=dict(color='#333', size=12),
            align='center',
            height=30
        )
    )])
    
    fig6.update_layout(
        title=dict(text=f'Algorithm Performance Summary (n={len(all_results)} samples)', 
                   font=dict(size=18, color='#333'), x=0.5),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20),
        height=300
    )
    fig6.write_html(f'{OUTPUT_DIR}/fig6_summary_table.html')
    fig6.write_image(f'{OUTPUT_DIR}/fig6_summary_table.png', scale=2)
    
    # ========================================================================
    # 3D MESH COMPARISONS
    # ========================================================================
    print("\n" + "=" * 60)
    print("GENERATING 3D MESH VISUALIZATIONS")
    print("=" * 60)
    
    for sample_name, mesh_data in sample_meshes.items():
        print(f"\nCreating 3D visualizations for: {sample_name}")
        
        orig_verts = mesh_data['orig_verts']
        faces = mesh_data['faces']
        results = mesh_data['results']
        
        # Create Taubin comparison (recommended algorithm)
        taubin_fig = create_comparison_figure(
            orig_verts, results['Taubin']['verts'], faces,
            f'Original ({len(orig_verts):,} vertices)',
            f'Taubin Smoothed (Vol: {results["Taubin"]["vol_change_pct"]:+.3f}%)'
        )
        taubin_fig.write_html(f'{MESH_DIR}/{sample_name}_taubin_comparison.html')
        taubin_fig.write_image(f'{MESH_DIR}/{sample_name}_taubin_comparison.png', scale=2)
        # Close-up PNG for slide clarity
        taubin_close = create_comparison_figure(
            orig_verts, results['Taubin']['verts'], faces,
            f'Original ({len(orig_verts):,} vertices)',
            f'Taubin Smoothed (Vol: {results["Taubin"]["vol_change_pct"]:+.3f}%)',
            zoom=1.35
        )
        taubin_close.write_image(f'{MESH_DIR}/{sample_name}_taubin_comparison_close.png', scale=2)
        print(f"  Saved: {sample_name}_taubin_comparison.html/.png")
        
        # Create all algorithms comparison
        all_algo_fig = create_all_algorithms_comparison(orig_verts, results, faces, sample_name)
        all_algo_fig.write_html(f'{MESH_DIR}/{sample_name}_all_algorithms.html')
        all_algo_fig.write_image(f'{MESH_DIR}/{sample_name}_all_algorithms.png', scale=2)
        all_algo_close = create_all_algorithms_comparison(orig_verts, results, faces, sample_name, zoom=1.35)
        all_algo_close.write_image(f'{MESH_DIR}/{sample_name}_all_algorithms_close.png', scale=2)
        print(f"  Saved: {sample_name}_all_algorithms.html/.png")
    
    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nFigures saved to: {OUTPUT_DIR}/")
    print(f"3D comparisons saved to: {MESH_DIR}/")


if __name__ == '__main__':
    main()
