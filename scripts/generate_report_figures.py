#!/usr/bin/env python3
"""
Generate figures for the final project report.
Creates publication-quality visualizations of mesh smoothing results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import process_nifti_to_mesh, compute_aspect_ratios
from src.algorithms import smoothing
from src.algorithms.metrics import (
    compute_mean_curvature, 
    compute_gaussian_curvature, 
    compute_curvature_error,
    hausdorff_distance
)
from src.algorithms.processing import coarsen_label_volume, map_labels_to_vertices
import pyvista as pv

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme
COLORS = {
    'primary': '#500000',      # Texas A&M Maroon
    'secondary': '#998542',    # Texas A&M Gold
    'laplacian': '#e74c3c',    # Red
    'taubin': '#3498db',       # Blue
    'semantic': '#2ecc71',     # Green
    'bilateral': '#9b59b6',    # Purple
}

OUTPUT_DIR = 'outputs/figures/report'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sample_data():
    """Load a BraTS sample for analysis."""
    data_dir = 'data/labels'
    files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
    if not files:
        raise FileNotFoundError("No BraTS samples found in data/labels/")
    
    path = os.path.join(data_dir, files[0])
    print(f"Loading: {files[0]}")
    mesh, aspect_ratios, verts, faces, label_volume, affine = process_nifti_to_mesh(path)
    return {
        'mesh': mesh,
        'verts': verts,
        'faces': faces,
        'aspect_ratios': aspect_ratios,
        'label_volume': label_volume,
        'affine': affine,
        'filename': files[0]
    }


def compute_all_metrics(data):
    """Compute comprehensive metrics for all algorithms."""
    verts = data['verts']
    faces = data['faces']
    label_volume = data['label_volume']
    affine = data['affine']
    
    # Prepare face array for PyVista
    faces_padded = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int64), 
        faces
    ]).astype(np.int64)
    
    orig_vol = float(pv.PolyData(verts, faces_padded).volume)
    orig_ar = float(np.mean(data['aspect_ratios']))
    
    # Get vertex labels for semantic smoothing
    vertex_labels = map_labels_to_vertices(
        coarsen_label_volume(label_volume), 
        affine, 
        verts
    )
    
    algorithms = {
        'Original': lambda v: v.copy(),
        'Laplacian (15 iter)': lambda v: smoothing.laplacian_smoothing(v.copy(), faces, 15),
        'Taubin (15 iter)': lambda v: smoothing.taubin_smoothing(v.copy(), faces, 15),
        'Semantic Taubin': lambda v: smoothing.taubin_smoothing(
            v.copy(), faces, 15, vertex_labels=vertex_labels
        ),
    }
    
    results = {}
    for name, fn in algorithms.items():
        print(f"  Processing: {name}")
        new_v = fn(verts)
        new_vol = float(pv.PolyData(new_v, faces_padded).volume)
        new_ar = float(np.mean(compute_aspect_ratios(new_v, faces)))
        
        H, H_mean, H_std = compute_mean_curvature(new_v, faces)
        K, K_mean, K_std = compute_gaussian_curvature(new_v, faces)
        
        if name != 'Original':
            curv_err = compute_curvature_error(verts, new_v, faces)
            hd = hausdorff_distance(verts, new_v)
        else:
            curv_err = {'mean_curvature_correlation': 1.0}
            hd = 0.0
        
        results[name] = {
            'vertices': new_v,
            'volume': new_vol,
            'vol_change': (new_vol - orig_vol) / orig_vol * 100,
            'aspect_ratio': new_ar,
            'ar_improvement': (orig_ar - new_ar) / orig_ar * 100,
            'mean_curvature': H,
            'H_mean': H_mean,
            'H_std': H_std,
            'gaussian_curvature': K,
            'K_mean': K_mean,
            'K_std': K_std,
            'curvature_correlation': curv_err['mean_curvature_correlation'],
            'hausdorff': hd,
        }
    
    return results


def fig1_algorithm_comparison_bar(results):
    """Figure 1: Bar chart comparing algorithm performance metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    algorithms = ['Laplacian (15 iter)', 'Taubin (15 iter)', 'Semantic Taubin']
    colors = [COLORS['laplacian'], COLORS['taubin'], COLORS['semantic']]
    x = np.arange(len(algorithms))
    
    # Volume Change
    ax = axes[0]
    vol_changes = [results[a]['vol_change'] for a in algorithms]
    bars = ax.bar(x, vol_changes, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Volume Change (%)')
    ax.set_title('(a) Volume Preservation')
    ax.set_xticks(x)
    ax.set_xticklabels(['Laplacian', 'Taubin', 'Semantic'], rotation=15, ha='right')
    ax.set_ylim(-0.5, 0.1)
    
    # Add value labels
    for bar, val in zip(bars, vol_changes):
        ax.annotate(f'{val:+.2f}%', 
                   xy=(bar.get_x() + bar.get_width()/2, val),
                   ha='center', va='bottom' if val > 0 else 'top',
                   fontsize=8, fontweight='bold')
    
    # AR Improvement
    ax = axes[1]
    ar_improvements = [results[a]['ar_improvement'] for a in algorithms]
    bars = ax.bar(x, ar_improvements, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Aspect Ratio Improvement (%)')
    ax.set_title('(b) Mesh Quality')
    ax.set_xticks(x)
    ax.set_xticklabels(['Laplacian', 'Taubin', 'Semantic'], rotation=15, ha='right')
    
    for bar, val in zip(bars, ar_improvements):
        ax.annotate(f'+{val:.1f}%', 
                   xy=(bar.get_x() + bar.get_width()/2, val),
                   ha='center', va='bottom',
                   fontsize=8, fontweight='bold')
    
    # Curvature Correlation
    ax = axes[2]
    curv_corrs = [results[a]['curvature_correlation'] for a in algorithms]
    bars = ax.bar(x, curv_corrs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Curvature Correlation')
    ax.set_title('(c) Feature Preservation')
    ax.set_xticks(x)
    ax.set_xticklabels(['Laplacian', 'Taubin', 'Semantic'], rotation=15, ha='right')
    ax.set_ylim(0, 0.2)
    
    for bar, val in zip(bars, curv_corrs):
        ax.annotate(f'{val:.3f}', 
                   xy=(bar.get_x() + bar.get_width()/2, val),
                   ha='center', va='bottom',
                   fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig1_algorithm_comparison.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved: {path}")
    plt.close()


def fig2_curvature_distribution(results, data):
    """Figure 2: Curvature distribution histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    # Original curvature
    H_orig = results['Original']['mean_curvature']
    K_orig = results['Original']['gaussian_curvature']
    
    # After Taubin
    H_taubin = results['Taubin (15 iter)']['mean_curvature']
    K_taubin = results['Taubin (15 iter)']['gaussian_curvature']
    
    # Mean Curvature - Original
    ax = axes[0, 0]
    ax.hist(H_orig, bins=100, range=(-1, 2), color=COLORS['primary'], 
            alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=np.mean(H_orig), color='red', linestyle='--', linewidth=1.5, 
               label=f'μ = {np.mean(H_orig):.3f}')
    ax.set_xlabel('Mean Curvature (H)')
    ax.set_ylabel('Vertex Count')
    ax.set_title('(a) Mean Curvature - Original')
    ax.legend(loc='upper right')
    
    # Mean Curvature - Smoothed
    ax = axes[0, 1]
    ax.hist(H_taubin, bins=100, range=(-1, 2), color=COLORS['taubin'], 
            alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=np.mean(H_taubin), color='red', linestyle='--', linewidth=1.5,
               label=f'μ = {np.mean(H_taubin):.3f}')
    ax.set_xlabel('Mean Curvature (H)')
    ax.set_ylabel('Vertex Count')
    ax.set_title('(b) Mean Curvature - After Taubin')
    ax.legend(loc='upper right')
    
    # Gaussian Curvature - Original
    ax = axes[1, 0]
    ax.hist(K_orig, bins=100, range=(-0.5, 0.5), color=COLORS['primary'], 
            alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=np.mean(K_orig), color='red', linestyle='--', linewidth=1.5,
               label=f'μ = {np.mean(K_orig):.4f}')
    ax.set_xlabel('Gaussian Curvature (K)')
    ax.set_ylabel('Vertex Count')
    ax.set_title('(c) Gaussian Curvature - Original')
    ax.legend(loc='upper right')
    
    # Gaussian Curvature - Smoothed
    ax = axes[1, 1]
    ax.hist(K_taubin, bins=100, range=(-0.5, 0.5), color=COLORS['taubin'], 
            alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=np.mean(K_taubin), color='red', linestyle='--', linewidth=1.5,
               label=f'μ = {np.mean(K_taubin):.4f}')
    ax.set_xlabel('Gaussian Curvature (K)')
    ax.set_ylabel('Vertex Count')
    ax.set_title('(d) Gaussian Curvature - After Taubin')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_curvature_distribution.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved: {path}")
    plt.close()


def fig3_aspect_ratio_comparison(results, data):
    """Figure 3: Aspect ratio distribution before/after smoothing."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # Compute AR for each algorithm
    faces = data['faces']
    
    ar_orig = data['aspect_ratios']
    ar_taubin = compute_aspect_ratios(results['Taubin (15 iter)']['vertices'], faces)
    
    # Before smoothing
    ax = axes[0]
    ax.hist(ar_orig, bins=50, range=(1, 4), color=COLORS['primary'], 
            alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=np.mean(ar_orig), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(ar_orig):.3f}')
    ax.axvline(x=1.0, color='green', linestyle='-', linewidth=1.5, alpha=0.5,
               label='Ideal (Equilateral)')
    ax.set_xlabel('Aspect Ratio')
    ax.set_ylabel('Triangle Count')
    ax.set_title('(a) Before Smoothing')
    ax.legend(loc='upper right')
    ax.set_xlim(1, 4)
    
    # After Taubin
    ax = axes[1]
    ax.hist(ar_taubin, bins=50, range=(1, 4), color=COLORS['taubin'], 
            alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=np.mean(ar_taubin), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(ar_taubin):.3f}')
    ax.axvline(x=1.0, color='green', linestyle='-', linewidth=1.5, alpha=0.5,
               label='Ideal (Equilateral)')
    ax.set_xlabel('Aspect Ratio')
    ax.set_ylabel('Triangle Count')
    ax.set_title('(b) After Taubin Smoothing (15 iterations)')
    ax.legend(loc='upper right')
    ax.set_xlim(1, 4)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_aspect_ratio.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved: {path}")
    plt.close()


def fig4_iteration_convergence():
    """Figure 4: Convergence analysis over iterations."""
    data = load_sample_data()
    verts = data['verts']
    faces = data['faces']
    
    faces_padded = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int64), 
        faces
    ]).astype(np.int64)
    
    orig_vol = float(pv.PolyData(verts, faces_padded).volume)
    orig_ar = float(np.mean(data['aspect_ratios']))
    
    iterations = [1, 3, 5, 10, 15, 20, 30, 50]
    
    results = {'laplacian': {'vol': [], 'ar': []}, 
               'taubin': {'vol': [], 'ar': []}}
    
    print("  Computing convergence data...")
    for n_iter in iterations:
        # Laplacian
        v_lap = smoothing.laplacian_smoothing(verts.copy(), faces, n_iter)
        vol_lap = float(pv.PolyData(v_lap, faces_padded).volume)
        ar_lap = float(np.mean(compute_aspect_ratios(v_lap, faces)))
        results['laplacian']['vol'].append((vol_lap - orig_vol) / orig_vol * 100)
        results['laplacian']['ar'].append((orig_ar - ar_lap) / orig_ar * 100)
        
        # Taubin
        v_tau = smoothing.taubin_smoothing(verts.copy(), faces, n_iter)
        vol_tau = float(pv.PolyData(v_tau, faces_padded).volume)
        ar_tau = float(np.mean(compute_aspect_ratios(v_tau, faces)))
        results['taubin']['vol'].append((vol_tau - orig_vol) / orig_vol * 100)
        results['taubin']['ar'].append((orig_ar - ar_tau) / orig_ar * 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # Volume Change
    ax = axes[0]
    ax.plot(iterations, results['laplacian']['vol'], 'o-', color=COLORS['laplacian'], 
            linewidth=2, markersize=6, label='Laplacian')
    ax.plot(iterations, results['taubin']['vol'], 's-', color=COLORS['taubin'], 
            linewidth=2, markersize=6, label='Taubin')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.axhline(y=-1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='±1% threshold')
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.fill_between(iterations, -1, 1, alpha=0.1, color='green')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Volume Change (%)')
    ax.set_title('(a) Volume Preservation vs. Iterations')
    ax.legend(loc='lower left')
    ax.set_xlim(0, 52)
    ax.set_ylim(-2.5, 0.5)
    
    # AR Improvement
    ax = axes[1]
    ax.plot(iterations, results['laplacian']['ar'], 'o-', color=COLORS['laplacian'], 
            linewidth=2, markersize=6, label='Laplacian')
    ax.plot(iterations, results['taubin']['ar'], 's-', color=COLORS['taubin'], 
            linewidth=2, markersize=6, label='Taubin')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Aspect Ratio Improvement (%)')
    ax.set_title('(b) Mesh Quality vs. Iterations')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 52)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_convergence.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved: {path}")
    plt.close()


def fig5_mesh_visualization(results, data):
    """Figure 5: 3D mesh renderings (wireframe + surface)."""
    print("  Generating mesh visualizations...")
    
    verts_orig = results['Original']['vertices']
    verts_taubin = results['Taubin (15 iter)']['vertices']
    faces = data['faces']
    
    # Use PyVista for off-screen rendering
    pv.start_xvfb()  # For headless rendering if needed
    
    faces_padded = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int64), 
        faces
    ]).astype(np.int64)
    
    # Create meshes
    mesh_orig = pv.PolyData(verts_orig, faces_padded)
    mesh_taubin = pv.PolyData(verts_taubin, faces_padded)
    
    # Add curvature data
    H_orig = results['Original']['mean_curvature']
    H_taubin = results['Taubin (15 iter)']['mean_curvature']
    mesh_orig['Mean Curvature'] = H_orig
    mesh_taubin['Mean Curvature'] = H_taubin
    
    # Create plotter for original
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(mesh_orig, scalars='Mean Curvature', cmap='coolwarm',
                     clim=[-0.5, 1.5], show_scalar_bar=True, 
                     scalar_bar_args={'title': 'Mean Curvature'})
    plotter.camera_position = 'iso'
    plotter.add_text('Original Mesh', font_size=12)
    path = os.path.join(OUTPUT_DIR, 'fig5a_mesh_original.png')
    plotter.screenshot(path)
    plotter.close()
    print(f"  Saved: {path}")
    
    # Create plotter for smoothed
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(mesh_taubin, scalars='Mean Curvature', cmap='coolwarm',
                     clim=[-0.5, 1.5], show_scalar_bar=True,
                     scalar_bar_args={'title': 'Mean Curvature'})
    plotter.camera_position = 'iso'
    plotter.add_text('After Taubin Smoothing (15 iter)', font_size=12)
    path = os.path.join(OUTPUT_DIR, 'fig5b_mesh_smoothed.png')
    plotter.screenshot(path)
    plotter.close()
    print(f"  Saved: {path}")
    
    # Wireframe comparison - zoomed region
    # Extract a small region for detail view
    center = np.mean(verts_orig, axis=0)
    radius = np.max(np.linalg.norm(verts_orig - center, axis=1)) * 0.2
    
    # Create detail view
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600], shape=(1, 2))
    
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_orig, style='wireframe', color='black', line_width=0.5)
    plotter.add_mesh(mesh_orig, color='lightblue', opacity=0.7)
    plotter.add_text('Original (Detail)', font_size=10)
    plotter.camera.zoom(2.5)
    
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_taubin, style='wireframe', color='black', line_width=0.5)
    plotter.add_mesh(mesh_taubin, color='lightgreen', opacity=0.7)
    plotter.add_text('Smoothed (Detail)', font_size=10)
    plotter.camera.zoom(2.5)
    
    plotter.link_views()
    path = os.path.join(OUTPUT_DIR, 'fig5c_mesh_detail.png')
    plotter.screenshot(path)
    plotter.close()
    print(f"  Saved: {path}")


def fig6_pipeline_diagram():
    """Figure 6: Pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')
    
    steps = [
        ('NIfTI\nInput', 0.5),
        ('Volume\nProcessing', 2.3),
        ('Marching\nCubes', 4.1),
        ('Mesh\nSmoothing', 5.9),
        ('Quality\nMetrics', 7.7),
        ('Export\n& Viz', 9.5),
    ]
    
    box_width = 1.4
    box_height = 0.8
    
    for i, (label, x) in enumerate(steps):
        # Draw box
        rect = mpatches.FancyBboxPatch(
            (x - box_width/2, 1 - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=COLORS['primary'] if i in [0, 5] else 'white',
            edgecolor=COLORS['primary'],
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, 1, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white' if i in [0, 5] else COLORS['primary'])
        
        # Draw arrow
        if i < len(steps) - 1:
            next_x = steps[i + 1][1]
            ax.annotate('', xy=(next_x - box_width/2 - 0.1, 1),
                       xytext=(x + box_width/2 + 0.1, 1),
                       arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2))
    
    ax.text(5, 0.1, 'Figure 6: Processing Pipeline Overview', ha='center', 
            fontsize=11, style='italic')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig6_pipeline.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved: {path}")
    plt.close()


def fig7_multi_sample_validation():
    """Figure 7: Box plots of results across multiple samples."""
    data_dir = 'data/labels'
    files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
    
    all_vol_changes = []
    all_ar_improvements = []
    
    print("  Processing multiple samples for validation...")
    for fname in files[:5]:
        fpath = os.path.join(data_dir, fname)
        try:
            mesh, ar, v, f, lv, af = process_nifti_to_mesh(fpath)
            fp = np.hstack([np.full((f.shape[0], 1), 3, dtype=np.int64), f]).astype(np.int64)
            ov = float(pv.PolyData(v, fp).volume)
            oar = float(np.mean(ar))
            
            new_v = smoothing.taubin_smoothing(v.copy(), f, 15)
            nv = float(pv.PolyData(new_v, fp).volume)
            nar = float(np.mean(compute_aspect_ratios(new_v, f)))
            
            all_vol_changes.append((nv - ov) / ov * 100)
            all_ar_improvements.append((oar - nar) / oar * 100)
        except Exception as e:
            print(f"    Error processing {fname}: {e}")
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
    # Volume Change
    ax = axes[0]
    bp = ax.boxplot([all_vol_changes], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(COLORS['taubin'])
    bp['boxes'][0].set_alpha(0.7)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('Volume Change (%)')
    ax.set_title('(a) Volume Preservation')
    ax.set_xticklabels(['Taubin\n(n=5 samples)'])
    ax.set_ylim(-0.1, 0.1)
    
    # AR Improvement
    ax = axes[1]
    bp = ax.boxplot([all_ar_improvements], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(COLORS['taubin'])
    bp['boxes'][0].set_alpha(0.7)
    ax.set_ylabel('AR Improvement (%)')
    ax.set_title('(b) Mesh Quality')
    ax.set_xticklabels(['Taubin\n(n=5 samples)'])
    
    # Add statistics annotation
    vol_mean = np.mean(all_vol_changes)
    vol_std = np.std(all_vol_changes)
    ar_mean = np.mean(all_ar_improvements)
    ar_std = np.std(all_ar_improvements)
    
    axes[0].text(0.95, 0.95, f'μ={vol_mean:+.3f}%\nσ={vol_std:.3f}%', 
                 transform=axes[0].transAxes, ha='right', va='top',
                 fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].text(0.95, 0.95, f'μ={ar_mean:+.1f}%\nσ={ar_std:.1f}%', 
                 transform=axes[1].transAxes, ha='right', va='top',
                 fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig7_multi_sample.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved: {path}")
    plt.close()


def main():
    """Generate all figures for the report."""
    print("=" * 60)
    print("GENERATING REPORT FIGURES")
    print("=" * 60)
    
    # Load data
    print("\n[1/8] Loading sample data...")
    data = load_sample_data()
    print(f"      Vertices: {data['verts'].shape[0]:,}")
    print(f"      Faces: {data['faces'].shape[0]:,}")
    
    # Compute metrics
    print("\n[2/8] Computing metrics for all algorithms...")
    results = compute_all_metrics(data)
    
    # Generate figures
    print("\n[3/8] Figure 1: Algorithm Comparison...")
    fig1_algorithm_comparison_bar(results)
    
    print("\n[4/8] Figure 2: Curvature Distribution...")
    fig2_curvature_distribution(results, data)
    
    print("\n[5/8] Figure 3: Aspect Ratio Comparison...")
    fig3_aspect_ratio_comparison(results, data)
    
    print("\n[6/8] Figure 4: Convergence Analysis...")
    fig4_iteration_convergence()
    
    print("\n[7/8] Figure 5: 3D Mesh Visualization...")
    try:
        fig5_mesh_visualization(results, data)
    except Exception as e:
        print(f"      Warning: Could not generate 3D figures: {e}")
    
    print("\n[8/8] Figure 6: Pipeline Diagram...")
    fig6_pipeline_diagram()
    
    print("\n[Bonus] Figure 7: Multi-Sample Validation...")
    fig7_multi_sample_validation()
    
    print("\n" + "=" * 60)
    print(f"ALL FIGURES SAVED TO: {OUTPUT_DIR}/")
    print("=" * 60)
    
    # Print summary table for LaTeX
    print("\n\nLATEX TABLE DATA:")
    print("-" * 60)
    print("Algorithm & Vol Δ & AR Δ & H Corr & Hausdorff \\\\")
    print("\\hline")
    for name in ['Laplacian (15 iter)', 'Taubin (15 iter)', 'Semantic Taubin']:
        r = results[name]
        print(f"{name.replace('(15 iter)', '').strip()} & {r['vol_change']:+.2f}\\% & "
              f"{r['ar_improvement']:+.1f}\\% & {r['curvature_correlation']:.3f} & "
              f"{r['hausdorff']:.3f} \\\\")


if __name__ == '__main__':
    main()
