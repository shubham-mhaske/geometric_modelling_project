import argparse
import csv
import datetime
import itertools
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import nibabel as nib
import numpy as np
import pyvista as pv
from skimage import measure

# Add project root to path so we can import from src
sys.path.append(os.getcwd())

try:
    from src.algorithms import smoothing, simplification, metrics
    from src.ml import get_ml_optimizer
except ImportError as e:
    print(f"Error importing src modules: {e}")
    print("Make sure you are running this script from the project root.")
    sys.exit(1)

# Ensure PyVista is off-screen
pv.OFF_SCREEN = True

@dataclass
class ExperimentParams:
    algorithm: str  # 'Laplacian', 'Taubin', 'None'
    iterations: int
    lambda_val: float
    mu_val: float = 0.0
    reduction_target: float = 0.0 # 0.0 to 0.95

def load_nifti_as_mesh(file_path: str):
    """Load NIfTI file and convert to mesh using marching cubes."""
    nii = nib.load(file_path)
    data = nii.get_fdata()
    
    # Create binary mask (labels 1, 2, 4)
    binary_mask = np.isin(data, [1, 2, 4]).astype(np.uint8)
    if binary_mask.sum() == 0:
        binary_mask = (data != 0).astype(np.uint8)
        
    zooms = nii.header.get_zooms()[:3]
    spacing = tuple(zooms)
    
    verts, faces, normals, values = measure.marching_cubes(binary_mask, level=0.5, spacing=spacing)
    return verts, faces

def compute_mesh_metrics(verts, faces, original_verts=None, original_volume=None):
    """Compute geometric metrics for the mesh."""
    # Create PyVista mesh for volume/area
    faces_padded = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, faces_padded)
    
    metrics_dict = {
        'vertex_count': verts.shape[0],
        'triangle_count': faces.shape[0],
        'surface_area_mm2': float(mesh.area),
        'volume_mm3': float(mesh.volume) if hasattr(mesh, 'volume') else 0.0,
    }
    
    # Hausdorff distance
    if original_verts is not None:
        # Sampling for speed if needed, but metrics.hausdorff_distance handles it?
        # Let's use the one from src
        h_dist = metrics.hausdorff_distance(original_verts, verts)
        metrics_dict['hausdorff_mm'] = h_dist
        
    # Volume change
    if original_volume is not None and original_volume > 0:
        metrics_dict['volume_change_pct'] = ((metrics_dict['volume_mm3'] - original_volume) / original_volume) * 100.0
    else:
        metrics_dict['volume_change_pct'] = 0.0
        
    # Aspect ratios
    # We need to implement aspect ratio calculation here or import it?
    # app.py has compute_aspect_ratios. Let's replicate it briefly or assume metrics has it?
    # metrics.py doesn't seem to have it based on previous reads (it had hausdorff).
    # Let's implement a simple one here.
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    max_e = np.maximum(np.maximum(e0, e1), e2)
    min_e = np.minimum(np.minimum(e0, e1), e2)
    min_e = np.where(min_e == 0, 1e-12, min_e)
    ratios = max_e / min_e
    
    metrics_dict['aspect_ratio_mean'] = float(np.mean(ratios))
    metrics_dict['aspect_ratio_min'] = float(np.min(ratios))
    # metrics_dict['aspect_ratio_max'] = float(np.max(ratios))
    
    return metrics_dict

def run_single_experiment(file_path: str, params: ExperimentParams, output_dir: str, save_artifacts: bool = True):
    """Run a single experiment configuration on a file."""
    start_time = time.time()
    result = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'success',
        'error_message': '',
        'input_file': file_path,
        'algorithm': params.algorithm,
        'iterations': params.iterations,
        'lambda': params.lambda_val,
        'mu': params.mu_val,
        'reduction_target': params.reduction_target
    }
    
    try:
        # Load original
        orig_verts, orig_faces = load_nifti_as_mesh(file_path)
        
        # Compute original metrics for baseline
        faces_padded = np.hstack([np.full((orig_faces.shape[0], 1), 3, dtype=np.int64), orig_faces]).astype(np.int64)
        orig_mesh = pv.PolyData(orig_verts, faces_padded)
        orig_volume = float(orig_mesh.volume)
        
        # Apply Smoothing
        curr_verts = orig_verts.copy()
        curr_faces = orig_faces.copy()
        
        if params.algorithm == 'Laplacian':
            curr_verts = smoothing.laplacian_smoothing(curr_verts, curr_faces, params.iterations, params.lambda_val)
        elif params.algorithm == 'Taubin':
            curr_verts = smoothing.taubin_smoothing(curr_verts, curr_faces, params.iterations, params.lambda_val, params.mu_val)
            
        # Apply Simplification
        if params.reduction_target > 0:
            curr_verts, curr_faces = simplification.qem_simplification(curr_verts, curr_faces, params.reduction_target)
            
        # Compute Metrics
        m = compute_mesh_metrics(curr_verts, curr_faces, original_verts=orig_verts, original_volume=orig_volume)
        result.update(m)
        
        # Save Artifact
        if save_artifacts:
            filename = os.path.basename(file_path).replace('.nii.gz', '').replace('.nii', '')
            artifact_name = f"{filename}_{params.algorithm}_{params.iterations:02d}it_L{int(params.lambda_val*100)}_M{int(abs(params.mu_val)*100)}_R{int(params.reduction_target*100)}.stl"
            artifact_path = os.path.join(output_dir, artifact_name)
            
            faces_padded = np.hstack([np.full((curr_faces.shape[0], 1), 3, dtype=np.int64), curr_faces]).astype(np.int64)
            mesh_out = pv.PolyData(curr_verts, faces_padded)
            mesh_out.save(artifact_path)
            result['artifact_path'] = artifact_path
        else:
            result['artifact_path'] = 'skipped'
            
    except Exception as e:
        result['status'] = 'failed'
        result['error_message'] = str(e)
        
    result['execution_time_sec'] = time.time() - start_time
    return result

def generate_param_combinations():
    """Generate a list of experiment parameters to sweep."""
    experiments = []
    
    # Baseline (No processing)
    experiments.append(ExperimentParams('None', 0, 0.0, 0.0, 0.0))
    experiments.append(ExperimentParams('None', 0, 0.0, 0.0, 0.3)) # Just simplification
    experiments.append(ExperimentParams('None', 0, 0.0, 0.0, 0.5))
    
    # Laplacian
    for iters in [10, 20]:
        for l in [0.3, 0.5]:
            for r in [0.0, 0.3, 0.5]:
                experiments.append(ExperimentParams('Laplacian', iters, l, 0.0, r))
                
    # Taubin
    for iters in [10, 20]:
        for l in [0.3, 0.5]:
            for r in [0.0, 0.3, 0.5]:
                experiments.append(ExperimentParams('Taubin', iters, l, -0.53, r))
                
    return experiments

def main():
    parser = argparse.ArgumentParser(description="Geometric Modelling Experiment Runner")
    parser.add_argument('--files', nargs='+', help='List of input NIfTI files')
    parser.add_argument('--data-dir', help='Directory containing NIfTI files (recursive search)')
    parser.add_argument('--output-dir', default='experiments/results', help='Directory to save artifacts')
    parser.add_argument('--log-path', default='experiments/results.csv', help='Path to CSV log file')
    parser.add_argument('--no-artifacts', action='store_true', help='Do not save STL artifacts')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    
    args = parser.parse_args()
    
    # Collect files
    files = []
    if args.files:
        files.extend(args.files)
    if args.data_dir:
        for root, _, filenames in os.walk(args.data_dir):
            for f in filenames:
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    files.append(os.path.join(root, f))
    
    files = sorted(list(set(files)))
    
    if not files:
        print("No input files found. Use --files or --data-dir.")
        sys.exit(1)
        
    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    
    params_list = generate_param_combinations()
    total_exps = len(files) * len(params_list)
    
    if args.verbose:
        print(f"Found {len(files)} files.")
        print(f"Generated {len(params_list)} parameter combinations.")
        print(f"Total experiments to run: {total_exps}")
        
    # Run
    results = []
    completed = 0
    
    # Check if log exists to append or write header
    file_exists = os.path.isfile(args.log_path)
    
    with open(args.log_path, 'a' if file_exists else 'w', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'status', 'error_message', 'input_file', 
            'algorithm', 'iterations', 'lambda', 'mu', 'reduction_target',
            'vertex_count', 'triangle_count', 'surface_area_mm2', 'volume_mm3',
            'hausdorff_mm', 'volume_change_pct', 
            'aspect_ratio_mean', 'aspect_ratio_min',
            'execution_time_sec', 'artifact_path'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        for f in files:
            for p in params_list:
                if args.verbose:
                    print(f"[{completed+1}/{total_exps}] Processing {os.path.basename(f)} | {p.algorithm} (it={p.iterations}, r={p.reduction_target})...")
                
                res = run_single_experiment(f, p, args.output_dir, save_artifacts=not args.no_artifacts)
                writer.writerow(res)
                csvfile.flush() # Ensure data is written
                results.append(res)
                completed += 1
                
    print(f"Experiments completed. Log written to {args.log_path}")

if __name__ == "__main__":
    main()
