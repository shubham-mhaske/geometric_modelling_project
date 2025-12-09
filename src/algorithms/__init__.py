"""
Core geometric algorithms for mesh processing.
"""

from .smoothing import (
    laplacian_smoothing, 
    taubin_smoothing,
    constrained_smoothing,
    adaptive_smoothing
)
from .simplification import qem_simplification
from .metrics import (
    hausdorff_distance, 
    compute_volume_change_percent,
    compute_mean_curvature,
    compute_gaussian_curvature,
    compute_curvature_error
)
from .bilateral_smoothing import bilateral_smoothing, guided_smoothing
from .neural_smoothing import (
    neural_smoothing,
    gnn_mesh_smoothing,
    diffusion_mesh_smoothing,
    transformer_mesh_smoothing,
    spectral_mesh_smoothing,
    neural_inspired_smoothing,
)

__all__ = [
    # Smoothing algorithms
    'laplacian_smoothing',
    'taubin_smoothing',
    'bilateral_smoothing',
    'guided_smoothing',
    'constrained_smoothing',
    'adaptive_smoothing',
    # Neural smoothing (latest techniques)
    'neural_smoothing',
    'gnn_mesh_smoothing',
    'diffusion_mesh_smoothing',
    'transformer_mesh_smoothing',
    'spectral_mesh_smoothing',
    'neural_inspired_smoothing',
    # Simplification
    'qem_simplification',
    # Metrics
    'hausdorff_distance',
    'compute_volume_change_percent',
    'compute_mean_curvature',
    'compute_gaussian_curvature',
    'compute_curvature_error',
]
