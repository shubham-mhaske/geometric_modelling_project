"""
Core geometric algorithms for mesh processing.
"""

from .smoothing import laplacian_smoothing, taubin_smoothing
from .simplification import qem_simplification
from .metrics import hausdorff_distance, compute_volume_change_percent

__all__ = [
    'laplacian_smoothing',
    'taubin_smoothing',
    'qem_simplification',
    'hausdorff_distance',
    'compute_volume_change_percent'
]
