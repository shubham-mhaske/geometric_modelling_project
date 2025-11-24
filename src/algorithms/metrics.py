import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(verts1, verts2, sample_size=5000):
    """
    Compute Hausdorff distance between two point clouds.
    Uses sampling for large meshes to keep computation tractable.
    
    Args:
        verts1: (N, 3) array of vertices
        verts2: (M, 3) array of vertices
        sample_size: max points to use for computation
    
    Returns:
        float: Hausdorff distance in mm
    """
    # Sample if too many vertices
    if verts1.shape[0] > sample_size:
        idx = np.random.choice(verts1.shape[0], sample_size, replace=False)
        verts1 = verts1[idx]
    
    if verts2.shape[0] > sample_size:
        idx = np.random.choice(verts2.shape[0], sample_size, replace=False)
        verts2 = verts2[idx]
    
    # Compute directed Hausdorff distances
    d1 = directed_hausdorff(verts1, verts2)[0]
    d2 = directed_hausdorff(verts2, verts1)[0]
    
    # Hausdorff distance is the maximum of the two directed distances
    return max(d1, d2)

def compute_volume_change_percent(original_volume, new_volume):
    """Compute percentage change in volume."""
    if original_volume == 0:
        return 0.0
    return ((new_volume - original_volume) / original_volume) * 100
