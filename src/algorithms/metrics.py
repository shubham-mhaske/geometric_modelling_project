import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy import sparse

def compute_mean_curvature(verts, faces):
    """
    Compute discrete mean curvature at each vertex using the Laplace-Beltrami operator.
    Vectorized implementation for performance on large meshes.
    
    Args:
        verts: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle indices
    
    Returns:
        H: (N,) array of mean curvature values at each vertex
        H_mean: scalar mean curvature across the mesh
        H_std: scalar standard deviation of curvature
    """
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    
    # Get vertices for all faces
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    # Edge vectors
    e01 = v1 - v0  # edge from v0 to v1
    e02 = v2 - v0  # edge from v0 to v2
    e12 = v2 - v1  # edge from v1 to v2
    e10 = -e01
    e20 = -e02
    e21 = -e12
    
    # Cross products for cotangent calculation
    cross_0 = np.cross(e01, e02)  # at vertex 0
    cross_1 = np.cross(e12, e10)  # at vertex 1
    cross_2 = np.cross(e20, e21)  # at vertex 2
    
    # Areas (magnitude of cross product)
    area_0 = np.linalg.norm(cross_0, axis=1)
    area_1 = np.linalg.norm(cross_1, axis=1)
    area_2 = np.linalg.norm(cross_2, axis=1)
    
    eps = 1e-12
    
    # Cotangent weights
    cot_0 = np.sum(e01 * e02, axis=1) / (area_0 + eps)  # cot angle at v0
    cot_1 = np.sum(e12 * e10, axis=1) / (area_1 + eps)  # cot angle at v1
    cot_2 = np.sum(e20 * e21, axis=1) / (area_2 + eps)  # cot angle at v2
    
    # Clamp cotangents
    cot_0 = np.clip(cot_0, -1e6, 1e6)
    cot_1 = np.clip(cot_1, -1e6, 1e6)
    cot_2 = np.clip(cot_2, -1e6, 1e6)
    
    # Triangle areas
    triangle_areas = area_0 / 2
    
    # Accumulate vertex areas
    vertex_areas = np.zeros(num_verts)
    np.add.at(vertex_areas, faces[:, 0], triangle_areas / 3)
    np.add.at(vertex_areas, faces[:, 1], triangle_areas / 3)
    np.add.at(vertex_areas, faces[:, 2], triangle_areas / 3)
    
    # Build sparse Laplacian using cotangent weights
    # Edge (1,2) opposite to vertex 0: weight = cot_0/2
    # Edge (2,0) opposite to vertex 1: weight = cot_1/2
    # Edge (0,1) opposite to vertex 2: weight = cot_2/2
    
    rows = np.concatenate([
        faces[:, 1], faces[:, 2],  # edge 1-2
        faces[:, 2], faces[:, 0],  # edge 2-0
        faces[:, 0], faces[:, 1],  # edge 0-1
    ])
    cols = np.concatenate([
        faces[:, 2], faces[:, 1],  # edge 1-2
        faces[:, 0], faces[:, 2],  # edge 2-0
        faces[:, 1], faces[:, 0],  # edge 0-1
    ])
    data = np.concatenate([
        cot_0 / 2, cot_0 / 2,
        cot_1 / 2, cot_1 / 2,
        cot_2 / 2, cot_2 / 2,
    ])
    
    # Off-diagonal entries
    L = sparse.coo_matrix((data, (rows, cols)), shape=(num_verts, num_verts))
    L = L.tocsr()
    L.sum_duplicates()
    
    # Diagonal: negative sum of row
    diag = -np.array(L.sum(axis=1)).flatten()
    L = L + sparse.diags(diag)
    
    # Compute Laplacian of vertex positions
    laplacian_verts = L @ verts
    
    # Mean curvature: H = |Δv| / (2 * area)
    vertex_areas = np.maximum(vertex_areas, eps)
    H = np.linalg.norm(laplacian_verts, axis=1) / (2 * vertex_areas)
    
    # Clamp extreme values
    H = np.clip(H, 0, np.percentile(H, 99))
    
    return H, float(np.mean(H)), float(np.std(H))


def compute_gaussian_curvature(verts, faces):
    """
    Compute discrete Gaussian curvature at each vertex using angle defect.
    Vectorized implementation for performance.
    
    K = (2π - Σθ) / A
    where Σθ is the sum of angles around the vertex and A is the vertex area.
    
    Args:
        verts: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle indices
    
    Returns:
        K: (N,) array of Gaussian curvature values at each vertex
        K_mean: scalar mean Gaussian curvature
        K_std: scalar standard deviation
    """
    num_verts = verts.shape[0]
    
    # Get vertices for all faces
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    # Edge vectors
    e01 = v1 - v0
    e02 = v2 - v0
    e10 = -e01
    e12 = v2 - v1
    e20 = -e02
    e21 = -e12
    
    eps = 1e-12
    
    # Normalize edge vectors
    e01_n = e01 / (np.linalg.norm(e01, axis=1, keepdims=True) + eps)
    e02_n = e02 / (np.linalg.norm(e02, axis=1, keepdims=True) + eps)
    e10_n = e10 / (np.linalg.norm(e10, axis=1, keepdims=True) + eps)
    e12_n = e12 / (np.linalg.norm(e12, axis=1, keepdims=True) + eps)
    e20_n = e20 / (np.linalg.norm(e20, axis=1, keepdims=True) + eps)
    e21_n = e21 / (np.linalg.norm(e21, axis=1, keepdims=True) + eps)
    
    # Compute angles at each vertex
    angle_0 = np.arccos(np.clip(np.sum(e01_n * e02_n, axis=1), -1, 1))
    angle_1 = np.arccos(np.clip(np.sum(e10_n * e12_n, axis=1), -1, 1))
    angle_2 = np.arccos(np.clip(np.sum(e20_n * e21_n, axis=1), -1, 1))
    
    # Accumulate angles
    angle_sum = np.zeros(num_verts)
    np.add.at(angle_sum, faces[:, 0], angle_0)
    np.add.at(angle_sum, faces[:, 1], angle_1)
    np.add.at(angle_sum, faces[:, 2], angle_2)
    
    # Triangle areas
    triangle_areas = np.linalg.norm(np.cross(e01, e02), axis=1) / 2
    
    # Vertex areas
    vertex_areas = np.zeros(num_verts)
    np.add.at(vertex_areas, faces[:, 0], triangle_areas / 3)
    np.add.at(vertex_areas, faces[:, 1], triangle_areas / 3)
    np.add.at(vertex_areas, faces[:, 2], triangle_areas / 3)
    
    # Gaussian curvature = angle defect / area
    vertex_areas = np.maximum(vertex_areas, eps)
    K = (2 * np.pi - angle_sum) / vertex_areas
    
    # Clamp extreme values
    K = np.clip(K, np.percentile(K, 1), np.percentile(K, 99))
    
    return K, float(np.mean(K)), float(np.std(K))


def compute_curvature_error(verts_orig, verts_new, faces):
    """
    Compute the curvature preservation error between original and processed mesh.
    
    Args:
        verts_orig: (N, 3) original vertex positions
        verts_new: (N, 3) processed vertex positions  
        faces: (M, 3) triangle indices (assumed same topology)
    
    Returns:
        dict with mean curvature error, Gaussian curvature error, and correlation
    """
    H_orig, H_orig_mean, _ = compute_mean_curvature(verts_orig, faces)
    H_new, H_new_mean, _ = compute_mean_curvature(verts_new, faces)
    
    K_orig, K_orig_mean, _ = compute_gaussian_curvature(verts_orig, faces)
    K_new, K_new_mean, _ = compute_gaussian_curvature(verts_new, faces)
    
    # Compute errors
    H_error = np.abs(H_new - H_orig)
    K_error = np.abs(K_new - K_orig)
    
    # Correlation (how well curvature pattern is preserved)
    H_corr = np.corrcoef(H_orig, H_new)[0, 1] if np.std(H_orig) > 0 and np.std(H_new) > 0 else 0
    K_corr = np.corrcoef(K_orig, K_new)[0, 1] if np.std(K_orig) > 0 and np.std(K_new) > 0 else 0
    
    return {
        "mean_curvature_error": float(np.mean(H_error)),
        "mean_curvature_max_error": float(np.max(H_error)),
        "mean_curvature_correlation": float(H_corr),
        "gaussian_curvature_error": float(np.mean(K_error)),
        "gaussian_curvature_max_error": float(np.max(K_error)),
        "gaussian_curvature_correlation": float(K_corr),
    }


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
