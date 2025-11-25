"""
Bilateral Mesh Smoothing

Feature-preserving smoothing that respects edges and high-curvature regions.
Unlike uniform Laplacian, bilateral smoothing weighs neighbors by both
spatial distance AND normal similarity, preserving sharp features.

Reference: Fleishman et al., "Bilateral Mesh Denoising" (2003)
"""

import numpy as np
from scipy import sparse


def compute_vertex_normals(verts, faces):
    """
    Compute per-vertex normals by averaging incident face normals.
    
    Args:
        verts: (N, 3) vertex positions
        faces: (M, 3) face indices
    
    Returns:
        normals: (N, 3) normalized vertex normals
    """
    num_verts = verts.shape[0]
    normals = np.zeros((num_verts, 3))
    
    # Compute face normals and accumulate to vertices
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    e1 = v1 - v0
    e2 = v2 - v0
    face_normals = np.cross(e1, e2)
    
    # Normalize face normals
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1, norms)
    face_normals = face_normals / norms
    
    # Accumulate to vertices
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    
    # Normalize vertex normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1, norms)
    normals = normals / norms
    
    return normals


def bilateral_smoothing(verts, faces, iterations=10, sigma_c=None, sigma_s=0.3):
    """
    Apply bilateral mesh smoothing (feature-preserving).
    
    The bilateral filter computes new vertex positions as:
        v' = v + n * Σ(W_c * W_s * h) / Σ(W_c * W_s)
    
    where:
        W_c = exp(-||p - q||² / (2 * σ_c²))  # spatial weight
        W_s = exp(-(<n, p-q>)² / (2 * σ_s²))  # range/normal weight  
        h = <n, q - p>  # height difference along normal
    
    Args:
        verts: (N, 3) vertex positions
        faces: (M, 3) face indices
        iterations: number of smoothing passes
        sigma_c: spatial bandwidth (auto-computed from edge length if None)
        sigma_s: range bandwidth (controls feature preservation, 0.1-0.5)
    
    Returns:
        smoothed_verts: (N, 3) smoothed vertex positions
    """
    verts = verts.copy().astype(np.float64)
    num_verts = verts.shape[0]
    
    # Build adjacency list
    adjacency = [set() for _ in range(num_verts)]
    for face in faces:
        i, j, k = face
        adjacency[i].update([j, k])
        adjacency[j].update([i, k])
        adjacency[k].update([i, j])
    
    # Compute average edge length for sigma_c if not provided
    if sigma_c is None:
        edge_lengths = []
        for face in faces:
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                edge_lengths.append(np.linalg.norm(verts[face[a]] - verts[face[b]]))
        sigma_c = np.mean(edge_lengths) * 2.0
    
    sigma_c_sq = sigma_c ** 2
    sigma_s_sq = sigma_s ** 2
    
    for _ in range(iterations):
        normals = compute_vertex_normals(verts, faces)
        new_verts = verts.copy()
        
        for i in range(num_verts):
            if len(adjacency[i]) == 0:
                continue
            
            p = verts[i]
            n = normals[i]
            
            sum_weights = 0.0
            sum_weighted_h = 0.0
            
            # Two-ring neighborhood for better smoothing
            neighbors = adjacency[i].copy()
            for j in list(adjacency[i]):
                neighbors.update(adjacency[j])
            neighbors.discard(i)
            
            for j in neighbors:
                q = verts[j]
                diff = q - p
                
                # Spatial distance
                dist_sq = np.dot(diff, diff)
                
                # Height along normal
                h = np.dot(n, diff)
                
                # Bilateral weights
                w_c = np.exp(-dist_sq / (2 * sigma_c_sq))
                w_s = np.exp(-h * h / (2 * sigma_s_sq))
                
                weight = w_c * w_s
                sum_weights += weight
                sum_weighted_h += weight * h
            
            if sum_weights > 1e-12:
                # Move vertex along normal direction
                offset = (sum_weighted_h / sum_weights) * n
                new_verts[i] = p + offset
        
        verts = new_verts
    
    return verts


def guided_smoothing(verts, faces, iterations=10, lambda_val=0.5, 
                     curvature_threshold=0.5, vertex_labels=None):
    """
    Curvature-guided adaptive smoothing.
    
    Smooths more in flat regions, less in high-curvature regions.
    Optionally respects label boundaries.
    
    Args:
        verts: (N, 3) vertex positions
        faces: (M, 3) face indices
        iterations: number of smoothing passes
        lambda_val: base smoothing strength
        curvature_threshold: curvature value above which smoothing is reduced
        vertex_labels: (N,) optional label array for boundary preservation
    
    Returns:
        smoothed_verts: (N, 3) smoothed vertex positions
    """
    from .metrics import compute_mean_curvature
    
    verts = verts.copy().astype(np.float64)
    num_verts = verts.shape[0]
    
    # Build adjacency
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], 
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows))
    
    A = sparse.coo_matrix((data, (rows, cols)), shape=(num_verts, num_verts))
    A = A.tocsr()
    A.data[:] = 1.0
    
    for _ in range(iterations):
        # Compute curvature-based weights
        H, _, _ = compute_mean_curvature(verts, faces)
        
        # Adaptive lambda: reduce smoothing in high-curvature regions
        # lambda_adaptive = lambda_val * exp(-H / threshold)
        curvature_weight = np.exp(-H / curvature_threshold)
        curvature_weight = np.clip(curvature_weight, 0.1, 1.0)
        
        # Apply label boundary constraint if provided
        if vertex_labels is not None:
            labels = np.asarray(vertex_labels).reshape(-1)
            # Reduce smoothing at label boundaries
            for i in range(num_verts):
                neighbors = A.indices[A.indptr[i]:A.indptr[i+1]]
                if len(neighbors) > 0:
                    different_labels = labels[neighbors] != labels[i]
                    if np.any(different_labels):
                        curvature_weight[i] *= 0.1  # Strong damping at boundaries
        
        adaptive_lambda = lambda_val * curvature_weight
        
        # Compute weighted Laplacian
        degrees = np.array(A.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv = sparse.diags(1.0 / degrees)
        W = D_inv @ A
        
        # Per-vertex adaptive smoothing
        neighbor_avg = W @ verts
        new_verts = np.zeros_like(verts)
        
        for i in range(num_verts):
            lam = adaptive_lambda[i]
            new_verts[i] = (1 - lam) * verts[i] + lam * neighbor_avg[i]
        
        verts = new_verts
    
    return verts
