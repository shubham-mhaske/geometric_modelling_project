"""
ACANS v2: Optimized Adaptive Curvature-Aware Neural Smoothing

Improved version with:
1. Vectorized operations for speed
2. Better curvature-adaptive weighting
3. Enhanced feature preservation
4. Hybrid approach combining best of classical and neural

Author: Shubham Vikas Mhaske
Course: CSCE 645 Geometric Modeling (Fall 2025)
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional, Dict
from collections import defaultdict


def _build_adjacency_sparse(num_verts: int, faces: np.ndarray) -> sparse.csr_matrix:
    """Build sparse adjacency matrix."""
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows))
    
    A = sparse.coo_matrix((data, (rows, cols)), shape=(num_verts, num_verts))
    A = A.tocsr()
    return A


def _compute_vertex_normals_vectorized(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute vertex normals using vectorized operations."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    # Face normals (not normalized, magnitude = 2*area)
    face_normals = np.cross(v1 - v0, v2 - v0)
    
    # Accumulate to vertices
    vertex_normals = np.zeros_like(verts)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    
    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    return vertex_normals / norms


def _compute_curvature_vectorized(verts: np.ndarray, 
                                   adj: sparse.csr_matrix,
                                   degrees: np.ndarray) -> np.ndarray:
    """Compute discrete curvature using sparse matrix operations."""
    # Neighbor average (normalized Laplacian target)
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    neighbor_avg = D_inv @ adj @ verts
    
    # Laplacian (displacement from average)
    laplacian = verts - neighbor_avg
    
    # Curvature magnitude normalized by local scale
    curv_magnitude = np.linalg.norm(laplacian, axis=1)
    
    # Normalize by mean edge length (approximate)
    mean_edge = np.mean(curv_magnitude[curv_magnitude > 0])
    if mean_edge > 1e-10:
        curv_magnitude = curv_magnitude / mean_edge
    
    return curv_magnitude


def acans_v2_smoothing(verts: np.ndarray,
                       faces: np.ndarray,
                       iterations: int = 10,
                       curvature_threshold: float = 0.2,
                       curvature_sharpness: float = 5.0,
                       normal_weight: float = 0.4,
                       lambda_smooth: float = 0.6,
                       mu_inflate: float = -0.55,
                       use_taubin_base: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    ACANS v2: Optimized Adaptive Curvature-Aware Smoothing
    
    Key Innovation: Combines Taubin's volume preservation with
    curvature-adaptive weighting and normal-guided updates.
    
    Args:
        verts: Vertex positions [N, 3]
        faces: Face indices [M, 3]  
        iterations: Number of smoothing iterations
        curvature_threshold: Curvature level separating features from noise
        curvature_sharpness: Sharpness of adaptive weight transition
        normal_weight: Weight for normal-projected component (0-1)
        lambda_smooth: Smoothing strength (Taubin λ)
        mu_inflate: Inflation strength (Taubin μ, should be negative)
        use_taubin_base: Whether to use Taubin two-step as base
    
    Returns:
        Tuple of (smoothed_verts, info_dict)
    """
    import time
    start_time = time.time()
    
    N = len(verts)
    current = verts.copy().astype(np.float64)
    
    # Build adjacency
    adj = _build_adjacency_sparse(N, faces)
    degrees = np.array(adj.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    W = D_inv @ adj  # Normalized adjacency
    
    stats = {
        'curvature_per_iter': [],
        'volume_per_iter': []
    }
    
    for it in range(iterations):
        # Compute normals
        normals = _compute_vertex_normals_vectorized(current, faces)
        
        # Compute curvature
        curvature = _compute_curvature_vectorized(current, adj, degrees)
        
        # Compute adaptive weights (sigmoid)
        # High curvature -> low weight (preserve features)
        # Low curvature -> high weight (smooth noise)
        x = curvature_sharpness * (curvature - curvature_threshold)
        adaptive_weights = 1.0 / (1.0 + np.exp(x))
        adaptive_weights = np.clip(adaptive_weights, 0.1, 1.0)
        
        # Compute Laplacian displacement
        neighbor_avg = W @ current
        displacement = neighbor_avg - current
        
        # Normal-guided projection
        # Project displacement onto normal to reduce tangential drift
        normal_component = np.sum(displacement * normals, axis=1, keepdims=True) * normals
        tangent_component = displacement - normal_component
        
        # Blend normal and tangent components
        guided_displacement = normal_weight * normal_component + (1 - normal_weight) * tangent_component
        
        # Apply adaptive weights
        weighted_displacement = adaptive_weights.reshape(-1, 1) * guided_displacement
        
        if use_taubin_base:
            # Taubin-style two-step update with adaptive modification
            # Step 1: Smooth (shrink)
            current = current + lambda_smooth * weighted_displacement
            
            # Recompute displacement for inflation step
            neighbor_avg = W @ current
            displacement = neighbor_avg - current
            
            # Apply same adaptive weights to inflation
            weighted_displacement = adaptive_weights.reshape(-1, 1) * displacement
            
            # Step 2: Inflate (expand back)
            current = current + mu_inflate * weighted_displacement
        else:
            # Simple adaptive smoothing
            current = current + lambda_smooth * weighted_displacement
        
        # Track statistics
        stats['curvature_per_iter'].append(float(np.mean(curvature)))
    
    elapsed = time.time() - start_time
    
    info = {
        'method': 'ACANS v2',
        'time_seconds': elapsed,
        'iterations': iterations,
        'parameters': {
            'curvature_threshold': curvature_threshold,
            'curvature_sharpness': curvature_sharpness,
            'normal_weight': normal_weight,
            'lambda': lambda_smooth,
            'mu': mu_inflate
        },
        'stats': stats
    }
    
    return current.astype(np.float32), info


def hybrid_neural_classical_smoothing(verts: np.ndarray,
                                      faces: np.ndarray,
                                      iterations: int = 5,
                                      neural_iterations: int = 2,
                                      classical_iterations: int = 3) -> Tuple[np.ndarray, Dict]:
    """
    Hybrid approach: Neural feature detection + Classical smoothing
    
    Novel idea: Use neural-style analysis to detect features,
    then apply classical smoothing with spatially-varying parameters.
    
    This combines:
    - Interpretability of classical methods
    - Adaptivity of learning-based approaches
    - No training required (unsupervised feature detection)
    """
    import time
    start_time = time.time()
    
    N = len(verts)
    current = verts.copy().astype(np.float64)
    
    # Build adjacency
    adj = _build_adjacency_sparse(N, faces)
    degrees = np.array(adj.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    W = D_inv @ adj
    
    # Phase 1: Feature detection (neural-inspired)
    # Compute multi-scale curvature features
    curvature_1ring = _compute_curvature_vectorized(current, adj, degrees)
    
    # 2-ring curvature (apply Laplacian twice)
    W2 = W @ W
    neighbor_avg_2 = W2 @ current
    curvature_2ring = np.linalg.norm(current - neighbor_avg_2, axis=1)
    
    # Combine multi-scale features
    feature_strength = 0.7 * curvature_1ring + 0.3 * curvature_2ring
    
    # Detect features using adaptive threshold
    threshold = np.percentile(feature_strength, 70)
    feature_mask = feature_strength > threshold
    
    # Phase 2: Adaptive classical smoothing
    for it in range(iterations):
        normals = _compute_vertex_normals_vectorized(current, faces)
        
        # Recompute features periodically
        if it % 2 == 0:
            curvature = _compute_curvature_vectorized(current, adj, degrees)
            adaptive_weights = 1.0 - 0.8 * (curvature / (np.max(curvature) + 1e-10))
            adaptive_weights = np.clip(adaptive_weights, 0.1, 1.0)
            
            # Reduce weights for detected features
            adaptive_weights[feature_mask] *= 0.3
        
        # Laplacian step
        neighbor_avg = W @ current
        displacement = neighbor_avg - current
        
        # Normal-guided update
        normal_comp = np.sum(displacement * normals, axis=1, keepdims=True) * normals
        guided = 0.7 * normal_comp + 0.3 * displacement
        
        # Taubin-style with adaptive weights
        current = current + 0.5 * adaptive_weights.reshape(-1, 1) * guided
        
        # Inflation step
        neighbor_avg = W @ current
        displacement = neighbor_avg - current
        current = current - 0.53 * adaptive_weights.reshape(-1, 1) * displacement
    
    elapsed = time.time() - start_time
    
    info = {
        'method': 'Hybrid Neural-Classical',
        'time_seconds': elapsed,
        'iterations': iterations,
        'features_detected': int(np.sum(feature_mask)),
        'feature_ratio': float(np.mean(feature_mask))
    }
    
    return current.astype(np.float32), info


def gradient_domain_smoothing(verts: np.ndarray,
                              faces: np.ndarray,
                              iterations: int = 5,
                              gradient_weight: float = 0.3,
                              position_weight: float = 0.7) -> Tuple[np.ndarray, Dict]:
    """
    Gradient Domain Mesh Smoothing
    
    Novel idea: Smooth the mesh gradients (edge vectors) instead of
    positions directly, then reconstruct positions.
    
    This better preserves local shape while removing noise.
    Similar to gradient domain image editing but for meshes.
    """
    import time
    start_time = time.time()
    
    N = len(verts)
    current = verts.copy().astype(np.float64)
    
    # Build adjacency
    adj = _build_adjacency_sparse(N, faces)
    degrees = np.array(adj.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    W = D_inv @ adj
    
    for it in range(iterations):
        # Compute current gradients (Laplacian = divergence of gradients)
        neighbor_avg = W @ current
        laplacian = current - neighbor_avg  # This encodes local shape
        
        # Smooth the Laplacian field
        smooth_laplacian = W @ laplacian
        
        # Blend original and smoothed Laplacian
        target_laplacian = gradient_weight * smooth_laplacian + (1 - gradient_weight) * laplacian
        
        # Reconstruct positions: solve v - W*v = target_laplacian
        # Using iterative update: v_new = W*v + target_laplacian
        # This is gradient descent on ||Lv - target||^2
        for sub_it in range(3):
            current = position_weight * (W @ current + target_laplacian) + (1 - position_weight) * current
    
    elapsed = time.time() - start_time
    
    info = {
        'method': 'Gradient Domain Smoothing',
        'time_seconds': elapsed,
        'iterations': iterations
    }
    
    return current.astype(np.float32), info


def edge_aware_bilateral_smoothing(verts: np.ndarray,
                                   faces: np.ndarray,
                                   iterations: int = 3,
                                   sigma_space: float = None,
                                   sigma_normal: float = 0.5) -> Tuple[np.ndarray, Dict]:
    """
    Edge-Aware Bilateral Mesh Smoothing (Optimized)
    
    Extends bilateral filtering with:
    1. Vectorized operations for speed
    2. Normal-based edge detection
    3. Adaptive spatial sigma
    """
    import time
    start_time = time.time()
    
    N = len(verts)
    current = verts.copy().astype(np.float64)
    
    # Build neighbor lists
    neighbors = defaultdict(list)
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            neighbors[v1].append(v2)
            neighbors[v2].append(v1)
    
    # Convert to arrays for faster access
    neighbor_arrays = {}
    for v in range(N):
        neighbor_arrays[v] = np.array(list(set(neighbors[v])))
    
    # Compute initial edge lengths for spatial sigma
    edge_lengths = []
    for face in faces[:1000]:  # Sample
        for i in range(3):
            e = np.linalg.norm(verts[face[i]] - verts[face[(i+1)%3]])
            edge_lengths.append(e)
    
    if sigma_space is None:
        sigma_space = np.mean(edge_lengths) * 2
    
    for it in range(iterations):
        normals = _compute_vertex_normals_vectorized(current, faces)
        new_verts = current.copy()
        
        for i in range(N):
            neighbor_idx = neighbor_arrays[i]
            if len(neighbor_idx) == 0:
                continue
            
            neighbor_verts = current[neighbor_idx]
            neighbor_normals = normals[neighbor_idx]
            
            # Spatial weights
            diffs = neighbor_verts - current[i]
            spatial_dists = np.linalg.norm(diffs, axis=1)
            spatial_weights = np.exp(-spatial_dists**2 / (2 * sigma_space**2))
            
            # Normal-based range weights (edge preservation)
            normal_dots = np.sum(neighbor_normals * normals[i], axis=1)
            normal_diffs = 1 - np.clip(normal_dots, -1, 1)
            range_weights = np.exp(-normal_diffs**2 / (2 * sigma_normal**2))
            
            # Combined bilateral weights
            weights = spatial_weights * range_weights
            weights = weights / (np.sum(weights) + 1e-10)
            
            # Weighted average
            new_verts[i] = np.sum(neighbor_verts * weights.reshape(-1, 1), axis=0)
        
        current = new_verts
    
    elapsed = time.time() - start_time
    
    info = {
        'method': 'Edge-Aware Bilateral',
        'time_seconds': elapsed,
        'iterations': iterations
    }
    
    return current.astype(np.float32), info


# Export all methods
__all__ = [
    'acans_v2_smoothing',
    'hybrid_neural_classical_smoothing', 
    'gradient_domain_smoothing',
    'edge_aware_bilateral_smoothing'
]
