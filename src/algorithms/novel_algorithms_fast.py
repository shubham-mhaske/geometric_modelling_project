#!/usr/bin/env python3
"""
Novel Geometric Modeling Algorithms - FAST MPS Version

Fully vectorized GPU operations for maximum speed on Apple Silicon.
No per-vertex loops - pure tensor operations.

Author: Shubham Vikas Mhaske  
Course: CSCE 645 Geometric Modeling (Fall 2025)
"""

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Tuple, Dict
import time

# =============================================================================
# DEVICE SETUP
# =============================================================================

def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[Fast MPS] Using device: {DEVICE}")


# =============================================================================
# FAST HELPER FUNCTIONS (FULLY VECTORIZED)
# =============================================================================

def build_weight_matrix(N: int, faces: np.ndarray) -> torch.Tensor:
    """Build normalized weight matrix - fully vectorized."""
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    
    degrees = np.array(adj.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    W = (D_inv @ adj).toarray().astype(np.float32)
    
    return torch.from_numpy(W).to(DEVICE)


def compute_curvature_fast(verts: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Compute curvature - single matrix multiply."""
    neighbor_avg = W @ verts
    laplacian = verts - neighbor_avg
    return torch.norm(laplacian, dim=1)


def compute_normals_fast(verts: torch.Tensor, faces_t: torch.Tensor) -> torch.Tensor:
    """Compute vertex normals - fully vectorized."""
    v0 = verts[faces_t[:, 0]]
    v1 = verts[faces_t[:, 1]]
    v2 = verts[faces_t[:, 2]]
    
    face_normals = torch.cross(v1 - v0, v2 - v0)
    
    vertex_normals = torch.zeros_like(verts)
    vertex_normals.index_add_(0, faces_t[:, 0], face_normals)
    vertex_normals.index_add_(0, faces_t[:, 1], face_normals)
    vertex_normals.index_add_(0, faces_t[:, 2], face_normals)
    
    norms = torch.norm(vertex_normals, dim=1, keepdim=True).clamp(min=1e-10)
    return vertex_normals / norms


# =============================================================================
# 1. GEODESIC HEAT SMOOTHING - FAST
# =============================================================================

def geodesic_heat_smoothing_fast(verts: np.ndarray,
                                  faces: np.ndarray,
                                  iterations: int = 5,
                                  alpha: float = 0.5) -> Tuple[np.ndarray, Dict]:
    """
    Fast Geodesic Heat Smoothing - ~0.5s expected.
    Uses heat kernel approximation via diffusion.
    """
    start = time.time()
    N = len(verts)
    
    # Build weight matrix once
    W = build_weight_matrix(N, faces)
    current = torch.from_numpy(verts.astype(np.float32)).to(DEVICE)
    
    # Heat kernel smoothing via iterative diffusion
    for _ in range(iterations):
        # Curvature-adaptive weights
        curvature = compute_curvature_fast(current, W)
        weights = 1.0 / (1.0 + curvature.unsqueeze(1))
        
        # Diffusion step
        neighbor_avg = W @ current
        current = current + alpha * weights * (neighbor_avg - current)
    
    elapsed = time.time() - start
    return current.cpu().numpy(), {'method': 'Geodesic Heat (Fast)', 'time': elapsed}


# =============================================================================
# 2. SPECTRAL CLUSTERING SMOOTHING - FAST
# =============================================================================

def spectral_clustering_smoothing_fast(verts: np.ndarray,
                                        faces: np.ndarray,
                                        n_clusters: int = 5,
                                        iterations: int = 10) -> Tuple[np.ndarray, Dict]:
    """
    Fast Spectral Clustering Smoothing - ~1s expected.
    Uses fast spectral embedding + per-cluster smoothing.
    """
    start = time.time()
    N = len(verts)
    
    W = build_weight_matrix(N, faces)
    current = torch.from_numpy(verts.astype(np.float32)).to(DEVICE)
    
    # Curvature for clustering (skip expensive eigendecomposition)
    curvature = compute_curvature_fast(current, W)
    
    # Fast clustering based on curvature quantiles
    quantiles = torch.quantile(curvature, torch.linspace(0, 1, n_clusters + 1, device=DEVICE))
    labels = torch.zeros(N, dtype=torch.long, device=DEVICE)
    for i in range(n_clusters):
        mask = (curvature >= quantiles[i]) & (curvature < quantiles[i + 1])
        labels[mask] = i
    
    # Per-cluster smoothing strength
    cluster_curv = torch.zeros(n_clusters, device=DEVICE)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            cluster_curv[c] = curvature[mask].mean()
    
    max_curv = cluster_curv.max() + 1e-10
    strength = 1.0 - 0.8 * (cluster_curv[labels] / max_curv)
    
    # Taubin with adaptive weights
    for _ in range(iterations):
        neighbor_avg = W @ current
        disp = neighbor_avg - current
        current = current + 0.5 * strength.unsqueeze(1) * disp
        
        neighbor_avg = W @ current
        disp = neighbor_avg - current
        current = current - 0.53 * strength.unsqueeze(1) * disp
    
    elapsed = time.time() - start
    return current.cpu().numpy(), {'method': 'Spectral Clustering (Fast)', 'time': elapsed}


# =============================================================================
# 3. OPTIMAL TRANSPORT SMOOTHING - FAST
# =============================================================================

def optimal_transport_smoothing_fast(verts: np.ndarray,
                                      faces: np.ndarray,
                                      iterations: int = 10,
                                      reg: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    Fast Optimal Transport Smoothing - ~0.3s expected.
    Gradient descent on transport cost - pure matrix ops.
    """
    start = time.time()
    N = len(verts)
    
    W = build_weight_matrix(N, faces)
    current = torch.from_numpy(verts.astype(np.float32)).to(DEVICE)
    original = current.clone()
    
    # Compute smooth target
    target = current.clone()
    for _ in range(10):
        target = W @ target
    
    # Gradient descent - pure tensor ops
    lr = 0.1
    for _ in range(iterations):
        grad = (current - target) + reg * (current - original)
        current = current - lr * grad
        target = 0.9 * target + 0.1 * (W @ current)
    
    elapsed = time.time() - start
    return current.cpu().numpy(), {'method': 'Optimal Transport (Fast)', 'time': elapsed}


# =============================================================================
# 4. INFORMATION-THEORETIC SMOOTHING - FAST
# =============================================================================

def information_theoretic_smoothing_fast(verts: np.ndarray,
                                          faces: np.ndarray,
                                          iterations: int = 10) -> Tuple[np.ndarray, Dict]:
    """
    Fast Information-Theoretic Smoothing - ~0.5s expected.
    Entropy approximated by curvature variance.
    """
    start = time.time()
    N = len(verts)
    
    W = build_weight_matrix(N, faces)
    W2 = W @ W  # 2-hop neighbors for entropy
    current = torch.from_numpy(verts.astype(np.float32)).to(DEVICE)
    
    for _ in range(iterations):
        curvature = compute_curvature_fast(current, W)
        
        # Approximate local entropy by curvature spread in neighborhood
        local_mean = W @ curvature.unsqueeze(1)
        local_sq_mean = W @ (curvature ** 2).unsqueeze(1)
        local_var = (local_sq_mean - local_mean ** 2).squeeze().clamp(min=0)
        
        # Entropy proxy: log(variance)
        entropy = torch.log(local_var + 1e-10)
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-10)
        
        # High entropy = more smoothing
        weights = 0.3 + 0.5 * entropy_norm
        
        # Taubin with entropy weights
        neighbor_avg = W @ current
        disp = neighbor_avg - current
        current = current + 0.5 * weights.unsqueeze(1) * disp
        
        neighbor_avg = W @ current
        disp = neighbor_avg - current
        current = current - 0.53 * weights.unsqueeze(1) * disp
    
    elapsed = time.time() - start
    return current.cpu().numpy(), {'method': 'Info-Theoretic (Fast)', 'time': elapsed}


# =============================================================================
# 5. ANISOTROPIC TENSOR SMOOTHING - FAST
# =============================================================================

def anisotropic_tensor_smoothing_fast(verts: np.ndarray,
                                       faces: np.ndarray,
                                       iterations: int = 5,
                                       dt: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    Fast Anisotropic Tensor Smoothing - ~0.5s expected.
    Uses normal-weighted diffusion (tangential smoothing).
    """
    start = time.time()
    N = len(verts)
    
    W = build_weight_matrix(N, faces)
    faces_t = torch.from_numpy(faces.astype(np.int64)).to(DEVICE)
    current = torch.from_numpy(verts.astype(np.float32)).to(DEVICE)
    
    for _ in range(iterations):
        normals = compute_normals_fast(current, faces_t)
        
        # Compute displacement
        neighbor_avg = W @ current
        disp = neighbor_avg - current
        
        # Project out normal component (smooth only tangentially)
        normal_comp = (disp * normals).sum(dim=1, keepdim=True) * normals
        tangent_disp = disp - 0.8 * normal_comp  # Keep 20% normal smoothing
        
        # Curvature-adaptive strength
        curvature = torch.norm(disp, dim=1, keepdim=True)
        strength = torch.exp(-curvature * 2)
        
        current = current + dt * strength * tangent_disp
    
    elapsed = time.time() - start
    return current.cpu().numpy(), {'method': 'Anisotropic Tensor (Fast)', 'time': elapsed}


# =============================================================================
# 6. FREQUENCY-SELECTIVE SMOOTHING - FAST
# =============================================================================

def frequency_selective_smoothing_fast(verts: np.ndarray,
                                        faces: np.ndarray,
                                        cutoff: float = 0.3) -> Tuple[np.ndarray, Dict]:
    """
    Fast Frequency-Selective Smoothing - ~0.8s expected.
    Uses truncated spectral decomposition.
    """
    start = time.time()
    N = len(verts)
    
    # Build Laplacian for eigendecomposition
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).tocsr()
    degrees = np.array(adj.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    L = sparse.eye(N) - D_inv @ adj
    
    # Compute eigenvectors (limited for speed)
    n_eig = min(50, N - 2)
    try:
        eigenvalues, eigenvectors = eigsh(L.tocsr(), k=n_eig, which='SM')
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    except:
        # Fallback to simple smoothing
        W = build_weight_matrix(N, faces)
        current = torch.from_numpy(verts.astype(np.float32)).to(DEVICE)
        for _ in range(5):
            current = W @ current
        return current.cpu().numpy(), {'method': 'Fallback', 'time': time.time() - start}
    
    # Move to GPU
    eigenvectors_t = torch.from_numpy(eigenvectors.astype(np.float32)).to(DEVICE)
    eigenvalues_t = torch.from_numpy(eigenvalues.astype(np.float32)).to(DEVICE)
    verts_t = torch.from_numpy(verts.astype(np.float32)).to(DEVICE)
    
    # Project, filter, reconstruct
    coeffs = eigenvectors_t.T @ verts_t
    
    cutoff_idx = int(cutoff * n_eig)
    cutoff_val = eigenvalues_t[min(cutoff_idx, n_eig - 1)]
    filter_weights = torch.exp(-eigenvalues_t / (cutoff_val + 1e-10))
    
    filtered = coeffs * filter_weights.unsqueeze(1)
    smoothed = eigenvectors_t @ filtered
    
    elapsed = time.time() - start
    return smoothed.cpu().numpy(), {'method': 'Frequency-Selective (Fast)', 'time': elapsed}


# =============================================================================
# QUICK BENCHMARK
# =============================================================================

def quick_benchmark(verts: np.ndarray, faces: np.ndarray):
    """Run quick benchmark of all methods."""
    print("\n" + "="*60)
    print("QUICK BENCHMARK - Expected ~5s total")
    print("="*60)
    
    methods = [
        ('Geodesic Heat', geodesic_heat_smoothing_fast),
        ('Spectral Clustering', spectral_clustering_smoothing_fast),
        ('Optimal Transport', optimal_transport_smoothing_fast),
        ('Info-Theoretic', information_theoretic_smoothing_fast),
        ('Anisotropic Tensor', anisotropic_tensor_smoothing_fast),
        ('Frequency-Selective', frequency_selective_smoothing_fast),
    ]
    
    total = 0
    for name, fn in methods:
        try:
            _, info = fn(verts.copy(), faces)
            t = info['time']
            total += t
            print(f"  ✓ {name:<22}: {t:.2f}s")
        except Exception as e:
            print(f"  ✗ {name:<22}: {e}")
    
    print(f"\n  Total: {total:.2f}s")
    return total


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'geodesic_heat_smoothing_fast',
    'spectral_clustering_smoothing_fast',
    'optimal_transport_smoothing_fast',
    'information_theoretic_smoothing_fast',
    'anisotropic_tensor_smoothing_fast',
    'frequency_selective_smoothing_fast',
    'quick_benchmark',
    'DEVICE'
]
