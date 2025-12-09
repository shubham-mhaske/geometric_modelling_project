"""
Novel Geometric Modeling Algorithms - MPS Accelerated Version

GPU-accelerated versions using Apple Metal Performance Shaders (MPS) backend.
These algorithms leverage PyTorch's MPS support for fast matrix operations
on Apple Silicon GPUs.

Author: Shubham Vikas Mhaske
Course: CSCE 645 Geometric Modeling (Fall 2025)
Instructor: Professor John Keyser

=============================================================================
MPS ACCELERATION
=============================================================================
Heavy computations are offloaded to GPU via PyTorch MPS backend:
- Matrix multiplications
- Eigenvalue decomposition (via iterative methods)
- Large-scale vector operations
- Spectral clustering
=============================================================================
"""

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Tuple, Dict
import warnings

# =============================================================================
# MPS DEVICE SETUP
# =============================================================================

def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"[Novel Algorithms] Using device: {DEVICE}")


def to_torch(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array to torch tensor on device."""
    return torch.from_numpy(arr.astype(np.float32)).to(DEVICE)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.cpu().numpy()


# =============================================================================
# 1. GEODESIC HEAT DIFFUSION SMOOTHING (MPS)
# =============================================================================

def geodesic_heat_smoothing_mps(verts: np.ndarray,
                                 faces: np.ndarray,
                                 iterations: int = 5,
                                 time_scale: float = 1.0,
                                 feature_threshold: float = 0.3) -> Tuple[np.ndarray, Dict]:
    """
    GPU-accelerated Geodesic Heat Diffusion Smoothing.
    
    Uses MPS backend for fast heat kernel computation and vertex updates.
    """
    import time
    start_time = time.time()
    
    N = len(verts)
    
    # Move data to GPU
    verts_t = to_torch(verts)
    current = verts_t.clone()
    
    # Build adjacency on CPU, then create sparse operations
    adj_sparse = build_adjacency_sparse(N, faces)
    degrees = np.array(adj_sparse.sum(axis=1)).flatten()
    
    # Create normalized weight matrix as dense on GPU
    # For MPS, we use a batched approach with neighbor indices
    adj_indices, adj_weights = sparse_to_indexed(adj_sparse, N)
    adj_indices_t = torch.from_numpy(adj_indices).long().to(DEVICE)
    
    # Compute mean edge length for time scale
    edge_lengths = compute_edge_lengths_torch(verts_t, faces)
    t = time_scale * (edge_lengths.mean() ** 2).item()
    
    for it in range(iterations):
        # Compute curvature on GPU
        curvature = compute_curvature_torch(current, adj_indices_t, degrees)
        
        # Heat kernel smoothing
        new_verts = torch.zeros_like(current)
        
        for i in range(N):
            neighbors = adj_indices_t[i]
            valid_mask = neighbors >= 0
            valid_neighbors = neighbors[valid_mask]
            
            if len(valid_neighbors) == 0:
                new_verts[i] = current[i]
                continue
            
            neighbor_verts = current[valid_neighbors]
            
            # Heat kernel weights
            edge_vecs = neighbor_verts - current[i]
            dists = torch.norm(edge_vecs, dim=1)
            heat_weights = torch.exp(-dists**2 / (4 * t))
            
            # Feature-aware modulation
            neighbor_curvatures = curvature[valid_neighbors]
            feature_weights = 1.0 / (1.0 + feature_threshold * neighbor_curvatures)
            
            # Combined weights
            weights = heat_weights * feature_weights
            weights = weights / (weights.sum() + 1e-10)
            
            # Weighted update
            new_verts[i] = (neighbor_verts * weights.unsqueeze(1)).sum(dim=0)
        
        # Blend with original
        alpha = 0.7
        current = alpha * new_verts + (1 - alpha) * current
    
    elapsed = time.time() - start_time
    
    info = {
        'method': 'Geodesic Heat Diffusion (MPS)',
        'device': str(DEVICE),
        'time_seconds': elapsed,
        'iterations': iterations
    }
    
    return to_numpy(current), info


# =============================================================================
# 2. SPECTRAL CLUSTERING-GUIDED SMOOTHING (MPS)
# =============================================================================

def spectral_clustering_smoothing_mps(verts: np.ndarray,
                                       faces: np.ndarray,
                                       n_clusters: int = 5,
                                       iterations: int = 10) -> Tuple[np.ndarray, Dict]:
    """
    GPU-accelerated Spectral Clustering-Guided Smoothing.
    
    Uses MPS for fast matrix operations in spectral clustering and smoothing.
    """
    import time
    start_time = time.time()
    
    N = len(verts)
    
    # Move to GPU
    verts_t = to_torch(verts)
    current = verts_t.clone()
    
    # Build Laplacian
    adj_sparse = build_adjacency_sparse(N, faces)
    degrees = np.array(adj_sparse.sum(axis=1)).flatten()
    
    # Compute eigenvectors on CPU (scipy is faster for sparse eigensolvers)
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    L = sparse.eye(N) - D_inv @ adj_sparse
    
    k = min(n_clusters + 1, N - 2)
    try:
        eigenvalues, eigenvectors = eigsh(L.tocsr(), k=k, which='SM')
        embedding = eigenvectors[:, 1:k]
    except:
        embedding = np.random.randn(N, n_clusters - 1)
    
    # K-means on GPU
    embedding_t = to_torch(embedding)
    labels = kmeans_mps(embedding_t, n_clusters)
    
    # Compute curvature
    adj_indices, _ = sparse_to_indexed(adj_sparse, N)
    adj_indices_t = torch.from_numpy(adj_indices).long().to(DEVICE)
    curvature = compute_curvature_torch(current, adj_indices_t, degrees)
    
    # Per-cluster curvature
    labels_t = torch.from_numpy(labels).long().to(DEVICE)
    cluster_curvatures = torch.zeros(n_clusters, device=DEVICE)
    for c in range(n_clusters):
        mask = labels_t == c
        if mask.any():
            cluster_curvatures[c] = curvature[mask].mean()
    
    # Smoothing strength per vertex
    max_curv = cluster_curvatures.max() + 1e-10
    smoothing_strength = torch.zeros(N, device=DEVICE)
    for i in range(N):
        smoothing_strength[i] = 1.0 - 0.8 * (cluster_curvatures[labels[i]] / max_curv)
    
    # Build dense weight matrix on GPU for fast smoothing
    W_dense = build_weight_matrix_torch(adj_sparse, N)
    
    # Taubin smoothing with per-vertex weights
    for it in range(iterations):
        # Smoothing step
        neighbor_avg = W_dense @ current
        displacement = neighbor_avg - current
        current = current + 0.5 * smoothing_strength.unsqueeze(1) * displacement
        
        # Inflation step
        neighbor_avg = W_dense @ current
        displacement = neighbor_avg - current
        current = current - 0.53 * smoothing_strength.unsqueeze(1) * displacement
    
    elapsed = time.time() - start_time
    
    info = {
        'method': 'Spectral Clustering (MPS)',
        'device': str(DEVICE),
        'time_seconds': elapsed,
        'n_clusters': n_clusters
    }
    
    return to_numpy(current), info


# =============================================================================
# 3. OPTIMAL TRANSPORT SMOOTHING (MPS)
# =============================================================================

def optimal_transport_smoothing_mps(verts: np.ndarray,
                                     faces: np.ndarray,
                                     iterations: int = 10,
                                     regularization: float = 0.1,
                                     target_smoothness: float = 0.5) -> Tuple[np.ndarray, Dict]:
    """
    GPU-accelerated Optimal Transport Smoothing.
    
    Uses MPS for fast gradient computation and vertex updates.
    """
    import time
    start_time = time.time()
    
    N = len(verts)
    
    # Move to GPU
    verts_t = to_torch(verts)
    current = verts_t.clone()
    original = verts_t.clone()
    
    # Build weight matrix on GPU
    adj_sparse = build_adjacency_sparse(N, faces)
    W_dense = build_weight_matrix_torch(adj_sparse, N)
    
    # Compute smooth target
    target = current.clone()
    smooth_iters = int(target_smoothness * 20)
    for _ in range(smooth_iters):
        target = W_dense @ target
    
    # Gradient descent on GPU
    learning_rate = 0.1
    
    for it in range(iterations):
        # Gradient of transport cost (all on GPU)
        grad_target = current - target
        grad_preserve = current - original
        gradient = grad_target + regularization * grad_preserve
        
        # Update
        current = current - learning_rate * gradient
        
        # Update target
        target = 0.9 * target + 0.1 * (W_dense @ current)
    
    elapsed = time.time() - start_time
    
    info = {
        'method': 'Optimal Transport (MPS)',
        'device': str(DEVICE),
        'time_seconds': elapsed,
        'iterations': iterations
    }
    
    return to_numpy(current), info


# =============================================================================
# 4. INFORMATION-THEORETIC SMOOTHING (MPS)
# =============================================================================

def information_theoretic_smoothing_mps(verts: np.ndarray,
                                         faces: np.ndarray,
                                         iterations: int = 10,
                                         entropy_weight: float = 0.5) -> Tuple[np.ndarray, Dict]:
    """
    GPU-accelerated Information-Theoretic Smoothing.
    
    Uses MPS for fast entropy computation and smoothing operations.
    """
    import time
    start_time = time.time()
    
    N = len(verts)
    
    # Move to GPU
    verts_t = to_torch(verts)
    current = verts_t.clone()
    
    # Build adjacency
    adj_sparse = build_adjacency_sparse(N, faces)
    degrees = np.array(adj_sparse.sum(axis=1)).flatten()
    adj_indices, _ = sparse_to_indexed(adj_sparse, N)
    adj_indices_t = torch.from_numpy(adj_indices).long().to(DEVICE)
    
    # Weight matrix on GPU
    W_dense = build_weight_matrix_torch(adj_sparse, N)
    
    for it in range(iterations):
        # Compute curvature
        curvature = compute_curvature_torch(current, adj_indices_t, degrees)
        
        # Compute entropy per vertex (on GPU)
        entropy = compute_local_entropy_torch(curvature, adj_indices_t)
        
        # Normalize entropy
        entropy_min = entropy.min()
        entropy_max = entropy.max()
        entropy_norm = (entropy - entropy_min) / (entropy_max - entropy_min + 1e-10)
        
        # Smoothing weights
        smoothing_weights = entropy_weight * entropy_norm + (1 - entropy_weight) * 0.5
        
        # Taubin with adaptive weights
        neighbor_avg = W_dense @ current
        displacement = neighbor_avg - current
        current = current + 0.5 * smoothing_weights.unsqueeze(1) * displacement
        
        neighbor_avg = W_dense @ current
        displacement = neighbor_avg - current
        current = current - 0.53 * smoothing_weights.unsqueeze(1) * displacement
    
    elapsed = time.time() - start_time
    
    info = {
        'method': 'Information-Theoretic (MPS)',
        'device': str(DEVICE),
        'time_seconds': elapsed,
        'iterations': iterations
    }
    
    return to_numpy(current), info


# =============================================================================
# 5. ANISOTROPIC TENSOR DIFFUSION (MPS)
# =============================================================================

def anisotropic_tensor_smoothing_mps(verts: np.ndarray,
                                      faces: np.ndarray,
                                      iterations: int = 10,
                                      diffusion_time: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    GPU-accelerated Anisotropic Tensor Diffusion.
    
    Uses MPS for fast tensor operations and vertex updates.
    """
    import time
    start_time = time.time()
    
    N = len(verts)
    
    # Move to GPU
    verts_t = to_torch(verts)
    current = verts_t.clone()
    
    # Compute normals on GPU
    normals = compute_vertex_normals_torch(current, faces)
    
    # Build adjacency
    adj_sparse = build_adjacency_sparse(N, faces)
    adj_indices, _ = sparse_to_indexed(adj_sparse, N)
    adj_indices_t = torch.from_numpy(adj_indices).long().to(DEVICE)
    
    for it in range(iterations):
        new_verts = current.clone()
        
        # Update normals
        normals = compute_vertex_normals_torch(current, faces)
        
        # Batched anisotropic diffusion
        for i in range(N):
            neighbors = adj_indices_t[i]
            valid_mask = neighbors >= 0
            valid_neighbors = neighbors[valid_mask]
            
            if len(valid_neighbors) < 2:
                continue
            
            neighbor_verts = current[valid_neighbors]
            normal = normals[i]
            
            # Compute tangent frame
            arbitrary = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
            if torch.abs(normal[0]) > 0.9:
                arbitrary = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
            
            tangent1 = torch.cross(normal, arbitrary)
            tangent1 = tangent1 / (torch.norm(tangent1) + 1e-10)
            tangent2 = torch.cross(normal, tangent1)
            
            # Displacements
            displacements = neighbor_verts - current[i]
            
            # Directional variances
            tangent_disps = (displacements * tangent1).sum(dim=1)
            normal_disps = (displacements * normal).sum(dim=1)
            
            curv_tangent = tangent_disps.std()
            curv_normal = normal_disps.std()
            
            # Anisotropic coefficients
            max_curv = max(curv_tangent.item(), curv_normal.item()) + 1e-10
            c_tangent = torch.exp(-curv_tangent / max_curv)
            c_normal = 0.1 * torch.exp(-curv_normal / max_curv)
            
            # Weighted update
            weights = torch.exp(-torch.norm(displacements, dim=1) / max_curv)
            weights = weights / (weights.sum() + 1e-10)
            
            mean_disp = (displacements * weights.unsqueeze(1)).sum(dim=0)
            
            # Anisotropic projection
            t1_comp = (mean_disp @ tangent1) * tangent1
            t2_comp = (mean_disp @ tangent2) * tangent2
            n_comp = (mean_disp @ normal) * normal
            
            aniso_disp = c_tangent * (t1_comp + t2_comp) + c_normal * n_comp
            new_verts[i] = current[i] + diffusion_time * aniso_disp
        
        current = new_verts
    
    elapsed = time.time() - start_time
    
    info = {
        'method': 'Anisotropic Tensor (MPS)',
        'device': str(DEVICE),
        'time_seconds': elapsed,
        'iterations': iterations
    }
    
    return to_numpy(current), info


# =============================================================================
# 6. FREQUENCY-SELECTIVE FILTERING (MPS)
# =============================================================================

def frequency_selective_smoothing_mps(verts: np.ndarray,
                                       faces: np.ndarray,
                                       cutoff_percentile: float = 30,
                                       preserve_low_freq: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    GPU-accelerated Frequency-Selective Filtering.
    
    Uses MPS for fast spectral projection and reconstruction.
    """
    import time
    start_time = time.time()
    
    N = len(verts)
    
    # Build Laplacian
    adj_sparse = build_adjacency_sparse(N, faces)
    degrees = np.array(adj_sparse.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    L = sparse.eye(N) - D_inv @ adj_sparse
    
    # Compute eigenvectors (CPU - scipy is better for sparse)
    n_eigenvectors = min(100, N - 2)
    
    try:
        eigenvalues, eigenvectors = eigsh(L.tocsr(), k=n_eigenvectors, which='SM')
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    except:
        # Fallback
        adj_sparse = build_adjacency_sparse(N, faces)
        W_dense = build_weight_matrix_torch(adj_sparse, N)
        current = to_torch(verts)
        for _ in range(5):
            current = W_dense @ current
        return to_numpy(current), {'method': 'Fallback', 'time_seconds': 0}
    
    # Move to GPU for projection and filtering
    eigenvectors_t = to_torch(eigenvectors)
    eigenvalues_t = to_torch(eigenvalues)
    verts_t = to_torch(verts)
    
    # Project onto eigenbasis (GPU matrix multiply)
    coefficients = eigenvectors_t.T @ verts_t
    
    # Design filter
    cutoff_idx = int(cutoff_percentile / 100 * n_eigenvectors)
    cutoff_eigenvalue = eigenvalues_t[min(cutoff_idx, len(eigenvalues_t)-1)]
    
    if preserve_low_freq:
        filter_weights = torch.exp(-eigenvalues_t / (cutoff_eigenvalue + 1e-10))
    else:
        filter_weights = 1 - torch.exp(-eigenvalues_t / (cutoff_eigenvalue + 1e-10))
    
    # Apply filter
    filtered_coefficients = coefficients * filter_weights.unsqueeze(1)
    
    # Reconstruct (GPU matrix multiply)
    smoothed = eigenvectors_t @ filtered_coefficients
    
    elapsed = time.time() - start_time
    
    info = {
        'method': 'Frequency-Selective (MPS)',
        'device': str(DEVICE),
        'time_seconds': elapsed,
        'n_eigenvectors': n_eigenvectors,
        'cutoff_percentile': cutoff_percentile
    }
    
    return to_numpy(smoothed), info


# =============================================================================
# HELPER FUNCTIONS (GPU-ACCELERATED)
# =============================================================================

def build_adjacency_sparse(N: int, faces: np.ndarray) -> sparse.csr_matrix:
    """Build sparse adjacency matrix (CPU)."""
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows))
    return sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


def sparse_to_indexed(adj: sparse.csr_matrix, N: int, max_neighbors: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Convert sparse adjacency to indexed format for GPU processing."""
    indices = np.full((N, max_neighbors), -1, dtype=np.int64)
    weights = np.zeros((N, max_neighbors), dtype=np.float32)
    
    for i in range(N):
        neighbors = adj[i].indices
        n = min(len(neighbors), max_neighbors)
        indices[i, :n] = neighbors[:n]
        weights[i, :n] = 1.0 / (n + 1e-10)
    
    return indices, weights


def build_weight_matrix_torch(adj: sparse.csr_matrix, N: int) -> torch.Tensor:
    """Build dense normalized weight matrix on GPU."""
    degrees = np.array(adj.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    W = D_inv @ adj
    W_dense = W.toarray().astype(np.float32)
    return torch.from_numpy(W_dense).to(DEVICE)


def compute_edge_lengths_torch(verts: torch.Tensor, faces: np.ndarray) -> torch.Tensor:
    """Compute edge lengths on GPU."""
    faces_t = torch.from_numpy(faces).long().to(DEVICE)
    v0 = verts[faces_t[:, 0]]
    v1 = verts[faces_t[:, 1]]
    v2 = verts[faces_t[:, 2]]
    
    e1 = torch.norm(v1 - v0, dim=1)
    e2 = torch.norm(v2 - v1, dim=1)
    e3 = torch.norm(v0 - v2, dim=1)
    
    return torch.cat([e1, e2, e3])


def compute_curvature_torch(verts: torch.Tensor, 
                            adj_indices: torch.Tensor,
                            degrees: np.ndarray) -> torch.Tensor:
    """Compute discrete mean curvature on GPU."""
    N = len(verts)
    curvature = torch.zeros(N, device=DEVICE)
    
    for i in range(N):
        neighbors = adj_indices[i]
        valid_mask = neighbors >= 0
        valid_neighbors = neighbors[valid_mask]
        
        if len(valid_neighbors) == 0:
            continue
        
        neighbor_avg = verts[valid_neighbors].mean(dim=0)
        laplacian = verts[i] - neighbor_avg
        curvature[i] = torch.norm(laplacian)
    
    return curvature


def compute_local_entropy_torch(curvature: torch.Tensor, 
                                adj_indices: torch.Tensor) -> torch.Tensor:
    """Compute local entropy on GPU."""
    N = len(curvature)
    entropy = torch.zeros(N, device=DEVICE)
    
    for i in range(N):
        neighbors = adj_indices[i]
        valid_mask = neighbors >= 0
        valid_neighbors = neighbors[valid_mask]
        
        if len(valid_neighbors) < 3:
            continue
        
        local_curvatures = curvature[valid_neighbors]
        
        # Simple entropy estimate
        std = local_curvatures.std()
        entropy[i] = torch.log(std + 1e-10) + 0.5 * torch.log(torch.tensor(2 * 3.14159))
    
    return entropy


def compute_vertex_normals_torch(verts: torch.Tensor, faces: np.ndarray) -> torch.Tensor:
    """Compute vertex normals on GPU."""
    N = len(verts)
    faces_t = torch.from_numpy(faces).long().to(DEVICE)
    
    v0 = verts[faces_t[:, 0]]
    v1 = verts[faces_t[:, 1]]
    v2 = verts[faces_t[:, 2]]
    
    face_normals = torch.cross(v1 - v0, v2 - v0)
    
    vertex_normals = torch.zeros_like(verts)
    vertex_normals.index_add_(0, faces_t[:, 0], face_normals)
    vertex_normals.index_add_(0, faces_t[:, 1], face_normals)
    vertex_normals.index_add_(0, faces_t[:, 2], face_normals)
    
    norms = torch.norm(vertex_normals, dim=1, keepdim=True)
    norms[norms < 1e-10] = 1
    return vertex_normals / norms


def kmeans_mps(data: torch.Tensor, k: int, max_iters: int = 100) -> np.ndarray:
    """K-means clustering on GPU."""
    N = data.shape[0]
    
    # Initialize centroids
    indices = torch.randperm(N)[:k]
    centroids = data[indices].clone()
    
    for _ in range(max_iters):
        # Compute distances to centroids
        dists = torch.cdist(data, centroids)
        
        # Assign labels
        labels = torch.argmin(dists, dim=1)
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centroids[c] = data[mask].mean(dim=0)
            else:
                new_centroids[c] = centroids[c]
        
        # Check convergence
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        
        centroids = new_centroids
    
    return labels.cpu().numpy()


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def run_all_mps_algorithms(verts: np.ndarray, faces: np.ndarray) -> Dict:
    """Run all MPS-accelerated algorithms and return results."""
    
    results = {}
    
    algorithms = [
        ('Geodesic Heat (MPS)', geodesic_heat_smoothing_mps),
        ('Spectral Clustering (MPS)', spectral_clustering_smoothing_mps),
        ('Optimal Transport (MPS)', optimal_transport_smoothing_mps),
        ('Info-Theoretic (MPS)', information_theoretic_smoothing_mps),
        ('Anisotropic Tensor (MPS)', anisotropic_tensor_smoothing_mps),
        ('Frequency-Selective (MPS)', frequency_selective_smoothing_mps),
    ]
    
    for name, fn in algorithms:
        try:
            smoothed, info = fn(verts.copy(), faces)
            results[name] = {
                'smoothed': smoothed,
                'info': info
            }
            print(f"  ✓ {name}: {info['time_seconds']:.2f}s")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results[name] = None
    
    return results


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'geodesic_heat_smoothing_mps',
    'spectral_clustering_smoothing_mps',
    'optimal_transport_smoothing_mps',
    'information_theoretic_smoothing_mps',
    'anisotropic_tensor_smoothing_mps',
    'frequency_selective_smoothing_mps',
    'run_all_mps_algorithms',
    'DEVICE'
]
