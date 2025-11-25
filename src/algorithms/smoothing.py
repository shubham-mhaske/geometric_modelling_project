import numpy as np
from scipy import sparse

# Cross-label weight: controls smoothing across tissue boundaries
# Higher = more smoothing across boundaries (less feature preservation)
# Lower = less smoothing across boundaries (more feature preservation)
# Default 0.3 provides good balance between quality and boundary protection
_CROSS_LABEL_WEIGHT = 0.3

def build_adjacency_matrix(num_verts, faces, vertex_labels=None):
    """
    Build the row-normalized adjacency matrix W = D^-1 * A.
    """
    # Create adjacency matrix
    # faces is (N, 3)
    # We need to add edges (i, j) for every edge in the mesh
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])

    if vertex_labels is not None:
        labels = np.asarray(vertex_labels).reshape(-1)
        if labels.shape[0] != num_verts:
            raise ValueError("vertex_labels length must match num_verts")
        same_label = labels[rows] == labels[cols]
        data = np.where(same_label, 1.0, _CROSS_LABEL_WEIGHT)
    else:
        data = np.ones(len(rows))
    
    # A is the adjacency matrix
    A = sparse.coo_matrix((data, (rows, cols)), shape=(num_verts, num_verts))
    # Convert to CSR for efficient arithmetic
    A = A.tocsr()
    A.sum_duplicates()

    if vertex_labels is None:
        A.data[:] = 1.0
    else:
        high_mask = A.data >= 0.5
        A.data[high_mask] = 1.0
        A.data[~high_mask] = _CROSS_LABEL_WEIGHT
    
    # Degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    # Avoid division by zero
    degrees[degrees == 0] = 1
    
    # Inverse degree matrix
    D_inv = sparse.diags(1.0 / degrees)
    
    # Row-normalized adjacency matrix W
    W = D_inv @ A
    return W

def laplacian_smoothing(verts, faces, iterations, lambda_val=0.5, vertex_labels=None):
    """
    Apply Laplacian smoothing to the mesh.
    """
    num_verts = verts.shape[0]
    W = build_adjacency_matrix(num_verts, faces, vertex_labels=vertex_labels)
    I = sparse.eye(num_verts)
    
    # Operator: K = (1 - lambda) * I + lambda * W
    K = (1 - lambda_val) * I + lambda_val * W
    
    curr_verts = verts
    for _ in range(iterations):
        curr_verts = K @ curr_verts
        
    return curr_verts

def taubin_smoothing(verts, faces, iterations, lambda_val=0.5, mu_val=-0.53, vertex_labels=None):
    """
    Apply Taubin smoothing to the mesh.
    """
    num_verts = verts.shape[0]
    W = build_adjacency_matrix(num_verts, faces, vertex_labels=vertex_labels)
    I = sparse.eye(num_verts)
    
    # Operators
    K_lambda = (1 - lambda_val) * I + lambda_val * W
    K_mu = (1 - mu_val) * I + mu_val * W
    
    curr_verts = verts
    for _ in range(iterations):
        # Shrink
        curr_verts = K_lambda @ curr_verts
        # Expand
        curr_verts = K_mu @ curr_verts
        
    return curr_verts


def constrained_smoothing(verts, faces, iterations, lambda_val=0.5, 
                          fixed_vertices=None, vertex_labels=None,
                          algorithm='laplacian'):
    """
    Apply smoothing while keeping specified vertices fixed (landmark preservation).
    
    This implements the "potential extension" from the project proposal:
    "constrained smoothing method to explicitly preserve user-defined anatomical landmarks"
    
    Args:
        verts: (N, 3) vertex positions
        faces: (M, 3) face indices
        iterations: number of smoothing passes
        lambda_val: smoothing strength (0-1)
        fixed_vertices: array-like of vertex indices to keep fixed, or boolean mask
        vertex_labels: optional (N,) array for semantic boundary preservation
        algorithm: 'laplacian' or 'taubin'
    
    Returns:
        smoothed_verts: (N, 3) with fixed vertices unchanged
    """
    num_verts = verts.shape[0]
    verts = verts.copy().astype(np.float64)
    
    # Handle fixed vertices specification
    if fixed_vertices is None:
        fixed_mask = np.zeros(num_verts, dtype=bool)
    elif isinstance(fixed_vertices, np.ndarray) and fixed_vertices.dtype == bool:
        fixed_mask = fixed_vertices
    else:
        fixed_mask = np.zeros(num_verts, dtype=bool)
        fixed_mask[np.asarray(fixed_vertices)] = True
    
    # Store original positions of fixed vertices
    original_positions = verts[fixed_mask].copy()
    
    # Apply chosen smoothing algorithm
    if algorithm == 'taubin':
        smoothed = taubin_smoothing(verts, faces, iterations, 
                                    lambda_val=lambda_val, vertex_labels=vertex_labels)
    else:
        smoothed = laplacian_smoothing(verts, faces, iterations,
                                       lambda_val=lambda_val, vertex_labels=vertex_labels)
    
    # Restore fixed vertices to original positions
    smoothed[fixed_mask] = original_positions
    
    return smoothed


def adaptive_smoothing(verts, faces, iterations, lambda_range=(0.1, 0.7),
                       vertex_labels=None, curvature_weights=None):
    """
    Per-vertex adaptive smoothing with variable lambda based on local properties.
    
    Vertices with high curvature or at label boundaries receive less smoothing.
    
    Args:
        verts: (N, 3) vertex positions
        faces: (M, 3) face indices  
        iterations: number of smoothing passes
        lambda_range: (min_lambda, max_lambda) range for adaptive smoothing
        vertex_labels: optional (N,) array - reduce smoothing at boundaries
        curvature_weights: optional (N,) array of per-vertex weights (0-1)
    
    Returns:
        smoothed_verts: (N, 3) adaptively smoothed positions
    """
    num_verts = verts.shape[0]
    verts = verts.copy().astype(np.float64)
    
    min_lam, max_lam = lambda_range
    
    # Build adjacency
    W = build_adjacency_matrix(num_verts, faces, vertex_labels=None)
    
    # Compute per-vertex adaptive lambda
    if curvature_weights is not None:
        # High curvature = low lambda (less smoothing)
        weights = np.clip(curvature_weights, 0, 1)
        adaptive_lambda = min_lam + (max_lam - min_lam) * (1 - weights)
    else:
        adaptive_lambda = np.full(num_verts, (min_lam + max_lam) / 2)
    
    # Reduce lambda at label boundaries
    if vertex_labels is not None:
        labels = np.asarray(vertex_labels).reshape(-1)
        # Find boundary vertices
        for i in range(num_verts):
            neighbors = W.indices[W.indptr[i]:W.indptr[i+1]]
            if len(neighbors) > 0 and labels[i] > 0:
                neighbor_labels = labels[neighbors]
                # If any neighbor has a different label, this is a boundary vertex
                if np.any((neighbor_labels != labels[i]) & (neighbor_labels > 0)):
                    adaptive_lambda[i] *= 0.2  # Reduce smoothing at boundaries
    
    # Iterative smoothing with per-vertex lambda
    for _ in range(iterations):
        neighbor_avg = W @ verts
        
        # Vectorized adaptive update
        for d in range(3):  # x, y, z
            verts[:, d] = (1 - adaptive_lambda) * verts[:, d] + adaptive_lambda * neighbor_avg[:, d]
    
    return verts
