import numpy as np
from scipy import sparse

def build_adjacency_matrix(num_verts, faces):
    """
    Build the row-normalized adjacency matrix W = D^-1 * A.
    """
    # Create adjacency matrix
    # faces is (N, 3)
    # We need to add edges (i, j) for every edge in the mesh
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows))
    
    # A is the adjacency matrix
    A = sparse.coo_matrix((data, (rows, cols)), shape=(num_verts, num_verts))
    # Convert to CSR for efficient arithmetic
    A = A.tocsr()
    # Ensure binary (in case of duplicate edges)
    A.data[:] = 1
    
    # Degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    # Avoid division by zero
    degrees[degrees == 0] = 1
    
    # Inverse degree matrix
    D_inv = sparse.diags(1.0 / degrees)
    
    # Row-normalized adjacency matrix W
    W = D_inv @ A
    return W

def laplacian_smoothing(verts, faces, iterations, lambda_val=0.5):
    """
    Apply Laplacian smoothing to the mesh.
    """
    num_verts = verts.shape[0]
    W = build_adjacency_matrix(num_verts, faces)
    I = sparse.eye(num_verts)
    
    # Operator: K = (1 - lambda) * I + lambda * W
    K = (1 - lambda_val) * I + lambda_val * W
    
    curr_verts = verts
    for _ in range(iterations):
        curr_verts = K @ curr_verts
        
    return curr_verts

def taubin_smoothing(verts, faces, iterations, lambda_val=0.5, mu_val=-0.53):
    """
    Apply Taubin smoothing to the mesh.
    """
    num_verts = verts.shape[0]
    W = build_adjacency_matrix(num_verts, faces)
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
