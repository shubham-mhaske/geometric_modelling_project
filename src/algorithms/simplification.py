import numpy as np
from scipy import sparse
import heapq

def compute_quadric(verts, faces):
    """
    Compute the quadric error matrix Q for each vertex.
    Q represents the sum of squared distances from the vertex to all incident planes.
    """
    num_verts = verts.shape[0]
    Q = np.zeros((num_verts, 4, 4))
    
    # For each face, compute its plane equation and add to vertex quadrics
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        
        # Compute normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        
        if norm_len < 1e-12:
            continue  # Degenerate triangle
        
        normal = normal / norm_len
        d = -np.dot(normal, v0)
        
        # Plane equation: ax + by + cz + d = 0
        p = np.array([normal[0], normal[1], normal[2], d])
        
        # Quadric: Q = p * p^T
        Kp = np.outer(p, p)
        
        # Add to all three vertices of this face
        Q[face[0]] += Kp
        Q[face[1]] += Kp
        Q[face[2]] += Kp
    
    return Q

def edge_collapse_cost(v1_idx, v2_idx, verts, Q):
    """
    Compute the cost of collapsing edge (v1, v2) and the optimal new position.
    Returns: (cost, new_position)
    """
    Q_bar = Q[v1_idx] + Q[v2_idx]
    
    # Try to solve for optimal position: Q_bar * [x y z 1]^T = 0
    # This means solving the linear system Q_bar[:3, :3] * p = -Q_bar[:3, 3]
    A = Q_bar[:3, :3]
    b = -Q_bar[:3, 3]
    
    try:
        # Check if matrix is invertible
        if np.linalg.det(A) > 1e-6:
            v_new = np.linalg.solve(A, b)
        else:
            # Fall back to midpoint
            v_new = (verts[v1_idx] + verts[v2_idx]) / 2
    except:
        v_new = (verts[v1_idx] + verts[v2_idx]) / 2
    
    # Compute error: v_new^T * Q_bar * v_new (homogeneous coordinates)
    v_hom = np.append(v_new, 1.0)
    cost = v_hom @ Q_bar @ v_hom
    
    return cost, v_new

def qem_simplification(verts, faces, target_reduction=0.5):
    """
    Simplify mesh using Quadric Error Metrics.
    
    Args:
        verts: (N, 3) vertex array
        faces: (M, 3) face array (triangle indices)
        target_reduction: fraction of faces to remove (0 to 1)
    
    Returns:
        new_verts, new_faces
    """
    verts = verts.copy()
    faces = faces.copy().astype(np.int32)
    
    num_faces = faces.shape[0]
    target_faces = int(num_faces * (1 - target_reduction))
    
    if target_faces >= num_faces:
        return verts, faces
    
    # Compute quadric for each vertex
    Q = compute_quadric(verts, faces)
    
    # Build edge list and adjacency
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))
    
    # Priority queue: (cost, edge)
    heap = []
    edge_cost_map = {}
    edge_pos_map = {}
    
    for edge in edges:
        v1, v2 = edge
        cost, new_pos = edge_collapse_cost(v1, v2, verts, Q)
        edge_cost_map[edge] = cost
        edge_pos_map[edge] = new_pos
        heapq.heappush(heap, (cost, edge))
    
    # Track which vertices are removed
    removed_verts = set()
    vertex_mapping = {}  # Maps old vertex index to new vertex index
    
    # Iteratively collapse edges
    collapses = 0
    max_collapses = num_faces - target_faces
    
    while heap and collapses < max_collapses:
        cost, edge = heapq.heappop(heap)
        v1, v2 = edge
        
        # Skip if vertices already removed
        if v1 in removed_verts or v2 in removed_verts:
            continue
        
        # Skip if cost changed (stale entry)
        if edge not in edge_cost_map or edge_cost_map[edge] != cost:
            continue
        
        # Perform collapse: move v1 to optimal position, remove v2
        new_pos = edge_pos_map[edge]
        verts[v1] = new_pos
        removed_verts.add(v2)
        vertex_mapping[v2] = v1
        
        # Update quadric
        Q[v1] = Q[v1] + Q[v2]
        
        collapses += 1
        
        # Early exit if we've done enough
        if collapses >= max_collapses:
            break
    
    # Rebuild faces, skipping degenerate ones
    new_faces = []
    for face in faces:
        # Map vertices
        mapped_face = []
        for v in face:
            if v in removed_verts:
                # Find the vertex it was collapsed to
                curr = v
                visited = set()
                while curr in vertex_mapping:
                    if curr in visited:
                        # Cycle detected, break
                        curr = vertex_mapping[curr]
                        break
                    visited.add(curr)
                    curr = vertex_mapping[curr]
                mapped_face.append(curr)
            else:
                mapped_face.append(v)
        
        # Skip degenerate faces (where all vertices are the same or only 2 unique)
        if len(set(mapped_face)) == 3:
            new_faces.append(mapped_face)
    
    new_faces = np.array(new_faces)
    
    # Reindex vertices to remove gaps
    used_verts = set(new_faces.flatten())
    old_to_new = {}
    new_verts_list = []
    
    for new_idx, old_idx in enumerate(sorted(used_verts)):
        old_to_new[old_idx] = new_idx
        new_verts_list.append(verts[old_idx])
    
    new_verts = np.array(new_verts_list)
    
    # Remap face indices
    for i in range(new_faces.shape[0]):
        for j in range(3):
            new_faces[i, j] = old_to_new[new_faces[i, j]]
    
    return new_verts, new_faces
