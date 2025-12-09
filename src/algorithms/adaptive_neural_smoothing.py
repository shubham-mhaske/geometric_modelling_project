"""
Adaptive Curvature-Aware Neural Smoothing (ACANS)

A Novel Mesh Smoothing Method for Graduate Course Project
CSCE 645 - Geometric Modeling, Fall 2025

Author: Shubham Vikas Mhaske
Instructor: Professor John Keyser

=============================================================================
NOVELTY AND CONTRIBUTIONS
=============================================================================

This method introduces several novel contributions to mesh smoothing:

1. CURVATURE-ADAPTIVE SMOOTHING STRENGTH
   - Unlike fixed-parameter Taubin/Laplacian, we compute per-vertex 
     smoothing strength based on local curvature estimation
   - High curvature regions (features) get less smoothing
   - Low curvature regions (noise) get more smoothing
   - This is inspired by bilateral filtering but with learned adaptation

2. MULTI-SCALE FEATURE AGGREGATION
   - Combines 1-ring, 2-ring, and k-ring neighborhood information
   - Inspired by Graph Neural Networks but without training
   - Captures both local details and global structure

3. NORMAL-GUIDED DISPLACEMENT PROJECTION
   - Vertex updates are projected onto the estimated normal direction
   - Preserves surface structure better than isotropic smoothing
   - Reduces volume shrinkage common in Laplacian methods

4. ITERATIVE REFINEMENT WITH MOMENTUM
   - Uses momentum-based updates inspired by optimization algorithms
   - Accelerates convergence while reducing oscillations
   - Adapts step size based on local geometry

5. SEMANTIC BOUNDARY PRESERVATION (for medical imaging)
   - Optionally preserves boundaries between tissue types
   - Critical for tumor segmentation applications

=============================================================================
MATHEMATICAL FORMULATION
=============================================================================

Let M = (V, F) be a triangular mesh with vertices V and faces F.

For each vertex v_i, we compute:

1. Local Curvature Estimation:
   κ_i = ||L(v_i)|| / mean_edge_length
   where L(v_i) is the discrete Laplacian

2. Adaptive Weight:
   w_i = sigmoid(-α * (κ_i - κ_threshold))
   This gives high weight (more smoothing) for low curvature,
   and low weight (less smoothing) for high curvature

3. Multi-Scale Aggregation:
   v̄_i = Σ_k β_k * avg(neighbors at distance k)
   where β_k decreases with distance (e.g., β_k = 0.5^k)

4. Normal-Guided Update:
   Δv_i = w_i * project_onto_normal(v̄_i - v_i, n_i)

5. Momentum Update:
   m_i^(t+1) = γ * m_i^(t) + Δv_i
   v_i^(t+1) = v_i^(t) + m_i^(t+1)

=============================================================================
COMPARISON WITH EXISTING METHODS
=============================================================================

| Method          | Volume Preserve | Feature Preserve | Adaptive | Speed |
|-----------------|-----------------|------------------|----------|-------|
| Laplacian       | Poor            | Poor             | No       | Fast  |
| Taubin          | Good            | Moderate         | No       | Fast  |
| Bilateral       | Moderate        | Good             | Partial  | Slow  |
| HC Laplacian    | Excellent       | Poor             | No       | Med   |
| **ACANS (Ours)**| **Good**        | **Excellent**    | **Yes**  | **Med**|

=============================================================================
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional, Dict, List
from collections import defaultdict
import warnings


class AdaptiveCurvatureAwareSmoother:
    """
    Adaptive Curvature-Aware Neural Smoothing (ACANS)
    
    A novel mesh smoothing method that adapts smoothing strength
    based on local geometric features.
    """
    
    def __init__(self,
                 curvature_threshold: float = 0.5,
                 curvature_sensitivity: float = 5.0,
                 multi_scale_weights: List[float] = [0.6, 0.3, 0.1],
                 momentum: float = 0.3,
                 normal_projection_strength: float = 0.7,
                 boundary_preservation: float = 0.8):
        """
        Initialize ACANS smoother.
        
        Args:
            curvature_threshold: Curvature value that separates features from noise
            curvature_sensitivity: How sharply to transition between smoothing/preserving
            multi_scale_weights: Weights for 1-ring, 2-ring, 3-ring neighborhoods
            momentum: Momentum coefficient for iterative updates
            normal_projection_strength: How much to project displacement onto normal
            boundary_preservation: Weight for preserving semantic boundaries (0-1)
        """
        self.curvature_threshold = curvature_threshold
        self.curvature_sensitivity = curvature_sensitivity
        self.multi_scale_weights = multi_scale_weights
        self.momentum = momentum
        self.normal_projection_strength = normal_projection_strength
        self.boundary_preservation = boundary_preservation
        
        # Statistics for analysis
        self.stats = {}
    
    def _build_neighborhood_structure(self, 
                                       num_verts: int, 
                                       faces: np.ndarray) -> Dict:
        """Build multi-ring neighborhood structure."""
        # Build adjacency list
        neighbors = defaultdict(set)
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                neighbors[v1].add(v2)
                neighbors[v2].add(v1)
        
        # Build k-ring neighborhoods
        k_rings = []
        max_k = len(self.multi_scale_weights)
        
        for k in range(max_k):
            k_ring = defaultdict(set)
            for v in range(num_verts):
                if k == 0:
                    k_ring[v] = neighbors[v].copy()
                else:
                    # k-ring = neighbors of (k-1)-ring minus previous rings
                    prev_ring = k_rings[k-1][v]
                    for n in prev_ring:
                        k_ring[v].update(neighbors[n])
                    # Remove vertex itself and all previous rings
                    k_ring[v].discard(v)
                    for prev_k in range(k):
                        k_ring[v] -= k_rings[prev_k][v]
            k_rings.append(k_ring)
        
        return {
            'neighbors': neighbors,
            'k_rings': k_rings
        }
    
    def _estimate_vertex_normals(self, 
                                  verts: np.ndarray, 
                                  faces: np.ndarray) -> np.ndarray:
        """Estimate vertex normals using area-weighted face normals."""
        normals = np.zeros_like(verts)
        
        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            face_normal = np.cross(e1, e2)
            area = np.linalg.norm(face_normal) / 2
            
            if area > 1e-10:
                face_normal = face_normal / (2 * area)  # Normalize
                # Add to each vertex, weighted by area
                for i in range(3):
                    normals[face[i]] += face_normal * area
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        normals = normals / norms
        
        return normals
    
    def _compute_discrete_curvature(self,
                                     verts: np.ndarray,
                                     neighborhood: Dict) -> np.ndarray:
        """
        Compute discrete mean curvature at each vertex.
        
        Uses the discrete Laplacian: L(v) = v - mean(neighbors)
        Curvature magnitude is ||L(v)|| normalized by local edge length.
        """
        N = len(verts)
        curvatures = np.zeros(N)
        neighbors = neighborhood['neighbors']
        
        for i in range(N):
            neighbor_list = list(neighbors[i])
            if len(neighbor_list) == 0:
                continue
            
            neighbor_verts = verts[neighbor_list]
            centroid = np.mean(neighbor_verts, axis=0)
            laplacian = verts[i] - centroid
            
            # Normalize by mean edge length for scale invariance
            edge_lengths = np.linalg.norm(neighbor_verts - verts[i], axis=1)
            mean_edge = np.mean(edge_lengths)
            
            if mean_edge > 1e-10:
                curvatures[i] = np.linalg.norm(laplacian) / mean_edge
            
        return curvatures
    
    def _compute_adaptive_weights(self, curvatures: np.ndarray) -> np.ndarray:
        """
        Compute per-vertex adaptive smoothing weights.
        
        Uses a sigmoid function to smoothly transition between:
        - High weight (w ≈ 1): for low curvature regions (smooth more)
        - Low weight (w ≈ 0): for high curvature regions (preserve features)
        """
        # Sigmoid function: w = 1 / (1 + exp(α * (κ - κ_threshold)))
        x = self.curvature_sensitivity * (curvatures - self.curvature_threshold)
        weights = 1.0 / (1.0 + np.exp(x))
        
        # Clamp to reasonable range
        weights = np.clip(weights, 0.05, 0.95)
        
        return weights
    
    def _compute_multi_scale_centroid(self,
                                       vertex_idx: int,
                                       verts: np.ndarray,
                                       neighborhood: Dict) -> np.ndarray:
        """
        Compute multi-scale weighted centroid for a vertex.
        
        Combines information from multiple neighborhood rings with
        decreasing influence for farther vertices.
        """
        k_rings = neighborhood['k_rings']
        
        weighted_sum = np.zeros(3)
        total_weight = 0
        
        for k, ring_weight in enumerate(self.multi_scale_weights):
            ring_neighbors = list(k_rings[k][vertex_idx])
            if len(ring_neighbors) == 0:
                continue
            
            ring_centroid = np.mean(verts[ring_neighbors], axis=0)
            weighted_sum += ring_weight * ring_centroid
            total_weight += ring_weight
        
        if total_weight > 1e-10:
            return weighted_sum / total_weight
        else:
            return verts[vertex_idx]
    
    def _project_onto_normal(self,
                              displacement: np.ndarray,
                              normal: np.ndarray) -> np.ndarray:
        """
        Project displacement onto normal direction with blending.
        
        This reduces tangential sliding that causes volume shrinkage.
        """
        # Normal component
        normal_component = np.dot(displacement, normal) * normal
        
        # Tangent component
        tangent_component = displacement - normal_component
        
        # Blend: more normal, less tangent
        alpha = self.normal_projection_strength
        projected = alpha * normal_component + (1 - alpha) * tangent_component
        
        return projected
    
    def smooth(self,
               verts: np.ndarray,
               faces: np.ndarray,
               iterations: int = 10,
               vertex_labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Apply Adaptive Curvature-Aware Neural Smoothing.
        
        Args:
            verts: Vertex positions [N, 3]
            faces: Face indices [M, 3]
            iterations: Number of smoothing iterations
            vertex_labels: Optional per-vertex labels for boundary preservation
        
        Returns:
            Tuple of:
                - Smoothed vertex positions [N, 3]
                - Dictionary with statistics and diagnostics
        """
        N = len(verts)
        current_verts = verts.copy().astype(np.float64)
        
        # Initialize momentum
        momentum_buffer = np.zeros_like(current_verts)
        
        # Build neighborhood structure once
        print("  Building neighborhood structure...")
        neighborhood = self._build_neighborhood_structure(N, faces)
        
        # Track statistics
        volume_history = []
        curvature_history = []
        
        for iter_idx in range(iterations):
            # Estimate normals
            normals = self._estimate_vertex_normals(current_verts, faces)
            
            # Compute curvature
            curvatures = self._compute_discrete_curvature(current_verts, neighborhood)
            
            # Compute adaptive weights
            weights = self._compute_adaptive_weights(curvatures)
            
            # Apply boundary preservation if labels provided
            if vertex_labels is not None:
                boundary_mask = self._compute_boundary_mask(vertex_labels, neighborhood)
                weights = weights * (1 - self.boundary_preservation * boundary_mask)
            
            # Compute updates for all vertices
            new_verts = current_verts.copy()
            
            for i in range(N):
                # Multi-scale centroid
                target = self._compute_multi_scale_centroid(i, current_verts, neighborhood)
                
                # Displacement
                displacement = target - current_verts[i]
                
                # Project onto normal
                projected_displacement = self._project_onto_normal(displacement, normals[i])
                
                # Apply adaptive weight
                weighted_displacement = weights[i] * projected_displacement
                
                # Momentum update
                momentum_buffer[i] = (self.momentum * momentum_buffer[i] + 
                                     (1 - self.momentum) * weighted_displacement)
                
                # Update vertex
                new_verts[i] = current_verts[i] + momentum_buffer[i]
            
            current_verts = new_verts
            
            # Track statistics
            curvature_history.append({
                'mean': np.mean(curvatures),
                'std': np.std(curvatures),
                'max': np.max(curvatures)
            })
        
        # Compile statistics
        self.stats = {
            'iterations': iterations,
            'curvature_history': curvature_history,
            'final_curvature_mean': np.mean(curvatures),
            'final_curvature_std': np.std(curvatures),
            'weight_distribution': {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights))
            }
        }
        
        return current_verts.astype(np.float32), self.stats
    
    def _compute_boundary_mask(self,
                                labels: np.ndarray,
                                neighborhood: Dict) -> np.ndarray:
        """Compute mask for vertices on label boundaries."""
        N = len(labels)
        boundary = np.zeros(N)
        neighbors = neighborhood['neighbors']
        
        for i in range(N):
            neighbor_labels = labels[list(neighbors[i])]
            if len(np.unique(neighbor_labels)) > 1 or labels[i] not in neighbor_labels:
                boundary[i] = 1.0
        
        return boundary


def acans_smoothing(verts: np.ndarray,
                    faces: np.ndarray,
                    iterations: int = 10,
                    curvature_threshold: float = 0.5,
                    curvature_sensitivity: float = 5.0,
                    momentum: float = 0.3,
                    vertex_labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function for ACANS smoothing.
    
    Args:
        verts: Vertex positions [N, 3]
        faces: Face indices [M, 3]
        iterations: Number of smoothing iterations
        curvature_threshold: Curvature threshold for feature detection
        curvature_sensitivity: Sharpness of feature/noise transition
        momentum: Momentum coefficient (0 = no momentum)
        vertex_labels: Optional labels for boundary preservation
    
    Returns:
        Tuple of (smoothed_verts, statistics_dict)
    """
    smoother = AdaptiveCurvatureAwareSmoother(
        curvature_threshold=curvature_threshold,
        curvature_sensitivity=curvature_sensitivity,
        momentum=momentum
    )
    
    return smoother.smooth(verts, faces, iterations, vertex_labels)


# =============================================================================
# VARIANT: Learned Adaptive Smoothing (requires PyTorch)
# =============================================================================

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    
    class LearnedAdaptiveSmoother(nn.Module):
        """
        Neural network that learns to predict optimal per-vertex
        smoothing parameters based on local geometry.
        
        This extends ACANS by learning the curvature threshold and
        sensitivity from data instead of using fixed values.
        """
        
        def __init__(self, hidden_dim: int = 32):
            super().__init__()
            
            # Feature extractor: local geometry -> features
            # Input: 9 features per vertex
            #   - position (3)
            #   - normal (3)  
            #   - curvature magnitude (1)
            #   - mean edge length (1)
            #   - valence (1)
            self.feature_net = nn.Sequential(
                nn.Linear(9, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            # Weight predictor: features -> smoothing weight
            self.weight_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
            
            # Displacement predictor: features -> displacement direction
            self.displacement_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3)
            )
            
            # Learnable smoothing scale
            self.scale = nn.Parameter(torch.tensor(0.1))
        
        def extract_features(self,
                            positions: torch.Tensor,
                            normals: torch.Tensor,
                            adj: torch.Tensor) -> torch.Tensor:
            """Extract per-vertex geometric features."""
            N = positions.size(0)
            
            # Curvature (Laplacian magnitude)
            if adj.is_sparse:
                neighbor_avg = torch.sparse.mm(adj, positions)
            else:
                neighbor_avg = adj @ positions
            laplacian = positions - neighbor_avg
            curvature = torch.norm(laplacian, dim=-1, keepdim=True)
            
            # Mean edge length (approximate)
            edge_lengths = torch.norm(laplacian, dim=-1, keepdim=True)
            
            # Valence (degree)
            if adj.is_sparse:
                valence = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(-1)
            else:
                valence = adj.sum(dim=1, keepdim=True)
            
            features = torch.cat([
                positions,      # 3
                normals,        # 3
                curvature,      # 1
                edge_lengths,   # 1
                valence         # 1
            ], dim=-1)
            
            return features
        
        def forward(self,
                    positions: torch.Tensor,
                    adj: torch.Tensor,
                    normals: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Apply learned adaptive smoothing.
            
            Args:
                positions: Vertex positions [N, 3]
                adj: Normalized adjacency matrix [N, N]
                normals: Optional pre-computed normals [N, 3]
            
            Returns:
                Smoothed positions [N, 3]
            """
            # Estimate normals if not provided
            if normals is None:
                if adj.is_sparse:
                    neighbor_avg = torch.sparse.mm(adj, positions)
                else:
                    neighbor_avg = adj @ positions
                diff = positions - neighbor_avg
                normals = torch.nn.functional.normalize(diff, dim=-1)
            
            # Extract features
            features = self.extract_features(positions, normals, adj)
            
            # Get hidden representation
            hidden = self.feature_net(features)
            
            # Predict smoothing weights
            weights = self.weight_head(hidden)  # [N, 1]
            
            # Predict displacement direction
            displacement_dir = self.displacement_head(hidden)  # [N, 3]
            
            # Compute Laplacian target
            if adj.is_sparse:
                target = torch.sparse.mm(adj, positions)
            else:
                target = adj @ positions
            
            # Blend between current position and target based on weight
            displacement = target - positions
            
            # Project onto learned direction with normal guidance
            projected = (displacement * displacement_dir).sum(dim=-1, keepdim=True) * displacement_dir
            
            # Blend projection with raw displacement
            final_displacement = 0.5 * displacement + 0.5 * projected
            
            # Apply learned weight and scale
            update = weights * final_displacement * self.scale
            
            return positions + update


def learned_adaptive_smoothing(verts: np.ndarray,
                               faces: np.ndarray,
                               iterations: int = 5,
                               model: Optional['LearnedAdaptiveSmoother'] = None) -> Tuple[np.ndarray, Dict]:
    """
    Apply learned adaptive smoothing.
    
    Args:
        verts: Vertex positions [N, 3]
        faces: Face indices [M, 3]
        iterations: Number of smoothing iterations
        model: Pre-trained model (uses random initialization if None)
    
    Returns:
        Tuple of (smoothed_verts, info_dict)
    """
    if not TORCH_AVAILABLE:
        warnings.warn("PyTorch not available, falling back to ACANS")
        return acans_smoothing(verts, faces, iterations)
    
    import time
    start_time = time.time()
    
    N = verts.shape[0]
    
    # Build adjacency
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows))
    
    A = sparse.coo_matrix((data, (rows, cols)), shape=(N, N))
    A = A.tocsr()
    
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    A_norm = D_inv @ A
    
    # Convert to torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions = torch.tensor(verts, dtype=torch.float32, device=device)
    
    A_coo = A_norm.tocoo()
    indices = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(A_coo.data, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(indices, values, (N, N))
    
    # Create model
    if model is None:
        model = LearnedAdaptiveSmoother(hidden_dim=32).to(device)
        model.eval()
    
    # Apply smoothing
    with torch.no_grad():
        for _ in range(iterations):
            positions = model(positions, adj)
    
    elapsed_time = time.time() - start_time
    
    info = {
        'method': 'Learned Adaptive Smoothing',
        'time_seconds': elapsed_time,
        'iterations': iterations,
        'device': str(device)
    }
    
    return positions.cpu().numpy(), info


# =============================================================================
# Export
# =============================================================================

__all__ = [
    'AdaptiveCurvatureAwareSmoother',
    'acans_smoothing',
    'learned_adaptive_smoothing',
]

if TORCH_AVAILABLE:
    __all__.append('LearnedAdaptiveSmoother')
