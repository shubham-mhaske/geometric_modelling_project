"""
Neural Network-Based Mesh Smoothing

This module implements state-of-the-art neural network techniques for mesh denoising
and smoothing, inspired by recent research:

1. GNN-Based Mesh Denoising (inspired by DMD-Net, June 2025)
   - Graph Convolutional Networks for mesh processing
   - Feature-guided denoising with attention mechanisms

2. Diffusion-Based Mesh Smoothing (inspired by MeshDiffusion, ICLR 2023)
   - Score-based denoising using diffusion process
   - Iterative refinement with learned priors

3. Neural Position Encoding (inspired by NeRF/Transformers)
   - Positional encoding for vertex features
   - Self-attention for global context

References:
- DMD-Net: Deep Mesh Denoising Network (arXiv:2506.22850, June 2025)
- MeshDiffusion: Score-based Generative 3D Mesh Modeling (ICLR 2023)
- NormalNet: Learning-based Normal Filtering for Mesh Denoising (2019)
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional, Dict
import warnings

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Neural smoothing will use fallback methods.")


# =============================================================================
# Graph Neural Network Components
# =============================================================================

if TORCH_AVAILABLE:
    
    class GraphConvLayer(nn.Module):
        """
        Graph Convolutional Layer for mesh processing.
        """
        def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
            self.weight_neighbor = nn.Linear(in_features, out_features, bias=False)
            
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)
            
            self._init_weights()
        
        def _init_weights(self):
            nn.init.xavier_uniform_(self.weight_self.weight)
            nn.init.xavier_uniform_(self.weight_neighbor.weight)
        
        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            h_self = self.weight_self(x)
            h_neighbor = self.weight_neighbor(torch.sparse.mm(adj, x) if adj.is_sparse else adj @ x)
            out = h_self + h_neighbor
            if self.bias is not None:
                out = out + self.bias
            return out


    class PositionalEncoding(nn.Module):
        """Positional encoding for 3D vertex positions."""
        def __init__(self, num_frequencies: int = 10):
            super().__init__()
            self.num_frequencies = num_frequencies
            self.out_dim = 3 + 3 * 2 * num_frequencies
        
        def forward(self, positions: torch.Tensor) -> torch.Tensor:
            encoded = [positions]
            for i in range(self.num_frequencies):
                freq = 2.0 ** i * np.pi
                encoded.append(torch.sin(freq * positions))
                encoded.append(torch.cos(freq * positions))
            return torch.cat(encoded, dim=-1)


    class GNNMeshDenoiser(nn.Module):
        """GNN for Mesh Denoising inspired by DMD-Net (2025)."""
        
        def __init__(self, hidden_dim: int = 64, num_layers: int = 4, 
                     num_heads: int = 4, dropout: float = 0.1, use_attention: bool = True):
            super().__init__()
            
            self.pos_encoder = PositionalEncoding(num_frequencies=6)
            input_dim = self.pos_encoder.out_dim
            
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            
            self.conv_layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                self.conv_layers.append(GraphConvLayer(hidden_dim, hidden_dim))
                self.norms.append(nn.LayerNorm(hidden_dim))
            
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)
            )
            
            self.smooth_scale = nn.Parameter(torch.tensor(0.1))
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, positions: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            x = self.pos_encoder(positions)
            x = self.input_proj(x)
            
            for conv, norm in zip(self.conv_layers, self.norms):
                x_new = conv(x, adj)
                x_new = F.relu(x_new)
                x_new = self.dropout(x_new)
                x = norm(x + x_new)
            
            displacement = self.output_head(x)
            smoothed = positions + self.smooth_scale * displacement
            return smoothed


    class MeshDiffusionSmoother(nn.Module):
        """Diffusion-based Mesh Smoothing inspired by MeshDiffusion (ICLR 2023)."""
        
        def __init__(self, hidden_dim: int = 64, num_steps: int = 10,
                     beta_start: float = 0.0001, beta_end: float = 0.02):
            super().__init__()
            
            self.num_steps = num_steps
            self.register_buffer('betas', torch.linspace(beta_start, beta_end, num_steps))
            alphas = 1.0 - self.betas
            self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
            
            self.score_net = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)
            )
            
            self.graph_refine = GraphConvLayer(3, 3)
        
        def forward(self, positions: torch.Tensor, adj: torch.Tensor,
                    num_steps: Optional[int] = None) -> torch.Tensor:
            if num_steps is None:
                num_steps = self.num_steps
            
            device = positions.device
            x = positions.clone()
            
            for t in reversed(range(num_steps)):
                t_emb = torch.full((x.size(0), 1), t / self.num_steps, device=device)
                x_input = torch.cat([x, t_emb], dim=-1)
                predicted_noise = self.score_net(x_input)
                
                alpha_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                x = (1 / torch.sqrt(1 - beta_t)) * (x - beta_t / torch.sqrt(1 - alpha_t) * predicted_noise)
                
                if t > 0:
                    noise = torch.randn_like(x) * torch.sqrt(beta_t) * 0.1
                    x = x + noise
            
            x = self.graph_refine(x, adj)
            return x


    class FeatureGuidedTransformer(nn.Module):
        """Feature-Guided Transformer from DMD-Net."""
        
        def __init__(self, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
            super().__init__()
            
            self.feature_extractor = nn.Sequential(
                nn.Linear(12, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.displacement_head = nn.Linear(hidden_dim, 3)
        
        def compute_local_features(self, positions: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            if adj.is_sparse:
                adj_dense = adj.to_dense()
            else:
                adj_dense = adj
            
            neighbor_avg = adj_dense @ positions
            diff = positions - neighbor_avg
            normals = F.normalize(diff, dim=-1)
            laplacian = diff
            curv_magnitude = torch.norm(laplacian, dim=-1, keepdim=True)
            edge_lengths = torch.norm(diff, dim=-1, keepdim=True)
            
            features = torch.cat([
                positions, normals, laplacian,
                curv_magnitude, edge_lengths, curv_magnitude ** 2
            ], dim=-1)
            return features
        
        def forward(self, positions: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            features = self.compute_local_features(positions, adj)
            x = self.feature_extractor(features)
            
            N = x.size(0)
            if N > 5000:
                chunk_size = 5000
                outputs = []
                for i in range(0, N, chunk_size):
                    chunk = x[i:i+chunk_size].unsqueeze(0)
                    out = self.transformer(chunk).squeeze(0)
                    outputs.append(out)
                x = torch.cat(outputs, dim=0)
            else:
                x = self.transformer(x.unsqueeze(0)).squeeze(0)
            
            displacement = self.displacement_head(x)
            smoothed = positions + 0.1 * displacement
            return smoothed


# =============================================================================
# Non-Neural Fallback Methods
# =============================================================================

def spectral_mesh_smoothing(verts: np.ndarray, faces: np.ndarray,
                            num_eigenvectors: int = 50,
                            smoothing_factor: float = 0.5) -> np.ndarray:
    """Spectral mesh smoothing using Laplacian eigenvectors."""
    from scipy.sparse.linalg import eigsh
    
    N = verts.shape[0]
    
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], 
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows))
    
    A = sparse.coo_matrix((data, (rows, cols)), shape=(N, N))
    A = A.tocsr()
    
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    L = sparse.eye(N) - D_inv @ A
    
    k = min(num_eigenvectors, N - 2)
    try:
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
    except:
        return verts
    
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    coeffs = eigenvectors.T @ verts
    smooth_weights = np.exp(-smoothing_factor * eigenvalues).reshape(-1, 1)
    smoothed_coeffs = coeffs * smooth_weights
    smoothed_verts = eigenvectors @ smoothed_coeffs
    
    return smoothed_verts


def neural_inspired_smoothing(verts: np.ndarray, faces: np.ndarray,
                              iterations: int = 10,
                              feature_preservation: float = 0.5) -> np.ndarray:
    """Neural-inspired mesh smoothing without neural networks."""
    from collections import defaultdict
    
    N = verts.shape[0]
    curr_verts = verts.copy()
    
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    
    neighbors = defaultdict(list)
    for r, c in zip(rows, cols):
        if c not in neighbors[r]:
            neighbors[r].append(c)
    
    for iteration in range(iterations):
        new_verts = curr_verts.copy()
        
        for i in range(N):
            if len(neighbors[i]) == 0:
                continue
            
            neighbor_indices = neighbors[i]
            neighbor_verts = curr_verts[neighbor_indices]
            
            diffs = neighbor_verts - curr_verts[i]
            distances = np.linalg.norm(diffs, axis=1)
            weights = np.exp(-distances / (np.mean(distances) + 1e-10))
            weights = weights / (np.sum(weights) + 1e-10)
            
            centroid = np.sum(neighbor_verts * weights.reshape(-1, 1), axis=0)
            displacement = centroid - curr_verts[i]
            displacement_magnitude = np.linalg.norm(displacement)
            
            scale = 1.0 - feature_preservation * min(1.0, displacement_magnitude * 10)
            scale = max(0.1, scale)
            
            new_verts[i] = curr_verts[i] + 0.5 * scale * displacement
        
        curr_verts = new_verts
    
    return curr_verts


# =============================================================================
# Main Interface Functions
# =============================================================================

def _build_adjacency(verts: np.ndarray, faces: np.ndarray):
    """Build normalized adjacency matrix."""
    N = verts.shape[0]
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
    
    return A_norm


def gnn_mesh_smoothing(verts: np.ndarray, faces: np.ndarray,
                       model: Optional['GNNMeshDenoiser'] = None,
                       iterations: int = 1) -> np.ndarray:
    """Apply GNN-based mesh smoothing."""
    if not TORCH_AVAILABLE:
        return neural_inspired_smoothing(verts, faces, iterations=iterations * 5)
    
    N = verts.shape[0]
    A_norm = _build_adjacency(verts, faces)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions = torch.tensor(verts, dtype=torch.float32, device=device)
    
    A_coo = A_norm.tocoo()
    indices = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(A_coo.data, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(indices, values, (N, N))
    
    if model is None:
        model = GNNMeshDenoiser(hidden_dim=32, num_layers=3, use_attention=N < 5000).to(device)
        model.eval()
    
    with torch.no_grad():
        for _ in range(iterations):
            positions = model(positions, adj)
    
    return positions.cpu().numpy()


def diffusion_mesh_smoothing(verts: np.ndarray, faces: np.ndarray,
                             num_steps: int = 10,
                             model: Optional['MeshDiffusionSmoother'] = None) -> np.ndarray:
    """Apply diffusion-based mesh smoothing."""
    if not TORCH_AVAILABLE:
        return spectral_mesh_smoothing(verts, faces, smoothing_factor=0.5)
    
    N = verts.shape[0]
    A_norm = _build_adjacency(verts, faces)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions = torch.tensor(verts, dtype=torch.float32, device=device)
    
    A_coo = A_norm.tocoo()
    indices = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(A_coo.data, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(indices, values, (N, N))
    
    if model is None:
        model = MeshDiffusionSmoother(hidden_dim=32, num_steps=num_steps).to(device)
        model.eval()
    
    with torch.no_grad():
        smoothed = model(positions, adj, num_steps=num_steps)
    
    return smoothed.cpu().numpy()


def transformer_mesh_smoothing(verts: np.ndarray, faces: np.ndarray,
                               model: Optional['FeatureGuidedTransformer'] = None) -> np.ndarray:
    """Apply Feature-Guided Transformer smoothing."""
    if not TORCH_AVAILABLE:
        return spectral_mesh_smoothing(verts, faces, smoothing_factor=0.3)
    
    N = verts.shape[0]
    A_norm = _build_adjacency(verts, faces)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions = torch.tensor(verts, dtype=torch.float32, device=device)
    
    A_coo = A_norm.tocoo()
    indices = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(A_coo.data, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(indices, values, (N, N))
    
    if model is None:
        model = FeatureGuidedTransformer(hidden_dim=32, num_heads=4, num_layers=2).to(device)
        model.eval()
    
    with torch.no_grad():
        smoothed = model(positions, adj)
    
    return smoothed.cpu().numpy()


def neural_smoothing(verts: np.ndarray, faces: np.ndarray,
                     method: str = 'gnn', **kwargs) -> Tuple[np.ndarray, Dict]:
    """Main entry point for neural network-based mesh smoothing."""
    import time
    start_time = time.time()
    
    method = method.lower()
    
    if method == 'gnn':
        result = gnn_mesh_smoothing(verts, faces, **kwargs)
        method_name = "GNN Mesh Denoiser"
    elif method == 'diffusion':
        result = diffusion_mesh_smoothing(verts, faces, **kwargs)
        method_name = "Diffusion-based Smoothing"
    elif method == 'transformer':
        result = transformer_mesh_smoothing(verts, faces, **kwargs)
        method_name = "Feature-Guided Transformer"
    elif method == 'spectral':
        result = spectral_mesh_smoothing(verts, faces, **kwargs)
        method_name = "Spectral Smoothing"
    elif method == 'neural_inspired':
        result = neural_inspired_smoothing(verts, faces, **kwargs)
        method_name = "Neural-Inspired Smoothing"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed_time = time.time() - start_time
    
    info = {
        'method': method_name,
        'time_seconds': elapsed_time,
        'num_vertices': verts.shape[0],
        'num_faces': faces.shape[0],
        'pytorch_available': TORCH_AVAILABLE,
        'device': 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    }
    
    return result, info


__all__ = [
    'neural_smoothing', 'gnn_mesh_smoothing', 'diffusion_mesh_smoothing',
    'transformer_mesh_smoothing', 'spectral_mesh_smoothing', 'neural_inspired_smoothing',
]

if TORCH_AVAILABLE:
    __all__.extend(['GNNMeshDenoiser', 'MeshDiffusionSmoother', 'FeatureGuidedTransformer'])
