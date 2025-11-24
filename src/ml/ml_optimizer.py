"""
Machine Learning-Based Smoothing Parameter Optimizer

This module implements a neural network that predicts optimal smoothing parameters
based on mesh characteristics. The model learns from labeled examples to recommend
the best smoothing algorithm, iterations, and lambda values.

Key Features:
- Feature extraction from mesh geometry
- Neural network for parameter prediction
- Training on synthetic/real medical data
- Transfer learning capability
"""

import numpy as np
import pickle
import os
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. ML optimizer will use fallback heuristics.")


class MeshFeatureExtractor:
    """Extract geometric features from mesh for ML model."""
    
    @staticmethod
    def extract_features(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from mesh geometry.
        
        Returns 12-dimensional feature vector:
        [num_vertices, num_faces, avg_edge_length, std_edge_length,
         min_edge, max_edge, avg_aspect_ratio, std_aspect_ratio,
         surface_area_est, volume_est, bbox_ratio, compactness]
        """
        num_verts = verts.shape[0]
        num_faces = faces.shape[0]
        
        # Edge lengths
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)
        
        all_edges = np.concatenate([e0, e1, e2])
        
        avg_edge = np.mean(all_edges)
        std_edge = np.std(all_edges)
        min_edge = np.min(all_edges)
        max_edge = np.max(all_edges)
        
        # Aspect ratios
        max_e = np.maximum(np.maximum(e0, e1), e2)
        min_e = np.minimum(np.minimum(e0, e1), e2)
        min_e = np.where(min_e == 0, 1e-12, min_e)
        aspect_ratios = max_e / min_e
        
        avg_aspect = np.mean(aspect_ratios)
        std_aspect = np.std(aspect_ratios)
        
        # Surface area estimate (sum of triangle areas)
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        surface_area = np.sum(areas)
        
        # Volume estimate (sum of signed tetrahedron volumes)
        volume = np.abs(np.sum(np.sum(v0 * cross, axis=1) / 6.0))
        
        # Bounding box ratio
        bbox_min = np.min(verts, axis=0)
        bbox_max = np.max(verts, axis=0)
        bbox_dims = bbox_max - bbox_min
        bbox_ratio = np.max(bbox_dims) / (np.min(bbox_dims) + 1e-12)
        
        # Compactness (sphere-like = 1.0)
        compactness = (36 * np.pi * volume**2) / (surface_area**3 + 1e-12)
        
        features = np.array([
            num_verts,
            num_faces,
            avg_edge,
            std_edge,
            min_edge,
            max_edge,
            avg_aspect,
            std_aspect,
            surface_area,
            volume,
            bbox_ratio,
            compactness
        ])
        
        # Log-scale for large values, normalize
        features[0] = np.log10(features[0] + 1)  # num_verts
        features[1] = np.log10(features[1] + 1)  # num_faces
        features[8] = np.log10(features[8] + 1)  # surface_area
        features[9] = np.log10(features[9] + 1)  # volume
        
        return features


class SmoothingParameterNN(nn.Module):
    """Neural network for predicting optimal smoothing parameters."""
    
    def __init__(self, input_dim: int = 12):
        super(SmoothingParameterNN, self).__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Output heads
        self.algorithm_head = nn.Linear(64, 2)  # [Laplacian, Taubin]
        self.iterations_head = nn.Linear(64, 1)  # Continuous value
        self.lambda_head = nn.Linear(64, 1)      # Continuous value (0-1)
        
    def forward(self, x):
        features = self.feature_net(x)
        
        algorithm_logits = self.algorithm_head(features)
        iterations = torch.relu(self.iterations_head(features))  # Ensure positive
        lambda_val = torch.sigmoid(self.lambda_head(features))   # 0-1 range
        
        return algorithm_logits, iterations, lambda_val


class MLSmoothingOptimizer:
    """
    Machine Learning-based optimizer for smoothing parameters.
    
    Uses a neural network trained on medical mesh data to predict
    optimal smoothing parameters.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.extractor = MeshFeatureExtractor()
        self.model = None
        self.device = None
        self.is_trained = False
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = SmoothingParameterNN()
            self.model.to(self.device)
            
            # Load pretrained model if available
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
    
    def predict(self, verts: np.ndarray, faces: np.ndarray) -> Dict[str, any]:
        """
        Predict optimal smoothing parameters for a mesh.
        
        Returns:
            dict with keys:
                - algorithm: 'Laplacian' or 'Taubin'
                - iterations: int (recommended iterations)
                - lambda_val: float (smoothing strength)
                - confidence: float (0-1, model confidence)
        """
        features = self.extractor.extract_features(verts, faces)
        
        if not TORCH_AVAILABLE or self.model is None or not self.is_trained:
            # Fallback to heuristics
            return self._heuristic_predict(features, verts, faces)
        
        # Neural network prediction
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            algo_logits, iterations, lambda_val = self.model(features_tensor)
            
            algo_probs = torch.softmax(algo_logits, dim=1)
            algo_idx = torch.argmax(algo_probs, dim=1).item()
            confidence = algo_probs[0, algo_idx].item()
            
            algorithm = 'Taubin' if algo_idx == 1 else 'Laplacian'
            iterations_val = max(5, min(50, int(iterations.item())))
            lambda_val = lambda_val.item()
        
        return {
            'algorithm': algorithm,
            'iterations': iterations_val,
            'lambda': lambda_val,
            'confidence': confidence
        }
    
    def _heuristic_predict(self, features: np.ndarray, verts: np.ndarray, 
                          faces: np.ndarray) -> Dict[str, any]:
        """
        Fallback heuristic-based prediction when ML model not available.
        
        Rules based on domain knowledge:
        - Large meshes (>50k triangles): Use Taubin, moderate iterations
        - High aspect ratio: More iterations needed
        - Complex geometry (low compactness): Prefer Taubin
        - Simple geometry: Laplacian is sufficient
        """
        num_verts = verts.shape[0]
        num_faces = faces.shape[0]
        avg_aspect = features[6]
        compactness = features[11]
        
        # Decision tree based on mesh characteristics
        if num_faces > 50000:
            # Large mesh
            algorithm = 'Taubin'
            iterations = min(30, int(15 + num_faces / 5000))
            lambda_val = 0.5
        elif avg_aspect > 3.0:
            # Poor quality mesh
            algorithm = 'Taubin'
            iterations = min(40, int(20 + avg_aspect * 2))
            lambda_val = 0.6
        elif compactness < 0.5:
            # Complex geometry
            algorithm = 'Taubin'
            iterations = 20
            lambda_val = 0.5
        else:
            # Simple, small mesh
            algorithm = 'Laplacian'
            iterations = 15
            lambda_val = 0.5
        
        return {
            'algorithm': algorithm,
            'iterations': iterations,
            'lambda': lambda_val,
            'confidence': 0.7  # Heuristic confidence
        }
    
    def train(self, training_data: list, epochs: int = 100, 
              learning_rate: float = 0.001):
        """
        Train the neural network on labeled data.
        
        Args:
            training_data: List of tuples (verts, faces, labels)
                labels = {'algorithm': str, 'iterations': int, 'lambda': float,
                         'quality_score': float}
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Cannot train model.")
            return
        
        print(f"Training ML optimizer on {len(training_data)} samples...")
        
        # Prepare training data
        X = []
        y_algo = []
        y_iters = []
        y_lambda = []
        
        for verts, faces, labels in training_data:
            features = self.extractor.extract_features(verts, faces)
            X.append(features)
            
            # Encode algorithm (0=Laplacian, 1=Taubin)
            algo_idx = 1 if labels['algorithm'] == 'Taubin' else 0
            y_algo.append(algo_idx)
            y_iters.append(labels['iterations'])
            y_lambda.append(labels['lambda'])
        
        X = torch.FloatTensor(np.array(X)).to(self.device)
        y_algo = torch.LongTensor(y_algo).to(self.device)
        y_iters = torch.FloatTensor(y_iters).unsqueeze(1).to(self.device)
        y_lambda = torch.FloatTensor(y_lambda).unsqueeze(1).to(self.device)
        
        # Setup training
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_mse = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            algo_logits, pred_iters, pred_lambda = self.model(X)
            
            # Multi-task loss
            loss_algo = criterion_ce(algo_logits, y_algo)
            loss_iters = criterion_mse(pred_iters, y_iters)
            loss_lambda = criterion_mse(pred_lambda, y_lambda)
            
            # Weighted combination
            total_loss = loss_algo + 0.5 * loss_iters + 0.3 * loss_lambda
            
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")
        
        self.is_trained = True
        print("Training complete!")
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        if not TORCH_AVAILABLE or self.model is None:
            return
        
        torch.save({
            'model_state': self.model.state_dict(),
            'is_trained': self.is_trained
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from disk."""
        if not TORCH_AVAILABLE or self.model is None:
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.is_trained = checkpoint.get('is_trained', True)
        self.model.eval()
        print(f"Model loaded from {path}")


def generate_synthetic_training_data(num_samples: int = 100) -> list:
    """
    Generate synthetic training data for model development.
    
    Creates meshes with known optimal parameters based on:
    - Mesh size -> affects iterations
    - Complexity -> affects algorithm choice
    - Quality -> affects lambda value
    
    In production, replace with real medical annotations.
    """
    training_data = []
    
    print(f"Generating {num_samples} synthetic training samples...")
    
    for i in range(num_samples):
        # Random mesh characteristics
        num_verts = np.random.randint(500, 100000)
        num_faces = int(num_verts * 1.8)
        
        # Synthetic vertex positions
        verts = np.random.randn(num_verts, 3) * 10
        
        # Synthetic faces (random triangulation)
        faces = np.random.randint(0, num_verts, size=(num_faces, 3))
        
        # Generate ground truth labels based on characteristics
        if num_verts > 50000:
            optimal_algo = 'Taubin'
            optimal_iters = 25
            optimal_lambda = 0.5
        elif num_verts > 10000:
            optimal_algo = 'Taubin'
            optimal_iters = 20
            optimal_lambda = 0.6
        else:
            optimal_algo = 'Laplacian'
            optimal_iters = 15
            optimal_lambda = 0.5
        
        # Add some noise to make it realistic
        optimal_iters += np.random.randint(-3, 4)
        optimal_lambda += np.random.uniform(-0.1, 0.1)
        optimal_lambda = np.clip(optimal_lambda, 0.1, 0.9)
        
        labels = {
            'algorithm': optimal_algo,
            'iterations': optimal_iters,
            'lambda': optimal_lambda,
            'quality_score': np.random.uniform(0.7, 1.0)
        }
        
        training_data.append((verts, faces, labels))
    
    print("Synthetic data generation complete!")
    return training_data


# Singleton instance for easy access
_ml_optimizer = None

def get_ml_optimizer(model_path: Optional[str] = None) -> MLSmoothingOptimizer:
    """Get or create the ML optimizer singleton."""
    global _ml_optimizer
    if _ml_optimizer is None:
        _ml_optimizer = MLSmoothingOptimizer(model_path)
    return _ml_optimizer
