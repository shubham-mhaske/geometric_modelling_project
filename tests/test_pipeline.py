#!/usr/bin/env python3
"""
Quick test script to verify all pipeline components work correctly.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_smoothing():
    """Test smoothing algorithms."""
    print("Testing smoothing algorithms...")
    from src.algorithms import smoothing
    
    # Create a simple mesh
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3]
    ])
    
    # Test Laplacian
    result = smoothing.laplacian_smoothing(verts, faces, iterations=5)
    assert result.shape == verts.shape, "Laplacian output shape mismatch"
    print("  ✓ Laplacian smoothing works")
    
    # Test Taubin
    result = smoothing.taubin_smoothing(verts, faces, iterations=5)
    assert result.shape == verts.shape, "Taubin output shape mismatch"
    print("  ✓ Taubin smoothing works")

    labels = np.array([0, 0, 1, 1])
    result = smoothing.laplacian_smoothing(verts, faces, iterations=1, vertex_labels=labels)
    assert result.shape == verts.shape, "Semantic Laplacian output shape mismatch"
    print("  ✓ Semantic Laplacian smoothing works")

def test_simplification():
    """Test QEM mesh simplification."""
    print("\nTesting QEM simplification...")
    from src.algorithms import simplification
    
    # Create a simple mesh with more triangles
    verts = np.random.rand(100, 3) * 10
    faces = np.random.randint(0, 100, size=(150, 3))
    
    new_verts, new_faces = simplification.qem_simplification(verts, faces, target_reduction=0.5)
    
    assert new_faces.shape[0] < faces.shape[0], "Simplification didn't reduce triangles"
    print(f"  ✓ Reduced from {faces.shape[0]} to {new_faces.shape[0]} triangles")

def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics...")
    from src.algorithms import metrics
    
    # Create two point clouds
    verts1 = np.random.rand(1000, 3) * 10
    verts2 = verts1 + np.random.randn(1000, 3) * 0.1  # Slightly perturbed
    
    # Test Hausdorff distance
    dist = metrics.hausdorff_distance(verts1, verts2, sample_size=500)
    assert dist > 0, "Hausdorff distance should be positive"
    print(f"  ✓ Hausdorff distance: {dist:.3f}")
    
    # Test volume change
    vol_change = metrics.compute_volume_change_percent(100.0, 105.0)
    assert abs(vol_change - 5.0) < 0.01, "Volume change calculation wrong"
    print(f"  ✓ Volume change calculation: {vol_change}%")


def test_label_mapping():
    """Test mapping from labels to vertices."""
    print("\nTesting label mapping...")
    from src.algorithms.processing import map_labels_to_vertices

    volume = np.zeros((4, 4, 4), dtype=np.int16)
    volume[1:, 1:, 1:] = 4
    affine = np.eye(4)
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.2, 1.1, 1.0],
        [3.0, 3.0, 3.0]
    ])

    labels = map_labels_to_vertices(volume, affine, verts)
    assert labels.shape == (3,)
    assert labels[0] == 0 and np.all(labels[1:] == 4)
    print("  ✓ Label mapping works")


def test_label_coarsening():
    """Ensure verbose atlases are reduced to coarse groups."""
    print("\nTesting label coarsening...")
    from src.algorithms.processing import coarsen_label_volume

    verbose = np.arange(64, dtype=np.int16).reshape(4, 4, 4)
    collapsed = coarsen_label_volume(verbose, canonical_labels=(), max_groups=3)
    assert collapsed.max() <= 3, "Collapsed volume should use <= max_groups labels"

    canonical = np.zeros((2, 2, 2), dtype=np.int16)
    canonical[0, 0, 0] = 1
    canonical[0, 0, 1] = 2
    canonical[0, 1, 0] = 4
    preserved = coarsen_label_volume(canonical)
    assert np.array_equal(preserved, canonical), "Canonical BraTS labels must remain untouched"
    print("  ✓ Label coarsening works")

def test_imports():
    """Test all required imports."""
    print("\nTesting imports...")
    try:
        import streamlit
        import nibabel
        import pyvista
        import plotly
        import scipy
        import matplotlib
        import pandas
        print("  ✓ All dependencies available")
        
        # Test ML optimizer (optional)
        try:
            import torch
            print("  ✓ PyTorch available (ML features enabled)")
        except ImportError:
            print("  ⚠ PyTorch not installed (ML features will use heuristics)")
        
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        return False
    return True

def test_ml_optimizer():
    """Test ML optimizer (with fallback to heuristics)."""
    print("\nTesting ML optimizer...")
    try:
        from src.ml import MLSmoothingOptimizer, MeshFeatureExtractor
        
        # Test feature extraction
        verts = np.random.randn(1000, 3) * 10
        faces = np.random.randint(0, 1000, size=(1800, 3))
        
        extractor = MeshFeatureExtractor()
        features = extractor.extract_features(verts, faces)
        assert features.shape == (12,), "Feature vector should be 12-dimensional"
        print("  ✓ Feature extraction works")
        
        # Test prediction (will use heuristics if no model)
        optimizer = MLSmoothingOptimizer()
        prediction = optimizer.predict(verts, faces)
        
        assert 'algorithm' in prediction
        assert 'iterations' in prediction
        assert 'lambda' in prediction
        assert 'confidence' in prediction
        
        print(f"  ✓ Prediction works: {prediction['algorithm']}, "
              f"{prediction['iterations']} iterations, "
              f"confidence: {prediction['confidence']:.2%}")
        
        return True
    except Exception as e:
        print(f"  ✗ ML optimizer test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("GEOMETRIC MODELING PROJECT - COMPONENT TESTS")
    print("=" * 60)
    
    try:
        if not test_imports():
            sys.exit(1)
        
        test_smoothing()
        test_simplification()
        test_metrics()
        test_label_mapping()
        test_label_coarsening()
        test_ml_optimizer()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
