#!/usr/bin/env python3
"""
Training script for the ML-based smoothing optimizer.

This script:
1. Generates synthetic training data (or loads real annotations)
2. Trains the neural network
3. Saves the trained model
4. Evaluates performance

Usage:
    python3 train_ml_model.py [--samples 500] [--epochs 100]
"""

import argparse
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml import MLSmoothingOptimizer, generate_synthetic_training_data

def main():
    parser = argparse.ArgumentParser(description='Train ML smoothing optimizer')
    parser.add_argument('--samples', type=int, default=200,
                       help='Number of training samples (default: 200)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--output', type=str, default='models/smoothing_optimizer.pth',
                       help='Output model path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ML SMOOTHING OPTIMIZER - TRAINING")
    print("=" * 60)
    
    # Generate training data
    print(f"\n1. Generating {args.samples} synthetic training samples...")
    training_data = generate_synthetic_training_data(args.samples)
    
    # Create optimizer
    print("\n2. Initializing neural network...")
    optimizer = MLSmoothingOptimizer()
    
    # Train
    print(f"\n3. Training for {args.epochs} epochs...")
    optimizer.train(training_data, epochs=args.epochs, learning_rate=0.001)
    
    # Save model
    print(f"\n4. Saving model to {args.output}...")
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    optimizer.save_model(args.output)
    
    # Quick evaluation
    print("\n5. Evaluating on test samples...")
    test_samples = 5
    for i in range(test_samples):
        verts = np.random.randn(np.random.randint(1000, 50000), 3) * 10
        faces = np.random.randint(0, verts.shape[0], size=(int(verts.shape[0] * 1.8), 3))
        
        prediction = optimizer.predict(verts, faces)
        print(f"\nTest {i+1}:")
        print(f"  Mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")
        print(f"  Predicted Algorithm: {prediction['algorithm']}")
        print(f"  Predicted Iterations: {prediction['iterations']}")
        print(f"  Predicted Lambda: {prediction['lambda']:.3f}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {args.output}")
    print("\nTo use in the app:")
    print("  1. The app will automatically load this model")
    print("  2. Enable 'ML-Optimized Parameters' in the sidebar")
    print("  3. Let AI predict the best smoothing settings!")

if __name__ == "__main__":
    main()
