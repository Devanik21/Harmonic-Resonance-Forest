"""Demo: Explainable Resonance Visualization

This demo demonstrates how to use the resonance visualization
system to explain HRF predictions through class-wise resonance
contributions.

Note: This demo is a reference implementation. Adapt X_test and y_test
to your actual dataset.
"""
from visualizations.resonance_plots import (
    plot_decision_boundary
)
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from visualizations.resonance_plots import plot_class_resonance


def demo_resonance_visualization(
    model: object,
    X_test: np.ndarray,
    num_samples: int = 3,
    figsize: tuple = (12, 5)
) -> None:
    """
    Run an interactive demo of resonance visualization.

    Parameters
    ----------
    model : object
        Trained HRF model with predict_proba() method.
    X_test : np.ndarray
        Test data samples.
    num_samples : int, optional
        Number of samples to visualize. Default: 3.
    figsize : tuple, optional
        Subplot figure size. Default: (12, 5).
    """
    # Ensure we don't exceed available samples
    num_samples = min(num_samples, len(X_test))

    # Create subplots for multiple samples
    if num_samples > 1:
        fig, axes = plt.subplots(1, num_samples, figsize=figsize)
        if num_samples == 1:
            axes = [axes]
    else:
        axes = [None]

    for i in range(num_samples):
        sample = X_test[i:i+1]
        print(f"\n[Sample {i+1}/{num_samples}]")
        print(f"  Shape: {sample.shape}")

        # Get prediction details
        proba = model.predict_proba(sample)[0]
        predicted_class = model.classes_[np.argmax(proba)]
        confidence = np.max(proba)

        print(f"  Predicted Class: {predicted_class}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Class Probabilities:")
        for cls, prob in zip(model.classes_, proba):
            print(f"    - Class {cls}: {prob:.4f}")

        # Visualize resonance
        fig = plot_class_resonance(model, sample, figsize=(10, 6))
        plt.show()


def create_demo_dataset(
    n_samples: int = 300,
    n_features: int = 20,
    n_classes: int = 3,
    random_state: int = 42
) -> tuple:
    """
    Create a synthetic dataset for demo purposes.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_features : int
        Number of features.
    n_classes : int
        Number of classes.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 5,
        n_redundant=5,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    """
    Main demo execution.

    This script:
    1. Creates a synthetic dataset
    2. Trains an HRF model (uses a placeholder here)
    3. Visualizes resonance contributions for sample predictions
    """
    print("=" * 70)
    print("HARMONIC RESONANCE FOREST - EXPLAINABLE VISUALIZATION DEMO")
    print("=" * 70)

    # Create demo dataset
    print("\n[1/3] Creating synthetic dataset...")
    X_train, X_test, y_train, y_test = create_demo_dataset(
        n_samples=300,
        n_features=20,
        n_classes=3
    )
    print(f"  ✓ Dataset created: {X_train.shape}")

    # Train HRF model
    print("\n[2/3] Training HRF model...")
    try:
        # Import the HRF model from the main codebase
        import sys
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generalized_hrf_v2",
            "../HRF-Engine/generalized_hrf_v2.py"
        )
        hrf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hrf_module)

        HarmonicResonanceForest_Ultimate = hrf_module.HarmonicResonanceForest_Ultimate

        model = HarmonicResonanceForest_Ultimate()
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"  ✓ Model trained")
        print(f"    - Train Accuracy: {train_acc:.4f}")
        print(f"    - Test Accuracy: {test_acc:.4f}")

    except ImportError as e:
        print(f"  ⚠ Warning: Could not import HRF model: {e}")
        print(f"  Using mock model for visualization demo...")

        # Mock model for demonstration
        class MockHRFModel:
            def __init__(self):
                self.classes_ = np.array([0, 1, 2])

            def predict_proba(self, X):
                """Generate mock probabilities for demo."""
                n_samples = len(X) if isinstance(X, np.ndarray) else 1
                # Simulate realistic probabilities
                probs = np.random.dirichlet([2, 1, 1], n_samples)
                return probs

        model = MockHRFModel()
        print(f"  ✓ Mock model created for visualization testing")

    # Visualize resonance contributions
    print("\n[3/3] Visualizing resonance contributions...")
    demo_resonance_visualization(model, X_test, num_samples=1)
    print("\n  ✓ Visualization complete!")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)



fig = plot_decision_boundary(
    model,
    X_test,
    y_test
)

plt.show()