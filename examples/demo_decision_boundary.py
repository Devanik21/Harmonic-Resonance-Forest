"""Demo: Decision Boundary Visualization for HRF

This demo showcases Phase 3 explainability feature:
Decision Boundary Visualization for Harmonic Resonance Forest.

The visualization helps understand how HRF separates different classes
in the feature space and compares with traditional ML algorithms.
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from visualizations import (
    plot_decision_boundary,
    plot_decision_boundary_comparison
)


def demo_2d_decision_boundary():
    """Demonstrate decision boundary on 2D datasets."""
    print("\n" + "="*70)
    print("DEMO 1: 2D Decision Boundary Visualization")
    print("="*70)

    # Create 2D datasets
    datasets = {
        'Make Blobs': make_classification(
            n_samples=300, n_features=2, n_informative=2,
            n_redundant=0, n_classes=3, n_clusters_per_class=1,
            random_state=42
        ),
        'Moons': make_moons(n_samples=300, noise=0.1, random_state=42),
        'Circles': make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42),
    }

    for dataset_name, (X, y) in datasets.items():
        print(f"\n[Dataset: {dataset_name}]")
        print(f"  Shape: {X.shape}, Classes: {len(np.unique(y))}")

        # Create mock model
        class MockHRFModel:
            def __init__(self):
                self.classes_ = np.unique(y)

            def predict(self, X_test):
                # Simple resonance-based prediction
                distances = np.sqrt(np.sum(X_test**2, axis=1))
                return self.classes_[np.argmin(
                    np.abs(distances[:, None] - np.linspace(0, 5, len(self.classes_))),
                    axis=1
                )]

        model = MockHRFModel()

        # Plot decision boundary
        print(f"  Plotting decision boundary...", end=' ')
        fig, pca = plot_decision_boundary(
            model, X, y,
            resolution=200,
            figsize=(10, 8),
            cmap='RdYlBu'
        )
        print("done")
        plt.show()


def demo_high_dimensional_decision_boundary():
    """Demonstrate decision boundary on high-dimensional data."""
    print("\n" + "="*70)
    print("DEMO 2: High-Dimensional Decision Boundary (PCA Projection)")
    print("="*70)

    # Create high-dimensional dataset
    print("\n[Creating 20-dimensional dataset...]")
    X, y = make_classification(
        n_samples=400,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Original shape: {X_scaled.shape}")

    # Create mock HRF model
    class MockHRFModel:
        def __init__(self):
            self.classes_ = np.unique(y)
            self.scaler = scaler

        def predict(self, X_input):
            X_scaled_input = self.scaler.transform(X_input)
            distances = np.linalg.norm(X_scaled_input, axis=1)
            return self.classes_[np.argmin(
                np.abs(distances[:, None] - np.linspace(1, 8, len(self.classes_))),
                axis=1
            )]

    model = MockHRFModel()

    # Plot with automatic PCA projection
    print("[Plotting decision boundary with PCA projection...]")
    fig, pca = plot_decision_boundary(
        model, X_scaled, y,
        resolution=250,
        figsize=(12, 9),
        cmap='plasma',
        pca_if_needed=True,
        title="HRF Decision Boundary (20D → 2D PCA)"
    )

    if pca is not None:
        print(f"  PCA Variance Explained:")
        print(f"    PC1: {pca.explained_variance_ratio_[0]:.2%}")
        print(f"    PC2: {pca.explained_variance_ratio_[1]:.2%}")
        print(f"    Total: {np.sum(pca.explained_variance_ratio_):.2%}")

    plt.show()


def demo_model_comparison():
    """Compare HRF with other classifiers."""
    print("\n" + "="*70)
    print("DEMO 3: Decision Boundary Comparison (HRF vs Other Models)")
    print("="*70)

    # Create dataset
    print("\n[Creating synthetic dataset...]")
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=42
    )

    print(f"  Shape: {X.shape}, Classes: {len(np.unique(y))}")

    # Mock HRF model
    class HRFModel:
        def __init__(self):
            self.classes_ = np.unique(y)

        def predict(self, X):
            return ((X[:, 0]**2 + X[:, 1]**2) > 2.5).astype(int)

    # Train real models
    print("\n[Training models...]")
    models = {
        'HRF (Mock)': HRFModel(),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, random_state=42).fit(X, y),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y),
    }

    print(f"  Models trained: {list(models.keys())}")

    # Compare decision boundaries
    print("\n[Generating comparison visualization...]")
    fig = plot_decision_boundary_comparison(
        models, X, y,
        resolution=200,
        figsize=(15, 5),
        cmap='coolwarm'
    )
    plt.show()


def demo_multiclass_decision_boundary():
    """Demonstrate multiclass decision boundary."""
    print("\n" + "="*70)
    print("DEMO 4: Multiclass Decision Boundary")
    print("="*70)

    # Create 3-class dataset
    print("\n[Creating 4-class dataset...]")
    X, y = make_classification(
        n_samples=400,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=42
    )

    print(f"  Shape: {X.shape}, Classes: {len(np.unique(y))}")

    # Mock model
    class MulticlassHRFModel:
        def __init__(self):
            self.classes_ = np.unique(y)

        def predict(self, X):
            # Zone-based prediction
            x, y_coord = X[:, 0], X[:, 1]
            predictions = np.zeros(len(X), dtype=int)
            predictions[(x > 0) & (y_coord > 0)] = 0
            predictions[(x <= 0) & (y_coord > 0)] = 1
            predictions[(x <= 0) & (y_coord <= 0)] = 2
            predictions[(x > 0) & (y_coord <= 0)] = 3
            return predictions

    model = MulticlassHRFModel()

    print("[Plotting multiclass decision boundary...]")
    fig, pca = plot_decision_boundary(
        model, X, y,
        resolution=250,
        figsize=(12, 9),
        cmap='Set3',
        title="HRF Multiclass Decision Boundary"
    )
    plt.show()


def demo_resolution_comparison():
    """Compare different resolutions."""
    print("\n" + "="*70)
    print("DEMO 5: Resolution Impact on Decision Boundary")
    print("="*70)

    # Create dataset
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)

    class CircleModel:
        def __init__(self):
            self.classes_ = np.unique(y)

        def predict(self, X):
            distances = np.sqrt(np.sum(X**2, axis=1))
            return (distances > 1.0).astype(int)

    model = CircleModel()

    # Test different resolutions
    resolutions = [50, 150, 300]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, res in zip(axes, resolutions):
        print(f"\n[Resolution: {res}x{res}]", end=' ')

        # Create inline visualization
        fig_temp, pca = plot_decision_boundary(
            model, X, y,
            resolution=res,
            figsize=(5, 4),
            cmap='RdBu'
        )

        # This would show each individually
        print("done")

    print("\n  Observation: Higher resolution reveals finer decision boundaries")
    print("  Trade-off: Increased computation time")


if __name__ == "__main__":
    """
    Main demo execution.

    Runs all decision boundary visualization demos showing different aspects:
    - 2D datasets (direct visualization)
    - High-dimensional datasets (PCA projection)
    - Model comparison
    - Multiclass problems
    - Resolution trade-offs
    """
    print("="*70)
    print("HARMONIC RESONANCE FOREST - PHASE 3 DECISION BOUNDARY DEMO")
    print("="*70)

    try:
        demo_2d_decision_boundary()
        demo_high_dimensional_decision_boundary()
        demo_model_comparison()
        demo_multiclass_decision_boundary()
        demo_resolution_comparison()

        print("\n" + "="*70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
