"""Demo: Resonance Heatmap Visualization

This demo demonstrates the Phase 2 explainability feature:
Resonance Intensity Heatmap Visualization for HRF models.

The heatmap shows resonance intensity patterns across samples
and classes, enabling users to understand prediction confidence
and class-specific resonance behavior.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from visualizations import (
    plot_resonance_heatmap,
    plot_resonance_comparison,
    ResonanceCaptureWrapper,
    extract_resonance_samples
)


def demo_basic_heatmap(
    model: object,
    X_test: np.ndarray,
    num_samples: int = 100
) -> None:
    """
    Demonstrate basic resonance heatmap visualization.

    Parameters
    ----------
    model : object
        Trained HRF model.
    X_test : np.ndarray
        Test samples.
    num_samples : int
        Number of samples to visualize. Default: 100.
    """
    print("\n" + "="*70)
    print("DEMO 1: Basic Resonance Heatmap")
    print("="*70)

    # Limit samples
    num_samples = min(num_samples, len(X_test))
    X_subset = X_test[:num_samples]

    print(f"\n[1/2] Generating heatmap for {num_samples} samples...")
    fig = plot_resonance_heatmap(model, X_subset, normalize_rows=True)
    print(f"  ✓ Heatmap created (row-normalized)")
    plt.show()

    print(f"\n[2/2] Generating raw resonance heatmap...")
    fig = plot_resonance_heatmap(model, X_subset, normalize_rows=False)
    print(f"  ✓ Heatmap created (raw intensities)")
    plt.show()


def demo_heatmap_variants(
    model: object,
    X_test: np.ndarray,
    num_samples: int = 50
) -> None:
    """
    Demonstrate heatmap visualization with different colormaps.

    Parameters
    ----------
    model : object
        Trained HRF model.
    X_test : np.ndarray
        Test samples.
    num_samples : int
        Number of samples to visualize. Default: 50.
    """
    print("\n" + "="*70)
    print("DEMO 2: Heatmap Variants (Different Colormaps)")
    print("="*70)

    num_samples = min(num_samples, len(X_test))
    X_subset = X_test[:num_samples]

    colormaps = ['YlOrRd', 'Blues', 'Greens', 'Purples']

    print(f"\n[Visualizing with {len(colormaps)} different colormaps...]")
    for cmap in colormaps:
        print(f"  • {cmap}...", end=' ')
        fig = plot_resonance_heatmap(
            model, X_subset, cmap=cmap, normalize_rows=True, figsize=(12, 5)
        )
        print("done")
        plt.show()


def demo_comparison_heatmap(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_samples: int = 50
) -> None:
    """
    Demonstrate comparison heatmap with ground truth labels.

    Parameters
    ----------
    model : object
        Trained HRF model.
    X_test : np.ndarray
        Test samples.
    y_test : np.ndarray
        True labels.
    num_samples : int
        Number of samples to visualize. Default: 50.
    """
    print("\n" + "="*70)
    print("DEMO 3: Comparison Heatmap (with Ground Truth)")
    print("="*70)

    num_samples = min(num_samples, len(X_test))
    X_subset = X_test[:num_samples]
    y_subset = y_test[:num_samples]

    print(f"\n[Generating comparison heatmap...]")
    fig = plot_resonance_comparison(model, X_subset, y_subset)
    print(f"  ✓ Comparison heatmap created")
    plt.show()

    # Analyze predictions
    predictions = model.predict(X_subset)
    correct = np.sum(predictions == y_subset)
    accuracy = correct / len(y_subset)
    print(f"\n[Analysis on {num_samples} samples]")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{num_samples})")


def demo_resonance_statistics(
    model: object,
    X_test: np.ndarray,
    num_samples: int = 100
) -> None:
    """
    Analyze and visualize resonance statistics.

    Parameters
    ----------
    model : object
        Trained HRF model.
    X_test : np.ndarray
        Test samples.
    num_samples : int
        Number of samples to analyze. Default: 100.
    """
    print("\n" + "="*70)
    print("DEMO 4: Resonance Statistics Analysis")
    print("="*70)

    num_samples = min(num_samples, len(X_test))
    X_subset = X_test[:num_samples]

    print(f"\n[1/4] Extracting resonance samples...")
    # Wrap model for resonance capture
    wrapped_model = ResonanceCaptureWrapper(model)
    resonance_map = extract_resonance_samples(wrapped_model, X_subset)
    print(f"  ✓ Resonance map shape: {resonance_map.shape}")

    print(f"\n[2/4] Computing statistics...")
    # Normalize
    resonance_normalized = resonance_map / (np.sum(resonance_map, axis=1, keepdims=True) + 1e-10)

    # Per-class statistics
    class_means = np.mean(resonance_normalized, axis=0)
    class_stds = np.std(resonance_normalized, axis=0)
    class_maxs = np.max(resonance_normalized, axis=0)

    print(f"  Per-class resonance statistics:")
    for i, cls in enumerate(model.classes_):
        print(f"    Class {cls}:")
        print(f"      Mean: {class_means[i]:.4f}")
        print(f"      Std:  {class_stds[i]:.4f}")
        print(f"      Max:  {class_maxs[i]:.4f}")

    # Sample confidence distribution
    max_resonance = np.max(resonance_normalized, axis=1)
    print(f"\n[3/4] Sample confidence distribution:")
    print(f"  Min confidence: {np.min(max_resonance):.4f}")
    print(f"  Mean confidence: {np.mean(max_resonance):.4f}")
    print(f"  Max confidence: {np.max(max_resonance):.4f}")

    # Visualize statistics
    print(f"\n[4/4] Plotting statistics...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Class-wise means
    axes[0].bar(range(len(model.classes_)), class_means, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Class', fontweight='bold')
    axes[0].set_ylabel('Mean Resonance', fontweight='bold')
    axes[0].set_title('Mean Resonance by Class', fontweight='bold')
    axes[0].set_xticks(range(len(model.classes_)))
    axes[0].set_xticklabels([str(c) for c in model.classes_])
    axes[0].grid(axis='y', alpha=0.3)

    # Confidence distribution
    axes[1].hist(max_resonance, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(max_resonance), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].set_xlabel('Max Resonance (Confidence)', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Sample Confidence Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    print(f"  ✓ Statistics visualization complete")


def create_demo_dataset(
    n_samples: int = 400,
    n_features: int = 20,
    n_classes: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic dataset for demo purposes.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_features : int
        Number of features.
    n_classes : int
        Number of classes.
    random_state : int
        Random seed.

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

    Runs all heatmap visualization demos showing different aspects
    of resonance intensity patterns.
    """
    print("="*70)
    print("HARMONIC RESONANCE FOREST - PHASE 2 HEATMAP VISUALIZATION DEMO")
    print("="*70)

    # Create dataset
    print("\n[Setup] Creating synthetic dataset...")
    X_train, X_test, y_train, y_test = create_demo_dataset(
        n_samples=400,
        n_features=20,
        n_classes=3
    )
    print(f"  ✓ Dataset created: train {X_train.shape}, test {X_test.shape}")

    # Train model
    print("\n[Setup] Training HRF model...")
    try:
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
        test_acc = model.score(X_test, y_test)
        print(f"  ✓ Model trained")
        print(f"    Test Accuracy: {test_acc:.4f}")

    except Exception as e:
        print(f"  ⚠ Warning: {e}")
        print(f"  Using mock model for demo...")

        class MockHRFModel:
            def __init__(self):
                self.classes_ = np.array([0, 1, 2])

            def predict_proba(self, X):
                n_samples = len(X)
                probs = np.random.dirichlet([2, 1.5, 1], n_samples)
                return probs

            def predict(self, X):
                return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

            def score(self, X, y):
                return np.mean(self.predict(X) == y)

        model = MockHRFModel()

    # Run demos
    print("\n" + "="*70)
    print("RUNNING DEMOS")
    print("="*70)

    demo_basic_heatmap(model, X_test, num_samples=50)
    demo_heatmap_variants(model, X_test, num_samples=30)
    demo_comparison_heatmap(model, X_test, y_test, num_samples=40)
    demo_resonance_statistics(model, X_test, num_samples=100)

    print("\n" + "="*70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*70)
