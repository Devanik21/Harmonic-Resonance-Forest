# CPU Fallback Workflow

## Overview
Harmonic Resonance Forest (HRF) heavily leverages NVIDIA GPU-accelerated libraries like `CuPy` and `RAPIDS cuML` for high-performance tensor operations and accelerated K-Nearest Neighbors (KNN) searches. However, the repository maintains full accessibility for contributors and researchers running CPU-only environments.

This guide provides instructions on how to set up, execute, and expect performance differences when running HRF without CUDA support.

## Dependency Alternatives

For systems without an NVIDIA GPU, you must omit `cupy` and `cuml` and use standard Python CPU libraries.

| GPU Dependency | CPU Fallback | Description |
|----------------|--------------|-------------|
| `cupy` | `numpy` | Used for array and resonance matrix calculations. |
| `cuml.neighbors` | `sklearn.neighbors` | Used for K-Nearest Neighbors in evolutionary searches. |

**Installation for CPU-only Systems:**
```bash
# Install core dependencies (excluding CuPy and RAPIDS)
pip install numpy scipy scikit-learn pandas matplotlib xgboost

# Optional: Ensure standard implementations are updated
pip install --upgrade scikit-learn numpy
```

## Execution Paths for Experiments

When executing HRF code without a GPU, simply replace GPU-accelerated calls with standard CPU libraries.
If you are developing or modifying scripts, ensure fallback mechanisms are implemented. For instance:

```python
try:
    import cupy as cp
    import cuml
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    from sklearn.neighbors import KNeighborsClassifier
    GPU_AVAILABLE = False
    print("CUDA dependencies not found. Falling back to CPU workflow.")
```

## Expected Performance Differences

Using CPU instead of GPU acceleration will introduce differences primarily in **speed** and **scaling**:

1. **Computational Speed**: Array interference operations and parallel distance calculations will be significantly slower. The Evolutionary Search for optimal physical laws may take minutes rather than seconds.
2. **Memory Scaling**: Large array operations may hit system RAM limits sooner than VRAM limits, depending on your system specifications.
3. **Accuracy Stability**: The **accuracy and mathematical validity of HRF will remain identical** whether executed on a GPU or CPU, provided the same random seeds and cross-validation techniques are used.

## Reduced-Scale Benchmark Example

To test the HRF algorithm on limited compute resources, you can use the Synthetic Moons dataset or a subsampled version of the OpenML 1471 corpus.

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from hrf_engine import HarmonicResonanceClassifier # Assuming basic engine import

# 1. Create a reduced-scale synthetic dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Run the HRF algorithm with fewer estimators and neighbors
# to speed up CPU computation
hrf = HarmonicResonanceClassifier(
    n_neighbors=3, 
    gamma=0.5, 
    n_estimators=10  # Reduced from 60
)

# 3. Train and test
hrf.fit(X_train, y_train)
accuracy = hrf.score(X_test, y_test)

print(f"CPU Fallback Test Accuracy: {accuracy:.4f}")
```

Following this guide ensures you can continue researching, developing, and contributing to Harmonic Resonance Forest from any hardware setup.
