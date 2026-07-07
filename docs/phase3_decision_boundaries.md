# Phase 3: Decision Boundary Visualization

## Overview

Phase 3 introduces **Decision Boundary Visualization**, enabling researchers to visualize how HRF separates classes in the feature space. This powerful technique reveals the "decision regions" where HRF assigns each class, providing intuitive understanding of model behavior.

## What Are Decision Boundaries?

### Definition

A decision boundary is the region in feature space where the model transitions from predicting one class to another. In 2D visualization:

- **Colored Regions**: Each color represents a predicted class
- **Boundaries**: Black contour lines separate regions
- **Sample Points**: Overlaid with different markers per class

### Why They Matter for HRF

1. **Understand Resonance Dynamics**: See how resonance patterns manifest as decision regions
2. **Verify Separability**: Confirm classes are well-separated
3. **Identify Non-linear Boundaries**: HRF can learn complex, non-linear decision surfaces
4. **Detect Overfitting**: Overly complex boundaries suggest overfitting
5. **Compare Models**: Side-by-side visualization shows HRF vs traditional ML

## Core Functionality

### `plot_decision_boundary()`

Main function for single-model decision boundary visualization.

```python
from visualizations import plot_decision_boundary

fig, pca = plot_decision_boundary(
    model=trained_model,
    X=X_train,
    y=y_train,
    resolution=300,
    figsize=(12, 8),
    cmap='viridis',
    pca_if_needed=True
)
```

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | object | Required | Trained HRF model |
| `X` | ndarray | Required | Training data |
| `y` | ndarray | Required | Class labels |
| `resolution` | int | 300 | Mesh grid resolution (higher = finer) |
| `figsize` | tuple | (12, 8) | Figure dimensions |
| `cmap` | str | 'viridis' | Colormap for regions |
| `pca_if_needed` | bool | True | Auto-reduce to 2D via PCA |
| `title` | str | See default | Custom title |

### `plot_decision_boundary_comparison()`

Compare multiple models' decision boundaries side-by-side.

```python
from visualizations import plot_decision_boundary_comparison

models = {
    'HRF': hrf_model,
    'SVM': svm_model,
    'Random Forest': rf_model
}

fig = plot_decision_boundary_comparison(
    models_dict=models,
    X=X_train,
    y=y_train,
    resolution=200,
    figsize=(16, 12)
)
```

## Dimensionality Handling

### 2D Data (Direct Visualization)

For 2D datasets, decision boundaries are plotted directly without transformation.

### High-Dimensional Data (PCA Projection)

For datasets with >2 features, PCA automatically projects to 2D:

1. Fit PCA to training data
2. Project training points to 2D
3. Create mesh in 2D space
4. Inverse-transform mesh to original feature space
5. Get predictions on original mesh
6. Visualize in 2D (PC1 vs PC2)

Axis labels show variance explained: "PC1 (72.3%)" indicates PC1 captures 72.3% of variance.

## Interpretation Guide

### Smooth, Well-Separated Boundaries

**Indicates**:
- Clear separation between classes
- Stable predictions
- Low uncertainty in decision regions
- Good model generalization

### Complex, Intertwined Boundaries

**Indicates**:
- Classes heavily overlap
- Model fits complex structure (good if separable, bad if overfitting)
- Check with test data to confirm generalization

### Linear Boundaries

**Indicates**:
- Simple, linear separation
- Classes are naturally linearly separable
- Likely no overfitting

### Radial/Concentric Boundaries

**Indicates**:
- Non-linear separation
- HRF captures complex patterns
- Handles periodic/resonance-based structures well

## Use Cases

### Model Validation

```python
fig, pca = plot_decision_boundary(model, X_test, y_test)
# Visually confirm decision regions match expected structure
```

### Hyperparameter Tuning

```python
# Compare with different resolutions
for res in [100, 200, 500]:
    fig, pca = plot_decision_boundary(model, X, y, resolution=res)
    # Finer resolution reveals overfitting details
```

### Algorithm Comparison

```python
competitors = {
    'HRF': hrf_model,
    'SVM RBF': svm_rbf,
    'Random Forest': rf_model
}

fig = plot_decision_boundary_comparison(competitors, X, y)
# Which model has most reasonable boundaries?
```

### Error Analysis

```python
# Visualize mispredictions
fig, pca = plot_decision_boundary(model, X_test, y_test)
# Sample points right on decision boundary = likely mispredictions
```

## Resolution Trade-offs

| Resolution | Time | Detail | Use Case |
|-----------|------|--------|----------|
| 50-100 | Fast | Smooth | Quick checks, presentations |
| 200-300 | Medium | Balanced | Default, most uses |
| 500+ | Slow | Fine | Detailed analysis, publication |

Memory scales as O(resolution²).

## Workflow Integration

### Complete Explainability Pipeline

```python
# 1. Understand single prediction (Phase 1)
plot_class_resonance(model, X_test[0:1])

# 2. Understand batch patterns (Phase 2)
plot_resonance_heatmap(model, X_test[:100])

# 3. Understand decision regions (Phase 3)
fig, pca = plot_decision_boundary(model, X_test, y_test)

# 4. Compare with others (Phase 3 advanced)
plot_decision_boundary_comparison(
    {'HRF': hrf_model, 'SVM': svm_model},
    X_test, y_test
)
```

## Examples

Complete examples in `examples/demo_decision_boundary.py`:

1. **demo_2d_decision_boundary()** - Direct 2D visualization
2. **demo_high_dimensional_decision_boundary()** - PCA projection
3. **demo_model_comparison()** - Side-by-side comparison
4. **demo_multiclass_decision_boundary()** - 3+ classes
5. **demo_resolution_comparison()** - Resolution trade-offs

## Performance

Typical timings (mock model, 300x300 resolution):

- Generation: <1 second
- Rendering: <1 second
- Total: <2 seconds

Depends on model complexity (HRF may be slower).

## Limitations

1. **2D Reduction**: High-D boundaries are approximate (PCA projection loss)
2. **Computational Cost**: O(resolution²) for mesh generation
3. **Overfitting Detection**: Can't distinguish overfitting from complex true structure
4. **Non-linear Projections**: PCA is linear; complex structure may be lost
5. **Multimodal Distributions**: Single mesh may miss disconnected regions

## Future Enhancements

1. Interactive Boundaries (Plotly/Bokeh)
2. 3D Visualization
3. Streaming Boundaries during training
4. Uncertainty Regions
5. Feature Importance overlay
6. Contour Labels

## References

- Visualization implementations: `visualizations/decision_boundary.py`
- Phase 1 (Class Resonance): `examples/demo_resonance_visualization.py`
- Phase 2 (Heatmaps): `examples/demo_heatmap.py`
- Phase 3 (Decision Boundaries): `examples/demo_decision_boundary.py`
