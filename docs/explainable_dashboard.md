# Explainable Resonance Dashboard

## Purpose

The Explainable Resonance Visualization system provides interpretability and explainability for Harmonic Resonance Forest (HRF) predictions through class-wise resonance contribution visualization.

### Why Explainability Matters

Machine learning models often act as "black boxes," making it difficult for practitioners to understand *why* predictions are made. The HRF model, while highly accurate, benefits from explainability mechanisms that:

- **Build Trust**: Users can verify that predictions align with domain expertise
- **Enable Debugging**: Identify when a model relies on unexpected features
- **Support Compliance**: Provide audit trails for regulated domains (healthcare, finance)
- **Guide Feature Engineering**: Understand which patterns drive predictions

## Core Concept: Class-wise Resonance Contributions

The HRF model computes predictions by aggregating "resonance energies" across 16 specialized units (Logic, Gradient, Kernel, Geometry, and Soul units). The final prediction for each class is the normalized sum of resonance contributions from all units.

### How It Works

1. **Prediction Probabilities**: The `predict_proba()` method returns normalized class probabilities
2. **Resonance Extraction**: Each probability represents the total "resonance energy" for that class
3. **Visualization**: A bar chart shows the magnitude of each class's resonance, with the predicted class highlighted

### Interpretation Guide

- **Bar Height**: Energy contribution for each class (higher = stronger signal)
- **Red Highlighted Bar**: The predicted class (with ★ marker)
- **Relative Heights**: Show the model's "confidence" in each prediction
- **Values**: Exact probabilities displayed above each bar

#### Example Scenario

```
Sample Classification Task: 3-class problem
- Class 0: 0.650 (predicted, ★)
- Class 1: 0.230
- Class 2: 0.120
```

**Interpretation**:
- The model strongly resonates with Class 0 (65% energy)
- Moderate resonance with Class 1 (23%)
- Weak resonance with Class 2 (12%)
- Prediction is relatively confident (≥50% probability)

## Usage

### Basic Usage

```python
from visualizations.resonance_plots import plot_class_resonance
import matplotlib.pyplot as plt

# Assume: model is a trained HRF instance
# Assume: X_test is test data (shape: n_samples × n_features)

# Visualize resonance for a single sample
fig = plot_class_resonance(model, X_test[0:1])
plt.show()
```

### Advanced Usage: Batch Visualization

```python
import matplotlib.pyplot as plt
from visualizations.resonance_plots import plot_class_resonance

# Create subplots for multiple samples
n_samples = 4
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i in range(n_samples):
    fig = plot_class_resonance(model, X_test[i:i+1], figsize=(6, 4))
    # Note: Each call creates a new figure (see advanced customization)

plt.tight_layout()
plt.show()
```

### Integration with Model Training

```python
from HRF-Engine.generalized_hrf_v2 import HarmonicResonanceForest_Ultimate
from visualizations.resonance_plots import plot_class_resonance

# Train model
model = HarmonicResonanceForest_Ultimate()
model.fit(X_train, y_train)

# After training, visualize predictions
fig = plot_class_resonance(model, X_test[0:1])
plt.show()
```

## API Reference

### `plot_class_resonance(model, sample, figsize=(10, 6), color_palette='viridis')`

**Parameters:**
- `model`: Trained HRF model with `predict_proba()` and `classes_` attributes
- `sample`: Single sample or batch (shape: 1 × n_features or n_features,)
- `figsize`: Figure size tuple (width, height) in inches
- `color_palette`: Matplotlib colormap name for bar colors

**Returns:**
- `fig`: matplotlib.figure.Figure object

**Raises:**
- `AttributeError`: If model lacks required methods/attributes
- `ValueError`: If sample is empty or malformed

**Example:**
```python
fig = plot_class_resonance(model, X_test[0:1], figsize=(12, 7), color_palette='plasma')
```

## Visualization Features

### 1. **Color Coding**
- Predicted class: Highlighted in red (high contrast)
- Other classes: Gradient colors (viridis, plasma, cool, etc.)
- Customizable via `color_palette` parameter

### 2. **Value Labels**
- Exact probabilities displayed above each bar
- Predicted class marked with ★ (star) symbol
- Precision: 3 decimal places (0.XXX)

### 3. **Legend Annotation**
- Text box in top-right corner
- Shows predicted class and confidence
- Formatted as: "Predicted: Class X (★)"

### 4. **Grid & Formatting**
- Y-axis grid for easy value reading
- Bold titles and axis labels
- Black bar edges for clarity
- Professional matplotlib styling

## Use Cases

### 1. Model Validation
Verify that predictions align with domain expectations:
```python
# Check if lung cancer prediction is reasonable
fig = plot_class_resonance(model, patient_sample)
# Should show high resonance for positive class if features indicate risk
```

### 2. Error Analysis
Investigate misclassifications by visualizing resonance for incorrect predictions:
```python
# Find false positives
incorrect_idx = np.where(y_pred != y_true)[0]
for idx in incorrect_idx[:3]:
    fig = plot_class_resonance(model, X_test[idx:idx+1])
    # Analyze why model was confused
```

### 3. Feature Importance via Perturbation
Combine with feature perturbation to understand which features drive predictions:
```python
# Baseline
fig1 = plot_class_resonance(model, sample)

# Perturb important feature
sample_perturbed = sample.copy()
sample_perturbed[:, important_feature_idx] = 0

# Recompute
fig2 = plot_class_resonance(model, sample_perturbed)

# Compare: if resonance shifts significantly, feature is important
```

### 4. Confidence Assessment
Use resonance distributions for decision-making:
```python
# High confidence (one class dominates)
fig = plot_class_resonance(model, X_test[0:1])  # One tall bar

# Low confidence (balanced probabilities)
fig = plot_class_resonance(model, X_test[1:2])  # Multiple medium bars
# Apply additional scrutiny or request human review
```

## Limitations & Considerations

### 1. Single-Sample View
- Visualizations are per-sample; aggregate statistics require additional analysis
- For dataset-level insights, use confusion matrices or ROC curves

### 2. Not True "Attribution"
- Bar heights show *probabilities*, not feature contributions
- To explain which features drive predictions, use SHAP/LIME in addition

### 3. Calibration Dependency
- Bar heights depend on model calibration
- Miscalibrated models may show overconfident resonances

### 4. Computational Efficiency
- `predict_proba()` calls may be slow on large batches
- Consider caching predictions for real-time dashboards

## Integration with Dashboards

### Example: Flask Web App
```python
from flask import Flask, render_template
from visualizations.resonance_plots import plot_class_resonance
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get sample from request
    sample = request.json['sample']
    
    # Generate prediction
    fig = plot_class_resonance(model, sample)
    
    # Convert to image
    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    
    return {'image': img_base64}
```

## Comparison with Other Explainability Methods

| Method | Pros | Cons |
|--------|------|------|
| **Resonance Visualization** | Model-agnostic, fast, intuitive | Doesn't show feature importance |
| **SHAP** | Feature-level attribution, theoretically sound | Computationally expensive |
| **LIME** | Local interpretability, flexible | Approximation may be inaccurate |
| **Attention Maps** | Great for NNs, visual | Not applicable to tree-based models |

**Recommendation**: Use resonance visualization as a quick check, combined with SHAP for feature-level insights.

## Future Enhancements

1. **Unit-Level Breakdown**: Show contributions from individual HRF units (Logic, Soul, etc.)
2. **Interactive Dashboard**: Web-based exploration with sample selection
3. **Confidence Intervals**: Bayesian estimates of probability uncertainty
4. **Comparison Mode**: Side-by-side visualization for similar samples
5. **Temporal Analysis**: Track how resonance evolves during model evolution

## References

- HRF Architecture: See `docs/architecture.md`
- Prediction Pipeline: See `docs/pipeline_overview.md`
- Soul Unit Details: `HRF-Engine/generalized_hrf_v2.py` (lines 46-196)

## Examples

See `examples/demo_resonance_visualization.py` for complete working examples.

## Questions?

For implementation details or debugging, refer to:
- `visualizations/resonance_plots.py` source code
- Model documentation in `HRF-Engine/generalized_hrf_v2.py`
- Test cases in `tests/` directory

---

# Phase 2: Resonance Heatmap Visualization

## Overview

Phase 2 extends the explainability system with **Resonance Intensity Heatmaps**, enabling batch-level analysis of resonance patterns across multiple samples and classes simultaneously.

### What It Shows

A heatmap visualizes:
- **Rows**: Individual test samples (indexed 0, 1, 2, ...)
- **Columns**: Predicted classes (labeled with class names)
- **Color Intensity**: Resonance energy for each sample-class pair
- **Overall Pattern**: Distribution of resonance strengths across the dataset

### Why It's Useful

1. **Batch Analysis**: See prediction patterns across 50-100+ samples at once
2. **Confidence Assessment**: Identify uncertain predictions (balanced colors) vs confident ones (concentrated intensity)
3. **Class-Specific Trends**: Notice if certain classes consistently activate across samples
4. **Error Analysis**: Compare heatmaps with ground truth to understand misclassifications
5. **Model Behavior**: Detect anomalies or unexpected resonance patterns

## Core Functionality

### `plot_resonance_heatmap()`

Main function for visualizing resonance patterns.

```python
from visualizations import plot_resonance_heatmap

fig = plot_resonance_heatmap(
    model=trained_model,
    X=X_test[:100],
    resonance_map=None,  # Auto-extracted if None
    figsize=(14, 6),
    cmap='YlOrRd',
    normalize_rows=True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | object | Required | Trained HRF model |
| `X` | ndarray | Required | Input samples (n_samples × n_features) |
| `resonance_map` | ndarray | None | Pre-computed resonance matrix (n_samples × n_classes) |
| `figsize` | tuple | (14, 6) | Figure dimensions in inches |
| `cmap` | str | 'YlOrRd' | Matplotlib colormap name |
| `normalize_rows` | bool | True | Row-wise normalize (relative) or raw (absolute) |

**Returns:**
- `fig` (matplotlib.figure.Figure): The visualization figure

**Available Colormaps:**
- `'YlOrRd'`: Yellow → Orange → Red (default, intuitive)
- `'Blues'`: Light → Dark Blue (cool, professional)
- `'Greens'`: Light → Dark Green (calming)
- `'Purples'`: Light → Dark Purple (elegant)
- `'viridis'`: Blue → Green → Yellow (perceptually uniform)
- `'plasma'`: Purple → Orange → Yellow (high contrast)

### `plot_resonance_comparison()`

Advanced visualization combining heatmap with ground truth labels.

```python
from visualizations import plot_resonance_comparison

fig = plot_resonance_comparison(
    model=trained_model,
    X=X_test,
    y=y_test,
    sample_indices=[0, 1, 5, 10, 20],  # Optional: specific samples
    figsize=(16, 10)
)
```

Shows:
- **Top**: Full resonance heatmap
- **Bottom**: Bar showing predicted vs true labels (✓ for correct, ✗ for incorrect)

### Resonance Capture Utilities

When the model doesn't natively expose resonance data, use the capture wrapper:

```python
from visualizations import ResonanceCaptureWrapper, enable_resonance_capture

# Method 1: Manual wrapping
wrapped_model = ResonanceCaptureWrapper(trained_model)
_ = wrapped_model.predict_proba(X_test)
resonance_map = wrapped_model.last_resonance_map_

# Method 2: Helper function
model = enable_resonance_capture(model)
_ = model.predict_proba(X_test)
resonance_map = model.last_resonance_map_
```

**Note**: The wrapper is non-breaking; predictions and accuracy remain unchanged.

## Interpretation Guide

### High Resonance (Bright Colors)

```
Sample 5, Class 0: ████████████ (0.85)
Sample 5, Class 1: ██ (0.10)
Sample 5, Class 2: ████ (0.05)
```

**Interpretation:**
- Model is very confident about Class 0
- Clear decision boundary
- Low risk of misclassification

### Low/Balanced Resonance (Pale Colors)

```
Sample 10, Class 0: ███ (0.33)
Sample 10, Class 1: ███ (0.33)
Sample 10, Class 2: ███ (0.34)
```

**Interpretation:**
- Model is uncertain; all classes equally plausible
- Sample may be ambiguous or on decision boundary
- Higher risk of misclassification
- Candidate for human review or additional features

### Multimodal Resonance (Multiple Peaks)

```
Sample 15, Class 0: ███████ (0.60)
Sample 15, Class 1: ██ (0.25)
Sample 15, Class 2: ██ (0.15)
```

**Interpretation:**
- Primary resonance with Class 0
- Secondary resonance with Class 1 (possible ambiguity)
- Model hedges between two classes
- Check if sample shares features with both classes

## Use Cases

### 1. Model Validation

Verify model behavior on new data:

```python
# Validate on held-out test set
fig = plot_resonance_heatmap(model, X_test[:100])
# Look for: concentrated color intensity = confident model
# Look for: scattered colors = uncertain or inconsistent model
```

### 2. Error Analysis

Understand misclassifications:

```python
# Compare resonance for correct vs incorrect predictions
predictions = model.predict(X_test)
incorrect_mask = predictions != y_test
incorrect_idx = np.where(incorrect_mask)[0]

fig = plot_resonance_comparison(model, X_test[incorrect_idx], y_test[incorrect_idx])
# Misclassifications often show balanced resonance (low confidence)
```

### 3. Class Imbalance Detection

Identify if model favors certain classes:

```python
fig = plot_resonance_heatmap(model, X_test, normalize_rows=False)
# If entire Class 2 column is bright red, model may be over-confident
# If entire Class 3 column is pale, model may under-activate that class
```

### 4. Confidence Thresholding

Select samples for further investigation:

```python
from visualizations import extract_resonance_samples

wrapped_model = enable_resonance_capture(model)
resonance_map = extract_resonance_samples(wrapped_model, X_test)

# Get max resonance per sample (confidence)
max_resonance = np.max(resonance_map, axis=1)

# Select uncertain samples
uncertain_mask = max_resonance < 0.6
uncertain_indices = np.where(uncertain_mask)[0]

fig = plot_resonance_comparison(model, X_test[uncertain_indices], y_test[uncertain_indices])
# All samples here have max resonance < 0.6 (low confidence)
```

### 5. Dataset Quality Assessment

Assess overall dataset characteristics:

```python
from visualizations import extract_resonance_samples

resonance_map = extract_resonance_samples(wrapped_model, X_test)
resonance_normalized = resonance_map / np.sum(resonance_map, axis=1, keepdims=True)

# Statistics
mean_confidence = np.mean(np.max(resonance_normalized, axis=1))
entropy_per_sample = -np.sum(resonance_normalized * np.log(resonance_normalized + 1e-10), axis=1)

print(f"Mean Confidence: {mean_confidence:.3f}")
print(f"Mean Entropy: {np.mean(entropy_per_sample):.3f}")
# High entropy = ambiguous dataset; low entropy = well-separated classes
```

## Normalization Modes

### Row-wise Normalized (Default)

```python
fig = plot_resonance_heatmap(model, X_test, normalize_rows=True)
```

- Resonance values sum to 1.0 per row (sample)
- Shows **relative** contributions within each sample
- All rows have comparable color ranges
- Useful for: confidence assessment, relative class importance
- Heatmap is more uniform in color intensity

### Raw Resonance (Unnormalized)

```python
fig = plot_resonance_heatmap(model, X_test, normalize_rows=False)
```

- Resonance values are absolute (not normalized)
- Shows **actual** energy magnitudes
- High-energy samples appear brighter
- Useful for: identifying unusually confident/uncertain samples
- Heatmap shows natural variation in model confidence

## Advanced Features

### Multi-Colormap Exploration

```python
import matplotlib.pyplot as plt

colormaps = ['YlOrRd', 'Blues', 'Greens', 'Purples', 'plasma']
fig, axes = plt.subplots(1, len(colormaps), figsize=(20, 4))

for ax, cmap in zip(axes, colormaps):
    # Create heatmap with each colormap
    fig = plot_resonance_heatmap(model, X_test[:50], cmap=cmap)

plt.show()
```

### Statistical Summary

```python
from visualizations import extract_resonance_samples

resonance_map = extract_resonance_samples(wrapped_model, X_test)
resonance_norm = resonance_map / np.sum(resonance_map, axis=1, keepdims=True)

# Per-class statistics
for i, cls in enumerate(model.classes_):
    print(f"Class {cls}:")
    print(f"  Mean resonance: {np.mean(resonance_norm[:, i]):.4f}")
    print(f"  Std dev: {np.std(resonance_norm[:, i]):.4f}")
    print(f"  Max: {np.max(resonance_norm[:, i]):.4f}")

# Confidence distribution
max_resonance = np.max(resonance_norm, axis=1)
print(f"\nOverall Confidence:")
print(f"  Mean: {np.mean(max_resonance):.4f}")
print(f"  Median: {np.median(max_resonance):.4f}")
print(f"  Min: {np.min(max_resonance):.4f}")
print(f"  Max: {np.max(max_resonance):.4f}")
```

## Limitations

1. **Scalability**: Heatmaps with >1000 rows become difficult to interpret (rows blur together)
   - **Solution**: Visualize stratified subsets or use aggregated statistics

2. **Color Perception**: Colorblind users may have difficulty distinguishing heatmap colors
   - **Solution**: Provide numeric annotations or use colorblind-friendly palettes (e.g., 'viridis')

3. **Static Visualization**: Cannot interact with individual samples
   - **Solution**: Use Jupyter widgets for interactive exploration

4. **Missing Ground Truth**: Without true labels, cannot assess correctness directly
   - **Solution**: Use `plot_resonance_comparison()` to compare with available labels

## Performance Considerations

- **Memory**: Heatmaps for >10,000 samples may require significant RAM
- **Rendering**: Large heatmaps (>1000×100) may render slowly
- **Batch Processing**: Use `extract_resonance_samples()` with `batch_size` parameter:

```python
resonance_map = extract_resonance_samples(wrapped_model, X_large, batch_size=512)
```

## Integration with Phase 1

Phase 1 (Class-wise Resonance Bars) and Phase 2 (Heatmaps) are complementary:

- **Phase 1**: Detailed view of single-sample predictions
- **Phase 2**: Aggregate view of batch patterns

Combined workflow:

```python
from visualizations import plot_class_resonance, plot_resonance_heatmap

# 1. Identify interesting sample via heatmap
fig1 = plot_resonance_heatmap(model, X_test[:100])
plt.show()

# 2. Drill down on specific sample with bar chart
sample_index = 42  # Identified from heatmap
fig2 = plot_class_resonance(model, X_test[sample_index:sample_index+1])
plt.show()
```

## Examples

See `examples/demo_heatmap.py` for comprehensive demos covering:
- Basic heatmap generation
- Colormap variants
- Comparison with ground truth
- Statistical analysis
- Error diagnosis

## API Summary

```python
from visualizations import (
    plot_resonance_heatmap,           # Main heatmap function
    plot_resonance_comparison,        # Heatmap + ground truth
    ResonanceCaptureWrapper,          # Wrapper for resonance capture
    enable_resonance_capture,         # Helper to enable capture
    extract_resonance_samples         # Extract resonance for batch
)

# Typical workflow
model = enable_resonance_capture(model)
resonance_map = extract_resonance_samples(model, X_test)
fig1 = plot_resonance_heatmap(model, X_test, resonance_map=resonance_map)
fig2 = plot_resonance_comparison(model, X_test, y_test)
```

## Future Enhancements

1. **Interactive Heatmaps**: Plotly/Bokeh for hover details
2. **Unit-Level Heatmaps**: Show contributions from individual HRF units
3. **Temporal Heatmaps**: Track resonance evolution during model training
4. **Uncertainty Estimation**: Confidence intervals on resonance values
5. **Anomaly Highlighting**: Automatically flag unusual resonance patterns

---

## References

- **Phase 1 Documentation**: See sections on class-wise resonance contributions
- **HRF Architecture**: `docs/architecture.md`
- **Prediction Pipeline**: `docs/pipeline_overview.md`
- **Implementation Details**: `visualizations/resonance_plots.py`, `visualizations/resonance_utils.py`

## See Also

- `examples/demo_resonance_visualization.py` — Phase 1 demos
- `examples/demo_heatmap.py` — Phase 2 demos
- `visualizations/__init__.py` — API reference
