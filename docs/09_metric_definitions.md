
# Metric Definitions

This document standardizes evaluation terminology used throughout the Harmonic Resonance Forest repository.

## Core Metrics

| Metric | Definition | Usage Context |
|--------|------------|---------------|
| **Peak Accuracy** | Highest accuracy achieved during training/validation | Reporting best model performance |
| **Final Test Accuracy** | Accuracy on held-out test set after training completes | Generalization assessment |
| **K-Fold Mean Accuracy** | Average accuracy across k-fold cross-validation | Robustness evaluation |
| **Benchmark Accuracy** | Accuracy on standard benchmark datasets | External comparison |
| **Accuracy** | General term — refer to specific metric above | **Do not use alone** |

## Reporting Guidelines

Whenever accuracy values are reported in documentation, the corresponding metric type **MUST** be explicitly stated.

### ✅ Correct Examples

- "HRF v16.0 achieved **98.93% Peak Accuracy**"
- "Cross-validation yielded **96.2% K-Fold Mean Accuracy**"
- "Final Test Accuracy: 94.7% ± 1.2%"

### ❌ Incorrect Examples

- "HRF v16.0 achieved 98.93% accuracy"
- "Model performance: 96.2%"
- "Accuracy = 98.84%" (without specifying metric type)

## Standard Implementation

### Training Report Format

```python
# Example from experiments/train.py
metrics = {
    "peak_accuracy": best_val_acc,       # Highest validation accuracy
    "final_test_accuracy": test_acc,     # Accuracy on test set
    "k_fold_mean": np.mean(k_fold_scores),  # Cross-validation average
    "benchmark_accuracy": bench_acc,     # External benchmark result
}