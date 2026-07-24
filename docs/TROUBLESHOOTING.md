# Troubleshooting & Error-Handling Guide

This guide helps users and contributors identify, debug, and resolve common exceptions encountered within the Harmonic Resonance Forest pipeline.

---

## 1. Tensor Shape Mismatch During Ingestion

### **Symptom**
* **Error Message:** `AssertionError: Input must have 3 dimensions: (batch, channels, time_steps)`
* **Cause:** The data loader or preprocessing function received a tensor or array with missing or extra dimensions (e.g., a 2D matrix or a 4D tensor).

### **Resolution**
Ensure your input data is correctly shaped before passing it into the model pipeline. You can use PyTorch's `unsqueeze` or `reshape` methods to fix dimensions:

```python
import torch

# Example: If your tensor is currently 2D (channels, time_steps), add a batch dimension
if tensor_data.ndim == 2:
    tensor_data = tensor_data.unsqueeze(0)