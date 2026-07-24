# API Reference Guide

This document outlines the core classes, methods, and parameters within the Harmonic Resonance Forest engine to assist with developer integration.

---

## 1. `GeneralizedHRF` Engine (`src/engine/generalized_hrf_v2.py`)

The primary processing engine responsible for handling resonance kernels and target signals.

### **Methods**

#### `_update_resonance_kernels(self, y: torch.Tensor) -> None`
Updates internal resonance kernels in-place based on the target signal values.

* **Parameters:**
  * `y` (`torch.Tensor`): The target labels or signal values of shape `(batch_size,)`. Values should be normalized within the range `[0, 1]`.
* **Returns:**
  * `None`: Updates internal model state in-place.
* **Numerical Stability Note:** Includes epsilon stabilization ($\epsilon = 1e-9$) during normalization to prevent `DivisionByZero` and `NaN` propagation when processing zero-amplitude inputs.

---

## 2. Preprocessing & Signal Pipeline

### **Bipolar Montage**
* **Role:** Differential signal extraction across adjacent electrode channels to eliminate common-mode noise.
* **Expected Input Shape:** `(batch_size, channels, time_steps)` of type `torch.float32`.

### **Fast Fourier Transform (FFT)**
* **Role:** Maps temporal windows into spectral resonance frequency bins.