# GPU Acceleration and cuML Integration Guide

This guide explains how to configure and utilize the v15.0+ hybrid CUDA-Python stack to achieve massively parallel resonance calculations and accelerated K-Nearest Neighbors (KNN) locality searches within the Harmonic Resonance Forest framework.

---

## 1. System Requirements

To leverage hardware acceleration, ensure your environment meets the following specifications:
* **Hardware:** CUDA-enabled NVIDIA GPU (Compute Capability 7.0+ recommended, e.g., RTX series, A100, H100).
* **Drivers & Toolkit:** Compatible NVIDIA CUDA Toolkit drivers installed (CUDA 12.x recommended).
* **Libraries:** NVIDIA RAPIDS ecosystem packages including `cuML` and `CuPy`.

---

## 2. Setting Up RAPIDS cuML and CuPy

You can install the required GPU dependencies via Conda:
```bash
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.08 cupy-cuda12x python=3.10