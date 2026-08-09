# GPU Acceleration and cuML Integration Guide (v15.0+)

The v15.0+ hybrid CUDA-Python stack in Harmonic-Resonance-Forest introduces hardware acceleration to dramatically speed up heavy computational workflows, specifically focusing on parallel resonance calculations and accelerated spatial indexing.

---

## 1. System Requirements

Before enabling GPU acceleration, ensure your environment meets the following specifications:
* **Hardware:** NVIDIA GPU with Compute Capability 6.0+ (Pascal architecture or newer) and sufficient VRAM.
* **Drivers:** Compatible NVIDIA CUDA Toolkit drivers installed on your host system.
* **Libraries:** 
  * `CUDA Toolkit` (v12.x recommended)
  * `cuML` (NVIDIA RAPIDS suite)
  * `CuPy` (for GPU-accelerated array computing)

### Installation via Conda
The cleanest way to install RAPIDS components (`cuML`, `cupy`) is via Anaconda or Miniconda:
```bash
conda create -n hrf-gpu -c rapidsai -c conda-forge -c nvidia \
    python=3.10 rapids=24.02 cuda-version=12.1 cupy
conda activate hrf-gpu