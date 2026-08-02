# Benchmark Reproduction Guide

This guide provides step-by-step instructions to reproduce the **98.84% EEG classification benchmark** using the OpenML 1471 dataset within the Harmonic Resonance Forest framework.

---

## 1. Prerequisites and Dataset Acquisition

1. Ensure your environment has the required dependencies installed (including PyTorch, NumPy, and scikit-learn).
2. Download or fetch the **OpenML 1471** dataset via the OpenML API or local data loader script provided in the repository:
   ```bash
   python scripts/fetch_openml_data.py --dataset_id 1471