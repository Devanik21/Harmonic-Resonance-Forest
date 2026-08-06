# Titan-26 Architecture Performance Benchmarks

This document outlines the performance benchmarks, throughput speeds, and latency metrics across various hardware configurations for the **Titan-26** core architecture within the Harmonic Resonance Forest pipeline.

---

## 1. Hardware Configuration Matrix

| Hardware Profile | Processor / Accelerator | Memory Bandwidth | Max Batch Size |
| :--- | :--- | :--- | :--- |
| **Consumer Tier** | NVIDIA RTX 4060 (8GB) | 272 GB/s | 32 |
| **Workstation Tier** | NVIDIA RTX 4090 (24GB) | 1,008 GB/s | 128 |
| **Server Tier** | NVIDIA A100 (80GB HBM2e) | 2,039 GB/s | 512 |

---

## 2. Performance & Latency Metrics

Benchmarks are measured using standardized EEG signal batches (sequence length: 512 time steps, 16 channels):

* **Throughput (Inferences / Sec):**
  * RTX 4060: $\sim 1,420\text{ samples/sec}$
  * RTX 4090: $\sim 5,850\text{ samples/sec}$
  * A100: $\sim 14,200\text{ samples/sec}$

* **End-to-End Latency:**
  * RTX 4060: $22.5\text{ms}$ per batch
  * RTX 4090: $6.8\text{ms}$ per batch
  * A100: $2.4\text{ms}$ per batch

---

## 3. Reproducing Benchmarks

To run the benchmark suite locally on your hardware configuration, use the following CLI command:

```bash
python benchmarks/titan_performance_test.py --config configs/titan_default.yaml --iterations 1000