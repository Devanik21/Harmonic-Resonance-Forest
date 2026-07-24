# Dataset Ingestion and Preprocessing Guide

This document outlines how raw datasets are ingested, parsed, and prepared before entering the Bipolar Montage and FFT processing layers in the Harmonic Resonance Forest pipeline.

## 1. Expected Data Schema & Shapes
Before ingestion, raw input data must adhere to strict tensor dimensions to ensure seamless processing through the resonance engines.

* **Input Type:** `torch.Tensor` or NumPy arrays converted to tensors.
* **Expected Shape:** `(batch_size, channels, time_steps)`
  * `batch_size`: Number of independent samples or trials.
  * `channels`: Number of recording electrodes or feature dimensions.
  * `time_steps`: Temporal length of each individual signal window.
* **Data Type:** `torch.float32`

## 2. Ingestion Pipeline Workflow