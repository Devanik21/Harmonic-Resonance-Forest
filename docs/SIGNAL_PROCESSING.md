# Signal-Processing Architecture Overview

This document outlines the core signal-processing pipeline within the Harmonic Resonance Forest engine, focusing on noise reduction and spectral feature extraction.

## 1. Bipolar Montage Preprocessing
The initial preprocessing stage applies a **Bipolar Montage** to raw input signals. 

* **Purpose:** To eliminate common-mode noise and artifact interference (such as power line noise or systemic electrode drift) shared across adjacent channels.
* **Mechanism:** Differential signal extraction is performed by calculating the potential differences between adjacent electrode pairs rather than referencing them to a single global ground.

## 2. Fast Fourier Transform (FFT) & Spectral Mapping
Once spatial noise is minimized through differential extraction, the temporal signals are mapped into the frequency domain.

* **Frequency Transformation:** An optimized Fast Fourier Transform (FFT) converts raw time-series data into spectral representations.
* **Resonance Inputs:** The resulting frequency bins serve as the spectral resonance inputs for the core model layers, allowing the engine to capture rhythmic patterns and harmonic structures.

## Summary Pipeline