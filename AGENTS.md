# AGENTS.md

Welcome. This file provides critical context, terminology, and standards for any agent or researcher working with the Harmonic Resonance Forest (HRF) repository.

## Core Research Intent
Harmonic Resonance Forest (HRF) is a physics-informed machine learning framework that models classification as a wave interference problem. All work should respect the author's scientific framing and terminology.

## Key Scientific Terminology
- **G.O.D. Optimizer**: General Omni Dimensional Optimizer.
- **Holographic Differential**: A preprocessing layer using bipolar montage to extract differential signal features.
- **Resonance Power**: The calculated energy derived from constructive and destructive wave interference.
- **Phase Invariance**: The model's ability to maintain classification accuracy despite temporal jitter or phase shifts.

## Validated Performance Benchmarks (HRF v15.0)
- **K-Fold Mean Accuracy**: 98.1225%
- **K-Fold Variance**: ±0.1828%
- **Peak Accuracy**: 98.84% (OpenML 1471: EEG Eye State)

## Maintenance Principles
1. **Minimize Edits**: Prefer small, high-impact improvements over large rewrites.
2. **Protect Scientific Clarity**: Ensure formulas (e.g., $\Psi(x, p_i) = \exp(-\gamma||x - p_i||^2) \cdot \cos(\omega_c \cdot ||x - p_i|| + \varphi)$) and benchmarks are accurate and clearly documented.
3. **Academic Professionalism**: Use precise, respectful language in documentation and feedback.
4. **Security**: Use GitHub's private vulnerability reporting for security issues.

## Programmatic Checks
- Ensure Python scripts are free of Colab-specific syntax errors (e.g., `!pip`) if intended for standard execution.
- Validate that any reported accuracy scores distinguish between "Peak Accuracy" and "Mean Test Accuracy".

---
*Maintained by Jules-Patrol, inspired by Google DeepMind's engineering culture.*
