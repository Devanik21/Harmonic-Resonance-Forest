## Ablation Study

## Objective

This study evaluates how individual HRF components contribute to overall model performance and robustness.

## Components Evaluated

- Damping term (γ): Measures influence of resonance locality control.
- Frequency term (ω): Evaluates resonance-frequency sensitivity.
- FFT preprocessing: Measures contribution of frequency-domain transformation.
- Bipolar montage preprocessing: Evaluates signal enhancement effects.
- GPU acceleration: Measures computational efficiency improvements.

## Evaluation Strategy

Each component is independently modified or removed while maintaining identical train/test splits and validation procedures.

## Goal

Quantify the contribution of each HRF component toward classification accuracy, stability, and robustness.