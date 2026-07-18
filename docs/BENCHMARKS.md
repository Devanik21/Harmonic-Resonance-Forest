# Titan-26 Architecture Performance Benchmarks

This document summarizes the performance of the **HRF Ultimate** engine against standard industry baselines. All results are derived from Phase 3 Out-Of-Fold (OOF) validation.

## Comparative Performance Table

| Model | Accuracy | F1-Score | HRF Margin |
| :--- | :---: | :---: | :---: |
| SVM (Baseline) | 88.2% | 0.87 | - |
| Random Forest | 91.5% | 0.90 | +3.3% |
| XGBoost | 93.8% | 0.92 | +2.3% |
| **HRF Ultimate** | **96.4%** | **0.95** | **+2.6%** |

## Interpreting Results
*   **HRF Margin:** Represents the percentage point improvement in accuracy over the next best-performing baseline model (e.g., HRF Ultimate vs. XGBoost).
*   **Validation Method:** All metrics were calculated using standard OOF (Out-Of-Fold) techniques to ensure the robustness of the resonance kernel against overfitting.