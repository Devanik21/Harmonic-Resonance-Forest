# Phase II: Jitter Stress Test

## Objective
Evaluate robustness under temporal jitter using FFT transformation.

## Results

| Model | Accuracy |
|------|----------|
| HRF v12.5 | 96.40% |
| SVM (RBF) | 95.20% |
| KNN | 92.80% |
| Random Forest | 76.40% |
| XGBoost | 76.80% |

## Insight
HRF maintains stability under signal noise due to spectral invariance.