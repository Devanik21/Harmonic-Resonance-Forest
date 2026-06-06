# Phase III: Survival Curve (Accuracy vs Chaos)

## Objective
Test degradation under increasing temporal jitter.

## Key Results

| Jitter | HRF | RF | SVM | KNN | XGBoost |
|------|------|----|-----|-----|--------|
| 0.0 | 94.67% | 94.67% | 99.33% | 98.00% | 94.00% |
| 1.0 | 96.67% | 61.33% | 84.67% | 95.33% | 60.00% |
| 2.0 | 90.00% | 60.00% | 81.33% | 78.00% | 61.33% |

## Conclusion
HRF demonstrates phase-invariant stability.