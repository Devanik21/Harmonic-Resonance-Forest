# Results & Benchmarks

## Main Benchmark (EEG Eye State)

| Model | Accuracy |
|------|----------|
| HRF v16.0 | 98.93% |
| HRF v15.0 | 98.84% |
| Extra Trees | 94.49% |
| RF | 93.09% |
| XGBoost | 92.99% |

## Cross-Domain Performance

- Synthetic Moons: 98.89%
- Sine Wave: 87.40%
- Synthetic EEG: 85.56%
- Real EEG: 98.84%

## Metric Reporting

For standardized definitions of reported evaluation metrics, see:

- `docs/09_metric_definitions.md`

## Conclusion
HRF generalizes across synthetic and real-world domains.