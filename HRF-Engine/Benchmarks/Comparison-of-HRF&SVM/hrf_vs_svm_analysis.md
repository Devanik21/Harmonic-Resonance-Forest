# HRF vs SVM Benchmark Analysis

## Overview

This analysis compares the performance of Harmonic Resonance Forest (HRF Ultimate GPU) and Support Vector Machine (SVM with RBF kernel) across 20 benchmark classification datasets. The objective is to evaluate the effectiveness of HRF relative to a widely used machine learning baseline and identify scenarios where HRF provides significant advantages.

---

## Evaluation Methodology

The benchmark results were collected from the official HRF benchmark suite. For each dataset, classification accuracy was recorded for the following models:

* Harmonic Resonance Forest (HRF Ultimate GPU)
* Support Vector Machine (SVM with RBF Kernel)
* Random Forest
* XGBoost

This study focuses specifically on the comparison between HRF and SVM.

---

## Results Summary

| Metric         | Value |
| -------------- | ----- |
| Total Datasets | 20    |
| HRF Wins       | 17    |
| SVM Wins       | 2     |
| Ties           | 1     |

Overall, HRF outperformed SVM on the majority of benchmark datasets, demonstrating strong generalization across diverse classification tasks.

---

## Largest Performance Improvements

### Madelon

* HRF Accuracy: 85.00%
* SVM Accuracy: 59.80%
* Improvement: +25.20%

Madelon is a high-dimensional artificial dataset containing many irrelevant features. HRF demonstrated substantially better performance, suggesting improved robustness in challenging feature spaces.

### Wall-Following Robot

* HRF Accuracy: 99.66%
* SVM Accuracy: 88.50%
* Improvement: +11.16%

HRF achieved near-perfect classification performance while maintaining a significant advantage over SVM.

### Phoneme

* HRF Accuracy: 92.00%
* SVM Accuracy: 81.66%
* Improvement: +10.34%

The benchmark indicates that HRF effectively captures complex decision boundaries in speech-related classification tasks.

---

## EEG Eye State Analysis

* HRF Accuracy: 92.66%
* SVM Accuracy: 85.33%
* Improvement: +7.33%

The EEG Eye State dataset contains noisy physiological signals. HRF demonstrated stronger predictive performance, indicating its ability to model complex patterns within sensor-generated data.

---

## Datasets Where SVM Remained Competitive

### Optdigits

* HRF Accuracy: 98.16%
* SVM Accuracy: 99.00%

SVM slightly outperformed HRF, suggesting that SVM remains highly effective for certain digit recognition tasks.

### HTRU2 Pulsar Detection

* HRF Accuracy: 76.68%
* SVM Accuracy: 77.72%

The performance difference was minimal, indicating comparable effectiveness between the two approaches.

### Solar Flare Evolution

* HRF Accuracy: 77.77%
* SVM Accuracy: 77.77%

Both models achieved identical accuracy.

---

## Key Observations

1. HRF consistently achieved higher accuracy across most benchmark datasets.
2. The largest gains were observed on high-dimensional and complex classification problems.
3. SVM remained competitive on a small subset of datasets, particularly those involving pattern recognition tasks.
4. HRF demonstrated strong performance across sensor, speech, robotics, astronomy, and industrial datasets.
5. The benchmark results suggest that HRF is a reliable alternative to traditional SVM-based classification methods.

---

## Conclusion

The benchmark analysis demonstrates that Harmonic Resonance Forest generally outperforms SVM (RBF) across a diverse collection of classification datasets. HRF achieved superior results on 17 of 20 datasets and delivered particularly strong improvements on challenging high-dimensional problems such as Madelon. While SVM remains competitive on selected datasets, the overall results indicate that HRF provides stronger classification performance and broader applicability across benchmark domains.
