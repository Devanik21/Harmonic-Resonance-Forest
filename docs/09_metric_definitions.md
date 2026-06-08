# Metric Definitions

This document standardizes the evaluation terminology used throughout the Harmonic Resonance Fields (HRF) repository.

## Accuracy

Accuracy represents the proportion of correctly classified samples among all evaluated samples.

**Formula:**

Accuracy = Correct Predictions / Total Predictions

---

## Peak Accuracy

Peak Accuracy refers to the highest accuracy achieved during experimentation or hyperparameter exploration.

**Note:** Peak Accuracy represents the best observed result and should not be considered the primary reproducibility metric.

---

## Final Test Accuracy

Final Test Accuracy is the accuracy obtained on the held-out test dataset after model training is complete.

This metric estimates real-world performance on unseen data.

---

## K-Fold Mean Accuracy

K-Fold Mean Accuracy is the average accuracy obtained across all folds during K-Fold Cross Validation.

This is the preferred metric for reproducibility because it reduces dependence on a single train-test split.

---

## Benchmark Accuracy

Benchmark Accuracy refers to performance reported when comparing HRF against baseline models under identical experimental conditions.

Examples include:

* Random Forest
* Extra Trees
* XGBoost
* SVM
* KNN

---

## Recommended Reporting Standard

For consistent scientific reporting, results should be presented in the following order:

1. K-Fold Mean Accuracy (Primary Reproducibility Metric)
2. Final Test Accuracy
3. Peak Accuracy (Supplementary Metric)

Whenever accuracy values are reported, the corresponding metric type should be explicitly stated.
