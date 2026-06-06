# Pipeline Overview — Harmonic Resonance Forest (HRF)

This document describes the end-to-end processing pipeline of the Harmonic Resonance Forest (HRF-26D / HRF-v15+), starting from raw input signals and ending in final resonance-based classification.

The system transforms biological signals into a physics-inspired resonance field, evaluates class-wise interference, and produces a final decision through a multi-strategy council ensemble.

# High-Level Architecture Flow

                                ┌────────────────────────────────────────────┐
                                │           RAW EEG SIGNAL INPUT             │
                                │        (Multi-channel brain data)          │
                                └─────────────────────┬──────────────────────┘
                                                      v
                                ┌────────────────────────────────────────────┐
                                │           PREPROCESSING LAYER              │
                                │  - Scaling (Standard/ Robust / MinMax)     │
                                │  - Noise filtering                         │
                                │  - Channel alignment / montage             │
                                │  - Feature normalization                   │
                                └─────────────────────┬──────────────────────┘
                                                      v
                                ┌────────────────────────────────────────────┐
                                │ GEOMETRIC–RESONANCE TRANSFORMATION LAYER   │
                                │  - Distance geometry (Euclidean / KNN)     │
                                │  - Kernel mapping (SVM / RBF)              │
                                │  - Wave-like resonance encoding            │
                                │  - Cosine-modulated similarity fields      │
                                └─────────────────────┬──────────────────────┘
                                                      v
                                ┌────────────────────────────────────────────┐
                                │      MODEL SECTOR EXECUTION LAYER          │
                                │  (Parallel specialized learners)           │
                                │                                            │
                                │  Logic: Trees / Boosting                   │
                                │  Kernel: SVM / RBF / Poly                  │
                                │  Geometry: KNN / QDA                       │
                                │  Neural: ELM / manifold activation         │
                                │  Soul: evolutionary resonance models       │
                                │  Cosmic: fractal / physics forests         │
                                └─────────────────────┬──────────────────────┘
                                                      v
                                ┌────────────────────────────────────────────┐
                                │      STRATEGY ROUTING ENGINE               │
                                │  - ACE / COUNCIL / LINEAR / BALANCE        │
                                │  - Per-dataset optimization                │
                                │  - Dynamic weight allocation               │
                                └─────────────────────┬──────────────────────┘
                                                      v
                                ┌────────────────────────────────────────────┐
                                │        COUNCIL / TITAN ENSEMBLE            │
                                │  - Top-K elite model selection             │
                                │  - Exponential weighted voting             │
                                │  - Titan Chorus convergence                │
                                └─────────────────────┬──────────────────────┘
                                                      v
                                ┌────────────────────────────────────────────┐
                                │   RESIDUAL CORRECTION (DEATH RAY)          │
                                │  - OOF-based correction                    │
                                │  - Geometric refinement                    │
                                │  - Error minimization layer                │
                                └─────────────────────┬──────────────────────┘
                                                      v
                                ┌────────────────────────────────────────────┐
                                │          FINAL PREDICTION OUTPUT           │
                                │   Eye Open / Eye Closed + Probabilities    │
                                └────────────────────────────────────────────┘ 

## 1. Raw EEG Signal Input

The pipeline begins with multi-channel EEG recordings representing electrical activity of the brain over time. These signals are inherently noisy, non-stationary, and highly sensitive to external interference, making direct classification ineffective without transformation.

At this stage, the system interprets EEG not as conventional time-series data but as a high-dimensional dynamic field containing latent frequency and geometric structures.

## 2. Preprocessing Layer

Before any learning occurs, the signal undergoes normalization and stabilization. This includes scaling techniques such as StandardScaler or RobustScaler, which ensure that channel-wise amplitude differences do not distort downstream computations.

Noise reduction and channel alignment are applied to ensure consistency across EEG sensors. In advanced versions of the architecture, bipolar montage conversion and channel weighting mechanisms further refine signal quality by emphasizing biologically meaningful regions.

This stage does not perform explicit feature extraction. Instead, it prepares the signal to exist in a stable mathematical space where geometric relationships can be meaningfully computed.

## 3. Geometric–Resonance Transformation Layer

Once stabilized, the EEG signals are projected into a resonance-aware feature space. Instead of relying solely on Fourier-based decomposition, HRF computes pairwise geometric relationships between data points and interprets them as wave interactions.

Distance metrics such as Euclidean or Minkowski form the basis of this transformation. These distances are then modulated using cosine-based resonance functions, allowing the system to model interaction patterns between data points using resonance-inspired modulation.

The result is a hybrid representation where classical geometry and wave physics coexist, enabling the system to capture nonlinear structures that traditional feature engineering often misses.

## 4. Model Sector Execution

At this stage, multiple specialized learning systems operate in parallel. Each sector interprets the transformed EEG data from a different mathematical perspective.

Tree-based models capture hierarchical decision boundaries, kernel machines map data into higher-dimensional separable spaces, and geometric learners focus on local neighborhood structures. Neural and evolutionary components introduce adaptive nonlinear transformations, while resonance-based “Soul” units model the data as frequency-driven patterns.

Stochastic ensemble models inspired by fractal and probabilistic structures enhance prediction diversity.

This multi-sector design ensures that no single inductive bias dominates the learning process.

## 5. Strategy Routing Engine

Rather than combining all models uniformly, HRF evaluates multiple weighting strategies to determine how information should flow through the system. Strategies such as ACE, COUNCIL, LINEAR, and BALANCE represent different philosophical approaches to ensemble decision-making, ranging from dominance-based selection to fully distributed voting.

Here, the system dynamically selects the optimal strategy based on dataset geometry and validation performance, allowing adaptive behavior across different problem domains.

## 6. Council and Titan Ensemble

Once individual predictions are generated, the Council system aggregates outputs from the most performant models. Instead of simple averaging, HRF applies exponential weighting to assign higher weights to more reliable models while reducing the influence of weaker predictions.

In advanced configurations, this stage evolves into the Titan Chorus, where multiple elite models converge through harmonic weighting to form a unified predictive consensus.

## 7. Residual Correction Layer (Death Ray)

As a final refinement step, residual prediction errors are analyzed and corrected using a specialized geometric optimization module. This layer performs final correction of systematic prediction errors, improving classification boundaries by learning from systematic mispredictions.

It operates only when it can guarantee measurable improvement, ensuring that final outputs are stable and not overfitted.

## 8. Final Prediction Output

The pipeline concludes with a probabilistic classification output, typically distinguishing between eye-open and eye-closed states in EEG datasets. Alongside the predicted class, the system also provides confidence scores derived from ensemble agreement and resonance strength.