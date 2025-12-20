# Harmonic Resonance Fields: A Physics-Informed Classification Framework

**Debanik Debnath** • National Institute of Technology Agartala • December 2025

---

## The Fundamental Question

*What if we designed classifiers around how waves interfere, rather than how lines divide space?*

Traditional machine learning constructs decision boundaries through geometric partitioning—hyperplanes, recursive splits, or learned manifolds. This paradigm fails when signals exhibit temporal phase variations. I propose **Harmonic Resonance Fields (HRF)**: a classifier that models training points as damped oscillators generating class-specific interference patterns.

---

## Mathematical Framework

### I. Wave Potential Function

Each training point $\mathbf{x}_i \in \mathbb{R}^d$ with label $y_i \in \{1,\ldots,C\}$ generates a class-specific wave field:

$$\Psi_c(\mathbf{q}, \mathbf{x}_i) = \exp\left(-\gamma \|\mathbf{q} - \mathbf{x}_i\|^2\right) \cdot \left(1 + \cos(\omega_c \|\mathbf{q} - \mathbf{x}_i\| + \varphi)\right)$$

where:
- $\mathbf{q}$: query point
- $\gamma > 0$: spatial damping (controls locality)
- $\omega_c = \omega_0(c+1)$: class-specific frequency
- $\varphi$: phase offset (temporal invariance)

**Physical Interpretation**: Gaussian envelope ensures exponential decay; cosine modulation encodes oscillatory resonance; constant offset prevents destructive interference.

### II. Resonance Energy Accumulation

Total energy from class $c$ at query $\mathbf{q}$:

$$E_c(\mathbf{q}) = \sum_{i: y_i = c} \Psi_c(\mathbf{q}, \mathbf{x}_i)$$

**Sparse Approximation** (computational efficiency):

$$E_c(\mathbf{q}) = \sum_{i \in \mathcal{N}_k(\mathbf{q}, c)} \Psi_c(\mathbf{q}, \mathbf{x}_i)$$

where $\mathcal{N}_k(\mathbf{q}, c)$ denotes $k$ nearest neighbors of class $c$.

### III. Classification Rule

$$\hat{y}(\mathbf{q}) = \arg\max_{c \in \{1,\ldots,C\}} E_c(\mathbf{q})$$

**Distinction from k-NN**: Neighbors contribute weighted by oscillatory phase, not mere distance; summation over class members, not global $k$-nearest.

---

## Signal Preprocessing

### Bipolar Montage (Common-Mode Rejection)

For EEG signals $\mathbf{x}_{\text{raw}} \in \mathbb{R}^d$:

$$\mathbf{x}_{\text{diff}}[i] = \mathbf{x}_{\text{raw}}[i] - \mathbf{x}_{\text{raw}}[i+1], \quad i = 1,\ldots,d-1$$

$$x_{\text{coh}} = \text{Var}(\mathbf{x}_{\text{raw}})$$

$$\mathbf{x}_{\text{enhanced}} = [\mathbf{x}_{\text{raw}}, \mathbf{x}_{\text{diff}}, x_{\text{coh}}] \in \mathbb{R}^{2d}$$

**Rationale**: Differential signals cancel body artifacts; variance captures global coherence.

### Spectral Transformation (Phase Invariance)

$$\mathbf{X}_{\text{freq}} = |\text{FFT}(\mathbf{x}_{\text{raw}})|$$

**Invariance Property**: If $x(t) = s(t - \tau)$, then $|\mathcal{F}\{x\}| = |\mathcal{F}\{s\}|$ ∀$\tau$.

---

## Auto-Evolution Mechanism

Grid search over physics-informed parameter space:

$(\omega_0^{*}, \gamma^{*}, k^{*}) = \arg\max_{(\omega_0, \gamma, k) \in \mathcal{G}} \mathrm{Acc_{val}}(\omega_0, \gamma, k)$

where $\mathcal{G}$ includes neurophysiologically meaningful frequencies:
- **Delta**: 1 Hz (deep sleep)
- **Theta**: 4 Hz (drowsiness)  
- **Alpha**: 10 Hz (relaxed wakefulness)
- **Beta**: 14 Hz (active thinking)
- **Gamma**: 30 Hz (cognitive processing)

---

## GPU-Accelerated Architecture (v15.0)

**Parallel Resonance Computation**:

$$E_c^{\text{GPU}}(\mathbf{q}) = \text{CuPy.sum}\left[\Psi_c^{\parallel}(\mathbf{q}, \mathbf{X}_c)\right]$$

Leverages NVIDIA RAPIDS (cuML, CuPy) for:
1. GPU-accelerated k-NN search
2. Vectorized wave interference calculation
3. Parallel evolutionary parameter search

**Ensemble Strategy**:

$$\hat{y}_{\text{ensemble}}(\mathbf{q}) = \text{Majority}\{\hat{y}_m(\mathbf{q})\}_{m=1}^M$$

with $M = 100$ estimators, `max_features=1.0` (full holography).

---

## Empirical Validation

**Dataset**: EEG Eye State Corpus (OpenML 1471)  
**Samples**: 14,980 real-world recordings  
**5-Fold Stratified Cross-Validation**:

$$\text{Acc}_{\text{mean}} = 98.12\% \quad (\sigma = \pm 0.18\%)$$

**Test Performance**: 98.53% accuracy

**Benchmarks** (test set):
- Extra Trees: 94.49% (Δ = +4.04%)
- Random Forest: 93.09% (Δ = +5.44%)
- XGBoost: 92.99% (Δ = +5.54%)

**Phase Invariance**: Under 2.0s temporal jitter, HRF maintains 90% accuracy; Random Forest degrades to 60%.

---

## Clinical Metrics

- **ROC-AUC**: 0.9849 (near-perfect class separation)
- **F1-Score**: 0.9836 (balanced precision/recall)
- **Sensitivity**: 98.07% (eyes-closed detection)
- **Specificity**: 98.91% (false alarm rejection)

**Statistical Significance**: Paired t-test vs. all baselines, $p < 0.001$.

---

## Theoretical Foundations

**Why Resonance Outperforms Trees**:

Decision trees partition via axis-aligned cuts: "if voltage at $t = 0.5$s $> 10\mu$V, classify as eyes-closed." Phase jitter shifts signal to $t = 0.7$s, examining wrong features.

**HRF's Fourier Invariance**:

$$\mathcal{F}\{x(t - \tau)\} = e^{-i\omega\tau} \mathcal{F}\{x(t)\}$$

Taking magnitudes eliminates phase factor $e^{-i\omega\tau}$, achieving mathematical time-shift invariance—not learned robustness, but inherent immunity.

---

## Conclusion

By modeling classification as wave interference rather than spatial partitioning, HRF achieves medical-grade accuracy (98.53%) with rigorous statistical validation (±0.18% CV variance). The physics-informed architecture inherently solves temporal jitter through spectral transformation, while GPU acceleration enables comprehensive hyperparameter optimization previously computationally infeasible. This represents a paradigm shift: **when AI listens to the physics of signals, it unlocks understanding inaccessible to purely statistical optimization**.

**Open Source**: Full implementation available at [github.com/Devanik21/Harmonic-Resonance-Forest](https://github.com/Devanik21/Harmonic-Resonance-Forest)

---

**Contact**: devanik2005@gmail.com | [linkedin.com/in/devanik](https://linkedin.com/in/devanik)
