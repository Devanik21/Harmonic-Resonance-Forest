# HRF Architecture: The Titan-26 Unified Manifold

This document provides a technical overview of the **Harmonic Resonance Fields (HRF)** system architecture, focusing on the **Titan-26** unified manifold.

## 1. Architectural Philosophy

HRF moves beyond traditional statistical machine learning by treating classification as a physical wave interference problem. The architecture is designed to be **physics-informed**, meaning it uses physical laws (like damping and resonance) to model data relationships.

The core of the system is the **Titan-26 Manifold**, a 26-dimensional integration of diverse physical and statistical topologies.

## 2. The Titan-26 Manifold

The "26 dimensions" refer to 26 distinct algorithmic units or "fields" that are orchestrated by the **G.O.D. (General Omni Dimensional) Optimizer**. These units are categorized into five major sectors:

### Sector A: Classical & Boosting Topologies
These units handle the "particle-like" aspects of data—discrete decision boundaries and statistical splitting.
1. **ExtraTrees Ensemble**: High-variance randomized trees for robust baseline.
2. **Random Forest**: Standard bagging for variance reduction.
3. **Histogram Gradient Boosting**: Fast, bin-based gradient boosting.
4. **XGBoost Deep**: Deep gradient boosted trees for complex patterns.
5. **XGBoost Fast**: Shallow gradient boosted trees for speed.

### Sector B: Topological & Geometric Manifolds
These units focus on the "shape" and distance metrics of the feature space.
6. **Nu Warp**: Nonlinear manifold warping (NuSVC).
7. **Polynomial SVM**: Captures curved relationships.
8. **KNN Local**: Micro-geometric locality.
9. **KNN Regional**: Macro-geometric structures.
10. **QDA**: Quadratic Spacetime Modeling for curved density.

### Sector C: Harmonic Wave Units
The heart of HRF, these units model data as wave potentials.
11. **RBF Resonance**: Radial Basis Function kernels.
12. **Soul Original**: The primary Holographic Resonance unit.
13. **TwinA Resonance**: Chaotic-seeded mirror soul.
14. **TwinB Resonance**: Order-seeded mirror soul.
15. **Wave Dimension D**: Higher-order harmonic field.
16. **Wave Dimension E**: Higher-order harmonic field.
17. **Wave Dimension F**: Higher-order harmonic field.

### Sector D: Macro-Physical Layers
Experimental layers that map data to physical constants and phenomena.
18. **Golden Phi**: Biological spiral mapping.
19. **Entropy Modeling**: Information theory-based stability.
20. **Quantum Superposition**: Probabilistic state modeling.
21. **Gravity Potential**: Inverse-square law attraction.
22. **Omega Point**: Ultimate convergence trajectory.

### Sector E: Advanced Architectures
High-precision correction and specialized filters.
23. **Fractal Mirror**: Self-similar recursive mapping.
24. **Dimension Z**: Latent space representation.
25. **Omega Neural ELM**: Extremely fast neural projection.
26. **Death Ray Sniper**: High-precision residual error correction.

---

## 3. Visual Architecture Diagram

The following diagram illustrates how the various sectors are integrated by the G.O.D. Optimizer into a final prediction.

```mermaid
graph TD
    Input[Input Data] --> Scaler[Robust Scaler]
    Scaler --> SectorA[Sector A: Boosting & Logic]
    Scaler --> SectorB[Sector B: Geometry & Manifold]
    Scaler --> SectorC[Sector C: Harmonic Wave Units]
    Scaler --> SectorD[Sector D: Macro-Physics]
    Scaler --> SectorE[Sector E: Advanced Correction]

    subgraph "Titan-26 Manifold"
        SectorA
        SectorB
        SectorC
        SectorD
        SectorE
    end

    SectorA --> GOD[G.O.D. Optimizer / Weight Manager]
    SectorB --> GOD
    SectorC --> GOD
    SectorD --> GOD
    SectorE --> GOD

    GOD --> Output[Final Resonance Prediction]
```

---

## 4. Repository Structure

For new contributors, here is how the project is organized:

```text
.
├── 1/                        # Early prototypes and benchmarks
├── docs/                     # Documentation and Monograph
├── HRF Codes/                # Applied HRF (EEG, Conference papers)
├── HRF-Engine/               # Core HRF algorithm implementations
│   ├── HRF 21D/              # Legacy 21-dimensional version
│   ├── HRF 26D/              # Stable 26-dimensional version
│   └── Generalized_HRF_V2.py # Current production engine
├── Research Paper/           # Published papers and whitepapers
├── README.md                 # Project introduction
├── CONTRIBUTING.md           # Contribution guidelines
├── AGENTS.md                 # Guidelines for AI/Agents
└── SECURITY.md               # Security policy
```

- **HRF-Engine/** is where the core logic resides. If you want to modify the algorithm, look here.
- **HRF Codes/** contains Jupyter notebooks and scripts for specific datasets like EEG.
- **docs/** is where you can find detailed mathematical explanations (the Monograph).
