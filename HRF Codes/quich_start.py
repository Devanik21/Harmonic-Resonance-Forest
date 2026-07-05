import numpy as np
from src.engine.generalized_hrf_v2 import HolographicSoulUnit

# 1. Initialize the model
model = HolographicSoulUnit(dim_reduction='none')

# 2. Create synthetic data
X = np.random.rand(10, 2)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# 3. Fit and Predict
model.fit(X, y)
print("Model trained successfully!")
print(f"Resonance output: {model._calculate_resonance(X[:2])}")