import sys
import os
import pytest
import numpy as np

# Ensure Python can find the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.generalized_hrf_v2 import HolographicSoulUnit

@pytest.fixture
def model():
    return HolographicSoulUnit()

def test_resonance_kernel_output(model):
    input_vector = np.array([[1.0, 0.5]])
    # Replace 0.85 with the actual expected output for your specific kernel
    output = model._calculate_resonance(input_vector)
    assert np.isclose(output[0], 0.85, atol=1e-2)

def test_model_stability_with_noise(model):
    X_clean = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model.fit(X_clean, y)
    baseline_score = model.score(X_clean, y)
    
    # Inject 10% noise
    X_noisy = X_clean + np.random.normal(0, 0.1, X_clean.shape)
    noisy_score = model.score(X_noisy, y)
    
    assert noisy_score > (baseline_score * 0.90)