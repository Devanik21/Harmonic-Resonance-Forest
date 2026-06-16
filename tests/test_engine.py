import numpy as np
from HRF_Engine.engine import HarmonicResonanceClassifier # Adjust the import path!

def test_energy_calculation_basic():
    # Setup: Create a tiny instance
    model = HarmonicResonanceClassifier(base_freq=1.0, gamma=1.0, decay_type='gaussian')
    
    # Simple input
    dists = np.array([1.0, 2.0])
    class_id = 0
    
    # Run the function
    energy = model._calculate_energy(dists, class_id)
    
    # Assert: Check if the result is a number (not NaN or error)
    assert not np.isnan(energy)
    assert isinstance(energy, float)