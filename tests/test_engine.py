import pytest
import numpy as np
# Assuming the engine is located in the HRF-Engine folder
from HRF_Engine import hrf_eeg # Adjust import based on your actual file structure

def test_hrf_initialization():
    """Check if the classifier initializes without errors."""
    try:
        # Replace 'HarmonicResonanceClassifier' with the actual class name from the repo
        model = hrf_eeg.HarmonicResonanceClassifier() 
        assert model is not None
    except Exception as e:
        pytest.fail(f"Initialization failed: {e}")