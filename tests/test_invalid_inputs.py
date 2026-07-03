import pytest
import numpy as np
from hrf_kernels import make_sigmoid_svc
from hrf_config import ENABLE_SIGMOID_KERNEL


@pytest.mark.skipif(not ENABLE_SIGMOID_KERNEL, reason="Sigmoid kernel is disabled in configuration")
def test_sigmoid_svc_empty_input_handling():
    """
    Ensure the sigmoid SVC safely handles or raises an error when 
    presented with completely empty training data arrays.
    """
    clf = make_sigmoid_svc()
    X_empty = np.empty((0, 4))
    y_empty = np.empty((0,))
    
    # It should raise a ValueError when trying to fit an empty dataset
    with pytest.raises(ValueError):
        clf.fit(X_empty, y_empty)


@pytest.mark.skipif(not ENABLE_SIGMOID_KERNEL, reason="Sigmoid kernel is disabled in configuration")
def test_sigmoid_svc_mismatched_dimensions():
    """
    Ensure the model raises an error if the features (X) and 
    labels (y) have mismatched lengths.
    """
    clf = make_sigmoid_svc()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_mismatch = np.array([1]) # Only 1 label for 2 data points
    
    with pytest.raises(ValueError):
        clf.fit(X, y_mismatch)
