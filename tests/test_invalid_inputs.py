import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from hrf_kernels import make_sigmoid_svc
from hrf_config import ENABLE_SIGMOID_KERNEL


def test_sigmoid_svc_empty_input_handling():
    """
    Ensure the sigmoid SVC safely handles or raises an error when 
    presented with completely empty training data arrays.
    """
    if not ENABLE_SIGMOID_KERNEL:
        return
        
    clf = make_sigmoid_svc()
    X_empty = np.empty((0, 4))
    y_empty = np.empty((0,))
    
    # It should raise a ValueError when trying to fit an empty dataset
    with pytest.raises(ValueError):
        clf.fit(X_empty, y_empty)


def test_sigmoid_svc_mismatched_dimensions():
    """
    Ensure the model raises an error if the features (X) and 
    labels (y) have mismatched lengths.
    """
    if not ENABLE_SIGMOID_KERNEL:
        return
        
    clf = make_sigmoid_svc()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_mismatch = np.array([1]) # Only 1 label for 2 data points
    
    with pytest.raises(ValueError):
        clf.fit(X, y_mismatch)

