import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from wavelet_resonance import WaveletResonanceTransformer
from hrf_config import ENABLE_WAVELET_RESONANCE


def test_wavelet_transformer_shape():
    # Enable flag may be False by default; test transform behavior
    X, y = make_circles(n_samples=100, factor=0.5, noise=0.05)
    tr = WaveletResonanceTransformer()
    if not ENABLE_WAVELET_RESONANCE:
        out = tr.transform(X)
        # should be pass-through
        assert out.shape == X.shape
    else:
        out = tr.fit_transform(X)
        assert out.shape[0] == X.shape[0]
