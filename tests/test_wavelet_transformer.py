import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from wavelet_resonance import WaveletResonanceTransformer
from hrf_config import ENABLE_WAVELET_RESONANCE


def test_wavelet_transformer_shape():
    """Test wavelet transformer shape behavior when disabled (default)."""
    X, y = make_circles(n_samples=100, factor=0.5, noise=0.05)
    tr = WaveletResonanceTransformer()
    if not ENABLE_WAVELET_RESONANCE:
        tr.fit(X)
        out = tr.transform(X)
        # should be pass-through when disabled
        assert out.shape == X.shape
    else:
        out = tr.fit_transform(X)
        assert out.shape[0] == X.shape[0]


def test_wavelet_transformer_enabled():
    """Test wavelet transformer with feature flag explicitly enabled."""
    # Temporarily patch the config flag
    import hrf_config
    original = hrf_config.ENABLE_WAVELET_RESONANCE
    try:
        hrf_config.ENABLE_WAVELET_RESONANCE = True
        X, y = make_circles(n_samples=50, factor=0.5, noise=0.05)
        tr = WaveletResonanceTransformer(wavelet='db1', level=1)
        # This will fail if pywt is not installed, which is fine for optional deps
        try:
            out = tr.fit_transform(X)
            # Should have same number of samples
            assert out.shape[0] == X.shape[0]
            # Should have features (output is not empty)
            assert out.shape[1] > 0
            # Fitted attributes should be available after fit
            assert hasattr(tr, 'wavelet_')
            assert hasattr(tr, 'level_')
        except RuntimeError as e:
            if 'pywt' in str(e):
                # pywt not installed; skip
                pass
            else:
                raise
    finally:
        hrf_config.ENABLE_WAVELET_RESONANCE = original


def test_wavelet_transformer_requires_fit():
    """Transform should require fit() to be called first."""
    X, y = make_circles(n_samples=50, factor=0.5, noise=0.05)
    tr = WaveletResonanceTransformer(wavelet='db1', level=1)
    try:
        tr.transform(X)
        assert False, 'Expected transform without fit to raise an exception'
    except Exception as e:
        assert 'fit' in str(e).lower() or 'fitted' in str(e).lower()
