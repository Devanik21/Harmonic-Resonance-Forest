"""Wavelet-inspired resonance preprocessing transformer.

Lightweight transformer that applies a single-level discrete wavelet
transform (DWT) and returns a compact representation suitable for
time-series signals. Designed to be optional and low-overhead.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

try:
    import pywt
except Exception:  # pragma: no cover - optional dependency
    pywt = None

from hrf_config import WAVELET_DEFAULTS, ENABLE_WAVELET_RESONANCE


class WaveletResonanceTransformer(BaseEstimator, TransformerMixin):
    """Apply a lightweight DWT-based preprocessing.

    The transformer performs a level-1 DWT (db1 by default) and returns the
    concatenation of approximation and detail coefficients for each sample.
    """

    def __init__(self, wavelet=None, level=None):
        # Store parameters directly for sklearn API compliance
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not ENABLE_WAVELET_RESONANCE:
            # Pass-through when disabled
            return X

        if pywt is None:
            raise RuntimeError('pywt (PyWavelets) is required for WaveletResonanceTransformer')

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError('X must be 2D: (n_samples, n_features)')

        # Resolve defaults from config if not provided
        wavelet = self.wavelet or WAVELET_DEFAULTS['wavelet']
        level = self.level if self.level is not None else WAVELET_DEFAULTS['level']

        # Vectorized DWT: apply to all rows at once
        # pywt.wavedec performs DWT decomposition and returns coefficients
        coeffs = pywt.wavedec(X, wavelet, level=level, axis=-1)
        # Concatenate coefficients along feature axis for all samples at once
        return np.concatenate(coeffs, axis=-1).astype(np.float64)
