"""Wavelet-inspired resonance preprocessing transformer.

Lightweight transformer that applies a single-level discrete wavelet
transform (DWT) and returns a compact representation suitable for
time-series signals. Designed to be optional and low-overhead.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

try:
    import pywt
except ImportError:  # pragma: no cover - optional dependency
    pywt = None

from sklearn.utils.validation import check_is_fitted
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
        self.wavelet_ = self.wavelet or WAVELET_DEFAULTS['wavelet']
        self.level_ = self.level if self.level is not None else WAVELET_DEFAULTS['level']
        return self

    def transform(self, X):
        check_is_fitted(self, ['wavelet_', 'level_'])

        if not ENABLE_WAVELET_RESONANCE:
            # Pass-through when disabled
            return X

        if pywt is None:
            raise RuntimeError('pywt (PyWavelets) is required for WaveletResonanceTransformer')

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError('X must be 2D: (n_samples, n_features)')

        wavelet = self.wavelet_
        level = self.level_

        # Vectorized DWT: apply to all rows at once
        # pywt.wavedec performs DWT decomposition and returns coefficients
        coeffs = pywt.wavedec(X, wavelet, level=level, axis=-1)
        # Concatenate coefficients along feature axis for all samples at once
        return np.concatenate(coeffs, axis=-1).astype(np.float64)
