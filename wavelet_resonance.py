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
        self.wavelet = wavelet or WAVELET_DEFAULTS['wavelet']
        self.level = level if level is not None else WAVELET_DEFAULTS['level']

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

        out = []
        for i in range(X.shape[0]):
            sig = X[i]
            # perform single-level DWT (approx, detail)
            coeffs = pywt.wavedec(sig, self.wavelet, level=self.level)
            # flatten and concatenate coefficients (keeps representation small)
            flat = np.concatenate([c.ravel() for c in coeffs])
            out.append(flat)

        # Pad/truncate to keep a consistent width: choose the median length
        lengths = [len(r) for r in out]
        target = int(np.median(lengths)) if lengths else 0
        arr = np.zeros((len(out), target), dtype=float)
        for i, r in enumerate(out):
            if len(r) >= target:
                arr[i] = r[:target]
            else:
                arr[i, : len(r)] = r
        return arr
