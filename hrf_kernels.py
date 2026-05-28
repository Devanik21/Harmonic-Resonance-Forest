"""Lightweight kernel utilities for HRF.

Provides a small wrapper to create an sklearn SVC with sigmoid kernel
using safe defaults. Kept minimal to avoid heavy overhead.
"""
from sklearn.svm import SVC
from hrf_config import ENABLE_SIGMOID_KERNEL, SIGMOID_KERNEL_DEFAULTS


def make_sigmoid_svc(**overrides):
    """Create a lightweight SVC with sigmoid kernel.

    Returns an sklearn.svm.SVC instance configured with conservative defaults.
    """
    if not ENABLE_SIGMOID_KERNEL:
        raise RuntimeError("Sigmoid kernel is disabled in configuration.")

    cfg = SIGMOID_KERNEL_DEFAULTS.copy()
    cfg.update(overrides)

    return SVC(kernel='sigmoid', C=cfg['C'], gamma=cfg['gamma'], coef0=cfg['coef0'], probability=False)
