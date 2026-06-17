"""Lightweight kernel utilities for HRF.

Provides a small wrapper to create an sklearn SVC with sigmoid kernel
using safe defaults. Kept minimal to avoid heavy overhead.
"""
from sklearn.svm import SVC
from hrf_config import ENABLE_SIGMOID_KERNEL, SIGMOID_KERNEL_DEFAULTS


def make_sigmoid_svc(**overrides):
    """Create a lightweight SVC with sigmoid kernel for use in the HRF ensemble.

    Returns an sklearn.svm.SVC instance configured with conservative defaults.
    Additional keyword arguments are forwarded to SVC.

    Notes
    -----
    ``probability`` defaults to ``True`` because the HRF ensemble calls
    ``predict_proba()`` on every unit during both weight optimisation (fit)
    and inference (predict_proba). Setting ``probability=False`` raises
    ``AttributeError`` at runtime and silently breaks the entire ensemble.
    Pass ``probability=False`` explicitly only if this unit will never be
    used inside ``HarmonicResonanceClassifier_BEAST_14D``.
    """
    if not ENABLE_SIGMOID_KERNEL:
        raise RuntimeError("Sigmoid kernel is disabled in configuration.")

    cfg = SIGMOID_KERNEL_DEFAULTS.copy()
    cfg.update(overrides)
    cfg['kernel'] = 'sigmoid'
    # Must be True: HRF ensemble requires predict_proba() on every unit.
    # Callers may override with probability=False only for standalone use.
    cfg.setdefault('probability', True)

    return SVC(**cfg)
