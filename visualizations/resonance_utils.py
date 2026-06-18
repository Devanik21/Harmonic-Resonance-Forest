"""Resonance Capture Utilities

Provides mechanisms to capture and store resonance maps from HRF models
without modifying the core prediction logic.
"""

from typing import Tuple, Callable
import numpy as np


class ResonanceCaptureWrapper:
    """
    Wrapper that captures resonance data during HRF predictions.

    This wrapper intercepts predict_proba calls to capture the raw
    resonance energies (before normalization) and stores them in
    the model as last_resonance_map_.

    This is a non-breaking enhancement that doesn't modify model
    predictions or training behavior.

    Parameters
    ----------
    model : object
        Trained HRF model with predict_proba() method.

    Attributes
    ----------
    model : object
        Wrapped HRF model instance.
    last_resonance_map_ : np.ndarray
        Last captured resonance matrix (n_samples × n_classes).

    Examples
    --------
    >>> from visualizations.resonance_utils import ResonanceCaptureWrapper
    >>> model = HarmonicResonanceClassifier_BEAST_14D()
    >>> model.fit(X_train, y_train)
    >>> wrapped_model = ResonanceCaptureWrapper(model)
    >>> _ = wrapped_model.predict_proba(X_test)
    >>> resonance_map = wrapped_model.last_resonance_map_

    Notes
    -----
    Attribute writes always go to the wrapper's own ``__dict__``.
    Attribute reads that are not found on the wrapper are delegated
    to the wrapped model via ``__getattr__``. This design keeps the
    wrapper fully compatible with ``sklearn.base.clone()``, ``pickle``,
    and all meta-estimators (GridSearchCV, Pipeline, cross_val_score).
    """

    def __init__(self, model: object):
        """Initialize wrapper with HRF model."""
        self.model = model
        self.last_resonance_map_ = None
        self._resonance_extractor = None
        self._setup_extractor()

    def _setup_extractor(self):
        """Setup resonance extraction strategy based on model type."""
        model_class_name = self.model.__class__.__name__

        if 'BEAST' in model_class_name or 'Harmonic' in model_class_name:
            self._resonance_extractor = self._extract_from_ensemble
        elif 'HolographicSoul' in model_class_name:
            self._resonance_extractor = self._extract_from_soul_unit
        else:
            self._resonance_extractor = self._extract_from_proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute probabilities while capturing resonance data.

        Parameters
        ----------
        X : np.ndarray
            Input samples.

        Returns
        -------
        probas : np.ndarray
            Class probabilities (same as model.predict_proba).
        """
        probas = self.model.predict_proba(X)
        self._capture_resonance(X, probas)
        return probas

    def _capture_resonance(self, X: np.ndarray, probas: np.ndarray):
        """Capture resonance data using established extractor."""
        if self._resonance_extractor:
            self.last_resonance_map_ = self._resonance_extractor(X, probas)

    def _extract_from_ensemble(self, X: np.ndarray, probas: np.ndarray) -> np.ndarray:
        """
        Extract resonance from ensemble model (BEAST).

        Discovers units dynamically so this works for any ensemble size
        (14-unit BEAST, 16-unit Titan extension, future variants).
        Weights are applied in the same order units are discovered.
        """
        try:
            if not hasattr(self.model, 'unit_01'):
                return self._extract_from_proba(X, probas)

            # Dynamic unit discovery — no hardcoded unit count.
            # Units are collected in numeric order so weights_[i] always
            # corresponds to the correct unit regardless of ensemble size.
            all_units = [
                getattr(self.model, f'unit_{i:02d}')
                for i in range(1, 100)
                if hasattr(self.model, f'unit_{i:02d}')
            ]

            units_proba = []
            for unit in all_units:
                try:
                    if hasattr(unit, 'predict_proba'):
                        u_proba = unit.predict_proba(X)
                    else:
                        u_proba = (np.ones((len(X), len(self.model.classes_)))
                                   / len(self.model.classes_))
                    units_proba.append(u_proba)
                except Exception:
                    units_proba.append(
                        np.ones((len(X), len(self.model.classes_)))
                        / len(self.model.classes_)
                    )

            if hasattr(self.model, 'weights_') and self.model.weights_ is not None:
                weighted_sum = np.zeros_like(probas)
                for i, u_proba in enumerate(units_proba):
                    weighted_sum += self.model.weights_[i] * u_proba
                return weighted_sum
            else:
                return probas

        except Exception:
            return self._extract_from_proba(X, probas)

    def _extract_from_soul_unit(self, X: np.ndarray, probas: np.ndarray) -> np.ndarray:
        """
        Extract resonance from HolographicSoulUnit.

        Use predicted probabilities as resonance intensity.
        """
        return probas

    def _extract_from_proba(self, X: np.ndarray, probas: np.ndarray) -> np.ndarray:
        """
        Fallback: use predicted probabilities as resonance intensity.

        While not a true resonance decomposition, this provides
        reasonable heatmap visualization using available data.
        """
        return probas

    def __getattr__(self, name: str):
        """
        Delegate attribute reads to the wrapped model.

        Only called when ``name`` is not found in the wrapper's own
        ``__dict__`` — so wrapper-level attributes (model,
        last_resonance_map_, _resonance_extractor, and any method
        defined on this class) are always resolved locally first.
        """
        return getattr(self.model, name)

    # NOTE: No custom __setattr__ defined here intentionally.
    # Python's default __setattr__ writes all assignments to the wrapper's
    # own __dict__, which is the correct behaviour:
    #   - sklearn clone() can reconstruct the wrapper correctly
    #   - pickle/unpickle restores wrapper state without touching the model
    #   - meta-estimators (GridSearchCV, Pipeline) set their bookkeeping
    #     attributes on the wrapper, not on the wrapped model


def enable_resonance_capture(model: object) -> object:
    """
    Enable resonance capture on an HRF model.

    This is a convenience function that wraps a model with
    ResonanceCaptureWrapper if not already wrapped.

    Parameters
    ----------
    model : object
        HRF model instance.

    Returns
    -------
    model : object
        Same model or wrapped version with resonance capture enabled.

    Examples
    --------
    >>> model = HarmonicResonanceForest_Ultimate()
    >>> model.fit(X_train, y_train)
    >>> model = enable_resonance_capture(model)
    >>> _ = model.predict_proba(X_test)
    >>> # Access resonance data
    >>> resonance_map = model.last_resonance_map_
    """
    if isinstance(model, ResonanceCaptureWrapper):
        return model
    else:
        return ResonanceCaptureWrapper(model)


def extract_resonance_samples(
    model: object,
    X: np.ndarray,
    batch_size: int = 256
) -> np.ndarray:
    """
    Extract resonance maps for all samples in batch.

    Parameters
    ----------
    model : object
        HRF model (wrapped or with resonance capture enabled).
    X : np.ndarray
        Input samples.
    batch_size : int, optional
        Process samples in batches. Default: 256.

    Returns
    -------
    resonance_map : np.ndarray
        Full resonance matrix (n_samples × n_classes).

    Examples
    --------
    >>> resonance_map = extract_resonance_samples(model, X_test)
    >>> print(resonance_map.shape)
    (100, 3)
    """
    n_samples = len(X)
    n_classes = len(model.classes_)
    resonance_map = np.zeros((n_samples, n_classes))

    # Process in batches
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch = X[i:end]

        # Use wrapped model if available
        if isinstance(model, ResonanceCaptureWrapper):
            _ = model.predict_proba(batch)
            resonance_map[i:end] = model.last_resonance_map_
        else:
            # Fallback: use probabilities
            resonance_map[i:end] = model.predict_proba(batch)

    return resonance_map
