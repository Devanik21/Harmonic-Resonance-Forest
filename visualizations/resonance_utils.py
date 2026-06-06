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
    >>> model = HarmonicResonanceForest_Ultimate()
    >>> model.fit(X_train, y_train)
    >>> wrapped_model = ResonanceCaptureWrapper(model)
    >>> _ = wrapped_model.predict_proba(X_test)
    >>> resonance_map = wrapped_model.last_resonance_map_
    """

    def __init__(self, model: object):
        """Initialize wrapper with HRF model."""
        self.model = model
        self.last_resonance_map_ = None
        self._resonance_extractor = None
        self._setup_extractor()

    def _setup_extractor(self):
        """Setup resonance extraction strategy based on model type."""
        # Check model type and setup appropriate extractor
        model_class_name = self.model.__class__.__name__

        if 'BEAST' in model_class_name or 'Harmonic' in model_class_name:
            self._resonance_extractor = self._extract_from_ensemble
        elif 'HolographicSoul' in model_class_name:
            self._resonance_extractor = self._extract_from_soul_unit
        else:
            # Fallback: use predict_proba output as resonance proxy
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
        # Get predictions from wrapped model
        probas = self.model.predict_proba(X)

        # Capture resonance
        self._capture_resonance(X, probas)

        return probas

    def _capture_resonance(self, X: np.ndarray, probas: np.ndarray):
        """Capture resonance data using established extractor."""
        if self._resonance_extractor:
            self.last_resonance_map_ = self._resonance_extractor(X, probas)

    def _extract_from_ensemble(self, X: np.ndarray, probas: np.ndarray) -> np.ndarray:
        """
        Extract resonance from ensemble model (BEAST).

        For ensemble models, compute weighted unit contributions
        to show which units "voted" for each prediction.
        """
        try:
            if not hasattr(self.model, 'unit_12'):
                return self._extract_from_proba(X, probas)

            # Get individual unit predictions if available
            units_proba = []
            all_units = [
                self.model.unit_01, self.model.unit_02, self.model.unit_03,
                self.model.unit_04, self.model.unit_05, self.model.unit_06,
                self.model.unit_07, self.model.unit_15, self.model.unit_16,
                self.model.unit_08, self.model.unit_09, self.model.unit_10,
                self.model.unit_11, self.model.unit_12, self.model.unit_13,
                self.model.unit_14
            ]

            for unit in all_units:
                try:
                    if hasattr(unit, 'predict_proba'):
                        u_proba = unit.predict_proba(X)
                    else:
                        u_proba = np.ones((len(X), len(self.model.classes_))) / len(self.model.classes_)
                    units_proba.append(u_proba)
                except Exception:
                    units_proba.append(np.ones((len(X), len(self.model.classes_))) / len(self.model.classes_))

            # Weight by model weights
            if hasattr(self.model, 'weights_'):
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
        """Delegate attribute access to wrapped model."""
        return getattr(self.model, name)

    def __setattr__(self, name: str, value):
        """Handle both wrapper and model attributes."""
        if name in ['model', 'last_resonance_map_', '_resonance_extractor']:
            super().__setattr__(name, value)
        else:
            setattr(self.model, name, value)


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
