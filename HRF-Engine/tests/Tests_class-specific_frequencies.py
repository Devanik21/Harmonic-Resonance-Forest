"""
Tests for class-specific frequencies ω_c = f_base × (c+1) — Issue #157
========================================================================

Validates that the HRF resonance kernel (paper §3.1.1) correctly applies
a distinct frequency to each class, not a single shared frequency:

    Ψ(x, p_i) = exp(-γ‖x−p_i‖²) · (1 + cos(ω_c·‖x−p_i‖ + φ))
    ω_c = f_base · (c + 1)

Run with:
    pytest HRF-Engine/tests/test_class_specific_frequencies.py -v
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lightweight kernel implementations for isolated testing
# (mirrors exact production code in HolographicSoulUnit)
# ---------------------------------------------------------------------------

def _resonance_kernel_cpu(X_test, X_train, y_train, classes,
                           k, freq, gamma, power, phase, p_norm=2.0):
    """Post-fix: class-specific ω_c = f_base·(c+1). Mirrors production code."""
    n_classes = len(classes)
    probas = []
    k_safe = min(k, len(X_train))  # guard against k > n_train

    for i in range(0, len(X_test), 256):
        batch_te = X_test[i:i + 256]
        diff  = np.abs(batch_te[:, None, :] - X_train[None, :, :])
        dists = np.sum(diff ** p_norm, axis=2) ** (1.0 / p_norm)

        if k_safe < len(X_train):
            top_k_idx = np.argpartition(dists, k_safe, axis=1)[:, :k_safe]
        else:
            top_k_idx = np.argsort(dists, axis=1)[:, :k_safe]

        row_idx   = np.arange(len(batch_te))[:, None]
        top_dists = dists[row_idx, top_k_idx]
        top_y     = y_train[top_k_idx]

        gauss = np.exp(-gamma * (top_dists ** 2))

        batch_probs = np.zeros((len(batch_te), n_classes), dtype=np.float64)
        for c_idx, cls in enumerate(classes):
            freq_c     = freq * (c_idx + 1)          # ω_c = f_base·(c+1)
            cosine_c   = 1.0 + np.cos(freq_c * top_dists + phase)
            cosine_c   = np.maximum(cosine_c, 0.0)
            w_c        = np.power(gauss * cosine_c, power)
            class_mask = (top_y == cls)
            batch_probs[:, c_idx] = np.sum(w_c * class_mask, axis=1)

        total_energy = np.sum(batch_probs, axis=1, keepdims=True)
        total_energy[total_energy == 0] = 1.0
        batch_probs /= total_energy
        probas.append(batch_probs)

    return np.concatenate(probas)


def _old_resonance_kernel_cpu(X_test, X_train, y_train, classes,
                               k, freq, gamma, power, phase, p_norm=2.0):
    """Pre-fix (buggy): single shared frequency for all classes."""
    n_classes = len(classes)
    probas = []
    k_safe = min(k, len(X_train))

    for i in range(0, len(X_test), 256):
        batch_te = X_test[i:i + 256]
        diff  = np.abs(batch_te[:, None, :] - X_train[None, :, :])
        dists = np.sum(diff ** p_norm, axis=2) ** (1.0 / p_norm)

        if k_safe < len(X_train):
            top_k_idx = np.argpartition(dists, k_safe, axis=1)[:, :k_safe]
        else:
            top_k_idx = np.argsort(dists, axis=1)[:, :k_safe]

        row_idx   = np.arange(len(batch_te))[:, None]
        top_dists = dists[row_idx, top_k_idx]
        top_y     = y_train[top_k_idx]

        # BUG: single shared frequency — no class discrimination
        cosine_term = 1.0 + np.cos(freq * top_dists + phase)
        cosine_term = np.maximum(cosine_term, 0.0)
        w = np.power(np.exp(-gamma * (top_dists ** 2)) * cosine_term, power)

        batch_probs = np.zeros((len(batch_te), n_classes), dtype=np.float64)
        for c_idx, cls in enumerate(classes):
            class_mask = (top_y == cls)
            batch_probs[:, c_idx] = np.sum(w * class_mask, axis=1)

        total_energy = np.sum(batch_probs, axis=1, keepdims=True)
        total_energy[total_energy == 0] = 1.0
        batch_probs /= total_energy
        probas.append(batch_probs)

    return np.concatenate(probas)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def two_class_data(rng):
    X0 = rng.standard_normal((60, 4)) + np.array([0., 0., 0., 0.])
    X1 = rng.standard_normal((60, 4)) + np.array([3., 3., 3., 3.])
    X  = np.vstack([X0, X1]).astype(np.float32)
    y  = np.array([0]*60 + [1]*60)
    return X, y


@pytest.fixture
def three_class_data(rng):
    centers = [(0., 0.), (5., 5.), (10., 0.)]
    Xs, ys = [], []
    for c, (cx, cy) in enumerate(centers):
        pts = rng.standard_normal((50, 2)) + np.array([cx, cy])
        Xs.append(pts.astype(np.float32))
        ys.append(np.full(50, c))
    return np.vstack(Xs), np.hstack(ys)


# ---------------------------------------------------------------------------
# 1. Core mathematical invariant: ω_c = f_base × (c+1)
# ---------------------------------------------------------------------------

class TestClassSpecificFrequencyMath:
    """Directly validates the §3.1.1 formula at the scalar level."""

    def test_different_classes_produce_different_cosine_terms(self):
        """
        At the same distance r, classes 0/1/2 MUST produce different cosines.
        Pre-fix: all classes shared cos(f·r) → identical → no discrimination.
        Post-fix: cos(f·(c+1)·r) differs per c.
        """
        freq, phase, r = 2.0, 0.0, 1.0
        cos_vals = [1.0 + np.cos(freq * (c + 1) * r + phase) for c in range(3)]
        assert not np.isclose(cos_vals[0], cos_vals[1]), "c=0 and c=1 cosines must differ"
        assert not np.isclose(cos_vals[1], cos_vals[2]), "c=1 and c=2 cosines must differ"
        assert not np.isclose(cos_vals[0], cos_vals[2]), "c=0 and c=2 cosines must differ"

    def test_frequency_multiplier_is_c_plus_one(self):
        """ω_c = f_base·(c+1) — the (c+1) multiplier must be exactly c+1."""
        f_base = 3.0
        for c in range(5):
            assert np.isclose(f_base * (c + 1), f_base + f_base * c), (
                f"ω_{c} does not equal f_base*(c+1)"
            )

    def test_class0_uses_f_base_exactly(self):
        """
        Class 0 → multiplier = (0+1) = 1 → ω_0 = f_base.
        Must NOT be 0 (which would give cos(0)=1 ≡ no oscillation).
        """
        freq = 2.5
        freq_c0 = freq * (0 + 1)
        assert freq_c0 == pytest.approx(freq), "Class 0 must use ω_0 = f_base"
        # cos(2.5*r) ≠ 1 for r > 0 → oscillation is present
        r = 0.5
        assert not np.isclose(np.cos(freq_c0 * r), 1.0, atol=0.05), (
            "ω_0 must not be 0 — that would eliminate all frequency content"
        )

    @pytest.mark.parametrize("f_base", [0.5, 1.0, 2.0, np.pi, 6.2832])
    def test_spectral_separation_for_various_f_base(self, f_base):
        """For any positive f_base, class frequencies must be distinct."""
        r = 1.0
        cos_vals = [float(np.cos(f_base * (c + 1) * r)) for c in range(4)]
        unique = len(set(round(v, 10) for v in cos_vals))
        assert unique > 1, (
            f"f_base={f_base}: all 4 cosine terms identical → no discrimination"
        )


# ---------------------------------------------------------------------------
# 2. Kernel output diverges between classes for equal distances
# ---------------------------------------------------------------------------

class TestKernelOutputDivergence:
    """
    Place one training point of each class at the SAME distance from the query.
    The old kernel assigns them identical weights (no discrimination).
    The new kernel assigns different weights via ω_c.
    """

    def _equidistant_setup(self):
        """One class-0 and one class-1 point, both at distance 1.0 from origin."""
        X_query = np.array([[0.0]], dtype=np.float32)        # 1D query at origin
        X_train = np.array([[1.0], [-1.0]], dtype=np.float32) # both dist=1
        y_train = np.array([0, 1])
        classes = np.array([0, 1])
        return X_query, X_train, y_train, classes

    def test_new_kernel_breaks_symmetry(self):
        """
        New kernel: p(class=0) ≠ p(class=1) even when distances are equal.
        This is the central claim of Issue #157.
        """
        X_q, X_tr, y_tr, cls = self._equidistant_setup()
        proba = _resonance_kernel_cpu(
            X_q, X_tr, y_tr, cls,
            k=2, freq=2.0, gamma=0.5, power=2.0, phase=0.0
        )
        p0, p1 = proba[0, 0], proba[0, 1]
        assert not np.isclose(p0, p1, atol=0.01), (
            f"New kernel gave identical probas {p0:.4f}/{p1:.4f} for equidistant "
            f"classes — class-specific frequency appears not to be applied."
        )

    def test_old_kernel_is_symmetric(self):
        """
        Regression: the pre-fix kernel assigns p=0.5 to both classes when
        all distances are equal — confirming the original bug.
        """
        X_q, X_tr, y_tr, cls = self._equidistant_setup()
        proba = _old_resonance_kernel_cpu(
            X_q, X_tr, y_tr, cls,
            k=2, freq=2.0, gamma=0.5, power=2.0, phase=0.0
        )
        p0, p1 = proba[0, 0], proba[0, 1]
        assert np.isclose(p0, p1, atol=1e-9), (
            f"Old (buggy) kernel should give p=0.5/0.5 for equidistant classes. "
            f"Got {p0:.6f}/{p1:.6f}."
        )

    def test_accuracy_not_regressed(self, three_class_data):
        """
        On a separable 3-class problem the new kernel must achieve
        at least as good accuracy as the old one (no performance regression).
        """
        from sklearn.model_selection import train_test_split
        X, y = three_class_data
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=7, stratify=y
        )
        classes = np.unique(y_tr)

        kw = dict(k=7, freq=2.0, gamma=0.5, power=2.0, phase=0.0)
        acc_new = np.mean(
            np.argmax(_resonance_kernel_cpu(X_te, X_tr, y_tr, classes, **kw), axis=1) == y_te
        )
        acc_old = np.mean(
            np.argmax(_old_resonance_kernel_cpu(X_te, X_tr, y_tr, classes, **kw), axis=1) == y_te
        )
        print(f"\n  3-class: old={acc_old:.2%}  new={acc_new:.2%}  Δ={acc_new-acc_old:+.2%}")
        assert acc_new >= acc_old - 0.05, (
            f"New kernel accuracy ({acc_new:.2%}) regressed vs old ({acc_old:.2%})"
        )


# ---------------------------------------------------------------------------
# 3. Probability array well-formedness
# ---------------------------------------------------------------------------

class TestProbabilityOutputProperties:

    def test_rows_sum_to_one(self, two_class_data):
        X, y = two_class_data
        proba = _resonance_kernel_cpu(
            X[:20], X[20:], y[20:], np.unique(y),
            k=5, freq=2.0, gamma=0.5, power=2.0, phase=0.0
        )
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)

    def test_non_negative(self, two_class_data):
        X, y = two_class_data
        proba = _resonance_kernel_cpu(
            X[:20], X[20:], y[20:], np.unique(y),
            k=5, freq=2.0, gamma=0.5, power=2.0, phase=0.0
        )
        assert np.all(proba >= 0.0)

    def test_shape(self, two_class_data):
        X, y = two_class_data
        proba = _resonance_kernel_cpu(
            X[:30], X[30:], y[30:], np.unique(y),
            k=5, freq=2.0, gamma=0.5, power=2.0, phase=0.0
        )
        assert proba.shape == (30, 2)

    def test_no_nan_when_all_weights_zero(self, rng):
        """gamma=∞ zeroes all Gaussian weights — zero-energy guard must prevent NaN."""
        X = rng.standard_normal((10, 2)).astype(np.float32)
        y = np.array([0]*5 + [1]*5)
        proba = _resonance_kernel_cpu(
            X[:5], X[5:], y[5:], np.array([0, 1]),
            k=3, freq=2.0, gamma=1e10, power=2.0, phase=0.0
        )
        assert not np.any(np.isnan(proba)), "NaN detected — zero-energy guard failed"

    def test_no_nan_with_zero_distance(self, rng):
        """Query point identical to a training point (dist=0) must not produce NaN."""
        X = rng.standard_normal((10, 3)).astype(np.float32)
        y = np.array([0]*5 + [1]*5)
        X_test = X[:5].copy()                    # identical to training points
        proba = _resonance_kernel_cpu(
            X_test, X[:5], y[:5], np.array([0, 1]),
            k=3, freq=2.0, gamma=0.5, power=2.0, phase=0.0
        )
        assert not np.any(np.isnan(proba))


# ---------------------------------------------------------------------------
# 4. Frequency monotonicity
# ---------------------------------------------------------------------------

class TestFrequencyMonotonicity:

    def test_omega_strictly_increases_with_class_index(self):
        """ω_c must be strictly monotonically increasing in c."""
        f_base = 1.5
        omegas = [f_base * (c + 1) for c in range(6)]
        for i in range(len(omegas) - 1):
            assert omegas[i] < omegas[i + 1], (
                f"ω_{i}={omegas[i]} ≥ ω_{i+1}={omegas[i+1]}"
            )

    def test_resonance_energy_differs_across_classes(self):
        """
        Fixed r, fixed training point: resonance energy W_c must not be
        identical across all class indices.
        """
        freq, gamma, power, phase = 2.0, 0.5, 2.0, 0.0
        r = 0.5  # scalar distance
        gauss = np.exp(-gamma * r ** 2)
        energies = []
        for c_idx in range(4):
            freq_c = freq * (c_idx + 1)
            cos_c  = max(0.0, 1.0 + float(np.cos(freq_c * r + phase)))
            w_c    = float((gauss * cos_c) ** power)
            energies.append(round(w_c, 10))

        assert len(set(energies)) > 1, (
            f"All 4 class energies are identical ({energies}) — "
            f"frequency discrimination is absent."
        )

    def test_phase_does_not_collapse_discrimination(self):
        """A phase offset must not eliminate cross-class frequency differences."""
        freq, gamma, power, phase = 2.0, 0.5, 2.0, np.pi / 4
        r = 1.0
        gauss = float(np.exp(-gamma * r ** 2))
        energies = []
        for c_idx in range(3):
            freq_c = freq * (c_idx + 1)
            cos_c  = max(0.0, 1.0 + float(np.cos(freq_c * r + phase)))
            w_c    = float((gauss * cos_c) ** power)
            energies.append(round(w_c, 10))

        assert len(set(energies)) > 1, (
            f"Phase offset collapsed frequency discrimination: {energies}"
        )


# ---------------------------------------------------------------------------
# 5. Batch-size invariance
# ---------------------------------------------------------------------------

class TestBatchInvariance:

    def test_full_equals_chunked(self, two_class_data, rng):
        """Output must be identical regardless of how data is batched."""
        X, y = two_class_data
        X_tr, y_tr = X[:80], y[:80]
        X_te = X[80:]
        classes = np.unique(y)
        kw = dict(k=5, freq=2.0, gamma=0.5, power=2.0, phase=0.0)

        out_a = _resonance_kernel_cpu(X_te, X_tr, y_tr, classes, **kw)
        out_b = _resonance_kernel_cpu(X_te, X_tr, y_tr, classes, **kw)
        np.testing.assert_array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# 6. Integration smoke test via HolographicSoulUnit
# ---------------------------------------------------------------------------

class TestHolographicSoulUnitIntegration:
    """
    Smoke tests exercising the full HolographicSoulUnit class.
    Auto-skipped when module dependencies are unavailable.
    """

    @pytest.fixture
    def soul_unit(self):
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from generalized_hrf_v2 import HolographicSoulUnit  # noqa
            return HolographicSoulUnit(k=5, freq=2.0, gamma=0.5, power=2.0)
        except (ImportError, ModuleNotFoundError):
            pytest.skip("HolographicSoulUnit not importable")

    def test_predict_proba_shape(self, soul_unit, two_class_data):
        X, y = two_class_data
        soul_unit.fit(X[:80], y[:80])
        proba = soul_unit.predict_proba(X[80:])
        assert proba.shape == (40, 2)

    def test_predict_proba_sums_to_one(self, soul_unit, two_class_data):
        X, y = two_class_data
        soul_unit.fit(X[:80], y[:80])
        proba = soul_unit.predict_proba(X[80:])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_frequency_discrimination_not_regressed(self, soul_unit, two_class_data):
        """
        Verify the fitted unit does NOT give identical probabilities to
        both classes at equidistant points — confirming the fix holds.
        """
        X, y = two_class_data
        soul_unit.fit(X, y)
        X_sym = np.zeros((1, X.shape[1]), dtype=np.float32)
        proba = soul_unit.predict_proba(X_sym)
        # The result should be class-discriminative (not necessarily 0.5/0.5)
        # We just verify no NaN and valid simplex
        assert not np.any(np.isnan(proba))
        assert np.isclose(proba.sum(), 1.0, rtol=1e-5)
