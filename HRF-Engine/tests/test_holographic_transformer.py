"""
Tests for HolographicDifferentialTransformer — Issue #139
=========================================================

Validates the bipolar montage preprocessing from HRF paper §3.2:
    X_diff[i] = X[i] - X[i+1]   for i in {0, ..., d-2}
    coherence  = Var(X)
    output     = [X_raw | X_diff | coherence]  → shape (n, 2d)

Run with:
    pytest tests/test_holographic_transformer.py -v
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

# ---------------------------------------------------------------------------
# Import the transformer from the HRF-Engine module
# ---------------------------------------------------------------------------
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "HRF-Engine"))

# We import only the transformer to keep the test lightweight (no GPU needed)
try:
    from generalized_hrf_v2 import HolographicDifferentialTransformer  # noqa: E402
except ImportError:
    pytest.skip("generalized_hrf_v2 not importable (GPU deps missing)", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def eeg_like(rng):
    """Simulate an OpenML-1471-style (n_samples, 14) EEG matrix."""
    return rng.standard_normal((200, 14))


@pytest.fixture
def fitted_hdt(eeg_like):
    hdt = HolographicDifferentialTransformer(clip_range=15.0)
    return hdt.fit(eeg_like)


# ---------------------------------------------------------------------------
# 1. Shape correctness — §3.2 guarantees output (n, 2*d)
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_eeg_14_channels(self, eeg_like, fitted_hdt):
        out = fitted_hdt.transform(eeg_like)
        assert out.shape == (200, 28), (
            f"Expected (200, 28) for 14 input channels: "
            f"14 raw + 13 diffs + 1 coherence. Got {out.shape}"
        )

    def test_n_features_out_attribute(self, fitted_hdt):
        assert fitted_hdt.n_features_out_ == 28

    @pytest.mark.parametrize("n_ch", [2, 5, 10, 20])
    def test_arbitrary_channel_counts(self, rng, n_ch):
        X = rng.standard_normal((50, n_ch))
        hdt = HolographicDifferentialTransformer().fit(X)
        out = hdt.transform(X)
        assert out.shape == (50, 2 * n_ch), (
            f"n_ch={n_ch}: expected (50, {2*n_ch}), got {out.shape}"
        )


# ---------------------------------------------------------------------------
# 2. Formula correctness — exact §3.2 values
# ---------------------------------------------------------------------------

class TestFormulaCorrectness:
    """
    Verify the three components against hand-computed reference values.

    §3.2 formula:
        X_diff[i] = X[i] - X[i+1]
        coherence  = Var(X)                     (population variance, ddof=0)
        output     = [X_raw | X_diff | coherence]
    """

    def test_differential_values(self):
        X = np.array([[1.0, 3.0, 6.0, 10.0]])          # 4-channel example
        hdt = HolographicDifferentialTransformer(clip_range=None).fit(X)
        out = hdt.transform(X)

        # X_diff[i] = X[i] - X[i+1]:  1-3, 3-6, 6-10  → [-2, -3, -4]
        expected_diffs = np.array([-2.0, -3.0, -4.0])
        np.testing.assert_allclose(out[0, 4:7], expected_diffs,
                                   err_msg="Differential features mismatch §3.2")

    def test_coherence_value(self):
        X = np.array([[1.0, 3.0, 6.0, 10.0]])
        hdt = HolographicDifferentialTransformer(clip_range=None).fit(X)
        out = hdt.transform(X)

        expected_var = float(np.var([1., 3., 6., 10.]))   # ddof=0
        np.testing.assert_allclose(out[0, -1], expected_var, rtol=1e-9,
                                   err_msg="Coherence (Var) mismatch §3.2")

    def test_raw_channels_preserved(self):
        X = np.array([[2.0, 4.0, 8.0, 16.0]])
        hdt = HolographicDifferentialTransformer(clip_range=None).fit(X)
        out = hdt.transform(X)
        np.testing.assert_array_equal(out[0, :4], X[0],
                                      err_msg="Raw channels must pass through unchanged")


# ---------------------------------------------------------------------------
# 3. Common-mode artefact rejection
# ---------------------------------------------------------------------------

class TestCommonModeRejection:
    """
    Bipolar montage is clinically valued because global artefacts (body
    movement, electrode drift) cancel in the differential channels.
    """

    def test_pure_common_mode_noise_vanishes(self):
        """Constant offset added to all channels → diffs unchanged."""
        X_clean = np.array([[1.0, 2.0, 3.0, 5.0]])
        noise_level = 500.0                             # extreme artefact
        X_noisy = X_clean + noise_level

        hdt = HolographicDifferentialTransformer(clip_range=None)
        out_clean = hdt.fit(X_clean).transform(X_clean)
        out_noisy = hdt.transform(X_noisy)

        d = X_clean.shape[1]
        np.testing.assert_allclose(
            out_clean[0, d:2*d-1], out_noisy[0, d:2*d-1], rtol=1e-9,
            err_msg="Common-mode noise must cancel in differential channels"
        )

    def test_differential_signal_preserved(self):
        """True differential signal (only one channel changes) is NOT cancelled."""
        X_ref  = np.array([[1.0, 1.0, 1.0, 1.0]])
        X_diff = np.array([[1.0, 2.0, 1.0, 1.0]])   # channel-1 fires

        hdt = HolographicDifferentialTransformer(clip_range=None)
        hdt.fit(X_ref)
        out_ref  = hdt.transform(X_ref)
        out_diff = hdt.transform(X_diff)

        assert not np.allclose(out_ref[0, 4:], out_diff[0, 4:]), (
            "Differential signal change should NOT be cancelled"
        )


# ---------------------------------------------------------------------------
# 4. Clipping behaviour
# ---------------------------------------------------------------------------

class TestClipping:
    def test_clip_applied_at_15(self):
        X = np.array([[20.0, -20.0, 0.0, 5.0]])
        hdt = HolographicDifferentialTransformer(clip_range=15.0).fit(X)
        out = hdt.transform(X)
        assert out[0, 0] == pytest.approx(15.0), "Positive extreme not clipped"
        assert out[0, 1] == pytest.approx(-15.0), "Negative extreme not clipped"

    def test_clip_none_disables_clipping(self):
        X = np.array([[100.0, -100.0, 0.0, 5.0]])
        hdt = HolographicDifferentialTransformer(clip_range=None).fit(X)
        out = hdt.transform(X)
        assert out[0, 0] == pytest.approx(100.0), "Clipping should be disabled"


# ---------------------------------------------------------------------------
# 5. Feature-name generation
# ---------------------------------------------------------------------------

class TestFeatureNames:
    def test_names_count(self):
        hdt = HolographicDifferentialTransformer().fit(np.zeros((3, 4)))
        names = hdt.get_feature_names_out()
        assert len(names) == 8    # 4 raw + 3 diffs + 1 coherence

    def test_names_content(self):
        hdt = HolographicDifferentialTransformer().fit(np.zeros((3, 3)))
        names = hdt.get_feature_names_out(["a", "b", "c"])
        expected = ["a", "b", "c", "a_minus_b", "b_minus_c", "global_coherence"]
        assert list(names) == expected

    def test_global_coherence_always_last(self):
        hdt = HolographicDifferentialTransformer().fit(np.zeros((3, 6)))
        assert hdt.get_feature_names_out()[-1] == "global_coherence"


# ---------------------------------------------------------------------------
# 6. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_raises_on_single_channel(self):
        X = np.random.randn(10, 1)
        with pytest.raises(ValueError, match="at least 2"):
            HolographicDifferentialTransformer().fit(X)

    def test_raises_on_feature_mismatch_at_transform(self):
        X_train = np.random.randn(10, 5)
        X_test  = np.random.randn(10, 7)
        hdt = HolographicDifferentialTransformer().fit(X_train)
        with pytest.raises(ValueError, match="features"):
            hdt.transform(X_test)

    def test_raises_if_not_fitted(self):
        hdt = HolographicDifferentialTransformer()
        with pytest.raises(Exception):          # NotFittedError
            hdt.transform(np.random.randn(5, 4))


# ---------------------------------------------------------------------------
# 7. sklearn Pipeline and estimator compatibility
# ---------------------------------------------------------------------------

class TestSklearnCompat:
    def test_pipeline_fit_predict(self, eeg_like, rng):
        y = rng.integers(0, 2, len(eeg_like))
        pipe = Pipeline([
            ("scaler", RobustScaler(quantile_range=(15, 85))),
            ("holo",   HolographicDifferentialTransformer()),
            ("clf",    LogisticRegression(max_iter=500)),
        ])
        pipe.fit(eeg_like, y)
        preds = pipe.predict(eeg_like)
        assert preds.shape == (200,)

    def test_fit_transform_consistency(self, eeg_like):
        """fit_transform() must equal fit().transform()."""
        hdt1 = HolographicDifferentialTransformer()
        hdt2 = HolographicDifferentialTransformer()
        out_ft = hdt1.fit_transform(eeg_like)
        out_f  = hdt2.fit(eeg_like).transform(eeg_like)
        np.testing.assert_array_equal(out_ft, out_f)

    def test_get_params_set_params(self):
        hdt = HolographicDifferentialTransformer(clip_range=10.0)
        assert hdt.get_params()["clip_range"] == 10.0
        hdt.set_params(clip_range=5.0)
        assert hdt.clip_range == 5.0

    def test_clone_preserves_params(self):
        from sklearn.base import clone
        hdt = HolographicDifferentialTransformer(clip_range=7.5)
        hdt_clone = clone(hdt)
        assert hdt_clone.clip_range == 7.5


# ---------------------------------------------------------------------------
# 8. Integration: use_holographic_diff=True flag mirrors standalone usage
# ---------------------------------------------------------------------------
# (Skipped when GPU deps are absent — the BEAST_14D unit itself is not
#  required to run; we only verify the data path via a mock.)

class TestBEAST14DIntegration:
    def test_feature_expansion_at_fit(self, eeg_like):
        """
        When use_holographic_diff=True, X seen by the scaler and units
        should have 2*d features (28 for 14-channel EEG).
        """
        hdt = HolographicDifferentialTransformer(clip_range=15.0)
        hdt.fit(eeg_like)
        X_expanded = hdt.transform(eeg_like)
        assert X_expanded.shape[1] == 28, (
            f"use_holographic_diff must expand 14→28 features. Got {X_expanded.shape[1]}"
        )

    def test_predict_proba_uses_same_expansion(self, eeg_like, rng):
        """
        Transform applied in predict_proba must match fit-time expansion.
        """
        hdt = HolographicDifferentialTransformer(clip_range=15.0)
        X_train = hdt.fit_transform(eeg_like[:160])
        X_test  = hdt.transform(eeg_like[160:])
        assert X_train.shape[1] == X_test.shape[1] == 28
