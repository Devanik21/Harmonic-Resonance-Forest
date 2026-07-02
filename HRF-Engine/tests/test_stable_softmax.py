"""
Tests for numerically-stable softmax — overflow / NaN bug fix
==============================================================

Verifies the fix for the naive ``np.exp(d) / np.sum(np.exp(d))`` overflow
in ``HarmonicResonanceClassifier_BEAST_14D.fit()`` (Step C weight collection)
and ``predict_proba()`` (inference fallback).

Bug:   d values > ~710  →  np.exp(d) == inf  →  inf/inf == NaN
Fix:   ``_stable_softmax(d)`` uses log-sum-exp (max-subtraction) trick

Run with:
    pytest HRF-Engine/tests/test_stable_softmax.py -v
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import production helper — ensures tests run against actual code
# ---------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from generalized_hrf_v2 import _stable_softmax
except ImportError:
    pytest.skip("generalized_hrf_v2 not importable", allow_module_level=True)


def _naive_softmax(d: np.ndarray) -> np.ndarray:
    """Pre-fix buggy softmax — kept to prove the original failure."""
    return np.exp(d) / np.sum(np.exp(d), axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_scores():
    """Typical decision-function scores — no overflow risk."""
    return np.array([
        [2.0,  1.0,  0.5],
        [0.1,  0.8,  0.3],
        [-1.0, 0.0,  1.5],
    ])


@pytest.fixture
def overflow_scores():
    """Scores that trigger inf/NaN in the naive implementation (>710)."""
    return np.array([
        [800.0, 750.0, 720.0],
        [710.0, 680.0, 690.0],
        [1000.0, 999.0, 998.0],
    ])


@pytest.fixture
def extreme_negative():
    """Very negative scores — tests underflow / near-zero exp."""
    return np.array([
        [-800.0, -750.0, -720.0],
        [-1000.0, -999.9, -999.8],
    ])


@pytest.fixture
def binary_scores_1d():
    """Binary decision_function returns shape (n,) — must become (n, 2)."""
    return np.array([3.0, -2.0, 0.0, 710.0, -710.0])


# ---------------------------------------------------------------------------
# 1.  The core bug: naive softmax produces NaN for large scores
# ---------------------------------------------------------------------------

class TestNaiveSoftmaxFailure:
    """Documents the original bug. These tests prove the issue exists."""

    def test_naive_overflow_produces_nan(self, overflow_scores):
        """Regression: naive formula yields NaN for d > ~710."""
        result = _naive_softmax(overflow_scores)
        assert np.any(np.isnan(result)), (
            "Expected NaN from naive softmax on large scores — "
            "if this passes the test environment may not reproduce the overflow."
        )

    def test_naive_gives_inf_before_division(self, overflow_scores):
        """Direct evidence: np.exp(800) == inf."""
        assert np.isinf(np.exp(800.0)), "np.exp(800) must be inf in IEEE-754"
        assert np.isnan(np.inf / np.inf),  "inf/inf must be NaN in IEEE-754"


# ---------------------------------------------------------------------------
# 2.  _stable_softmax: NaN-free on overflow-triggering inputs
# ---------------------------------------------------------------------------

class TestStableSoftmaxNoNaN:

    def test_no_nan_on_large_positive_scores(self, overflow_scores):
        result = _stable_softmax(overflow_scores)
        assert not np.any(np.isnan(result)), (
            f"NaN detected in stable softmax output:\n{result}"
        )

    def test_no_inf_on_large_positive_scores(self, overflow_scores):
        result = _stable_softmax(overflow_scores)
        assert not np.any(np.isinf(result)), (
            f"Inf detected in stable softmax output:\n{result}"
        )

    def test_no_nan_on_extreme_negative_scores(self, extreme_negative):
        result = _stable_softmax(extreme_negative)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_no_nan_for_identical_scores(self):
        """All equal scores → uniform distribution (no 0/0)."""
        d = np.ones((4, 3)) * 500.0
        result = _stable_softmax(d)
        assert not np.any(np.isnan(result))
        np.testing.assert_allclose(result, np.ones((4, 3)) / 3.0, rtol=1e-9)

    def test_no_nan_for_extreme_spread(self):
        """One class dominates by a huge margin — others should be ~0, not NaN."""
        d = np.array([[1000.0, -1000.0, -1000.0]])
        result = _stable_softmax(d)
        assert not np.any(np.isnan(result))
        assert result[0, 0] == pytest.approx(1.0, abs=1e-9)
        assert result[0, 1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3.  Mathematical correctness — stable == naive for normal-range scores
# ---------------------------------------------------------------------------

class TestMathematicalCorrectness:
    """
    The log-sum-exp trick is mathematically equivalent to naive softmax.
    For normal-range inputs both must agree to machine precision.
    """

    def test_matches_naive_for_normal_scores(self, normal_scores):
        stable = _stable_softmax(normal_scores)
        naive  = _naive_softmax(normal_scores)
        np.testing.assert_allclose(stable, naive, rtol=1e-12,
                                   err_msg="Stable and naive softmax disagree on normal-range scores")

    def test_matches_scipy_softmax(self, normal_scores):
        from scipy.special import softmax
        expected = softmax(normal_scores, axis=1)
        result   = _stable_softmax(normal_scores)
        np.testing.assert_allclose(result, expected, rtol=1e-12,
                                   err_msg="Does not match scipy.special.softmax")

    def test_argmax_preserved_for_overflow_scores(self, overflow_scores):
        """
        Even when absolute values overflow, the argmax must be correct.
        The largest score in each row should have probability closest to 1.
        """
        result = _stable_softmax(overflow_scores)
        expected_argmax = np.argmax(overflow_scores, axis=1)
        actual_argmax   = np.argmax(result, axis=1)
        np.testing.assert_array_equal(actual_argmax, expected_argmax,
                                      err_msg="Argmax changed after stable softmax — ordering violated")


# ---------------------------------------------------------------------------
# 4.  Probability simplex properties
# ---------------------------------------------------------------------------

class TestProbabilitySimplex:

    @pytest.mark.parametrize("fixture_name", [
        "normal_scores", "overflow_scores", "extreme_negative"
    ])
    def test_rows_sum_to_one(self, request, fixture_name):
        d = request.getfixturevalue(fixture_name)
        result = _stable_softmax(d)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=1e-9,
                                   err_msg=f"Rows do not sum to 1.0 for {fixture_name}")

    @pytest.mark.parametrize("fixture_name", [
        "normal_scores", "overflow_scores", "extreme_negative"
    ])
    def test_all_values_in_unit_interval(self, request, fixture_name):
        d = request.getfixturevalue(fixture_name)
        result = _stable_softmax(d)
        assert np.all(result >= 0.0), f"Negative probability in {fixture_name}"
        assert np.all(result <= 1.0), f"Probability > 1 in {fixture_name}"

    def test_output_shape_preserved(self, normal_scores):
        result = _stable_softmax(normal_scores)
        assert result.shape == normal_scores.shape

    def test_single_sample(self):
        d = np.array([[1.5, -0.5, 2.0]])
        result = _stable_softmax(d)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-9)

    def test_large_batch(self):
        rng = np.random.default_rng(42)
        d = rng.standard_normal((10_000, 10)) * 1000.0  # huge range
        result = _stable_softmax(d)
        assert not np.any(np.isnan(result))
        np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# 5.  Binary (1-D) decision_function handling
# ---------------------------------------------------------------------------

class TestBinaryDecisionFunction:
    """
    sklearn's decision_function for binary problems returns shape (n_samples,),
    not (n_samples, 2).  _stable_softmax must handle this correctly.
    """

    def test_1d_input_gives_2_column_output(self, binary_scores_1d):
        result = _stable_softmax(binary_scores_1d)
        assert result.shape == (5, 2), (
            f"Expected (5, 2) for 1-D binary input, got {result.shape}"
        )

    def test_1d_rows_sum_to_one(self, binary_scores_1d):
        result = _stable_softmax(binary_scores_1d)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=1e-9)

    def test_1d_no_nan_for_extreme_scores(self, binary_scores_1d):
        """Scores ±710 via sigmoid — must not produce NaN."""
        result = _stable_softmax(binary_scores_1d)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_1d_positive_score_gives_higher_class1_prob(self, binary_scores_1d):
        """Positive d → sigmoid > 0.5 → P(class=1) > P(class=0)."""
        result = _stable_softmax(binary_scores_1d)
        positive_mask = binary_scores_1d > 0
        assert np.all(result[positive_mask, 1] > 0.5), (
            "Positive decision score should give P(class=1) > 0.5"
        )

    def test_1d_zero_score_gives_equal_probs(self):
        """d=0 → sigmoid=0.5 → [0.5, 0.5]."""
        result = _stable_softmax(np.array([0.0]))
        np.testing.assert_allclose(result[0], [0.5, 0.5], rtol=1e-9)

    def test_1d_large_positive_clips_to_one(self):
        """Extreme positive score must not give inf — clipped sigmoid → ~1."""
        result = _stable_softmax(np.array([1e6]))
        assert not np.isnan(result[0, 1])
        assert result[0, 1] == pytest.approx(1.0, abs=1e-9)

    def test_1d_large_negative_clips_to_zero(self):
        """Extreme negative score must not give NaN — clipped sigmoid → ~0."""
        result = _stable_softmax(np.array([-1e6]))
        assert not np.isnan(result[0, 1])
        assert result[0, 1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 6.  Reproduce the exact issue example from the bug report
# ---------------------------------------------------------------------------

class TestBugReportReproduction:
    """Exact inputs from the bug report — must now return valid probabilities."""

    def test_bug_report_inputs_no_longer_nan(self):
        d = np.array([
            [800.0, 750.0, 720.0],
            [710.0, 680.0, 690.0],
        ])
        # Prove naive fails
        naive = _naive_softmax(d)
        assert np.any(np.isnan(naive)), "Bug not reproducible — environment differs"

        # Prove fix works
        stable = _stable_softmax(d)
        assert not np.any(np.isnan(stable)), "Fix failed — NaN still present"
        assert not np.any(np.isinf(stable)), "Fix failed — Inf still present"
        np.testing.assert_allclose(stable.sum(axis=1), 1.0, rtol=1e-9)

    def test_bug_report_argmax_correct(self):
        """Row 0 max is 800.0 (col 0); row 1 max is 710.0 (col 0)."""
        d = np.array([[800.0, 750.0, 720.0], [710.0, 680.0, 690.0]])
        result = _stable_softmax(d)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert result[1, 0] == pytest.approx(1.0, abs=1e-6)

    def test_nan_propagation_now_blocked(self):
        """
        Simulate the ensemble aggregation step:
        weights × p must not be NaN after the fix.
        """
        d = np.array([[800.0, 750.0, 720.0]])
        p = _stable_softmax(d)
        weights = np.array([0.3, 0.2, 0.5])  # per-unit ensemble weights

        # Simulate: final_pred += weight * p
        # (in predict_proba loop — just one unit here for illustration)
        final_pred = weights[0] * p
        assert not np.any(np.isnan(final_pred)), (
            "NaN propagated into ensemble aggregation — fix incomplete"
        )


# ---------------------------------------------------------------------------
# 7.  Gradient / monotonicity properties
# ---------------------------------------------------------------------------

class TestSoftmaxProperties:

    def test_highest_score_gets_highest_probability(self):
        """Softmax is order-preserving: argmax(d) == argmax(p)."""
        rng = np.random.default_rng(7)
        d = rng.standard_normal((50, 5)) * 500.0
        p = _stable_softmax(d)
        np.testing.assert_array_equal(
            np.argmax(d, axis=1), np.argmax(p, axis=1),
            err_msg="Softmax must preserve the argmax"
        )

    def test_uniform_input_uniform_output(self):
        """If all d equal, all probabilities equal 1/C."""
        C = 7
        d = np.zeros((10, C))
        p = _stable_softmax(d)
        np.testing.assert_allclose(p, np.full((10, C), 1.0 / C), rtol=1e-12)

    def test_translation_invariant(self):
        """Adding a constant to all scores must not change probabilities."""
        d = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]])
        shift = 1000.0
        p_orig    = _stable_softmax(d)
        p_shifted = _stable_softmax(d + shift)
        np.testing.assert_allclose(p_orig, p_shifted, rtol=1e-9,
                                   err_msg="Softmax must be translation-invariant")

    def test_scale_changes_sharpness(self):
        """Multiplying scores by T > 1 sharpens the distribution."""
        d = np.array([[1.0, 2.0, 3.0]])
        p_soft  = _stable_softmax(d)
        p_sharp = _stable_softmax(d * 10.0)
        # The dominant class (col 2) should have higher probability when scaled
        assert p_sharp[0, 2] > p_soft[0, 2], "Scaling must sharpen the distribution"
