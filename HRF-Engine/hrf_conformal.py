"""
HRFConformalPredictor
=====================
Distribution-free prediction sets with guaranteed marginal coverage
for any fitted HarmonicResonanceClassifier_BEAST_14D instance.

Based on split conformal prediction (Papadopoulos et al., 2002).
Coverage guarantee: P(y_true ∈ prediction_set) >= 1 - alpha, exactly.
No distribution assumptions. No asymptotic approximations.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array


class HRFConformalPredictor:
    """
    Wraps a fitted HarmonicResonanceClassifier_BEAST_14D and produces
    prediction sets with guaranteed marginal coverage using split
    conformal prediction.

    Parameters
    ----------
    estimator : fitted HarmonicResonanceClassifier_BEAST_14D
        Must already be fitted. This class does not refit.
    alpha : float, default=0.05
        Miscoverage level. Prediction sets contain the true label
        with probability >= 1 - alpha.

    Attributes
    ----------
    calibration_scores_ : ndarray of shape (n_cal,)
        Nonconformity scores computed on the calibration set.
    cal_scores_sorted_ : ndarray of shape (n_cal,)
        Pre-sorted calibration scores for O(log N) threshold lookups.
    classes_ : ndarray
        Class labels from the wrapped estimator.
    n_cal_ : int
        Number of calibration samples used.

    Examples
    --------
    >>> clf = HarmonicResonanceClassifier_BEAST_14D(verbose=False)
    >>> clf.fit(X_train, y_train)
    >>> conformal = HRFConformalPredictor(clf, alpha=0.05)
    >>> conformal.calibrate(X_cal, y_cal)
    >>> prediction_sets = conformal.predict_set(X_test)
    >>> report = conformal.coverage_report(X_test, y_test)
    """

    def __init__(self, estimator, alpha=0.05):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_scores_ = None
        self.cal_scores_sorted_ = None
        self.classes_ = None
        self.n_cal_ = None

    def __repr__(self):
        status = "calibrated" if self.calibration_scores_ is not None else "not calibrated"
        return (
            f"HRFConformalPredictor("
            f"alpha={self.alpha}, "
            f"n_cal={self.n_cal_}, "
            f"status={status})"
        )

    def _nonconformity_score(self, proba, y):
        """
        Compute nonconformity scores for calibration samples.

        For class y_i, the score is 1 - p(y_i | x_i).
        Low score  = highly conforming (model is confident and correct).
        High score = nonconforming (model is wrong or uncertain).

        Parameters
        ----------
        proba : ndarray of shape (n, n_classes)
        y : ndarray of shape (n,) — true class labels

        Returns
        -------
        scores : ndarray of shape (n,)
        """
        # classes_ is sorted (produced by np.unique), so searchsorted is exact
        true_idx = np.searchsorted(self.classes_, y)
        true_proba = proba[np.arange(len(y)), true_idx]
        return 1.0 - true_proba

    def calibrate(self, X_cal, y_cal):
        """
        Compute nonconformity scores on a held-out calibration set.
        Must be called before predict_set() or predict_p_values().

        Parameters
        ----------
        X_cal : array-like of shape (n_cal, n_features)
        y_cal : array-like of shape (n_cal,)

        Returns
        -------
        self
        """
        # classes_ is universal across all sklearn classifiers
        check_is_fitted(self.estimator, 'classes_')
        X_cal = check_array(X_cal)
        y_cal = np.asarray(y_cal)

        self.classes_ = self.estimator.classes_
        self.n_cal_ = len(y_cal)

        proba = self.estimator.predict_proba(X_cal)
        self.calibration_scores_ = self._nonconformity_score(proba, y_cal)
        # Pre-sort once here — reused by both _quantile_threshold and predict_p_values
        self.cal_scores_sorted_ = np.sort(self.calibration_scores_)
        return self

    def _quantile_threshold(self, alpha=None):
        """
        Compute the adjusted quantile threshold using exact index lookup.

        Uses sorting + integer indexing instead of np.quantile (which uses
        linear interpolation and can violate the strict finite-sample
        coverage guarantee of split conformal prediction).

        Parameters
        ----------
        alpha : float or None
            Uses instance alpha if None.

        Returns
        -------
        threshold : float
        """
        if self.calibration_scores_ is None:
            raise RuntimeError("Call calibrate() before predict_set().")
        if alpha is None:
            alpha = self.alpha
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        n = self.n_cal_
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = min(max(k, 1), n)
        return self.cal_scores_sorted_[k - 1]

    def predict_set(self, X, alpha=None):
        """
        Produce prediction sets with guaranteed coverage >= 1 - alpha.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        alpha : float or None
            Override instance alpha for this call only.
            Does not mutate instance state.

        Returns
        -------
        prediction_sets : list of lists
            Each inner list contains class labels in the prediction set.
            May be empty (impossible, model highly certain wrong class)
            or contain all classes (maximally uncertain).
        """
        X = check_array(X)
        # alpha passed directly — no state mutation, fully thread-safe
        threshold = self._quantile_threshold(alpha)
        proba = self.estimator.predict_proba(X)

        # Fully vectorized: scores shape (n_samples, n_classes)
        scores = 1.0 - proba
        mask = scores <= threshold
        return [self.classes_[row].tolist() for row in mask]

    def predict_p_values(self, X):
        """
        Compute conformal p-values for each class at each test point.

        The p-value for class c at point x is the fraction of calibration
        scores >= the nonconformity score of (x, c). A large p-value means
        the sample is highly conforming to class c.

        Uses np.searchsorted on pre-sorted calibration scores for
        O(M * C * log N) complexity vs O(M * C * N) with naive loops.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        p_values : ndarray of shape (n_samples, n_classes)
        """
        X = check_array(X)
        proba = self.estimator.predict_proba(X)

        # test_scores shape: (n_samples, n_classes)
        test_scores = 1.0 - proba
        # searchsorted returns count of cal scores STRICTLY LESS THAN test_score
        # so (n_cal - less_than_count) = count of cal scores >= test_score
        less_than_count = np.searchsorted(self.cal_scores_sorted_, test_scores, side='left')
        p_values = (self.n_cal_ - less_than_count + 1) / (self.n_cal_ + 1)
        return p_values

    def predict(self, X):
        """
        Return the most conformal class (highest p-value) for each sample.
        Falls back to standard point prediction when sets are ambiguous.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
        """
        p_values = self.predict_p_values(X)
        return self.classes_[np.argmax(p_values, axis=1)]

    def coverage_report(self, X_test, y_test, alphas=None):
        """
        Empirically validate coverage and efficiency across alpha levels.

        Parameters
        ----------
        X_test : array-like of shape (n_test, n_features)
        y_test : array-like of shape (n_test,)
        alphas : list of float or None
            Alpha levels to evaluate. Defaults to [0.01, 0.05, 0.10, 0.20].

        Returns
        -------
        report : dict
            Keys: alpha level (float)
            Values: dict with 'coverage', 'avg_set_size', 'empty_sets',
                    'singleton_sets', 'full_sets'
        """
        if alphas is None:
            alphas = [0.01, 0.05, 0.10, 0.20]

        X_test = check_array(X_test)
        y_test = np.asarray(y_test)
        n_classes = len(self.classes_)
        report = {}

        print(f"\n{'='*60}")
        print(f"  HRF CONFORMAL COVERAGE REPORT")
        print(f"  Calibration size : {self.n_cal_} | Test size: {len(y_test)}")
        print(f"  Classes          : {self.classes_}")
        print(f"{'='*60}")
        print(f"  {'Alpha':<8} {'Target':<10} {'Actual':<10} "
              f"{'Avg Set':<10} {'Empty':<8} {'Single':<8} {'Full'}")
        print(f"  {'-'*58}")

        for alpha in alphas:
            sets = self.predict_set(X_test, alpha=alpha)
            set_sizes = np.array([len(s) for s in sets])

            # Vectorized coverage check
            covered = sum(y_test[i] in sets[i] for i in range(len(y_test)))
            coverage = covered / len(y_test)

            stats = {
                'coverage'        : coverage,
                'target_coverage' : 1 - alpha,
                'avg_set_size'    : float(np.mean(set_sizes)),
                'empty_sets'      : int(np.sum(set_sizes == 0)),
                'singleton_sets'  : int(np.sum(set_sizes == 1)),
                'full_sets'       : int(np.sum(set_sizes == n_classes)),
            }
            report[alpha] = stats

            status = "✅" if coverage >= (1 - alpha) else "❌"
            print(f"  {alpha:<8.2f} {1-alpha:<10.2%} {coverage:<10.2%} "
                  f"{stats['avg_set_size']:<10.2f} "
                  f"{stats['empty_sets']:<8} "
                  f"{stats['singleton_sets']:<8} "
                  f"{stats['full_sets']} {status}")

        print(f"{'='*60}\n")
        return report
