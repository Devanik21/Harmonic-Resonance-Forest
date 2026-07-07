import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hrf_kernels import make_sigmoid_svc
from hrf_config import ENABLE_SIGMOID_KERNEL


def test_sigmoid_svc_runs():
    if not ENABLE_SIGMOID_KERNEL:
        return
    data = load_iris()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = make_sigmoid_svc()
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert len(preds) == len(y_te)


def test_sigmoid_svc_predict_proba():
    """
    Regression test for: make_sigmoid_svc() defaulting probability=False.
    The HRF ensemble calls predict_proba() on every unit — a unit with
    probability=False raises AttributeError and breaks the entire ensemble.
    """
    if not ENABLE_SIGMOID_KERNEL:
        return
    data = load_iris()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = make_sigmoid_svc()
    clf.fit(X_tr, y_tr)

    # Must not raise AttributeError
    proba = clf.predict_proba(X_te)

    assert proba.shape == (len(y_te), len(clf.classes_)), (
        "predict_proba output shape should be (n_samples, n_classes)"
    )
    assert np.allclose(proba.sum(axis=1), 1.0), (
        "Each row of predict_proba must sum to 1.0"
    )


def test_sigmoid_svc_probability_default_is_true():
    """
    Ensure probability=True is the default so ensemble usage never breaks.
    Callers who explicitly pass probability=False override this intentionally.
    """
    if not ENABLE_SIGMOID_KERNEL:
        return
    clf = make_sigmoid_svc()
    assert clf.probability is True, (
        "make_sigmoid_svc() must default probability=True for HRF ensemble compatibility"
    )
