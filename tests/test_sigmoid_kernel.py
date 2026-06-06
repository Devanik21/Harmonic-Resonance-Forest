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
