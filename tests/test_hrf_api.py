"""
tests/test_hrf_api.py

Unit tests for HarmonicResonanceClassifier_v16 sklearn API compliance.

The source file (hrf_final_v16_hrf.py) is a converted Colab notebook with
hundreds of lines of top-level executable code and GPU-only dependencies
(cupy, cuml). This test module:

  1. Mocks cupy / cuml so all GPU calls run on CPU via numpy/sklearn.
  2. Uses AST parsing to dynamically locate and extract only the
     HarmonicResonanceClassifier_v16 class from the source file —
     avoiding the notebook's top-level training/plotting code entirely.

Run from repo root:
    pytest tests/test_hrf_api.py -v
"""

import sys
import os
import ast
import types
import importlib.util
import inspect
import textwrap

import numpy as np
import pytest
from unittest.mock import MagicMock
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification

# ---------------------------------------------------------------------------
# 1. GPU MOCKS  (must happen before any HRF code is imported)
# ---------------------------------------------------------------------------

class _FakeCupy(types.ModuleType):
    """Thin numpy-backed cupy shim."""
    def __init__(self):
        super().__init__("cupy")

    def asarray(self, x):            return np.asarray(x)
    def asnumpy(self, x):            return np.asarray(x)
    def zeros(self, shape, **kw):    return np.zeros(shape)
    def argmax(self, x, axis=None):  return np.argmax(x, axis=axis)
    def sum(self, x, **kw):          return np.sum(x, **kw)
    def exp(self, x):                return np.exp(x)
    def cos(self, x):                return np.cos(x)
    def clip(self, x, a, b):         return np.clip(x, a, b)
    ndarray = np.ndarray

    def __getattr__(self, name):
        try:
            return getattr(np, name)
        except AttributeError:
            return MagicMock()


class _FakeCuNN:
    """CPU NearestNeighbors with cuml's interface."""
    def __init__(self, n_neighbors=5, **kw):
        from sklearn.neighbors import NearestNeighbors
        self._nn = NearestNeighbors(n_neighbors=n_neighbors)

    def fit(self, X):
        self._nn.fit(np.asarray(X))
        return self

    def kneighbors(self, X):
        return self._nn.kneighbors(np.asarray(X))


class _FakeCumlNeighbors(types.ModuleType):
    def __init__(self):
        super().__init__("cuml.neighbors")
    NearestNeighbors = _FakeCuNN


class _FakeCuml(types.ModuleType):
    def __init__(self):
        super().__init__("cuml")
    neighbors = _FakeCumlNeighbors()


_cp   = _FakeCupy()
_cuml = _FakeCuml()

sys.modules["cupy"]                         = _cp
sys.modules["cuml"]                         = _cuml
sys.modules["cuml.neighbors"]               = _cuml.neighbors
sys.modules.setdefault("openml",            MagicMock())
sys.modules.setdefault("matplotlib",        MagicMock())
sys.modules.setdefault("matplotlib.pyplot", MagicMock())

# ---------------------------------------------------------------------------
# 2. EXTRACT & LOAD HarmonicResonanceClassifier_v16 via AST
# ---------------------------------------------------------------------------
_CLASS_NAME = "HarmonicResonanceClassifier_v16"
_REL_PATH   = os.path.join("HRF Codes", "hrf_final_v16_hrf.py")

_MINIMAL_IMPORTS = textwrap.dedent("""
import sys, numpy as np, cupy as cp
from cuml.neighbors import NearestNeighbors as cuNN
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
""")


def _load_hrf_class():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fpath = os.path.join(repo_root, _REL_PATH)

    if not os.path.exists(fpath):
        pytest.skip(
            f"Source not found: {fpath}. Run pytest from the repo root.",
            allow_module_level=True,
        )

    with open(fpath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
        class_node = next(
            (node for node in ast.walk(tree)
             if isinstance(node, ast.ClassDef) and node.name == _CLASS_NAME),
            None
        )
    except Exception as e:
        pytest.skip(f"Failed to parse source file: {e}", allow_module_level=True)

    if class_node is None:
        pytest.skip(f"'{_CLASS_NAME}' not found in source file.", allow_module_level=True)

    if hasattr(ast, "get_source_segment"):
        class_src = ast.get_source_segment(source, class_node)
    else:
        lines = source.splitlines()
        end_line = getattr(class_node, "end_lineno", len(lines))
        class_src = "\n".join(lines[class_node.lineno - 1 : end_line])

    synthetic_src = _MINIMAL_IMPORTS + "\n" + class_src

    mod_name = "hrf_v16_synthetic"
    mod = types.ModuleType(mod_name)
    mod.__file__ = fpath
    sys.modules[mod_name] = mod

    exec(compile(synthetic_src, fpath, "exec"), mod.__dict__)

    if not hasattr(mod, _CLASS_NAME):
        pytest.skip(
            f"'{_CLASS_NAME}' not found after extraction.",
            allow_module_level=True,
        )
    return getattr(mod, _CLASS_NAME)


HRF = _load_hrf_class()

# ---------------------------------------------------------------------------
# 3. FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_data():
    """100-sample binary dataset with 14 features (EEG-like dimensionality)."""
    X, y = make_classification(
        n_samples=100, n_features=14, n_informative=8,
        n_redundant=2, random_state=42
    )
    return X.astype(np.float64), y.astype(int)


@pytest.fixture(scope="module")
def multiclass_data():
    X, y = make_classification(
        n_samples=150, n_features=14, n_informative=8,
        n_classes=3, n_clusters_per_class=1, random_state=7
    )
    return X.astype(np.float64), y.astype(int)


@pytest.fixture(scope="module")
def fitted_model(binary_data):
    X, y = binary_data
    model = HRF(auto_evolve=False)
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# 4. TESTS
# ---------------------------------------------------------------------------

class TestInheritance:
    def test_is_base_estimator(self):
        assert issubclass(HRF, BaseEstimator)

    def test_is_classifier_mixin(self):
        assert issubclass(HRF, ClassifierMixin)


class TestFit:
    def test_fit_returns_self(self, binary_data):
        X, y = binary_data
        model = HRF(auto_evolve=False)
        assert model.fit(X, y) is model

    def test_fit_sets_classes_(self, fitted_model):
        model, _, y = fitted_model
        assert hasattr(model, "classes_")
        np.testing.assert_array_equal(np.sort(model.classes_), np.unique(y))

    def test_fit_accepts_float32(self, binary_data):
        X, y = binary_data
        HRF(auto_evolve=False).fit(X.astype(np.float32), y)

    def test_fit_sets_X_train_(self, fitted_model):
        model, _, _ = fitted_model
        assert hasattr(model, "X_train_")

    def test_fit_sets_y_train_(self, fitted_model):
        model, _, _ = fitted_model
        assert hasattr(model, "y_train_")


class TestPredict:
    def test_predict_shape(self, fitted_model):
        model, X, _ = fitted_model
        assert model.predict(X).shape == (X.shape[0],)

    def test_predict_known_classes(self, fitted_model):
        model, X, _ = fitted_model
        assert set(model.predict(X)).issubset(set(model.classes_))

    def test_predict_single_sample(self, fitted_model):
        model, X, _ = fitted_model
        assert model.predict(X[:1]).shape == (1,)

    def test_predict_dtype(self, fitted_model):
        model, X, _ = fitted_model
        assert model.predict(X).dtype.kind in ("i", "u", "O")


class TestScore:
    def test_score_is_float(self, fitted_model):
        model, X, y = fitted_model
        assert isinstance(model.score(X, y), float)

    def test_score_in_range(self, fitted_model):
        model, X, y = fitted_model
        s = model.score(X, y)
        assert 0.0 <= s <= 1.0

    def test_score_beats_random(self, fitted_model):
        model, X, y = fitted_model
        assert model.score(X, y) > 1.0 / len(np.unique(y))


class TestParams:
    def test_get_params_is_dict(self):
        assert isinstance(HRF().get_params(), dict)

    def test_auto_evolve_in_params(self):
        assert "auto_evolve" in HRF().get_params()

    def test_set_params_round_trip(self):
        m = HRF(auto_evolve=True)
        m.set_params(auto_evolve=False)
        assert m.get_params()["auto_evolve"] is False

    def test_clone_is_unfitted(self, fitted_model):
        model, _, _ = fitted_model
        assert not hasattr(clone(model), "classes_")


class TestRobustness:
    def test_refit_no_error(self, binary_data):
        X, y = binary_data
        model = HRF(auto_evolve=False)
        model.fit(X, y)
        model.fit(X, np.random.RandomState(99).permutation(y))

    def test_multiclass(self, multiclass_data):
        X, y = multiclass_data
        model = HRF(auto_evolve=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        assert set(preds).issubset({0, 1, 2})

    def test_input_not_mutated(self, binary_data):
        X, y = binary_data
        X_copy = X.copy()
        model = HRF(auto_evolve=False)
        model.fit(X, y)
        model.predict(X)
        np.testing.assert_array_equal(X, X_copy)