import numpy as np
import pytest
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from generalized_hrf_v2 import HolographicSoulUnit

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy is not installed or GPU unavailable")
def test_gpu_predict_matches_cpu():
    # 1. Generate synthetic dataset
    np.random.seed(42)
    X_train = np.random.rand(1000, 50).astype(np.float32)
    y_train = np.random.randint(0, 2, 1000)
    X_test = np.random.rand(500, 50).astype(np.float32)

    # 2. Initialize and Fit
    model = HolographicSoulUnit(k=15, seed=42)
    model.fit(X_train, y_train)

    # 3. Test Euclidean (L2) - Matrix Optimization
    model.dna_['p'] = 2.0
    cpu_pred_l2 = model._predict_proba_cpu(X_test)
    gpu_pred_l2 = model._predict_proba_gpu(X_test)
    
    np.testing.assert_allclose(
        cpu_pred_l2,
        gpu_pred_l2,
        rtol=1e-4,
        atol=1e-4,
        err_msg="GPU and CPU predictions diverge for L2 (Euclidean) distance."
    )

    # 4. Test Non-Euclidean (L1) - Chunking Optimization
    model.dna_['p'] = 1.0
    cpu_pred_l1 = model._predict_proba_cpu(X_test)
    gpu_pred_l1 = model._predict_proba_gpu(X_test)

    np.testing.assert_allclose(
        cpu_pred_l1,
        gpu_pred_l1,
        rtol=1e-4,
        atol=1e-4,
        err_msg="GPU and CPU predictions diverge for L1 (Manhattan) distance."
    )
    
    print("All VRAM-optimized GPU predictions match CPU baselines.")

if __name__ == "__main__":
    if GPU_AVAILABLE:
        test_gpu_predict_matches_cpu()