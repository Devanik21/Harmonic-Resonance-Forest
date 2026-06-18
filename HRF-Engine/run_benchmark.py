import numpy as np
import time
from sklearn.preprocessing import LabelEncoder


try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not found or no GPU detected. Benchmark will run in CPU mode (No VRAM tracking).")
from generalized_hrf_v2 import HolographicSoulUnit 

print("Preparing data...")
# Generate dummy data to test the memory limits
np.random.seed(42)
X_train = np.random.rand(5000, 100).astype(np.float32) # 5000 samples, 100 features
y_train = np.random.randint(0, 2, 5000)
X_test = np.random.rand(2000, 100).astype(np.float32)  

print("Fitting model...")
model = HolographicSoulUnit(k=15, seed=42)
model.fit(X_train, y_train)

print("Starting memory benchmark...")
import time

# 2. Run the Prediction (Timing it instead of tracking VRAM for CPU)
start_time = time.time()
# Assuming 'X_test' is defined in your benchmark script
predictions = model.predict_proba(X_test) 
end_time = time.time()

print("-" * 30)
print(f"Time Taken: {end_time - start_time:.2f} seconds")

# 3. Only track VRAM if CuPy successfully loaded
if GPU_AVAILABLE:
    final_mem = cp.get_default_memory_pool().used_bytes()
    print(f"Peak VRAM used: {(final_mem - initial_mem) / (1024**2):.2f} MB")
else:
    print("Peak VRAM used: N/A (Running on CPU)")
print("-" * 30)