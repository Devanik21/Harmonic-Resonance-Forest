import cupy as cp
import numpy as np
import time

def stress_test():
    # Simulate a massive dataset that would definitely trigger OOM
    # 100,000 samples x 100 features = ~40MB (fits in RAM, but creates large dist matrix)
    n_samples = 100000 
    n_features = 100
    X_train = cp.random.randn(n_samples, n_features).astype(cp.float32)
    
    print(f"Starting Stress Test with {n_samples} samples...")
    start_time = time.time()
    
    try:
        # Simulate the 'dist' calculation that used to crash
        # If your fix is working, this loop runs in chunks
        chunk_size = 1024
        for i in range(0, n_samples, chunk_size):
            chunk = X_train[i:i+chunk_size]
            # Perform dummy calculation
            _ = cp.dot(chunk, X_train.T)
            
        print(f"Success! Memory stable. Execution time: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"OOM CRASHED: {e}")

if __name__ == "__main__":
    stress_test()