# Troubleshooting FAQ

This guide covers common environment setup issues encountered by contributors.

## 1. CUDA Version Mismatch
**Error:** `RuntimeError: CUDA initialization: Unexpected error`
*   **Cause:** Your installed PyTorch/CuPy version does not match your system's CUDA driver.
*   **Fix:** 
    *   Check your driver: `nvidia-smi`
    *   Ensure your environment matches:
        ```bash
        # Example for CUDA 12.1
        pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
        ```

## 2. CuPy Installation Failures
**Error:** `cupy-cuda12x not found` or `nvcc not in path`
*   **Cause:** CuPy cannot find the CUDA compiler during installation.
*   **Fix:** Ensure `nvcc` is in your system `$PATH` or install the pre-compiled wheel:
    ```bash
    pip install cupy-cuda12x
    ```

## 3. Memory Overflows on Small GPUs
**Error:** `CUDA out of memory`
*   **Cause:** The default batch size is too large for your VRAM.
*   **Fix:** Modify your local configuration or environment variable to reduce batch size:
    ```bash
    export BATCH_SIZE=4
    ```

## 4. Permission Denied (Linux/macOS)
**Error:** `EACCES: permission denied`
*   **Cause:** Attempting to install global packages or write to protected system directories.
*   **Fix:** Always use a `venv` or `conda` environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

## 5. Dependency Conflicts
**Error:** `ResolutionImpossible` or `version mismatch`
*   **Cause:** Conflicting versions of `numpy` or `scipy`.
*   **Fix:** Reinstall dependencies from the requirements file:
    ```bash
    pip install -r requirements.txt --force-reinstall
    ```