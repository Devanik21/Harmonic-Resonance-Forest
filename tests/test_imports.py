import pytest
import sys
from pathlib import Path

@pytest.mark.unit
def test_python_version():
    """Verify Python 3.11+ is running."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info.major}.{sys.version_info.minor}"

@pytest.mark.unit
def test_requirements_installed():
    """Verify required packages are installed."""
    required_packages = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'xgboost',
        'openml',
    ]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            pytest.fail(f"Required package '{package}' is not installed")

@pytest.mark.unit
def test_hrf_code_files_exist():
    """Verify HRF code files exist in expected locations."""
    # Resolve project root relative to test file location
    project_root = Path(__file__).parent.parent

    expected_files = [
        project_root / "1/harmonic_resonance_fields_hrf (1).py",
        project_root / "HRF Codes/hrf_conference.py",
        project_root / "HRF Codes/hrf_final_v16_hrf.py",
        project_root / "HRF Codes/hrf_eeg.py",
    ]

    for file_path in expected_files:
        assert file_path.exists(), f"Expected file not found: {file_path}"

@pytest.mark.unit
def test_notebooks_exist():
    """Verify Jupyter notebooks exist in expected locations."""
    project_root = Path(__file__).parent.parent
    notebook_count = len(list(project_root.rglob("*.ipynb")))
    assert notebook_count > 0, "No Jupyter notebooks found in project"
