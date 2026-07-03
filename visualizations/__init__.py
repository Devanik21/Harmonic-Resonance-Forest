"""Resonance Visualization Module

Provides explainability tools for HRF predictions through
class-wise resonance contribution visualization, heatmaps,
and decision boundaries.
"""

from .resonance_plots import (
    plot_class_resonance,
    plot_resonance_heatmap,
    plot_resonance_comparison
)
from .resonance_utils import (
    ResonanceCaptureWrapper,
    enable_resonance_capture,
    extract_resonance_samples
)
from .decision_boundary import (
    plot_decision_boundary,
    plot_decision_boundary_comparison
)

__all__ = [
    'plot_class_resonance',
    'plot_resonance_heatmap',
    'plot_resonance_comparison',
    'ResonanceCaptureWrapper',
    'enable_resonance_capture',
    'extract_resonance_samples',
    'plot_decision_boundary',
    'plot_decision_boundary_comparison'
]
