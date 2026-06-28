"""Class-wise Resonance Contribution Visualization

This module provides visualization tools to explain HRF predictions
by displaying the resonance energy contributions for each class,
heatmaps, and decision boundaries.
"""

from typing import Tuple, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from .decision_boundary import plot_decision_boundary

def plot_class_resonance(
    model: object,
    sample: Union[np.ndarray, list],
    figsize: Tuple[int, int] = (10, 6),
    color_palette: str = 'viridis'
) -> Figure:
    """
    Visualize class-wise resonance contributions for a sample.

    This function extracts class probabilities (resonance energies) from
    an HRF model's predict_proba() output and creates a bar chart showing
    the contribution of each class to the final prediction, with the
    predicted class highlighted.

    Parameters
    ----------
    model : object
        A trained HRF model with predict_proba() method.
        Must have been fitted and contain classes_ attribute.
    sample : np.ndarray or list
        A single sample or batch (shape: (1, n_features)).
        Will be converted to 2D array if needed.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default: (10, 6).
    color_palette : str, optional
        Matplotlib colormap name. Default: 'viridis'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the visualization.

    Raises
    ------
    ValueError
        If sample is empty or model lacks required attributes.
    AttributeError
        If model does not have predict_proba() or classes_ attributes.

    Examples
    --------
    >>> from visualizations.resonance_plots import plot_class_resonance
    >>> model.fit(X_train, y_train)
    >>> fig = plot_class_resonance(model, X_test[0:1])
    >>> plt.show()
    """
    # Validate inputs
    if not hasattr(model, 'predict_proba'):
        raise AttributeError("Model must have a predict_proba() method")
    if not hasattr(model, 'classes_'):
        raise AttributeError("Model must have classes_ attribute")

    # Ensure sample is 2D
    sample = np.asarray(sample)
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    if len(sample) == 0:
        raise ValueError("Sample cannot be empty")

    # Get resonance probabilities (class-wise energies)
    resonance_proba = model.predict_proba(sample)[0]  # First (only) sample
    classes = model.classes_

    # Get predicted class
    predicted_idx = np.argmax(resonance_proba)
    predicted_class = classes[predicted_idx]

    # Create color array: highlight predicted class
    colors = plt.get_cmap(color_palette)(np.linspace(0.3, 0.9, len(classes)))
    highlight_color = plt.get_cmap('Reds')(0.8)
    bar_colors = [highlight_color if i == predicted_idx else colors[i]
                  for i in range(len(classes))]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar chart
    bars = ax.bar(range(len(classes)), resonance_proba, color=bar_colors,
                   edgecolor='black', linewidth=1.5, alpha=0.85)

    # Labels and title
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Resonance Energy (Probability)', fontsize=12, fontweight='bold')
    ax.set_title('Class-wise Resonance Contribution', fontsize=14, fontweight='bold',
                 pad=20)

    # Set x-axis ticks to class labels
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([str(c) for c in classes], fontsize=11)

    # Y-axis limits and grid
    ax.set_ylim(0, max(resonance_proba) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, resonance_proba)):
        height = bar.get_height()
        label_text = f'{prob:.3f}'
        if i == predicted_idx:
            label_text = f'★ {label_text}'
            fontweight = 'bold'
        else:
            fontweight = 'normal'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                label_text, ha='center', va='bottom', fontsize=10,
                fontweight=fontweight)

    # Add legend annotation
    predicted_text = f'Predicted: Class {predicted_class} (★)'
    ax.text(0.98, 0.97, predicted_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_resonance_heatmap(
    model: object,
    X: Union[np.ndarray, list],
    resonance_map: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = 'YlOrRd',
    normalize_rows: bool = True
) -> Figure:
    """
    Visualize resonance intensity patterns across samples and classes.

    This function generates a heatmap showing how resonance energies vary
    across different samples and predicted classes. Each row represents a
    sample, each column represents a class, and color intensity indicates
    the resonance strength for that sample-class pair.

    Parameters
    ----------
    model : object
        A trained HRF model with predict_proba() and classes_ attributes.
    X : np.ndarray or list
        Batch of samples (shape: (n_samples, n_features)).
    resonance_map : np.ndarray, optional
        Pre-computed resonance matrix (shape: n_samples × n_classes).
        If not provided, will be extracted from model.last_resonance_map_
        or computed from predict_proba(). Default: None.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default: (14, 6).
    cmap : str, optional
        Matplotlib colormap name. Default: 'YlOrRd' (yellow-orange-red).
    normalize_rows : bool, optional
        If True, normalize resonance values per sample (row-wise).
        Shows relative contributions. Default: True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the heatmap.

    Raises
    ------
    ValueError
        If sample batch is empty or resonance_map shape is invalid.
    AttributeError
        If model lacks required attributes and resonance_map not provided.

    Notes
    -----
    The function attempts to obtain resonance maps in this order:
    1. Provided resonance_map parameter
    2. model.last_resonance_map_ (if available)
    3. Extracted from predict_proba() output via capture mechanism

    Examples
    --------
    >>> from visualizations.resonance_plots import plot_resonance_heatmap
    >>> model.fit(X_train, y_train)
    >>> fig = plot_resonance_heatmap(model, X_test[:50])
    >>> plt.show()

    >>> # With pre-computed resonance map
    >>> resonance = np.random.rand(100, 3)
    >>> fig = plot_resonance_heatmap(model, X_test[:100], resonance_map=resonance)
    """
    # Validate inputs
    X = np.asarray(X)
    if len(X) == 0:
        raise ValueError("Sample batch cannot be empty")

    # Get resonance matrix
    if resonance_map is None:
        # Try to extract from model
        if hasattr(model, 'last_resonance_map_'):
            resonance_map = model.last_resonance_map_
        else:
            # Capture resonance via predict_proba
            resonance_map = _capture_resonance_map(model, X)

    if resonance_map is None:
        raise ValueError(
            "Could not obtain resonance map. Provide resonance_map parameter "
            "or use model with resonance capture enabled."
        )

    resonance_map = np.asarray(resonance_map)
    if resonance_map.shape[0] != len(X):
        raise ValueError(
            f"Resonance map shape {resonance_map.shape} does not match "
            f"sample batch size {len(X)}"
        )

    # Get class labels
    if not hasattr(model, 'classes_'):
        raise AttributeError("Model must have classes_ attribute")
    classes = model.classes_

    if resonance_map.shape[1] != len(classes):
        raise ValueError(
            f"Resonance map has {resonance_map.shape[1]} classes "
            f"but model has {len(classes)} classes"
        )

    # Normalize if requested
    if normalize_rows:
        row_sums = np.sum(resonance_map, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        resonance_normalized = resonance_map / row_sums
    else:
        resonance_normalized = resonance_map

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Generate heatmap
    im = ax.imshow(resonance_normalized, cmap=cmap, aspect='auto',
                   interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Resonance Intensity')

    # Labels
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_title('Resonance Intensity Heatmap', fontsize=14, fontweight='bold',
                 pad=20)

    # Set ticks
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([str(c) for c in classes], fontsize=11)

    # Y-axis: show every nth sample for readability
    n_samples = len(X)
    y_tick_step = max(1, n_samples // 15)  # ~15 ticks max
    y_ticks = range(0, n_samples, y_tick_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(i) for i in y_ticks], fontsize=9)

    # Add grid for better readability
    ax.set_xticks([x - 0.5 for x in range(len(classes) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(n_samples + 1)], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Normalization note
    norm_text = "Row-wise normalized" if normalize_rows else "Raw resonance energies"
    ax.text(0.98, 0.02, norm_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def _capture_resonance_map(model: object, X: np.ndarray) -> Optional[np.ndarray]:
    """
    Capture resonance map by running predict_proba and extracting metadata.

    This helper function attempts to extract resonance intensities from
    the model using multiple strategies, ensuring compatibility with
    different HRF model variants.

    Parameters
    ----------
    model : object
        HRF model instance.
    X : np.ndarray
        Input samples.

    Returns
    -------
    resonance_map : np.ndarray or None
        Resonance matrix (n_samples × n_classes) if available, else None.
    """
    # First, run predict_proba to potentially trigger resonance capture
    probas = model.predict_proba(X)

    # Check if model stored resonance during prediction
    if hasattr(model, 'last_resonance_map_'):
        return model.last_resonance_map_

    # Fallback: use probabilities as resonance proxy
    # This is not ideal but allows visualization to work
    return probas


def plot_resonance_comparison(
    model: object,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    sample_indices: Optional[list] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> Figure:
    """
    Create a comparison heatmap with ground truth labels.

    This advanced visualization shows resonance patterns alongside
    true and predicted class labels for error analysis.

    Parameters
    ----------
    model : object
        Trained HRF model.
    X : np.ndarray
        Input samples (shape: n_samples × n_features).
    y : np.ndarray, optional
        True labels (shape: n_samples,). If provided, included in visualization.
    sample_indices : list, optional
        Specific sample indices to visualize. If None, uses all samples.
    figsize : tuple, optional
        Figure size. Default: (16, 10).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> fig = plot_resonance_comparison(model, X_test, y_test)
    >>> plt.show()
    """
    X = np.asarray(X)
    if sample_indices is not None:
        X = X[sample_indices]
        if y is not None:
            y = y[sample_indices]

    # Get resonance and predictions
    resonance_map = _capture_resonance_map(model, X)
    if resonance_map is None:
        resonance_map = model.predict_proba(X)

    predictions = model.predict(X)
    classes = model.classes_

    # Normalize resonance
    resonance_normalized = resonance_map / (np.sum(resonance_map, axis=1, keepdims=True) + 1e-10)

    # Create figure with subplots
    if y is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [20, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    # Main heatmap
    im = ax1.imshow(resonance_normalized, cmap='YlOrRd', aspect='auto',
                    interpolation='nearest')
    plt.colorbar(im, ax=ax1, label='Resonance Intensity')

    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sample Index', fontsize=12, fontweight='bold')
    ax1.set_title('Resonance Heatmap with Predictions', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels([str(c) for c in classes], fontsize=11)

    # Predictions bar
    if ax2 is not None:
        pred_colors = plt.cm.Set3(predictions / max(predictions.max(), len(classes) - 1))
        ax2.bar(range(len(predictions)), [1] * len(predictions), color=pred_colors, width=1.0)
        ax2.set_ylabel('Pred', fontsize=10, fontweight='bold')
        ax2.set_xlim(-0.5, len(predictions) - 0.5)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])

        # Add true labels if available
        if y is not None:
            for i, (true, pred) in enumerate(zip(y, predictions)):
                match = "✓" if true == pred else "✗"
                color = 'green' if true == pred else 'red'
                ax2.text(i, 0.5, match, ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)

    plt.tight_layout()
    return fig
