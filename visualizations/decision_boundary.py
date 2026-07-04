"""Decision Boundary Visualization for HRF

Provides visualization of decision regions and boundaries,
supporting both 2D and high-dimensional data (via PCA).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.decomposition import PCA


def plot_decision_boundary(
    model,
    X,
    y,
    resolution=300,
    figsize=(12, 8),
    cmap='viridis',
    pca_if_needed=True,
    title="HRF Decision Boundary Visualization"
):
    """
    Visualize HRF decision boundaries in 2D feature space.

    This function generates a mesh grid over the input feature space,
    computes predictions for each grid point, and visualizes the
    decision regions. For high-dimensional data, PCA is automatically
    applied to project to 2D.

    Parameters
    ----------
    model : object
        Trained HRF model with predict() method.
    X : np.ndarray
        Training data (shape: n_samples × n_features).
    y : np.ndarray
        Class labels (shape: n_samples,).
    resolution : int, optional
        Grid resolution for mesh generation. Default: 300.
    figsize : tuple, optional
        Figure size in inches. Default: (12, 8).
    cmap : str, optional
        Matplotlib colormap name. Default: 'viridis'.
    pca_if_needed : bool, optional
        Auto-reduce high-dimensional data to 2D using PCA. Default: True.
    title : str, optional
        Figure title. Default: "HRF Decision Boundary Visualization".

    Returns
    -------
    tuple
        (fig, pca) - matplotlib figure and PCA transformer (or None).

    Raises
    ------
    ValueError
        If data is not 2D and pca_if_needed=False.
    AttributeError
        If model lacks predict() method.

    Examples
    --------
    >>> fig, pca = plot_decision_boundary(model, X_train, y_train)
    >>> plt.show()
    """
    # Validate inputs
    if not hasattr(model, 'predict'):
        raise AttributeError("Model must have predict() method")

    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != len(y):
        raise ValueError("X and y must have same number of samples")

    if len(X) == 0:
        raise ValueError("X cannot be empty")

    # Get classes
    if not hasattr(model, 'classes_'):
        classes = np.unique(y)
    else:
        classes = model.classes_

    n_classes = len(classes)

    # Handle dimensionality
    pca = None
    X_display = X

    if X.shape[1] != 2:
        if not pca_if_needed:
            raise ValueError(
                f"Data has {X.shape[1]} features. "
                "Set pca_if_needed=True to auto-reduce to 2D."
            )

        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        X_display = pca.fit_transform(X)

    # Generate mesh grid
    x_min, x_max = X_display[:, 0].min() - 0.5, X_display[:, 0].max() + 0.5
    y_min, y_max = X_display[:, 1].min() - 0.5, X_display[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Prepare mesh points for prediction
    mesh_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Transform mesh points if PCA was used
    if pca is not None:
        mesh_points_original = pca.inverse_transform(mesh_points)
    else:
        mesh_points_original = mesh_points

    # Predict on mesh
    predictions = model.predict(mesh_points_original)
    predictions_grid = predictions.reshape(xx.shape)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create color map for decision regions
    region_cmap = plt.get_cmap(cmap)
    region_colors = region_cmap(np.linspace(0, 1, n_classes))

    # Plot decision regions with contourf
    contour_levels = np.arange(n_classes + 1) - 0.5
    ax.contourf(
        xx, yy, predictions_grid,
        levels=contour_levels,
        colors=region_colors,
        alpha=0.6
    )

    # Add contour lines for boundary emphasis
    ax.contour(
        xx, yy, predictions_grid,
        levels=contour_levels,
        colors='black',
        linewidths=1.5,
        linestyles='solid'
    )

    # Define markers for different classes
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H']
    marker_colors = plt.get_cmap('tab10')(
        np.linspace(0, 1, max(n_classes, 10))
    )

    # Plot sample points
    for idx, cls in enumerate(classes):
        class_mask = y == cls
        marker = markers[idx % len(markers)]
        color = marker_colors[idx]

        ax.scatter(
            X_display[class_mask, 0],
            X_display[class_mask, 1],
            c=[color],
            marker=marker,
            s=100,
            edgecolors='black',
            linewidths=1.5,
            label=f'Class {cls}',
            zorder=5,
            alpha=0.9
        )

    # Labels and title
    if pca is not None:
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
    else:
        xlabel = 'Feature 1'
        ylabel = 'Feature 2'

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor='black')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Add info box
    info_text = f"Resolution: {resolution}x{resolution}\nSamples: {len(X)}\nClasses: {n_classes}"
    if pca is not None:
        info_text += f"\nDimensionality: {X.shape[1]} -> 2D (PCA)"
    else:
        info_text += f"\nDimensionality: {X.shape[1]}D"

    ax.text(
        0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()
    return fig, pca


def plot_decision_boundary_comparison(
    models_dict,
    X,
    y,
    resolution=200,
    figsize=(16, 12),
    cmap='viridis'
):
    """
    Compare decision boundaries of multiple models side-by-side.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to trained models.
    X : np.ndarray
        Training data.
    y : np.ndarray
        Class labels.
    resolution : int, optional
        Grid resolution. Default: 200.
    figsize : tuple, optional
        Figure size. Default: (16, 12).
    cmap : str, optional
        Colormap name. Default: 'viridis'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The comparison figure.

    Examples
    --------
    >>> fig = plot_decision_boundary_comparison(
    ...     {'HRF': hrf_model, 'SVM': svm_model},
    ...     X_test, y_test
    ... )
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(
        2, (n_models + 1) // 2,
        figsize=figsize,
        squeeze=False
    )
    axes = axes.ravel()

    # Handle dimensionality once
    pca = None
    X_display = X

    if X.shape[1] != 2:
        pca = PCA(n_components=2, random_state=42)
        X_display = pca.fit_transform(X)

    # Get mesh
    x_min, x_max = X_display[:, 0].min() - 0.5, X_display[:, 0].max() + 0.5
    y_min, y_max = X_display[:, 1].min() - 0.5, X_display[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    mesh_points = np.column_stack([xx.ravel(), yy.ravel()])
    if pca is not None:
        mesh_points_original = pca.inverse_transform(mesh_points)
    else:
        mesh_points_original = mesh_points

    # Plot each model
    classes = np.unique(y)
    n_classes = len(classes)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for ax_idx, (model_name, model) in enumerate(models_dict.items()):
        ax = axes[ax_idx]

        try:
            predictions = model.predict(mesh_points_original)
            predictions_grid = predictions.reshape(xx.shape)

            # Plot regions
            region_cmap = plt.cm.get_cmap(cmap)
            region_colors = region_cmap(np.linspace(0, 1, n_classes))
            contour_levels = np.arange(n_classes + 1) - 0.5

            ax.contourf(xx, yy, predictions_grid, levels=contour_levels,
                       colors=region_colors, alpha=0.6)
            ax.contour(xx, yy, predictions_grid, levels=contour_levels,
                      colors='black', linewidths=1.5)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {type(e).__name__}', ha='center', va='center')

        # Plot samples
        marker_colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
        for idx, cls in enumerate(classes):
            class_mask = y == cls
            ax.scatter(
                X_display[class_mask, 0],
                X_display[class_mask, 1],
                c=[marker_colors[idx]],
                marker=markers[idx % len(markers)],
                s=80,
                edgecolors='black',
                linewidths=1,
                alpha=0.8
            )

        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('PC1' if pca else 'Feature 1', fontsize=10)
        ax.set_ylabel('PC2' if pca else 'Feature 2', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Hide unused subplots
    for ax_idx in range(len(models_dict), len(axes)):
        axes[ax_idx].axis('off')

    plt.suptitle('Decision Boundary Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig
