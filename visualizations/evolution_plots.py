"""
Evolution Analytics Visualization Module

Provides plots for:
- Accuracy progression
- Gamma evolution
- Frequency evolution
- Generic parameter tracking
"""

from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_accuracy_history(
    scores: Sequence[float]
) -> Figure:
    """
    Plot accuracy progression across generations.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        scores,
        linewidth=2
    )

    ax.set_title(
        "Evolution Accuracy Progression",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Accuracy")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_parameter_evolution(
    values: Sequence[float],
    parameter_name: str
) -> Figure:
    """
    Plot evolution of a parameter over generations.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        values,
        linewidth=2
    )

    ax.set_title(
        f"{parameter_name} Evolution",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel(parameter_name)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_multi_parameter_evolution(
    gamma_values: Sequence[float],
    frequency_values: Sequence[float]
) -> Figure:
    """
    Compare gamma and frequency evolution.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        gamma_values,
        label="Gamma",
        linewidth=2
    )

    ax.plot(
        frequency_values,
        label="Frequency",
        linewidth=2
    )

    ax.set_title(
        "Parameter Evolution Comparison",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Value")

    ax.legend()

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig