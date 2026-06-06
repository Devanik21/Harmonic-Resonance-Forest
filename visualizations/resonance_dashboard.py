import streamlit as st
import pandas as pd
import numpy as np
from evolution_plots import (
    plot_accuracy_history,
    plot_parameter_evolution,
    plot_multi_parameter_evolution
)
from resonance_plots import (
    plot_class_resonance,
    plot_resonance_heatmap,
    plot_resonance_comparison,
    plot_decision_boundary
)
st.set_page_config(
    page_title="HRF Explainability Dashboard",
    page_icon="🌊",
    layout="wide"
)
# -------------------------
# Session State
# -------------------------

if "model" not in st.session_state:
    st.session_state.model = None

if "X" not in st.session_state:
    st.session_state.X = None

if "y" not in st.session_state:
    st.session_state.y = None

if st.sidebar.button("Load Demo Model"):

    from examples.demo_resonance_visualization import (
        create_demo_dataset
    )

    X_train, X_test, y_train, y_test = create_demo_dataset()

    # HRF load
    import importlib.util

    import os
    dir_path = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "generalized_hrf_v2",
        os.path.join(dir_path, "../HRF-Engine/generalized_hrf_v2.py")
    )

    hrf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hrf_module)

    ModelClass = hrf_module.HarmonicResonanceForest_Ultimate

    model = ModelClass()

    model.fit(X_train, y_train)

    st.session_state.model = model
    st.session_state.X = X_test
    st.session_state.y = y_test

    st.success("Demo HRF Model Loaded")

st.title("🌊 Harmonic Resonance Forest Explainability Dashboard")

st.markdown("""
Explore HRF predictions through:

- Class-wise Resonance Contributions
- Resonance Heatmaps
- Resonance Comparison Analysis
""")

# -------------------------
# Sidebar
# -------------------------
page = st.sidebar.selectbox(
    "Select Visualization",
    [
        "Overview",
        "Class Resonance",
        "Heatmap",
        "Resonance Comparison",
        "Decision Boundary",
        "Evolution Analytics"
    ]
)


# -------------------------
# Upload Dataset
# -------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.session_state.df = df

        st.sidebar.success(
            f"Loaded {df.shape[0]} rows"
        )

    except Exception as e:

        st.sidebar.error(str(e))

# -------------------------
# Overview
# -------------------------

if page == "Overview":

    st.header("About HRF")

    st.write("""
    Harmonic Resonance Forest (HRF) is a
    physics-informed machine learning framework.

    This dashboard provides visual explanations
    for resonance-based predictions.
    """)

    if hasattr(st.session_state, "df"):

        st.subheader("Dataset Preview")

        st.dataframe(
            st.session_state.df.head()
        )

# -------------------------
# Class Resonance
# -------------------------

elif page == "Class Resonance":

    st.header("Class-wise Resonance Contribution")

    st.info(
        "Requires a trained HRF model."
    )

    if st.session_state.model is None:

        st.warning(
            "Load a trained HRF model first."
        )

    else:

        sample_idx = st.number_input(
            "Sample Index",
            min_value=0,
            max_value=len(st.session_state.X) - 1,
            value=0
        )

        sample = (
            st.session_state.X[
                sample_idx:sample_idx+1
            ]
        )

        fig = plot_class_resonance(
            st.session_state.model,
            sample
        )

        st.pyplot(fig)

# -------------------------
# Heatmap
# -------------------------

elif page == "Heatmap":

    st.header("Resonance Intensity Heatmap")

    if st.session_state.model is None:

        st.warning(
            "Load a trained HRF model first."
        )

    else:

        n_samples = st.slider(
            "Number of Samples",
            min_value=10,
            max_value=min(
                200,
                len(st.session_state.X)
            ),
            value=50
        )

        fig = plot_resonance_heatmap(
            st.session_state.model,
            st.session_state.X[:n_samples]
        )

        st.pyplot(fig)

# -------------------------
# Resonance Comparison
# -------------------------

elif page == "Resonance Comparison":

    st.header(
        "Resonance Heatmap + Prediction Analysis"
    )

    if st.session_state.model is None:

        st.warning(
            "Load a trained HRF model first."
        )

    else:

        n_samples = st.slider(
            "Samples",
            min_value=10,
            max_value=min(
                100,
                len(st.session_state.X)
            ),
            value=30
        )

        fig = plot_resonance_comparison(
            st.session_state.model,
            st.session_state.X[:n_samples],
            st.session_state.y[:n_samples]
            if st.session_state.y is not None
            else None
        )

        st.pyplot(fig)

elif page == "Decision Boundary":

    st.header("Decision Boundary Visualization")

    if st.session_state.model is None:

        st.warning(
            "Load a trained HRF model first."
        )

    else:

        sample_limit = st.slider(
            "Samples",
            min_value=50,
            max_value=min(
                1000,
                len(st.session_state.X)
            ),
            value=min(
                300,
                len(st.session_state.X)
            )
        )

        fig = plot_decision_boundary(
            st.session_state.model,
            st.session_state.X[:sample_limit],
            st.session_state.y[:sample_limit]
        )

        st.pyplot(fig)
# -------------------------
# Evolution Analytics
# -------------------------

elif page == "Evolution Analytics":

    st.header("Evolution Optimization Analytics")

    st.write(
        "Track how HRF parameters evolve during optimization."
    )

    # Temporary mock data
    # Later replace with actual training logs

    accuracy_history = [
        0.72, 0.75, 0.77, 0.80, 0.82,
        0.84, 0.86, 0.88, 0.90, 0.92
    ]

    gamma_history = [
        0.5, 0.7, 0.9, 1.1, 1.3,
        1.5, 1.6, 1.7, 1.8, 1.9
    ]

    frequency_history = [
        2.0, 2.4, 2.8, 3.2, 3.8,
        4.1, 4.5, 5.0, 5.4, 5.8
    ]

    # ---------------------
    # Metrics
    # ---------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Accuracy",
            f"{max(accuracy_history):.2%}"
        )

    with col2:
        st.metric(
            "Best Gamma",
            f"{max(gamma_history):.2f}"
        )

    with col3:
        st.metric(
            "Best Frequency",
            f"{max(frequency_history):.2f}"
        )

    st.divider()

    # ---------------------
    # Accuracy Plot
    # ---------------------

    fig = plot_accuracy_history(
        accuracy_history
    )

    st.pyplot(fig)

    # ---------------------
    # Gamma Plot
    # ---------------------

    fig = plot_parameter_evolution(
        gamma_history,
        "Gamma"
    )

    st.pyplot(fig)

    # ---------------------
    # Frequency Plot
    # ---------------------

    fig = plot_parameter_evolution(
        frequency_history,
        "Frequency"
    )

    st.pyplot(fig)

    # ---------------------
    # Combined Plot
    # ---------------------

    fig = plot_multi_parameter_evolution(
        gamma_history,
        frequency_history
    )

    st.pyplot(fig)

st.sidebar.markdown("---")

if st.sidebar.button("Export Report"):

    report = """
HRF Explainability Dashboard Report

Features:
- Class Resonance Visualization
- Resonance Heatmap
- Resonance Comparison
- Decision Boundary Analysis
- Evolution Analytics

Generated Successfully.
"""

    with open(
        "hrf_dashboard_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    st.sidebar.success(
        "Report Exported"
    )


# -------------------------
# Footer
# -------------------------

st.sidebar.markdown("---")

st.sidebar.caption(
    "HRF Explainability Dashboard v1.0"
)