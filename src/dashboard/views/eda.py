"""Exploratory Data Analysis & Demand Sparsity view for Streamlit dashboard."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard import data_loader as dl


def render() -> None:
    """Render Demand Sparsity & EDA page."""
    st.markdown("## 🔍 Demand Sparsity & Intermittency Audit")
    st.markdown(
        """
        Measured over all **30,490 series × 1,913 days** under the
        **Syntetos-Boylan-Teunter (SBT)** demand categorization framework.
        """
    )

    # Key Sparsity Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Catalog Series", "30,490", "100%")
    with col2:
        st.metric("Overall Zero Demand", "68.2%", "Daily Sales = 0")
    with col3:
        st.metric("Sparse Series (SBT)", "95.2%", "Intermittent + Lumpy")
    with col4:
        st.metric("Smooth Series", "3.2%", "Traditional Time-Series")

    st.markdown("---")

    # Interactive SBT Breakdown Chart
    st.markdown("### 📊 Syntetos-Boylan-Teunter Demand Classification")
    sbt_data = {
        "Category": ["Intermittent", "Lumpy", "Smooth", "Erratic"],
        "Pct_Series": [75.6, 19.5, 3.2, 1.6],
        "Pct_Volume": [42.6, 27.7, 19.9, 9.9],
        "Description": [
            "Low volatility, infrequent demand (ADI >= 1.32, CV^2 < 0.49)",
            "High volatility, infrequent demand (ADI >= 1.32, CV^2 >= 0.49)",
            "Low volatility, regular demand (ADI < 1.32, CV^2 < 0.49)",
            "High volatility, regular demand (ADI < 1.32, CV^2 >= 0.49)",
        ],
    }

    tab1, tab2 = st.tabs(["Chart View", "Data Table"])
    with tab1:
        fig = px.bar(
            sbt_data,
            x="Category",
            y=["Pct_Series", "Pct_Volume"],
            barmode="group",
            labels={"value": "Percentage (%)", "Category": "Demand Category", "variable": "Metric"},
            title="SBT Demand Patterns: % of Series vs % of Sales Volume",
            color_discrete_sequence=["#1f77b4", "#ff7f0e"],
        )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        import pandas as pd
        st.dataframe(pd.DataFrame(sbt_data), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🖼️ Diagnostic Exploratory Figures")
    figure_options = {
        "Fig 3.1: Product Lifecycle & Zero-Demand Spans": "reports/eda/fig_3_1_lifecycle.png",
        "Fig 3.2: Daily Demand Distributions across Stores": "reports/eda/fig_3_2_demand_distribution.png",
        "Fig 3.3: Intermittency & Lumpy Pattern Scatter": "reports/eda/fig_3_3_intermittency.png",
        "Fig 3.4: Temporal Trends & Day-of-Week Seasonality": "reports/eda/fig_3_4_temporal.png",
        "Fig 3.5: Event Spikes & SNAP Welfare Impacts": "reports/eda/fig_3_5_events_snap.png",
        "Fig 3.8: Statistical Distribution Fits": "reports/eda/fig_3_8_distributions.png",
    }

    selected_fig = st.selectbox("Select Diagnostic Figure to Inspect", list(figure_options.keys()))
    if selected_fig:
        rel_path = figure_options[selected_fig]
        img_path = dl.get_image_path(rel_path)
        if img_path:
            st.image(str(img_path), caption=selected_fig, use_container_width=True)
        else:
            st.warning(f"Image not found at {rel_path}")
