"""Exploratory Data Analysis & Demand Sparsity view for Streamlit dashboard."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard import data_loader as dl


def render() -> None:
    """Render Demand Sparsity & EDA page."""
    st.markdown("## 🔍 Demand Sparsity & Intermittency Audit")

    # Every figure below is read from reports/eda/ (scripts/pipeline/run_eda.py), never typed here.
    findings = dl.load_eda_findings()
    sbc = dl.read_table("reports/eda/table_sbc_classification")
    if not findings or sbc.empty:
        st.info("EDA artifacts not found. Run `python scripts/pipeline/run_eda.py`.")
        return
    structure = findings.get("3.1_structure", {})
    demand = findings.get("3.2_demand", {})
    mix = findings.get("3.3_intermittency", {})
    st.markdown(
        f"""
        Measured over all **{structure.get("n_series", 0):,} series × {structure.get("n_days", 0):,} days** under the
        **Syntetos-Boylan-Croston (SBC)** demand categorization framework (full history, pre-launch zeros trimmed).
        """
    )

    # Key Sparsity Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Catalog Series", f"{structure.get('n_series', 0):,}")
    with col2:
        st.metric("Overall Zero Demand", f"{demand.get('overall_zero_rate', 0.0):.1%}", "Daily Sales = 0")
    with col3:
        st.metric("Sparse Series (SBC)", f"{mix.get('intermittent_or_lumpy_pct', 0.0):.1f}%", "Intermittent + Lumpy")
    with col4:
        st.metric("Smooth Series", f"{mix.get('pct_of_series', {}).get('SMOOTH', 0.0):.1f}%", "Traditional Time-Series")

    st.markdown("---")

    # Interactive SBC Breakdown Chart
    st.markdown("### 📊 Syntetos-Boylan-Croston Demand Classification")
    descriptions = {
        "INTERMITTENT": "Low volatility, infrequent demand (ADI >= 1.32, CV^2 < 0.49)",
        "LUMPY": "High volatility, infrequent demand (ADI >= 1.32, CV^2 >= 0.49)",
        "SMOOTH": "Low volatility, regular demand (ADI < 1.32, CV^2 < 0.49)",
        "ERRATIC": "High volatility, regular demand (ADI < 1.32, CV^2 >= 0.49)",
    }
    sbc_data = {
        "Category": [str(p).title() for p in sbc["pattern"]],
        "Pct_Series": [float(x) for x in sbc["pct_of_series"]],
        "Pct_Volume": [float(x) for x in sbc["pct_of_volume"]],
        "Description": [descriptions.get(str(p), "") for p in sbc["pattern"]],
    }

    tab1, tab2 = st.tabs(["Chart View", "Data Table"])
    with tab1:
        fig = px.bar(
            sbc_data,
            x="Category",
            y=["Pct_Series", "Pct_Volume"],
            barmode="group",
            labels={"value": "Percentage (%)", "Category": "Demand Category", "variable": "Metric"},
            title="SBC Demand Patterns: % of Series vs % of Sales Volume",
            color_discrete_sequence=["#1f77b4", "#ff7f0e"],
        )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        import pandas as pd
        st.dataframe(pd.DataFrame(sbc_data), use_container_width=True)

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
