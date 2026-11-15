"""Executive Overview view for Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from src.dashboard import data_loader as dl


def render() -> None:
    """Render Executive Overview page."""
    st.markdown("## 📊 Executive Overview & Platform KPIs")
    st.markdown(
        """
        **M5 Retail Demand & Inventory Intelligence Platform** bridges the gap between
        statistical/machine learning forecasts and **actionable inventory decisions**—solving
        the fundamental retail trade-off: **Stockout Risk vs. Working Capital**.
        """
    )

    # Dynamic KPI calculation from artifacts
    acc_df = dl.load_accuracy_overall()
    if not acc_df.empty:
        best_row = acc_df.sort_values(by="wape").iloc[0]
        acc_value = f"{best_row['wape']:.1f}% WAPE"
        acc_delta = f"{best_row['model']} (RMSSE: {best_row['rmsse']:.3f})"
    else:
        acc_value = "Pending"
        acc_delta = "Run backtest"

    portfolio_df = dl.load_portfolio_summary()
    if not portfolio_df.empty and "capital_released_total" in portfolio_df.columns:
        freed = portfolio_df["capital_released_total"].sum()
        inv_value = f"${freed:,.0f} Freed"
        inv_delta = "CA_1 & CA_2 Rebalancing"
    else:
        inv_value = "Pending"
        inv_delta = "Run inventory analysis"

    # Top KPI Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Catalog Scale",
            value="30,490 SKUs",
            delta="10 Walmart Stores",
            delta_color="off",
        )
    with col2:
        st.metric(
            label="Panel Observations",
            value="58.3M Records",
            delta="1,913 Days (30,490 Series)",
            delta_color="off",
        )
    with col3:
        st.metric(
            label="Champion Forecast",
            value=acc_value,
            delta=acc_delta,
            delta_color="off",
        )
    with col4:
        st.metric(
            label="Working Capital Impact",
            value=inv_value,
            delta=inv_delta,
            delta_color="off",
        )

    st.markdown("---")

    # Business Problem & Operational Solution
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("### ⚠️ The Core Retail Trade-Off")
        st.info(
            """
            * **Stockout Risk (Too Little Inventory)**:
              Empty shelves lose instant sales, damage customer loyalty, and surrender market share to competitors.
            * **Excess Inventory (Too Much Inventory)**:
              Over-ordering ties up critical working capital, inflates warehouse storage fees, and causes margin erosion from markdowns and obsolescence.
            * **The Operational Decision**:
              Not just *"predict sales"*, but *"which products/stores should receive additional inventory, which should have inventory reduced, and how much working capital can be safely released?"*
            """
        )
    with right_col:
        st.markdown("### 💡 The Platform Solution")
        st.success(
            """
            * **High-Throughput Streaming Engine**:
              Processes 58.3M rows via Polars out-of-core lazy execution, eliminating high memory consumption during feature generation.
            * **Multi-Tier Benchmark**:
              Benchmarks statistical, intermittent, and gradient-boosted architectures on walk-forward rolling origins with non-parametric significance testing.
            * **Differentiated ABC/XYZ Policies**:
              Segments catalog into a 3x3 matrix: ensures higher availability on high-revenue staples (AX) while dialing back erratic, slow-moving items (CZ), freeing capital with controlled stockout risk.
            """
        )

    st.markdown("---")
    st.markdown("### 🏗️ End-to-End System Architecture")

    st.markdown(
        """
        ```mermaid
        flowchart LR
            A[M5 Panel: 58.3M Rows] --> B[Polars Lazy Engine]
            B --> C[Forecasting Model Benchmark]
            C --> D[Friedman & Nemenyi Tests]
            D --> E[ABC/XYZ Segmentation]
            E --> F[Safety Stock & Capital Optimizer]
            F --> G[Differentiated Stocking Policies]
        ```
        """
    )

    # Portfolio Summary Table if available
    portfolio_df = dl.load_portfolio_summary()
    if not portfolio_df.empty:
        st.markdown("### 📋 Executive Portfolio Summary")
        st.dataframe(portfolio_df, use_container_width=True)
