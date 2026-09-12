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
            label="Transaction Data",
            value="58.3M Records",
            delta="1,913 Days (5 Yrs)",
            delta_color="off",
        )
    with col3:
        st.metric(
            label="Forecast Accuracy",
            value="0.812 RMSSE",
            delta="-11% vs AutoETS Baseline",
            delta_color="normal",
        )
    with col4:
        st.metric(
            label="Inventory Optimization",
            value="32% Buffer Cut",
            delta="$3,223 Capital Freed",
            delta_color="normal",
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
              Processes 58.3M rows in 113s at 10.7GB RAM via Polars out-of-core lazy execution, eliminating 17GB Pandas OOM crashes.
            * **Multi-Tier Model Zoo (15 Models)**:
              Benchmarks statistical, intermittent, gradient boosted, and deep learning architectures with nested TimeSeriesSplit validation.
            * **Differentiated ABC/XYZ Policies**:
              Segments catalog into a 3x3 matrix: ensures 98% availability on high-revenue staples (AX) while dialing back erratic, slow-moving items (CZ) to 90%, freeing capital with zero revenue loss.
            """
        )

    st.markdown("---")
    st.markdown("### 🏗️ End-to-End System Architecture")

    st.markdown(
        """
        ```mermaid
        flowchart LR
            A[M5 Data: 58.3M Rows] --> B[Polars Lazy Engine]
            B --> C[15 Forecasting Models]
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
