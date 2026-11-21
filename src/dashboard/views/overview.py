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

    findings = dl.load_eda_findings()
    structure = findings.get("3.1_structure", {})
    n_series = int(structure.get("n_series", 0))
    n_days = int(structure.get("n_days", 0))
    n_rows_m = structure.get("long_format_rows", 0) / 1e6
    n_stores = structure.get("hierarchy_cardinality", {}).get("store", 0)

    portfolio_df = dl.load_portfolio_summary()
    if not portfolio_df.empty and "capital_released_total" in portfolio_df.columns:
        freed = portfolio_df["capital_released_total"].sum()
        inv_value = f"${freed:,.0f} Released"
        # Capital released is never shown alone: next to it, the change in total annual cost.
        if "total_cost_delta_total" in portfolio_df.columns:
            inv_delta = f"{portfolio_df['total_cost_delta_total'].sum():+,.0f}/yr total annual cost"
        else:
            inv_delta = "CA_1 & CA_2 Rebalancing"
    else:
        inv_value = "Pending"
        inv_delta = "Run inventory analysis"

    # Top KPI Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Catalog Scale",
            value=f"{n_series:,} store-item series",
            delta=f"{n_stores} Walmart Stores",
            delta_color="off",
        )
    with col2:
        st.metric(
            label="Panel Observations",
            value=f"{n_rows_m:.1f}M Records",
            delta=f"{n_days:,} Days ({n_series:,} Series)",
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
              Processes the full panel via Polars out-of-core lazy execution, eliminating high memory consumption during feature generation.
            * **Multi-Tier Benchmark**:
              Benchmarks statistical, intermittent, and gradient-boosted architectures on walk-forward rolling origins with non-parametric significance testing.
            * **Differentiated ABC/XYZ Policies**:
              Segments the catalog into a 3x3 ABC/XYZ matrix and compares policies by the capital they release and the annual cost they add, never capital alone.
            """
        )

    st.markdown("---")
    st.markdown("### 🏗️ End-to-End System Architecture")

    st.markdown(
        """
        ```mermaid
        flowchart LR
            A[M5 Panel: store-item daily sales] --> B[Polars Lazy Engine]
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
