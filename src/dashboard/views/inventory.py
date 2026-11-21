"""Inventory Optimization & Working Capital Decision Lab for Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import data_loader as dl


def render() -> None:
    """Render Inventory Optimization page."""
    st.markdown("## 📦 Inventory Optimization & Working Capital Decision Lab")
    st.markdown(
        """
        Converts raw demand forecasts into **actionable inventory decisions**: which SKUs to stock more,
        which to cut, and how much working capital each policy releases.
        """
    )

    # Top KPI Metrics from portfolio summary
    port_df = dl.load_portfolio_summary()
    if not port_df.empty:
        try:
            curr_cap = float(port_df["current_wc_total"].sum())
            prop_cap = float(port_df["recommended_wc_total"].sum())
            freed_cap = float(port_df["capital_released_total"].sum())
            pct_freed = (freed_cap / curr_cap * 100.0) if curr_cap > 0 else 0.0
            # Capital released is never shown alone: next to it, the change in total annual cost.
            has_cost = "total_cost_delta_total" in port_df.columns
            cost_chg = float(port_df["total_cost_delta_total"].sum()) if has_cost else float("nan")
            released_delta = (
                f"{cost_chg:+,.0f}/yr total annual cost" if has_cost else f"{pct_freed:.1f}% of working capital"
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Baseline Working Capital", f"${curr_cap:,.2f}")
            with col2:
                st.metric("Optimized Working Capital", f"${prop_cap:,.2f}")
            with col3:
                st.metric(
                    "Released Working Capital", f"${freed_cap:,.2f}", delta=released_delta, delta_color="inverse"
                )
            with col4:
                st.metric("Cycle Service Level (CSL)", "98% (AX) / 90% (CZ)", "Capital-release rule")
        except Exception as e:
            st.warning(f"Unable to render portfolio KPIs: {e}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["3x3 ABC/XYZ Matrix", "Policy Comparison", "SKU Policy Explorer"])

    with tab1:
        st.markdown("### 🎯 3x3 ABC/XYZ Classification Matrix")
        st.markdown(
            """
            * **ABC (Revenue Pareto)**: **Class A** (Top 80% revenue), **Class B** (Next 15%), **Class C** (Bottom 5%).
            * **XYZ (Demand Variability)**: **Class X** (CV <= 0.50, predictable), **Class Y** (0.50 < CV <= 1.0, moderate), **Class Z** (CV > 1.0, erratic), **Class N** (no sales in the window: CV undefined, held at the current service level and flagged for review).
            """
        )

        # SKU counts per ABC x XYZ cell come from the classification table (the segment summary is
        # keyed by the combined class and by service level, so it cannot be pivoted into a 3x3).
        abc_df = dl.load_abc_xyz_classification()
        if not abc_df.empty and {"abc_class", "xyz_class"} <= set(abc_df.columns):
            pivot_count = pd.crosstab(abc_df["abc_class"].fillna("none"), abc_df["xyz_class"])
            fig = px.imshow(
                pivot_count,
                labels={"x": "XYZ (Demand Variability)", "y": "ABC (Revenue Importance)", "color": "SKU Count"},
                x=pivot_count.columns,
                y=pivot_count.index,
                text_auto=True,
                title="SKU Count Distribution across ABC/XYZ Matrix",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

        seg_df = dl.load_segment_summary()
        if not seg_df.empty:
            st.dataframe(seg_df, use_container_width=True)

        img_path = dl.get_image_path("reports/inventory/abc_xyz_heatmap.png")
        if img_path:
            st.image(str(img_path), caption="ABC/XYZ Inventory Matrix & Strategy", use_container_width=True)

    with tab2:
        st.markdown("### ⚖️ Policy Comparison & Margin Sensitivity")
        st.markdown(
            """
            Capital released is only meaningful next to what it costs. Both tables are read from `reports/inventory/`
            (`policy_comparison.csv`, `margin_sensitivity.csv`, `margin_break_even.csv`); nothing here is typed in.
            """
        )
        policy_df = dl.read_table("reports/inventory/policy_comparison")
        if not policy_df.empty:
            st.dataframe(policy_df, use_container_width=True)
        else:
            st.info("Policy comparison not found. Run `python scripts/run_inventory_analysis.py`.")
        sens_df = dl.read_table("reports/inventory/margin_sensitivity")
        be_df = dl.read_table("reports/inventory/margin_break_even")
        if not sens_df.empty:
            st.markdown("#### Does the answer survive a different gross margin?")
            st.dataframe(sens_df, use_container_width=True)
        if not be_df.empty:
            st.dataframe(be_df, use_container_width=True)

    with tab3:
        st.markdown("### 🔎 SKU Policy & Reorder Point Explorer")
        policies_df = dl.load_policy_recommendations()
        if not policies_df.empty:
            st.dataframe(policies_df, use_container_width=True)
        else:
            sku_df = dl.load_abc_xyz_classification()
            if not sku_df.empty:
                st.dataframe(sku_df.head(100), use_container_width=True)
            else:
                st.info("SKU classification data not found.")
