"""Statistical Significance & Hypothesis Testing view for Streamlit dashboard."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard import data_loader as dl


def render() -> None:
    """Render Statistical Significance page."""
    st.markdown("## 📊 Statistical Testing & Model Selection Rigor")
    st.markdown(
        """
        No model ranking is reported as a finding without rigorous statistical significance behind it.
        We employ non-parametric testing to account for violations of normality in multi-series retail hierarchies.
        """
    )

    # Every figure is read from significance_report.json (scripts/run_significance.py).
    report = dl.load_significance_report()
    friedman = report.get("friedman", {})
    nemenyi = report.get("nemenyi", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        stat = friedman.get("statistic")
        st.metric(
            "Friedman Test Statistic",
            f"chi-sq = {stat:,.2f}" if stat is not None else "Pending",
            f"{report.get('n_models', 0)} models x {report.get('n_series_complete', 0):,} complete series",
        )
    with col2:
        cd = nemenyi.get("critical_difference")
        st.metric(
            "Nemenyi Critical Difference",
            f"{cd:.4f}" if cd is not None else "Pending",
            "Average-rank gap needed for a real difference",
        )
    with col3:
        alpha = report.get("alpha")
        st.metric(
            "Significance Level",
            f"Alpha = {alpha}" if alpha is not None else "Pending",
            "Per-series Diebold-Mariano with Holm correction",
        )

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Nemenyi MCB Ranking", "Diebold-Mariano Tests", "Intermittent Demand Dynamics"])

    with tab1:
        st.markdown("### 🏆 Multiple Comparison with the Best (MCB)")
        mcb_df = dl.load_significance_mcb()
        if not mcb_df.empty:
            st.dataframe(mcb_df, use_container_width=True)
            rank_col = next((c for c in mcb_df.columns if "rank" in c.lower()), None)
            model_col = next((c for c in mcb_df.columns if "model" in c.lower()), None)
            if rank_col and model_col:
                fig = px.bar(
                    mcb_df.sort_values(by=rank_col),
                    x=rank_col,
                    y=model_col,
                    orientation="h",
                    title="Average Model Ranks across Series (Lower is Better)",
                    color=rank_col,
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("MCB significance table not found in reports.")

    with tab2:
        st.markdown("### 🔍 Pairwise Diebold-Mariano Tests")
        dm_df = dl.load_significance_dm()
        if not dm_df.empty:
            st.dataframe(dm_df, use_container_width=True)
        else:
            st.info("Diebold-Mariano pairwise matrix not found in reports.")

    with tab3:
        st.markdown("### 💡 Why Simple Intermittent Models Matched Global LightGBM")
        zero_rate = dl.load_eda_findings().get("3.2_demand", {}).get("overall_zero_rate")
        zero_txt = f"{zero_rate:.1%} of daily records across the full catalog are zero" if zero_rate else "Most daily records are zero"
        routing = dl.read_table("reports/backtest_main/routing_vs_best_single")
        route_txt = (
            "Routing series by SBC demand pattern did not beat applying ADIDA to every series "
            "(router versus best single model: README, section 6)."
        )
        if not routing.empty:
            pooled = routing[(routing["origin"].astype(str) == "all") & (routing["pattern"] == "ALL")]
            if len(pooled):
                r = pooled.iloc[0]
                route_txt = (
                    f"Routing series by SBC demand pattern gives {float(r['wape_routed']):.2f}% WAPE against "
                    f"{float(r['wape_adida']):.2f}% for ADIDA applied to every series "
                    f"({float(r['routed_minus_adida']):+.2f} pp, positive meaning the router is worse), so it did not beat the best single model."
                )
        st.markdown(
            f"""
            1. **The Intermittent Zero-Inflation Reality**:
               {zero_txt}. In high-sparsity panels, complex global models risk overfitting to stochastic non-zero impulses.
            2. **Temporal Aggregation Shrinkage (ADIDA)**:
               ADIDA aggregates high-frequency zero-inflated demand into lower-frequency time buckets, filters noise via exponential smoothing, and disaggregates back, acting as an effective shrinkage estimator.
            3. **Segment-Conditioned Routing (SBC)**:
               {route_txt}
            """
        )
