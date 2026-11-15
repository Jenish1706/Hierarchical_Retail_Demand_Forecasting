"""Model Leaderboard & Time Series Viewer for Streamlit dashboard."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import data_loader as dl


def render() -> None:
    """Render Model Leaderboard & Forecast page."""
    st.markdown("## 🤖 Model Zoo Leaderboard & Series Forecast Viewer")
    st.markdown(
        """
        Benchmarking forecasting models across statistical, intermittent, and gradient-boosted
        tiers using walk-forward rolling-origin evaluation and standardized accuracy metrics.
        """
    )

    # Leaderboard Table
    overall_df = dl.load_accuracy_overall()
    if not overall_df.empty:
        st.markdown("### 🏆 Overall Leaderboard")
        st.dataframe(overall_df, use_container_width=True)

        # Plot RMSSE comparison if column exists
        rmsse_col = next((c for c in overall_df.columns if "rmsse" in c.lower()), None)
        model_col = next((c for c in overall_df.columns if "model" in c.lower() or "name" in c.lower()), None)

        if rmsse_col and model_col:
            sorted_df = overall_df.sort_values(by=rmsse_col, ascending=True)
            fig = px.bar(
                sorted_df,
                x=model_col,
                y=rmsse_col,
                title="Model Accuracy (RMSSE — Lower is Better)",
                color=rmsse_col,
                color_continuous_scale="Blues_r",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Interactive Series Forecast Viewer")

    pred_sample = dl.load_series_predictions_sample(n_series=30)
    if not pred_sample.empty:
        series_col = next((c for c in pred_sample.columns if "series" in c.lower() or "id" in c.lower()), None)
        date_col = next((c for c in pred_sample.columns if "date" in c.lower() or "ds" in c.lower() or "d" in c.lower()), None)
        y_col = next((c for c in pred_sample.columns if "y_true" in c.lower() or "actual" in c.lower() or "sales" in c.lower()), None)

        if series_col:
            unique_series = sorted(pred_sample[series_col].dropna().unique())
            selected_series = st.selectbox("Select Series to Visualize", unique_series)

            subset = pred_sample[pred_sample[series_col] == selected_series].copy()
            if date_col:
                subset = subset.sort_values(by=date_col)

            fig = go.Figure()
            if y_col and date_col:
                fig.add_trace(
                    go.Scatter(
                        x=subset[date_col],
                        y=subset[y_col],
                        mode="lines+markers",
                        name="Actual Sales",
                        line={"color": "black", "width": 2},
                    )
                )

            # Add model prediction columns
            pred_cols = [c for c in subset.columns if c not in [series_col, date_col, y_col] and subset[c].dtype in ["float64", "float32", "int64"]]
            for pc in pred_cols[:5]:
                fig.add_trace(
                    go.Scatter(
                        x=subset[date_col] if date_col else list(range(len(subset))),
                        y=subset[pc],
                        mode="lines",
                        name=pc,
                    )
                )

            fig.update_layout(
                title=f"Forecast vs Actual for Series: {selected_series}",
                xaxis_title="Date / Step",
                yaxis_title="Units Sold",
                legend_title="Legend",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Interactive series forecast samples will display here once backtest predictions are generated.")
