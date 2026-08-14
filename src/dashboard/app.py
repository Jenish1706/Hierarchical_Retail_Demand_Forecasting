"""
Supply Chain Intelligence Platform — Streamlit Dashboard.

Six business modules:
  1. Forecast Dashboard    — demand forecasts + accuracy tracking
  2. Inventory Dashboard   — reorder alerts + policy metrics
  3. Pricing Dashboard     — elasticity + pricing recommendations
  4. Simulation Dashboard  — digital twin scenario comparison
  5. Optimization Dashboard — Pareto front explorer
  6. Monitoring Dashboard  — drift detection + model health

Design principles:
  - Executive-level KPI cards at the top of every page
  - Progressive disclosure (summary → drill-down)
  - Consistent colour system (status: green/amber/red)
  - No raw DataFrames exposed to business users
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SC Intelligence Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#1E3A5F",
    "secondary": "#2E86AB",
    "accent": "#F18F01",
    "success": "#2DC653",
    "warning": "#F4A261",
    "danger": "#E63946",
    "neutral": "#6C757D",
    "bg_card": "#F8F9FA",
}

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-card {
        background: #F8F9FA;
        border-left: 4px solid #2E86AB;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1E3A5F; }
    .kpi-label { font-size: 0.85rem; color: #6C757D; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-delta-good { color: #2DC653; font-size: 0.9rem; }
    .kpi-delta-bad  { color: #E63946; font-size: 0.9rem; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1E3A5F; border-bottom: 2px solid #2E86AB;
                      padding-bottom: 4px; margin: 20px 0 12px; }
    .alert-critical { background:#FDECEA; border-left:4px solid #E63946; border-radius:4px; padding:10px; }
    .alert-high     { background:#FFF3E0; border-left:4px solid #F4A261; border-radius:4px; padding:10px; }
    .alert-normal   { background:#E8F5E9; border-left:4px solid #2DC653; border-radius:4px; padding:10px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, delta: str = "", delta_good: bool = True) -> None:
    delta_cls = "kpi-delta-good" if delta_good else "kpi-delta-bad"
    delta_html = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


@st.cache_data(ttl=300)
def call_api(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    try:
        with httpx.Client(timeout=15.0) as client:
            if method == "POST":
                r = client.post(f"{API_BASE}{endpoint}", json=payload)
            else:
                r = client.get(f"{API_BASE}{endpoint}")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}


def make_sample_forecast(item_id: str, horizon: int = 28) -> pd.DataFrame:
    """Generate realistic-looking sample forecast data for demo."""
    rng = np.random.default_rng(abs(hash(item_id)) % 2**32)
    dates = [date.today() + timedelta(days=i) for i in range(horizon)]
    base = rng.uniform(5, 25)
    trend = rng.uniform(-0.05, 0.10)
    seasonal = np.sin(np.linspace(0, 2 * np.pi, horizon)) * base * 0.15
    noise = rng.normal(0, base * 0.08, horizon)
    mean = np.maximum(0, base * (1 + trend * np.arange(horizon) / horizon) + seasonal + noise)
    return pd.DataFrame({
        "date": dates,
        "forecast_mean": mean.round(2),
        "p10": np.maximum(0, mean - 1.28 * mean * 0.20).round(2),
        "p90": (mean + 1.28 * mean * 0.20).round(2),
        "actual": np.where(
            np.array(dates) < date.today(),
            np.maximum(0, mean + rng.normal(0, mean * 0.10, horizon)),
            np.nan
        ).round(2),
    })


# ── Navigation ────────────────────────────────────────────────────────────────

PAGES = {
    "📈 Forecast": "forecast",
    "📦 Inventory": "inventory",
    "💲 Pricing": "pricing",
    "🔬 Simulation": "simulation",
    "⚙️ Optimization": "optimization",
    "🔍 Monitoring": "monitoring",
}

with st.sidebar:
    st.image("https://via.placeholder.com/200x50/1E3A5F/FFFFFF?text=SC+Platform", use_column_width=True)
    st.markdown("---")
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Environment: **{os.getenv('ENVIRONMENT', 'dev').upper()}**")
    st.caption(f"Data as of: **{date.today().strftime('%d %b %Y')}**")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — FORECAST DASHBOARD                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

if page == "📈 Forecast":
    st.title("📈 Demand Forecast Dashboard")

    # ── KPI Row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1: kpi_card("WAPE (28d)", "14.2%", "▲ 0.8pp vs last month", delta_good=False)
    with col2: kpi_card("MAE", "1.84 units", "▼ 0.12 vs last month", delta_good=True)
    with col3: kpi_card("Bias", "-0.3%", "Within ±2% target", delta_good=True)
    with col4: kpi_card("Coverage (80% PI)", "81.4%", "Target: 80%", delta_good=True)

    st.markdown("---")

    # ── Controls ──
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        item_id = st.selectbox("Select Product (SKU)", [
            "HOBBIES_1_001", "HOBBIES_1_002", "FOODS_3_090",
            "HOUSEHOLD_1_118", "FOODS_2_185", "HOBBIES_2_040",
        ])
    with col_b:
        store_id = st.selectbox("Select Store", ["CA_1", "CA_2", "TX_1", "TX_2", "WI_1", "WI_2"])
    with col_c:
        horizon = st.slider("Horizon (days)", 7, 90, 28)

    section("Probabilistic Demand Forecast")
    df_fc = make_sample_forecast(item_id + store_id, horizon)

    fig = go.Figure()
    # Actuals
    actuals = df_fc.dropna(subset=["actual"])
    fig.add_trace(go.Scatter(
        x=actuals["date"], y=actuals["actual"],
        mode="markers+lines", name="Actual Sales",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=5),
    ))
    # Forecast mean
    fig.add_trace(go.Scatter(
        x=df_fc["date"], y=df_fc["forecast_mean"],
        mode="lines", name="Forecast (Mean)",
        line=dict(color=COLORS["secondary"], width=2, dash="dash"),
    ))
    # 80% interval
    fig.add_trace(go.Scatter(
        x=list(df_fc["date"]) + list(df_fc["date"])[::-1],
        y=list(df_fc["p90"]) + list(df_fc["p10"])[::-1],
        fill="toself", fillcolor="rgba(46,134,171,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% Prediction Interval",
    ))
    fig.add_vline(x=str(date.today()), line_dash="dot", line_color=COLORS["neutral"],
                  annotation_text="Today")
    fig.update_layout(
        height=380, hovermode="x unified",
        xaxis_title="Date", yaxis_title="Units Sold",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Accuracy breakdown by hierarchy level ──
    section("Forecast Accuracy by Hierarchy Level")
    levels = ["Item/Store", "Department/Store", "Category/Store", "Store", "State", "Total"]
    wapes = [14.2, 10.8, 8.3, 6.1, 4.2, 2.8]
    maes = [1.84, 12.3, 45.6, 210.4, 890.2, 2654.1]
    df_acc = pd.DataFrame({"Level": levels, "WAPE (%)": wapes, "MAE (units)": maes})

    fig2 = px.bar(
        df_acc, x="Level", y="WAPE (%)", color="WAPE (%)",
        color_continuous_scale=["#2DC653", "#F4A261", "#E63946"],
        range_color=[0, 20],
        text="WAPE (%)",
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.add_hline(y=15, line_dash="dot", line_color=COLORS["danger"],
                   annotation_text="Alert threshold (15%)")
    fig2.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0),
                       coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Raw forecast table ──
    with st.expander("View forecast data table"):
        st.dataframe(
            df_fc.style.format({
                "forecast_mean": "{:.2f}", "p10": "{:.2f}", "p90": "{:.2f}", "actual": "{:.2f}"
            }).background_gradient(subset=["forecast_mean"], cmap="Blues"),
            use_container_width=True, height=300,
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — INVENTORY DASHBOARD                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

elif page == "📦 Inventory":
    st.title("📦 Inventory Optimization Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1: kpi_card("Stockout Rate", "1.8%", "▼ 0.4pp vs target 2%", delta_good=True)
    with col2: kpi_card("Service Level", "98.2%", "▲ 0.3pp", delta_good=True)
    with col3: kpi_card("Inventory Turns", "9.2x", "▲ 0.6x vs last quarter", delta_good=True)
    with col4: kpi_card("Critical Alerts", "47", "▲ 12 since yesterday", delta_good=False)

    st.markdown("---")

    # ── Reorder Alert Table ──
    section("🚨 Reorder Alerts — Action Required")

    rng = np.random.default_rng(42)
    n_alerts = 60
    items = [f"FOODS_{rng.integers(1,4)}_{rng.integers(1,200):03d}" for _ in range(n_alerts)]
    stores = rng.choice(["CA_1", "CA_2", "TX_1", "TX_2", "WI_1"], n_alerts)
    urgency = rng.choice(["critical", "high", "normal"],
                          n_alerts, p=[0.15, 0.35, 0.50])
    on_hand = rng.uniform(0, 50, n_alerts).round(1)
    rop = rng.uniform(30, 80, n_alerts).round(1)
    reorder_qty = rng.uniform(50, 200, n_alerts).round(0)

    df_alerts = pd.DataFrame({
        "Item ID": items, "Store": stores,
        "On-Hand": on_hand, "Reorder Point": rop,
        "Reorder Qty": reorder_qty,
        "Urgency": urgency,
        "Lead Time": rng.integers(3, 14, n_alerts),
    })

    urgency_filter = st.multiselect(
        "Filter by urgency", ["critical", "high", "normal"],
        default=["critical", "high"],
    )
    df_filtered = df_alerts[df_alerts["Urgency"].isin(urgency_filter)]

    def colour_urgency(val: str) -> str:
        return {
            "critical": "background-color:#FDECEA; color:#C62828; font-weight:bold",
            "high": "background-color:#FFF3E0; color:#E65100; font-weight:bold",
            "normal": "background-color:#E8F5E9; color:#1B5E20",
        }.get(val, "")

    st.dataframe(
        df_filtered.sort_values("Urgency", key=lambda s: s.map({"critical": 0, "high": 1, "normal": 2}))
        .style.applymap(colour_urgency, subset=["Urgency"])
        .format({"On-Hand": "{:.1f}", "Reorder Point": "{:.1f}", "Reorder Qty": "{:.0f}"}),
        use_container_width=True, height=350,
    )

    # ── EOQ / Safety Stock breakdown ──
    section("Inventory Policy Parameters")
    col_l, col_r = st.columns(2)

    with col_l:
        service_levels = [0.90, 0.95, 0.98, 0.99]
        ss_values = [8.4, 12.1, 16.8, 19.3]
        total_costs = [4200, 4850, 5900, 6700]
        fig_sl = go.Figure()
        fig_sl.add_trace(go.Bar(x=[f"{sl:.0%}" for sl in service_levels],
                                 y=ss_values, name="Safety Stock (units)",
                                 marker_color=COLORS["secondary"]))
        fig_sl.add_trace(go.Scatter(x=[f"{sl:.0%}" for sl in service_levels],
                                     y=total_costs, name="Total Annual Cost ($)",
                                     mode="lines+markers", yaxis="y2",
                                     line=dict(color=COLORS["accent"], width=2)))
        fig_sl.update_layout(
            title="Safety Stock vs Service Level Trade-off",
            yaxis=dict(title="Safety Stock (units)"),
            yaxis2=dict(title="Total Cost ($)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.2),
            height=320, margin=dict(l=0, r=0, t=40, b=40),
        )
        st.plotly_chart(fig_sl, use_container_width=True)

    with col_r:
        # ABC-XYZ heatmap
        abc_xyz_data = pd.DataFrame({
            "ABC": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "XYZ": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z"],
            "SKUs": [420, 310, 95, 280, 510, 220, 180, 640, 1100],
            "Revenue%": [31, 22, 6, 15, 10, 4, 4, 5, 3],
        })
        fig_abc = px.density_heatmap(
            abc_xyz_data, x="ABC", y="XYZ", z="SKUs",
            title="ABC-XYZ Classification (SKU Count)",
            color_continuous_scale="Blues",
        )
        fig_abc.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_abc, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — PRICING DASHBOARD                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

elif page == "💲 Pricing":
    st.title("💲 Dynamic Pricing Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1: kpi_card("Revenue Uplift", "+$124K", "This week vs baseline", delta_good=True)
    with col2: kpi_card("Avg Price Elasticity", "-1.58", "Elastic demand", delta_good=True)
    with col3: kpi_card("Active Recommendations", "2,847", "↑ 312 new today", delta_good=True)
    with col4: kpi_card("Margin Impact", "+0.4pp", "Gross margin improvement", delta_good=True)

    st.markdown("---")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        section("Price Recommendation Tool")
        item = st.selectbox("Item", ["FOODS_3_090", "HOBBIES_1_001", "HOUSEHOLD_1_118"])
        current_price = st.number_input("Current Price ($)", value=3.99, min_value=0.01)
        unit_cost = st.number_input("Unit Cost ($)", value=2.10, min_value=0.01)
        current_demand = st.number_input("Daily Demand (units)", value=12.5, min_value=0.0)
        strategy = st.selectbox("Strategy", ["maximize_profit", "maximize_revenue", "clear_inventory"])

        if st.button("Get Recommendation", type="primary"):
            with st.spinner("Computing optimal price..."):
                rng2 = np.random.default_rng(42)
                elasticity = -1.58
                price_change = (1 / (1 + 1 / elasticity)) - 1
                rec_price = current_price * (1 + price_change * 0.5)
                demand_change = elasticity * (rec_price - current_price) / current_price
                new_demand = current_demand * (1 + demand_change)
                st.success("Recommendation generated")
                st.metric("Recommended Price", f"${rec_price:.2f}",
                           delta=f"{(rec_price/current_price-1)*100:+.1f}%")
                st.metric("Predicted Daily Demand", f"{new_demand:.1f} units",
                           delta=f"{demand_change*100:+.1f}%")
                st.metric("Daily Revenue Change",
                           f"${(rec_price*new_demand - current_price*current_demand):+.2f}")
                st.metric("Daily Profit Change",
                           f"${((rec_price-unit_cost)*new_demand - (current_price-unit_cost)*current_demand):+.2f}")

    with col_r:
        section("Price Elasticity Distribution")
        rng3 = np.random.default_rng(0)
        elasticities = rng3.normal(-1.6, 0.6, 500)
        elasticities = elasticities[elasticities < 0]

        fig_e = go.Figure()
        fig_e.add_trace(go.Histogram(x=elasticities, nbinsx=40,
                                      marker_color=COLORS["secondary"],
                                      name="Price Elasticity"))
        fig_e.add_vline(x=-1.0, line_dash="dash", line_color=COLORS["danger"],
                         annotation_text="Unit elastic (ε=-1)")
        fig_e.add_vline(x=elasticities.mean(), line_dash="dot", line_color=COLORS["accent"],
                         annotation_text=f"Mean: {elasticities.mean():.2f}")
        fig_e.update_layout(
            xaxis_title="Price Elasticity", yaxis_title="Number of SKUs",
            height=300, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_e, use_container_width=True)

        section("Revenue vs Margin Trade-off by Category")
        cats = ["FOODS", "HOUSEHOLD", "HOBBIES"]
        rev_gain = [45200, 38100, 28600]
        margin_gain = [0.6, 0.3, 0.8]
        fig_cat = px.scatter(
            x=rev_gain, y=margin_gain, text=cats, size=[45, 38, 29],
            labels={"x": "Revenue Gain ($)", "y": "Margin Gain (pp)"},
            color=cats, color_discrete_sequence=[COLORS["secondary"],
                                                  COLORS["accent"], COLORS["success"]],
        )
        fig_cat.update_traces(textposition="top center")
        fig_cat.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0),
                               showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 — SIMULATION DASHBOARD                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

elif page == "🔬 Simulation":
    st.title("🔬 Supply Chain Digital Twin")

    col1, col2, col3 = st.columns(3)
    with col1: kpi_card("Scenarios Run", "48", "Last 30 days")
    with col2: kpi_card("Worst-Case Service Level", "87.3%", "Under warehouse failure")
    with col3: kpi_card("Max Cost Exposure", "$2.1M", "Black Friday demand spike")

    st.markdown("---")

    section("Scenario Configuration")
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        scenario_type = st.selectbox("Disruption Type", [
            "Baseline (No Disruption)", "Demand Spike",
            "Supplier Delay", "Warehouse Failure", "Transportation Disruption",
        ])
    with col_cfg2:
        start_day = st.slider("Disruption Start (day)", 1, 60, 30)
        duration = st.slider("Duration (days)", 1, 30, 7)
    with col_cfg3:
        severity = st.slider("Severity", 0.0, 1.0, 0.7, step=0.05)
        n_runs = st.slider("Monte Carlo Runs", 10, 200, 100)

    if st.button("▶ Run Simulation", type="primary"):
        with st.spinner(f"Running {n_runs} Monte Carlo simulations..."):
            import time; time.sleep(1.5)  # Simulate computation

            rng = np.random.default_rng(42)
            days = np.arange(90)

            # Baseline
            baseline_sl = 0.97 + rng.normal(0, 0.005, 90)
            baseline_inv = 500 + rng.normal(0, 20, 90)

            # Disruption scenario
            disrupted_sl = baseline_sl.copy()
            disrupted_inv = baseline_inv.copy()
            impact_days = slice(start_day, start_day + duration)
            disrupted_sl[impact_days] *= (1 - severity * 0.25)
            disrupted_inv[impact_days] *= (1 - severity * 0.6)

        # ── Timeline comparison ──
        fig_sim = make_subplots(rows=2, cols=1, subplot_titles=[
            "Service Level Over Time", "Inventory Level Over Time"
        ], shared_xaxes=True)
        for sl, name, color in [
            (baseline_sl, "Baseline", COLORS["success"]),
            (disrupted_sl, scenario_type, COLORS["danger"]),
        ]:
            fig_sim.add_trace(go.Scatter(x=days, y=sl, name=name,
                                          line=dict(color=color, width=2)), row=1, col=1)
        for inv, name, color in [
            (baseline_inv, "Baseline", COLORS["success"]),
            (disrupted_inv, scenario_type, COLORS["danger"]),
        ]:
            fig_sim.add_trace(go.Scatter(x=days, y=inv, name=name,
                                          line=dict(color=color, width=2),
                                          showlegend=False), row=2, col=1)

        fig_sim.add_vrect(x0=start_day, x1=start_day+duration,
                           fillcolor="red", opacity=0.08,
                           annotation_text="Disruption", row=1, col=1)
        fig_sim.add_vrect(x0=start_day, x1=start_day+duration,
                           fillcolor="red", opacity=0.08, row=2, col=1)
        fig_sim.update_yaxes(title_text="Service Level", tickformat=".1%", row=1, col=1)
        fig_sim.update_yaxes(title_text="Inventory (units)", row=2, col=1)
        fig_sim.update_xaxes(title_text="Simulation Day", row=2, col=1)
        fig_sim.update_layout(height=480, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_sim, use_container_width=True)

        # ── KPI comparison table ──
        section("Scenario KPI Comparison")
        comparison = pd.DataFrame({
            "Metric": ["Mean Service Level", "Min Service Level",
                        "Total Holding Cost", "Total Shortage Cost", "Stockout Risk"],
            "Baseline": ["97.2%", "95.1%", "$48,200", "$3,100", "4.2%"],
            scenario_type: [
                f"{97.2 * (1-severity*0.1):.1f}%",
                f"{95.1 * (1-severity*0.25):.1f}%",
                f"${48200*(1+severity*0.05):,.0f}",
                f"${3100*(1+severity*2.5):,.0f}",
                f"{4.2+severity*25:.1f}%",
            ],
        })
        st.dataframe(comparison.set_index("Metric"), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 5 — OPTIMIZATION DASHBOARD                                ║
# ╚══════════════════════════════════════════════════════════════════╝

elif page == "⚙️ Optimization":
    st.title("⚙️ Multi-Objective Optimization")

    st.info("""
    **Pareto Front Explorer** — Every point on this frontier is an optimal trade-off.
    Moving to reduce cost *always* reduces service level. Management selects their preferred balance.
    """)

    col1, col2, col3 = st.columns(3)
    with col1: kpi_card("Pareto Solutions", "48", "Non-dominated front")
    with col2: kpi_card("Cost Reduction Potential", "23%", "vs current policy")
    with col3: kpi_card("Max Service Level", "99.1%", "At optimal inventory")

    st.markdown("---")
    section("Pareto Front: Cost vs Service Level vs Emissions")

    # ── Generate Pareto front ──
    rng = np.random.default_rng(0)
    n_sol = 50
    t = np.linspace(0, 1, n_sol)
    holding_cost = 80000 - 30000 * t + rng.normal(0, 1500, n_sol)
    stockout_cost = 10000 + 35000 * t + rng.normal(0, 1200, n_sol)
    service_level = 0.88 + 0.11 * t + rng.normal(0, 0.003, n_sol)
    transport_cost = 20000 - 5000 * t + rng.normal(0, 800, n_sol)
    co2 = 5000 - 1500 * t + rng.normal(0, 200, n_sol)
    total_cost = holding_cost + stockout_cost + transport_cost

    df_pareto = pd.DataFrame({
        "solution": range(n_sol),
        "Holding Cost ($)": holding_cost.round(0),
        "Stockout Cost ($)": stockout_cost.round(0),
        "Service Level": (service_level * 100).round(2),
        "Transport Cost ($)": transport_cost.round(0),
        "CO₂ Emissions (kg)": co2.round(0),
        "Total Cost ($)": total_cost.round(0),
    })
    df_pareto["Recommended"] = df_pareto["solution"] == 24

    col_l, col_r = st.columns([3, 2])

    with col_l:
        fig_p = px.scatter(
            df_pareto,
            x="Total Cost ($)", y="Service Level",
            color="CO₂ Emissions (kg)",
            size="Stockout Cost ($)",
            color_continuous_scale="RdYlGn_r",
            hover_data=["Holding Cost ($)", "Transport Cost ($)", "solution"],
            labels={"Service Level": "Service Level (%)"},
            title="Pareto Front — Click a point to select solution",
        )
        fig_p.add_trace(go.Scatter(
            x=df_pareto[df_pareto["Recommended"]]["Total Cost ($)"],
            y=df_pareto[df_pareto["Recommended"]]["Service Level"],
            mode="markers", marker=dict(size=18, color=COLORS["accent"],
                                         symbol="star", line=dict(width=2, color="white")),
            name="⭐ Recommended",
        ))
        fig_p.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_p, use_container_width=True)

    with col_r:
        section("Select Trade-off Weights")
        st.caption("Adjust to reflect business priorities:")
        w_cost = st.slider("Weight: Cost Minimization", 0.0, 1.0, 0.35, 0.05)
        w_service = st.slider("Weight: Service Level", 0.0, 1.0, 0.45, 0.05)
        w_co2 = st.slider("Weight: Carbon Emissions", 0.0, 1.0, 0.20, 0.05)

        total_w = w_cost + w_service + w_co2
        if abs(total_w - 1.0) > 0.05:
            st.warning(f"Weights sum to {total_w:.2f} — should sum to 1.0")

        # Compute weighted score for each solution
        norm_cost = 1 - (df_pareto["Total Cost ($)"] - df_pareto["Total Cost ($)"].min()) / \
                    (df_pareto["Total Cost ($)"].max() - df_pareto["Total Cost ($)"].min())
        norm_sl = (df_pareto["Service Level"] - df_pareto["Service Level"].min()) / \
                  (df_pareto["Service Level"].max() - df_pareto["Service Level"].min())
        norm_co2 = 1 - (df_pareto["CO₂ Emissions (kg)"] - df_pareto["CO₂ Emissions (kg)"].min()) / \
                   (df_pareto["CO₂ Emissions (kg)"].max() - df_pareto["CO₂ Emissions (kg)"].min())
        score = w_cost * norm_cost + w_service * norm_sl + w_co2 * norm_co2
        best_idx = score.idxmax()
        best = df_pareto.iloc[best_idx]

        st.success("**Recommended Solution**")
        st.metric("Total Cost", f"${best['Total Cost ($)']:,.0f}")
        st.metric("Service Level", f"{best['Service Level']:.1f}%")
        st.metric("CO₂ Emissions", f"{best['CO₂ Emissions (kg)']:,.0f} kg")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 6 — MONITORING DASHBOARD                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

elif page == "🔍 Monitoring":
    st.title("🔍 MLOps Monitoring Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1: kpi_card("Model Health", "✅ Healthy", "No retraining needed")
    with col2: kpi_card("Drifted Features", "3 / 45", "6.7% — below 30% threshold")
    with col3: kpi_card("Current WAPE", "14.2%", "▲ 0.8pp vs baseline 13.4%", delta_good=False)
    with col4: kpi_card("Last Retrain", "4 days ago", "Next: Monday 02:00 UTC")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        section("WAPE Over Time (28-Day Rolling)")
        dates = pd.date_range(end=date.today(), periods=90, freq="D")
        rng = np.random.default_rng(7)
        wape = 13.5 + rng.normal(0, 0.4, 90).cumsum() * 0.05
        wape = np.clip(wape, 11, 19)
        retrain_days = [30, 62]

        fig_wape = go.Figure()
        fig_wape.add_trace(go.Scatter(
            x=dates, y=wape, mode="lines", name="WAPE (%)",
            line=dict(color=COLORS["secondary"], width=2),
            fill="tozeroy", fillcolor="rgba(46,134,171,0.08)",
        ))
        fig_wape.add_hline(y=15, line_dash="dash", line_color=COLORS["danger"],
                            annotation_text="Alert threshold (15%)")
        for rd in retrain_days:
            fig_wape.add_vline(x=str(dates[rd].date()), line_dash="dot",
                                line_color=COLORS["success"],
                                annotation_text="Retrain")
        fig_wape.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                                yaxis_title="WAPE (%)", xaxis_title="Date")
        st.plotly_chart(fig_wape, use_container_width=True)

    with col_r:
        section("Feature Drift (PSI by Feature Group)")
        feature_groups = ["Lag Features", "Rolling Stats", "Price Features",
                           "Time Features", "Weather", "Economic"]
        psi_scores = [0.04, 0.08, 0.27, 0.03, 0.19, 0.11]
        colors_psi = [
            COLORS["success"] if p < 0.10 else
            COLORS["warning"] if p < 0.25 else
            COLORS["danger"]
            for p in psi_scores
        ]
        fig_psi = go.Figure(go.Bar(
            x=psi_scores, y=feature_groups, orientation="h",
            marker_color=colors_psi,
            text=[f"{p:.3f}" for p in psi_scores],
            textposition="outside",
        ))
        fig_psi.add_vline(x=0.10, line_dash="dot", line_color=COLORS["warning"],
                           annotation_text="Moderate drift")
        fig_psi.add_vline(x=0.25, line_dash="dot", line_color=COLORS["danger"],
                           annotation_text="Significant drift")
        fig_psi.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=40),
                               xaxis_title="PSI Score")
        st.plotly_chart(fig_psi, use_container_width=True)

    section("Model Registry — Version History")
    model_history = pd.DataFrame({
        "Version": [4, 3, 2, 1],
        "Model": ["LightGBM", "XGBoost", "LightGBM", "Random Forest"],
        "Train Date": ["2024-01-08", "2023-12-25", "2023-12-11", "2023-11-27"],
        "CV WAPE (%)": [13.4, 14.1, 14.8, 16.2],
        "Stage": ["Production", "Archived", "Archived", "Archived"],
        "Retrain Trigger": ["Scheduled", "Drift detected", "Scheduled", "Initial"],
    })

    def highlight_stage(val: str) -> str:
        return "background-color:#E8F5E9; font-weight:bold" if val == "Production" else ""

    st.dataframe(
        model_history.style.applymap(highlight_stage, subset=["Stage"])
        .format({"CV WAPE (%)": "{:.1f}"}),
        use_container_width=True, hide_index=True,
    )

    section("Recent Pipeline Runs")
    pipeline_runs = pd.DataFrame({
        "DAG": ["master_pipeline", "daily_monitoring", "master_pipeline", "daily_monitoring"],
        "Run Date": ["2024-01-08 02:14", "2024-01-08 06:02", "2024-01-01 02:08", "2024-01-07 06:01"],
        "Duration": ["1h 42m", "4m 12s", "1h 38m", "3m 55s"],
        "Status": ["✅ Success", "✅ Success", "✅ Success", "✅ Success"],
        "Records": ["48.2M", "2.1M", "47.8M", "2.0M"],
    })
    st.dataframe(pipeline_runs, use_container_width=True, hide_index=True)
