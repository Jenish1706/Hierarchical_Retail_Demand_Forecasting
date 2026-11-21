"""M5 Retail Demand & Inventory Intelligence Platform — Streamlit Dashboard."""

from __future__ import annotations

import streamlit as st

from src.dashboard import data_loader as dl
from src.dashboard.views import eda, inventory, models, overview, significance

# Set page configuration
st.set_page_config(
    page_title="M5 Retail Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stMetric {
        background-color: #F8FAFC;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.image("https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg", width=50)
st.sidebar.title("M5 Intelligence Lab")
st.sidebar.caption("Retail Demand & Inventory Optimization")

pages = {
    "📊 Executive Overview": overview.render,
    "🔍 Demand Sparsity & EDA": eda.render,
    "🤖 Model Zoo & Forecasts": models.render,
    "📈 Statistical Significance": significance.render,
    "📦 Inventory & Capital Lab": inventory.render,
}

selection = st.sidebar.radio("Navigate Module", list(pages.keys()))

st.sidebar.markdown("---")
_structure = dl.load_eda_findings().get("3.1_structure", {})
_data_line = (
    f"{_structure.get('long_format_rows', 0) / 1e6:.1f}M panel records / {_structure.get('n_series', 0):,} store-item series"
    if _structure
    else "run scripts/pipeline/run_eda.py to populate"
)
st.sidebar.markdown(
    f"""
    **Project Highlights**:
    * **Data**: {_data_line}
    * **Evaluation**: Rolling-origin walk-forward backtest
    * **Decision Layer**: ABC/XYZ policy comparison with annual cost and margin sensitivity
    * **Testing**: automated test suite (`pytest`)
    """
)
st.sidebar.caption("IIT Kharagpur Placement Portfolio")

# Render selected page
pages[selection]()
