"""Data loading and caching layer for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:
    def cache_data(func=None, **kwargs):  # type: ignore
        if func is None:
            return lambda f: f
        return func


def get_project_root() -> Path:
    """Return project root directory."""
    # src/dashboard/data_loader.py -> root is 2 levels up from src
    return Path(__file__).resolve().parents[2]


def read_table(rel_path_without_ext: str) -> pd.DataFrame:
    """Read canonical table (reads .csv as primary source of truth, falling back to .parquet)."""
    root = get_project_root()
    csv_path = root / f"{rel_path_without_ext}.csv"
    parquet_path = root / f"{rel_path_without_ext}.parquet"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    elif parquet_path.exists():
        try:
            return pl.read_parquet(parquet_path).to_pandas()
        except Exception:
            return pd.read_parquet(parquet_path)
    return pd.DataFrame()


@cache_data
def load_portfolio_summary() -> pd.DataFrame:
    """Load inventory portfolio summary (current vs proposed working capital)."""
    return read_table("reports/inventory/portfolio_summary")


@cache_data
def load_segment_summary() -> pd.DataFrame:
    """Load 3x3 ABC/XYZ segment summary metrics."""
    return read_table("reports/inventory/segment_summary")


@cache_data
def load_abc_xyz_classification() -> pd.DataFrame:
    """Load SKU-level ABC/XYZ classification."""
    return read_table("reports/inventory/abc_xyz_classification")


@cache_data
def load_safety_stock() -> pd.DataFrame:
    """Load SKU-level safety stock and reorder points across service levels."""
    return read_table("reports/inventory/safety_stock")


@cache_data
def load_service_level_scenarios() -> pd.DataFrame:
    """Load service level scenarios across 90%, 95%, 98%."""
    return read_table("reports/inventory/service_level_scenarios")


@cache_data
def load_marginal_analysis() -> pd.DataFrame:
    """Load marginal capital analysis per ABC/XYZ segment."""
    return read_table("reports/inventory/marginal_analysis")


@cache_data
def load_policy_recommendations() -> pd.DataFrame:
    """Load policy recommendations per ABC/XYZ cell."""
    return read_table("reports/inventory/policy_recommendations")


@cache_data
def load_accuracy_overall() -> pd.DataFrame:
    """Load cross-tier accuracy leaderboard."""
    df = read_table("reports/backtest_main/accuracy_overall")
    if df.empty:
        df = read_table("reports/tier1/tier1_overall")
    return df


@cache_data
def load_accuracy_by_level() -> pd.DataFrame:
    """Load accuracy broken down by hierarchy level."""
    df = read_table("reports/backtest_main/accuracy_by_hierarchy_level")
    if df.empty:
        df = read_table("reports/tier1/tier1_by_level")
    return df


@cache_data
def load_accuracy_by_category() -> pd.DataFrame:
    """Load accuracy broken down by category."""
    df = read_table("reports/backtest_main/accuracy_by_category")
    if df.empty:
        df = read_table("reports/tier1/tier1_by_category")
    return df


@cache_data
def load_accuracy_by_store() -> pd.DataFrame:
    """Load accuracy broken down by store."""
    return read_table("reports/backtest_main/accuracy_by_store")


@cache_data
def load_significance_mcb() -> pd.DataFrame:
    """Load Friedman-Nemenyi MCB results."""
    return read_table("reports/backtest_main/significance/mcb")


@cache_data
def load_significance_dm() -> pd.DataFrame:
    """Load pairwise Diebold-Mariano tests."""
    return read_table("reports/backtest_main/significance/diebold_mariano")


@cache_data
def load_eda_findings() -> dict[str, Any]:
    """Load EDA findings JSON."""
    root = get_project_root()
    path = root / "reports/eda/eda_findings.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@cache_data
def load_series_predictions_sample(n_series: int = 50) -> pd.DataFrame:
    """Load sample of backtest predictions for interactive time series plotting."""
    root = get_project_root()
    pred_path = root / "reports/backtest_main/predictions.parquet"
    if not pred_path.exists():
        pred_path = root / "reports/tier1/predictions.parquet"

    if pred_path.exists():
        try:
            # Polars lazy scan to quickly sample n_series distinct series_id
            lazy = pl.scan_parquet(pred_path)
            # Find first n_series
            distinct_series = (
                lazy.select("series_id")
                .unique()
                .limit(n_series)
                .collect()["series_id"]
                .to_list()
            )
            sample_df = (
                lazy.filter(pl.col("series_id").is_in(distinct_series))
                .collect()
                .to_pandas()
            )
            return sample_df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def get_image_path(rel_path: str) -> Path | None:
    """Get absolute path to an image if it exists."""
    p = get_project_root() / rel_path
    return p if p.exists() else None
