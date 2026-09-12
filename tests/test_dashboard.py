"""Tests for the Streamlit dashboard data loader and view modules."""

from __future__ import annotations

import pandas as pd

from src.dashboard import data_loader as dl
from src.dashboard.views import eda, inventory, models, overview, significance


def test_get_project_root():
    """Project root should exist and contain pyproject.toml."""
    root = dl.get_project_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()


def test_data_loader_functions():
    """All table loaders should return pandas DataFrames without raising exceptions."""
    loaders = [
        dl.load_portfolio_summary,
        dl.load_segment_summary,
        dl.load_abc_xyz_classification,
        dl.load_safety_stock,
        dl.load_service_level_scenarios,
        dl.load_marginal_analysis,
        dl.load_policy_recommendations,
        dl.load_accuracy_overall,
        dl.load_accuracy_by_level,
        dl.load_accuracy_by_category,
        dl.load_accuracy_by_store,
        dl.load_significance_mcb,
        dl.load_significance_dm,
    ]

    for loader in loaders:
        df = loader()
        assert isinstance(df, pd.DataFrame), f"{loader.__name__} did not return a DataFrame"


def test_eda_findings_loader():
    """EDA findings should load as a dictionary."""
    findings = dl.load_eda_findings()
    assert isinstance(findings, dict)


def test_views_have_render():
    """All view modules should provide a callable render() function."""
    views = [overview, eda, models, significance, inventory]
    for view in views:
        assert hasattr(view, "render")
        assert callable(view.render)
