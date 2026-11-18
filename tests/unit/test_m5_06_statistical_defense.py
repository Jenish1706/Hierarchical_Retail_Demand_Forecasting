"""
Unit tests for M5-06: Statistical evidence that can be defended.

Verifies:
1. Paired series bootstrap intervals for WAPE, RMSSE, and relative deltas.
2. Per-series pairwise head-to-head win-rates and Diebold-Mariano tests.
3. Generation of the critical difference diagram.
4. Completeness of significance_report.json schema.
5. README statistics section rendering from backing artifacts without hardcoding.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.significance import (
    paired_series_bootstrap,
    pairwise_per_series_tests,
)
from scripts.render_readme_tables import (
    render_statistics_section,
    check_readme_tables,
    SIGNIFICANCE_JSON,
)


@pytest.fixture
def synthetic_predictions() -> pd.DataFrame:
    """Create a minimal 3-model, 20-series, 28-day panel with known error differences."""
    rng = np.random.default_rng(123)
    n_series = 20
    n_days = 28
    models = ["model_best", "model_tie", "model_worse"]
    
    rows = []
    for s in range(n_series):
        sid = f"FOODS_1_{s:03d}_CA_1"
        base_demand = float(rng.poisson(lam=5.0))
        for d in range(n_days):
            y_true = float(rng.poisson(lam=base_demand))
            # model_best: small noise
            rows.append({
                "series_id": sid,
                "origin": 0,
                "model": "model_best",
                "date": f"2016-01-{d+1:02d}",
                "y_true": y_true,
                "y_pred": max(0.0, y_true + rng.normal(0, 0.5)),
                "scale": 1.5,
            })
            # model_tie: identical to best on some, slightly better/worse on others
            rows.append({
                "series_id": sid,
                "origin": 0,
                "model": "model_tie",
                "date": f"2016-01-{d+1:02d}",
                "y_true": y_true,
                "y_pred": max(0.0, y_true + rng.normal(0, 0.5)),
                "scale": 1.5,
            })
            # model_worse: large noise
            rows.append({
                "series_id": sid,
                "origin": 0,
                "model": "model_worse",
                "date": f"2016-01-{d+1:02d}",
                "y_true": y_true,
                "y_pred": max(0.0, y_true + rng.normal(5, 2.0)),
                "scale": 1.5,
            })
    return pd.DataFrame(rows)


def test_paired_series_bootstrap_structure(synthetic_predictions):
    boot = paired_series_bootstrap(
        synthetic_predictions,
        n_bootstraps=100,
        alpha=0.05,
        seed=42,
    )
    
    assert set(boot.keys()) == {"model_best", "model_tie", "model_worse"}
    
    for m, stats in boot.items():
        wape = stats["wape"]
        assert wape["ci_lower"] <= wape["boot_mean"] <= wape["ci_upper"]
        rmsse = stats["rmsse"]
        assert rmsse["ci_lower"] <= rmsse["boot_mean"] <= rmsse["ci_upper"]
        
    best_m = boot["model_best"]["delta_wape_vs_best"]["ref_model"]
    assert best_m in {"model_best", "model_tie"}
    # The reference model delta vs itself must be exactly 0
    ref_delta = boot[best_m]["delta_wape_vs_best"]
    assert ref_delta["point"] == 0.0
    assert ref_delta["ci_lower"] == 0.0
    assert ref_delta["ci_upper"] == 0.0
    assert not ref_delta["separable"]


def test_pairwise_per_series_tests(synthetic_predictions):
    df_h2h = pairwise_per_series_tests(synthetic_predictions, horizon=28, alpha=0.05)
    
    # 3 models => 3 pairs
    assert len(df_h2h) == 3
    expected_cols = {
        "model_a",
        "model_b",
        "a_win_pct",
        "b_win_pct",
        "tie_pct",
        "dm_a_beats_b_pct",
        "dm_b_beats_a_pct",
        "dm_indistinguishable_pct",
    }
    assert expected_cols <= set(df_h2h.columns)
    
    for _, row in df_h2h.iterrows():
        total_win = row["a_win_pct"] + row["b_win_pct"] + row["tie_pct"]
        assert total_win == pytest.approx(100.0, abs=1e-3)
        total_dm = (
            row["dm_a_beats_b_pct"]
            + row["dm_b_beats_a_pct"]
            + row["dm_indistinguishable_pct"]
        )
        assert total_dm == pytest.approx(100.0, abs=1e-3)


def test_significance_report_json_schema():
    assert SIGNIFICANCE_JSON.exists(), f"Missing {SIGNIFICANCE_JSON}"
    with open(SIGNIFICANCE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "friedman" in data
    assert "nemenyi" in data
    assert "mcb" in data
    assert "bootstrap" in data
    assert "pairwise_per_series" in data
    assert "verdict" in data
    
    # Check bootstrap contents
    boot = data["bootstrap"]
    assert "adida" in boot
    assert "croston_sba" in boot
    assert "routed_ensemble" in boot
    
    # Check verdict contents
    verdict = data["verdict"]
    assert "best_by_rank" in verdict
    assert "best_by_wape" in verdict


def test_critical_difference_figure_exists():
    cd_fig = Path("reports/backtest_main/significance/critical_difference.png")
    assert cd_fig.exists(), f"Missing critical difference plot at {cd_fig}"
    assert cd_fig.stat().st_size > 5000, "Critical difference plot is suspiciously small"


def test_readme_statistics_rendered_from_artifacts():
    with open(SIGNIFICANCE_JSON, "r", encoding="utf-8") as f:
        sig = json.load(f)
    stats_text = render_statistics_section(sig)
    assert len(stats_text) > 100
    assert "Friedman" in stats_text
    assert "Nemenyi" in stats_text
    assert "Bootstrap" in stats_text or "bootstrap" in stats_text
    
    is_synced, msg = check_readme_tables()
    assert is_synced, f"README out of sync: {msg}"
