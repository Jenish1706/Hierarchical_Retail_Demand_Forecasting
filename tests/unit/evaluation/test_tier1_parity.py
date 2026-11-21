"""
Test parity and metric consistency for M5-03: Tier-1 results on the same metric.
Verifies that:
1. Shared models between tier1 and tier1_esbu have identical RMSSE scores under canonical scales.
2. Shared baseline models between tier1 and backtest_main have identical RMSSE scores.
3. Manifests in tier1 and tier1_esbu record consistent autoets_rmsse_level12.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_tier1_and_tier1_esbu_shared_models_parity():
    t1 = pd.read_csv(PROJECT_ROOT / "reports" / "tier1" / "tier1_overall.csv").set_index("model")
    esbu = pd.read_csv(PROJECT_ROOT / "reports" / "tier1_esbu" / "tier1_overall.csv").set_index("model")

    common = sorted(set(t1.index).intersection(set(esbu.index)))
    assert len(common) >= 5, f"Expected at least 5 shared models, found {len(common)}"

    for model in common:
        # RMSSE must match within floating point tolerance
        assert t1.loc[model, "rmsse"] == pytest.approx(esbu.loc[model, "rmsse"], abs=1e-3), (
            f"Model {model} RMSSE mismatch: tier1={t1.loc[model, 'rmsse']}, esbu={esbu.loc[model, 'rmsse']}"
        )
        assert t1.loc[model, "wape"] == pytest.approx(esbu.loc[model, "wape"], abs=1e-3), (
            f"Model {model} WAPE mismatch: tier1={t1.loc[model, 'wape']}, esbu={esbu.loc[model, 'wape']}"
        )


def test_tier1_and_backtest_baseline_parity():
    t1 = pd.read_csv(PROJECT_ROOT / "reports" / "tier1" / "tier1_overall.csv").set_index("model")
    bt = pd.read_csv(PROJECT_ROOT / "reports" / "backtest_main" / "accuracy_overall.csv").set_index("model")

    # Map model names where applicable
    model_mapping = {
        "croston_sba": "croston_sba",
        "sf_seasonal_naive": "seasonal_naive_weekly",
        "sf_naive": "naive",
        "sf_window_average": "moving_average",
    }

    for t1_model, bt_model in model_mapping.items():
        if t1_model in t1.index and bt_model in bt.index:
            assert t1.loc[t1_model, "rmsse"] == pytest.approx(bt.loc[bt_model, "rmsse"], abs=1e-4), (
                f"Mismatch for {t1_model} vs {bt_model}: {t1.loc[t1_model, 'rmsse']} vs {bt.loc[bt_model, 'rmsse']}"
            )


def test_tier1_manifest_autoets_rmsse_consistency():
    m_t1 = json.loads((PROJECT_ROOT / "reports" / "tier1" / "manifest.json").read_text(encoding="utf-8"))
    m_esbu = json.loads((PROJECT_ROOT / "reports" / "tier1_esbu" / "manifest.json").read_text(encoding="utf-8"))

    t1_ets = m_t1.get("autoets_rmsse_level12")
    esbu_ets = m_esbu.get("autoets_rmsse_level12")

    assert t1_ets is not None
    assert esbu_ets is not None
    assert t1_ets == pytest.approx(esbu_ets, abs=1e-3)
    assert t1_ets == pytest.approx(0.7675, abs=1e-2)
