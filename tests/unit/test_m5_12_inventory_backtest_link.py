"""Unit tests for M5-12: connecting inventory layer to benchmarked forecast errors."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_inventory_analysis import compute_safety_stock_by_model
from src.inventory.safety_stock import SafetyStockCalculator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_DIR = PROJECT_ROOT / "reports" / "inventory"
README_PATH = PROJECT_ROOT / "README.md"


def test_safety_stock_records_model_column():
    """safety_stock.csv must explicitly record which model's errors it used."""
    csv_path = INVENTORY_DIR / "safety_stock.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    df = pd.read_csv(csv_path)
    assert "model" in df.columns, "safety_stock.csv is missing required 'model' column"
    models = df["model"].unique().tolist()
    assert models == ["adida"], f"Expected model 'adida', found {models}"


def test_readme_inventory_names_forecast_model_and_shows_by_model_table():
    """README inventory section must name the forecast model and report safety stock by model."""
    assert README_PATH.exists(), f"Missing {README_PATH}"
    content = README_PATH.read_text(encoding="utf-8")

    # 1. Names the forecast model
    assert "champion **ADIDA** forecast errors" in content or "**ADIDA**" in content

    # 2. Contains safety stock model table
    assert "<!-- BEGIN_SAFETY_STOCK_MODEL_TABLE -->" in content
    assert "<!-- END_SAFETY_STOCK_MODEL_TABLE -->" in content

    # 3. Artifact exists and shows dollar savings from better forecasts
    model_csv = INVENTORY_DIR / "safety_stock_by_model.csv"
    assert model_csv.exists(), f"Missing {model_csv}"
    df_models = pd.read_csv(model_csv)

    assert "model" in df_models.columns
    assert "total_ss_capital" in df_models.columns
    assert "capital_reduction_vs_naive" in df_models.columns

    models_present = set(df_models["model"].unique())
    assert {"adida", "croston_sba", "naive"}.issubset(models_present)

    naive_cap = df_models.loc[df_models["model"] == "naive", "total_ss_capital"].iloc[0]
    adida_cap = df_models.loc[df_models["model"] == "adida", "total_ss_capital"].iloc[0]
    assert adida_cap < naive_cap, f"Expected ADIDA capital ({adida_cap}) < Naive capital ({naive_cap})"


def test_inventory_pipeline_consumes_backtest_predictions():
    """Assert SafetyStockCalculator directly consumes backtest predictions DataFrame."""
    dates = pd.date_range("2016-01-01", periods=28)
    preds_df = pd.DataFrame({
        "series_id": ["FOODS_1_001_CA_1"] * 28 + ["FOODS_1_002_CA_1"] * 28,
        "date": list(dates) * 2,
        "y_true": [1.0, 0.0, 2.0, 0.0] * 14,
        "y_pred": [0.8, 0.2, 1.5, 0.1] * 14,
        "model": ["adida"] * 56,
    })

    calc = SafetyStockCalculator(lead_time_days=7, service_levels=(0.90, 0.95, 0.98))
    ss_df = calc.from_forecast_errors(preds_df, model="adida")

    assert not ss_df.empty
    assert "model" in ss_df.columns
    assert (ss_df["model"] == "adida").all()
    assert (ss_df["forecast_error_std"] > 0).all()
    assert (ss_df["safety_stock_units"] > 0).all()


def test_compute_safety_stock_by_model_logic():
    """Verify compute_safety_stock_by_model calculates capital reduction vs naive."""
    dates = pd.date_range("2016-01-01", periods=28)
    preds_data = []
    # Naive with higher errors (std = 2.0)
    for d in dates:
        preds_data.append({
            "series_id": "FOODS_1_001_CA_1", "date": d,
            "y_true": 5.0, "y_pred": 3.0, "model": "naive"
        })
    # ADIDA with lower errors (std = 1.0)
    for d in dates:
        preds_data.append({
            "series_id": "FOODS_1_001_CA_1", "date": d,
            "y_true": 5.0, "y_pred": 4.0, "model": "adida"
        })
    preds_df = pd.DataFrame(preds_data)
    prices_df = pd.DataFrame({
        "item_id": ["FOODS_1_001"],
        "store_id": ["CA_1"],
        "sell_price": [10.0],
    })

    summary = compute_safety_stock_by_model(preds_df, prices_df, lead_time=7, csl=0.95)
    assert len(summary) == 2
    naive_row = summary[summary["model"] == "naive"].iloc[0]
    adida_row = summary[summary["model"] == "adida"].iloc[0]

    assert adida_row["total_ss_capital"] <= naive_row["total_ss_capital"]
    assert adida_row["capital_reduction_vs_naive"] >= 0
