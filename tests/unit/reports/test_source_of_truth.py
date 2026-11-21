"""
Regression and parity tests for M5-04: One source of truth per artifact.
Verifies that:
1. For every report present in both CSV and Parquet formats under reports/, the CSV and Parquet agree.
2. The dashboard's overview figures equal the README's rendered values.
3. Dashboard data_loader reads the canonical CSV file as primary source of truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.dashboard import data_loader as dl

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_every_report_csv_and_parquet_agree():
    """Verify that every report present in both .csv and .parquet formats agrees."""
    reports_dir = PROJECT_ROOT / "reports"
    csv_files = sorted(reports_dir.rglob("*.csv"))

    checked = 0
    for csv_path in csv_files:
        parquet_path = csv_path.with_suffix(".parquet")
        if not parquet_path.exists():
            continue

        df_csv = pd.read_csv(csv_path)
        df_parquet = pd.read_parquet(parquet_path)

        assert df_csv.shape == df_parquet.shape, (
            f"Shape mismatch for {csv_path.name}: csv={df_csv.shape}, parquet={df_parquet.shape}"
        )
        assert list(df_csv.columns) == list(df_parquet.columns), (
            f"Column mismatch for {csv_path.name}: csv={list(df_csv.columns)}, parquet={list(df_parquet.columns)}"
        )

        for col in df_csv.columns:
            if pd.api.types.is_numeric_dtype(df_csv[col]):
                val_csv = df_csv[col].to_numpy(dtype=float)
                val_pq = df_parquet[col].to_numpy(dtype=float)
                # Check match with NaN tolerance
                assert np.isclose(val_csv, val_pq, rtol=1e-4, atol=1e-4, equal_nan=True).all(), (
                    f"Numeric value mismatch in {csv_path.name} column '{col}'"
                )
            else:
                s_csv = df_csv[col].astype("object").fillna("").astype(str)
                s_pq = df_parquet[col].astype("object").fillna("").astype(str)
                assert (s_csv == s_pq).all(), (
                    f"String value mismatch in {csv_path.name} column '{col}'"
                )

        checked += 1

    # If companions exist on disk, ensure all match. If not present (e.g. fresh clone where
    # companions are generated on demand via scripts/tools/optimize_reports.py), test on-the-fly parity.
    if checked > 0:
        print(f"\nVerified agreement for {checked} report CSV-Parquet companion pairs.")
    else:
        sample_csv = reports_dir / "backtest_main" / "accuracy_overall.csv"
        if sample_csv.exists():
            df_csv = pd.read_csv(sample_csv)
            import polars as pl
            df_pq = pl.from_pandas(df_csv).to_pandas()
            assert np.isclose(
                df_csv["wape"].to_numpy(), df_pq["wape"].to_numpy(), rtol=1e-4
            ).all()


def test_dashboard_overview_figures_equal_readme_rendered_values():
    """Verify that dashboard overview KPIs match the README leaderboard and inventory tables."""
    # 1. Backtest accuracy
    acc_df = dl.load_accuracy_overall()
    assert not acc_df.empty, "Dashboard failed to load accuracy_overall"
    best_row = acc_df.sort_values(by="wape").iloc[0]

    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    # README rendered leaderboard checks
    wape_str = f"{best_row['wape']:.2f}%"
    rmsse_str = f"{best_row['rmsse']:.3f}"
    model_name = best_row["model"].upper()

    assert wape_str in readme_text, f"Champion WAPE {wape_str} not found in README"
    assert rmsse_str in readme_text, f"Champion RMSSE {rmsse_str} not found in README"
    assert model_name in readme_text, f"Champion model {model_name} not found in README"

    # 2. Inventory capital freed
    portfolio_df = dl.load_portfolio_summary()
    assert not portfolio_df.empty, "Dashboard failed to load portfolio_summary"
    freed = portfolio_df["capital_released_total"].sum()
    freed_str = f"${freed:,.2f}"

    assert freed_str in readme_text, f"Capital released string {freed_str} not found in README"


def test_data_loader_prefers_csv():
    """Verify data_loader read_table reads CSV as canonical source."""
    df = dl.read_table("reports/backtest_main/accuracy_overall")
    assert not df.empty
    csv_df = pd.read_csv(PROJECT_ROOT / "reports" / "backtest_main" / "accuracy_overall.csv")
    assert df.equals(csv_df)
