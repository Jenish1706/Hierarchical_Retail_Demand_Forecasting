"""Format report CSV files to clean, strict ASCII representations and
generate companion compressed Parquet files for maximum performance and
storage efficiency.

Usage:
    python scripts/optimize_reports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean unrounded floating point jitter while preserving precision."""
    df_clean = df.copy()

    for col in df_clean.columns:
        if pd.api.types.is_float_dtype(df_clean[col]):
            series = df_clean[col]
            # Check if all non-null values are exact integers
            non_null = series.dropna()
            if len(non_null) > 0 and (non_null == non_null.round()).all() and (non_null.abs() < 1e12).all():
                # Integer-compatible float (e.g., counts or IDs)
                if series.isna().any():
                    df_clean[col] = series.round(0)
                else:
                    df_clean[col] = series.astype(int)
            else:
                # Format floats cleanly: round to 4 decimals by default
                # If metric needs more precision (like small p-values), preserve up to 6 decimals
                if col.endswith(("_pct", "_pvalue", "p_value", "p_adj", "weight", "share")):
                    df_clean[col] = series.round(6)
                elif col.endswith(("_cost", "_value", "_capital", "_demand", "_stock", "_units", "rop")):
                    df_clean[col] = series.round(2)
                else:
                    df_clean[col] = series.round(4)

    return df_clean


def process_reports(reports_dir: Path | str = "reports") -> None:
    reports_dir = Path(reports_dir)
    if not reports_dir.exists():
        print(f"Directory {reports_dir} does not exist.")
        return

    csv_files = sorted(reports_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {reports_dir}\n")

    total_csv_before = 0
    total_csv_after = 0
    total_parquet_size = 0

    for csv_path in csv_files:
        size_before = csv_path.stat().st_size
        total_csv_before += size_before

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            df_clean = optimize_dataframe(df)

            # 1. Write clean ASCII CSV
            df_clean.to_csv(csv_path, index=False, encoding="ascii", lineterminator="\n")
            size_after = csv_path.stat().st_size
            total_csv_after += size_after

            # 2. Write companion Parquet file with Snappy compression
            parquet_path = csv_path.with_suffix(".parquet")
            # Convert through Polars for high-quality Arrow typing
            pl_df = pl.from_pandas(df_clean)
            pl_df.write_parquet(parquet_path, compression="snappy")
            parquet_size = parquet_path.stat().st_size
            total_parquet_size += parquet_size

            pct_change = (size_after - size_before) / size_before * 100 if size_before > 0 else 0
            rel_csv = csv_path.relative_to(reports_dir)
            print(
                f"  {rel_csv}: {size_before/1024:.1f}KB -> {size_after/1024:.1f}KB ({pct_change:+.1f}%) | "
                f"Parquet: {parquet_size/1024:.1f}KB"
            )

        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total CSV size before:   {total_csv_before / (1024 * 1024):.2f} MB")
    print(f"  Total CSV size after:    {total_csv_after / (1024 * 1024):.2f} MB")
    print(f"  Total Parquet companion: {total_parquet_size / (1024 * 1024):.2f} MB")
    if total_csv_before > 0:
        savings = (1 - (total_parquet_size / total_csv_before)) * 100
        print(f"  Parquet storage savings vs original CSV: {savings:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    process_reports()
