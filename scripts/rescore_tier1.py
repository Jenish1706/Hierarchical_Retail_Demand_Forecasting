"""
Re-score tier1 and tier1_esbu reports using canonical M5-01 scale semantics.
Aligns reports/tier1/ and reports/tier1_esbu/ with the verified backtest scales.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.point_metrics import metrics_by_group


def rescore_tier1_directory(
    target_dir: Path,
    canonical_scale_map: pd.Series,
    meta_df: pd.DataFrame,
) -> None:
    preds_path = target_dir / "predictions.parquet"
    if not preds_path.exists():
        print(f"Skipping {target_dir}: no predictions.parquet")
        return

    print(f"\n--- Rescoring {target_dir.name} ---")
    preds = pd.read_parquet(preds_path)

    # Map canonical scale by (series_id, date)
    assert preds.set_index(["series_id", "date"]).index.isin(canonical_scale_map.index).all(), (
        "Some (series_id, date) pairs missing in canonical scale map!"
    )
    preds["scale"] = preds.set_index(["series_id", "date"]).index.map(canonical_scale_map).values

    # Save updated predictions.parquet
    preds.to_parquet(preds_path, index=False)
    print(f"Updated {preds_path} with canonical scales")

    # Re-compute overall
    overall = metrics_by_group(preds, ["model"], scale_col="scale")
    overall_csv_path = target_dir / "tier1_overall.csv"
    if overall_csv_path.exists():
        existing_overall = pd.read_csv(overall_csv_path)
        timing_cols = [c for c in ["mean_seconds", "group"] if c in existing_overall.columns]
        if timing_cols:
            overall = overall.merge(
                existing_overall[["model"] + timing_cols],
                on="model",
                how="left",
            )
    overall.to_csv(overall_csv_path, index=False)
    print(f"Wrote {overall_csv_path}")

    # Re-compute by origin
    by_origin = metrics_by_group(preds, ["model", "origin"], scale_col="scale")
    by_origin.to_csv(target_dir / "tier1_by_origin.csv", index=False)
    print(f"Wrote {target_dir / 'tier1_by_origin.csv'}")

    # Re-compute by category
    if "cat_id" not in preds.columns:
        preds_meta = preds.merge(meta_df[["series_id", "cat_id"]].drop_duplicates(), on="series_id", how="left")
    else:
        preds_meta = preds
    by_category = metrics_by_group(preds_meta, ["model", "cat_id"], scale_col="scale")
    by_category.to_csv(target_dir / "tier1_by_category.csv", index=False)
    print(f"Wrote {target_dir / 'tier1_by_category.csv'}")

    # Re-compute by level (item_store)
    by_level_path = target_dir / "tier1_by_level.csv"
    if by_level_path.exists():
        existing_level = pd.read_csv(by_level_path)
        higher_levels = existing_level[existing_level["level"] != "item_store"]
        item_level_rows = overall.copy()
        item_level_rows["level"] = "item_store"
        item_level_rows["n_aggregated_series"] = np.nan

        # Align columns
        new_by_level = pd.concat([item_level_rows, higher_levels], ignore_index=True)
        # Ensure consistent column ordering
        new_by_level = new_by_level[[c for c in existing_level.columns if c in new_by_level.columns]]
        new_by_level.to_csv(by_level_path, index=False)
        print(f"Wrote {by_level_path}")

    # Update manifest.json
    manifest_path = target_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_at"] = datetime.now().isoformat(timespec="seconds")

        ets_row = overall[overall["model"] == "autoets"]
        if len(ets_row):
            manifest["autoets_rmsse_level12"] = float(ets_row["rmsse"].iloc[0])

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {manifest_path} (autoets_rmsse_level12 = {manifest.get('autoets_rmsse_level12')})")


def main() -> None:
    bt_preds_path = PROJECT_ROOT / "reports" / "backtest_main" / "predictions.parquet"
    if not bt_preds_path.exists():
        raise FileNotFoundError(f"Missing {bt_preds_path}")

    print("Loading canonical backtest scales...")
    bt_preds = pd.read_parquet(bt_preds_path)
    canonical_scale_map = (
        bt_preds[["series_id", "date", "scale"]]
        .drop_duplicates()
        .set_index(["series_id", "date"])["scale"]
    )
    meta_df = bt_preds[["series_id", "cat_id", "store_id", "dept_id", "state_id"]].drop_duplicates()

    rescore_tier1_directory(PROJECT_ROOT / "reports" / "tier1", canonical_scale_map, meta_df)
    rescore_tier1_directory(PROJECT_ROOT / "reports" / "tier1_esbu", canonical_scale_map, meta_df)
    print("\nRescoring of tier1 and tier1_esbu complete!")


if __name__ == "__main__":
    main()
