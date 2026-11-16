"""
Rescore backtest predictions using canonical per-series M5 RMSSE metrics.

Reads:
    reports/backtest_main/predictions.parquet
Writes:
    reports/backtest_main/accuracy_overall.csv
    reports/backtest_main/accuracy_by_origin.csv
    reports/backtest_main/accuracy_by_category.csv
    reports/backtest_main/accuracy_by_store.csv
    reports/backtest_main/accuracy_by_hierarchy_level.csv
    reports/backtest_main/manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.point_metrics import metrics_by_group
from src.evaluation.backtest import evaluate_by_hierarchy_level


def rescore_backtest(backtest_dir: Path) -> None:
    preds_path = backtest_dir / "predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")

    print(f"Loading {preds_path}...")
    preds = pd.read_parquet(preds_path)

    print("Computing metrics_by_group for overall...")
    overall = metrics_by_group(preds, ["model"], scale_col="scale")
    overall.to_csv(backtest_dir / "accuracy_overall.csv", index=False)

    print("Computing metrics_by_group by origin...")
    by_origin = metrics_by_group(preds, ["model", "origin"], scale_col="scale")
    by_origin.to_csv(backtest_dir / "accuracy_by_origin.csv", index=False)

    print("Computing metrics_by_group by category...")
    by_category = metrics_by_group(preds, ["model", "cat_id"], scale_col="scale")
    by_category.to_csv(backtest_dir / "accuracy_by_category.csv", index=False)

    print("Computing metrics_by_group by store...")
    by_store = metrics_by_group(preds, ["model", "store_id"], scale_col="scale")
    by_store.to_csv(backtest_dir / "accuracy_by_store.csv", index=False)

    print("Computing evaluate_by_hierarchy_level...")
    by_level = evaluate_by_hierarchy_level(preds)
    by_level.to_csv(backtest_dir / "accuracy_by_hierarchy_level.csv", index=False)

    manifest_path = backtest_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["champion"] = str(overall.iloc[0]["model"])
        manifest["champion_wape"] = float(overall.iloc[0]["wape"])
        manifest["champion_rmsse"] = float(overall.iloc[0]["rmsse"])
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Updated {manifest_path}")

    print("Rescoring complete!")


if __name__ == "__main__":
    backtest_dir = PROJECT_ROOT / "reports" / "backtest_main"
    rescore_backtest(backtest_dir)
