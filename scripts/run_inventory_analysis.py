"""Run the full inventory optimisation pipeline.

Usage
-----
    python scripts/run_inventory_analysis.py                     # CA_1 + CA_2
    python scripts/run_inventory_analysis.py --stores CA_1       # single store
    python scripts/run_inventory_analysis.py --sample-series 500 # quick test

Outputs are saved to ``reports/inventory/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path for absolute imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging import get_logger  # noqa: E402
from src.config.settings import M5, settings  # noqa: E402
from src.data.loader import M5Loader  # noqa: E402
from src.inventory.abc_xyz import ABCXYZClassifier  # noqa: E402
from src.inventory.safety_stock import SafetyStockCalculator  # noqa: E402
from src.inventory.service_level import ServiceLevelAnalyzer  # noqa: E402
from src.inventory.working_capital import WorkingCapitalAnalyzer  # noqa: E402
from src.visualization.inventory_plots import (  # noqa: E402
    plot_abc_pareto,
    plot_abc_xyz_heatmap,
    plot_service_level_tradeoff,
    plot_working_capital_waterfall,
)

log = get_logger("inventory_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory optimisation pipeline")
    parser.add_argument(
        "--stores", nargs="*", default=list(M5.STORES_WITH_PRICE_DATA),
        help="Stores to analyse. Default: CA_1, CA_2 (stores with price data).",
    )
    parser.add_argument(
        "--sample-series", type=int, default=None,
        help="Random sample of series for quick testing.",
    )
    parser.add_argument(
        "--lead-time", type=int, default=7,
        help="Replenishment lead time in days (default 7).",
    )
    parser.add_argument(
        "--current-sl", type=float, default=0.95,
        help="Current blanket service level to compare against (default 0.95).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Default: reports/inventory/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else settings.base_dir / "reports" / "inventory"
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("output_dir", path=str(output_dir))

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    log.info("loading_data", stores=args.stores, sample=args.sample_series)
    loader = M5Loader()
    panel = loader.load_panel(
        stores=args.stores,
        sample_series=args.sample_series,
        with_prices=True,
        with_calendar=True,
    )
    log.info("panel_loaded", rows=len(panel), series=panel.groupby(["item_id", "store_id"], observed=True).ngroups)

    # Load raw prices for ABC
    prices = loader.load_prices()

    # ---------------------------------------------------------------
    # 2. ABC/XYZ classification
    # ---------------------------------------------------------------
    log.info("running_abc_xyz")
    classifier = ABCXYZClassifier()
    abc_xyz_df = classifier.classify(panel, prices=prices)
    abc_xyz_df.to_csv(output_dir / "abc_xyz_classification.csv", index=False)
    log.info("abc_xyz_saved", rows=len(abc_xyz_df))

    # Plots
    if abc_xyz_df["abc_class"].notna().any():
        plot_abc_pareto(abc_xyz_df, output_dir / "abc_pareto_curve.png")
        plot_abc_xyz_heatmap(abc_xyz_df, output_dir / "abc_xyz_heatmap.png")

    # ---------------------------------------------------------------
    # 3. Safety stock (classical approach using synthetic forecast errors)
    # ---------------------------------------------------------------
    # Without actual backtest forecast files, use panel demand statistics
    # as a proxy for forecast error (std of demand ≈ forecast error std
    # for a naive model).
    log.info("running_safety_stock")
    calc = SafetyStockCalculator(
        lead_time_days=args.lead_time,
        service_levels=(0.90, 0.95, 0.98),
    )

    # Build a synthetic forecast/actuals pair from the panel itself.
    # This uses the last 28 days as "test" and the preceding period for
    # a simple mean forecast -- giving genuine residuals.
    max_date = panel["date"].max()
    test_start = max_date - __import__("pandas").Timedelta(days=27)

    actuals = panel[panel["date"] >= test_start][["item_id", "store_id", "date", "sales"]].copy()
    train = panel[panel["date"] < test_start]

    mean_forecast = (
        train.groupby(["item_id", "store_id"], observed=True)["sales"]
        .mean()
        .reset_index()
        .rename(columns={"sales": "forecast"})
    )
    forecasts = actuals[["item_id", "store_id", "date"]].merge(
        mean_forecast, on=["item_id", "store_id"], how="left"
    )
    forecasts["forecast"] = forecasts["forecast"].fillna(0.0)

    ss_df = calc.from_forecast_errors(forecasts, actuals, panel)
    ss_df.to_csv(output_dir / "safety_stock.csv", index=False)
    log.info("safety_stock_saved", rows=len(ss_df))

    # ---------------------------------------------------------------
    # 4. Service-level scenario analysis
    # ---------------------------------------------------------------
    log.info("running_service_level_analysis")
    sla = ServiceLevelAnalyzer()
    scenarios = sla.analyze(ss_df, abc_xyz_df, prices=prices, panel=panel)
    scenarios.to_csv(output_dir / "service_level_scenarios.csv", index=False)

    seg_summary = sla.segment_summary(scenarios)
    seg_summary.to_csv(output_dir / "segment_summary.csv", index=False)

    marginals = sla.marginal_analysis(scenarios)
    marginals.to_csv(output_dir / "marginal_analysis.csv", index=False)

    if len(seg_summary) > 0:
        plot_service_level_tradeoff(seg_summary, output_dir / "service_level_tradeoff.png")

    # ---------------------------------------------------------------
    # 5. Working capital analysis
    # ---------------------------------------------------------------
    log.info("running_working_capital")
    wca = WorkingCapitalAnalyzer()
    kpis = wca.compute_kpis(scenarios)
    kpis.to_csv(output_dir / "inventory_kpis.csv", index=False)

    recs = wca.recommend_policy(kpis, current_sl=args.current_sl)
    recs.to_csv(output_dir / "policy_recommendations.csv", index=False)

    portfolio = wca.portfolio_summary(recs)
    portfolio.to_csv(output_dir / "portfolio_summary.csv", index=False)

    if len(portfolio) > 0:
        plot_working_capital_waterfall(portfolio, output_dir / "working_capital_waterfall.png")

    # ---------------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------------
    log.info(
        "pipeline_complete",
        output_dir=str(output_dir),
        files_written=len(list(output_dir.iterdir())),
        total_skus=len(abc_xyz_df),
        capital_released_total=round(float(portfolio["capital_released_total"].sum()), 2) if len(portfolio) > 0 else "N/A",
    )
    print(f"\n{'='*60}")
    print(f"Inventory analysis complete.  Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
