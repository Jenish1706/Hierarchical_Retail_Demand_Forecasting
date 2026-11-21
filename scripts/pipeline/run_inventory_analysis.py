"""Run the full inventory optimisation pipeline.

Usage
-----
    python scripts/pipeline/run_inventory_analysis.py                     # CA_1 + CA_2
    python scripts/pipeline/run_inventory_analysis.py --stores CA_1       # single store
    python scripts/pipeline/run_inventory_analysis.py --sample-series 500 # quick test

Outputs are saved to ``reports/inventory/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path for absolute imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.logging import get_logger  # noqa: E402
from src.config.settings import M5, settings  # noqa: E402
from src.data.loader import M5Loader  # noqa: E402
from src.inventory.abc_xyz import ABCXYZClassifier  # noqa: E402
from src.inventory.margin_sensitivity import (  # noqa: E402
    break_even_table,
    margin_sensitivity,
)
from src.inventory.safety_stock import (  # noqa: E402
    SafetyStockCalculator,
    _z_score,
    summarise_forecast_errors,
)
from src.inventory.service_level import DEFAULT_COST_MARGINS, ServiceLevelAnalyzer  # noqa: E402
from src.inventory.working_capital import WorkingCapitalAnalyzer  # noqa: E402
from src.visualization.inventory_plots import (  # noqa: E402
    plot_abc_pareto,
    plot_abc_xyz_heatmap,
    plot_service_level_tradeoff,
    plot_working_capital_waterfall,
)

log = get_logger("inventory_pipeline")

# The backtest evaluation window: N_ORIGINS consecutive HORIZON-day test blocks ending at the
# panel's last date.  These are the *scoring* days (M5-10).  Decisions are made without them:
#   decision window : everything up to the day before the first test day.  ABC revenue uses that
#                     history; XYZ, mean demand for the classes and the no-sales (N) check use its
#                     last EVAL_WINDOW_DAYS days (ending 2016-01-31 on M5).
#   planning demand : the champion's mean forecast over the scoring days (a planner has forecasts,
#                     not future sales).
#   sigma_e         : out-of-sample forecast-error size measured on the scoring days, the standard
#                     backtest proxy; it is not a realised inventory outcome.
N_ORIGINS = 3
HORIZON = 28
EVAL_WINDOW_DAYS = N_ORIGINS * HORIZON


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
    parser.add_argument(
        "--ca-predictions", type=str, default=None,
        help="Path to precomputed CA backtest predictions parquet file.",
    )
    parser.add_argument(
        "--rerun-backtest", action="store_true",
        help="Force re-running the rolling-origin backtest for CA stores.",
    )
    parser.add_argument(
        "--champion-model", type=str, default="adida",
        help="Champion model to use for inventory safety stock (default: adida).",
    )
    return parser.parse_args()


def _save_table(df: pd.DataFrame, file_path: Path) -> None:
    """Save DataFrame as clean ASCII CSV and compressed Parquet."""
    file_path = Path(file_path)
    csv_path = file_path.with_suffix(".csv")
    parquet_path = file_path.with_suffix(".parquet")
    df.to_csv(csv_path, index=False, encoding="ascii", lineterminator="\n")
    try:
        df.to_parquet(parquet_path, compression="snappy")
    except Exception as e:
        log.warning("parquet_save_fallback", path=str(parquet_path), error=str(e))


def ensure_ca_backtest_predictions(
    panel: pd.DataFrame,
    loader: M5Loader,
    output_path: Path,
    force_rerun: bool = False,
    n_origins: int = N_ORIGINS,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Load or generate rolling-origin backtest predictions on California stores.

    Evaluates the statistical benchmark models (naive, seasonal_naive_weekly,
    seasonal_naive, moving_average, ses, croston_sba, adida) plus the historical_mean
    baseline across canonical rolling origins.
    """
    output_path = Path(output_path)
    if not force_rerun and output_path.exists():
        log.info("loading_cached_ca_predictions", path=str(output_path))
        return pd.read_parquet(output_path)

    log.info("running_ca_backtest", n_origins=n_origins, horizon=horizon)
    from src.evaluation.backtest import build_origins, run_series_models

    p = panel.copy()
    if "series_id" not in p.columns:
        p["series_id"] = p["item_id"].astype(str) + "_" + p["store_id"].astype(str)

    origins = build_origins(p["date"], horizon=horizon, n_origins=n_origins)
    fs_dates = loader.load_first_sale_dates()

    all_preds: list[pd.DataFrame] = []
    for orig in origins:
        log.info(
            "running_origin",
            origin=orig.index,
            train_end=str(orig.train_end.date()),
            test_start=str(orig.test_start.date()),
        )
        preds, scales, _ = run_series_models(
            p, orig, horizon=horizon, first_sale_dates=fs_dates
        )

        # Include historical mean baseline for comparison with legacy synthetic path
        train = p[p["date"] <= orig.train_end]
        test = p[(p["date"] >= orig.test_start) & (p["date"] <= orig.test_end)]
        means = train.groupby("series_id", observed=True)["sales"].mean().to_dict()
        test_dates = pd.to_datetime(sorted(test["date"].unique()))
        mean_rows = []
        for sid, g in test.groupby("series_id", observed=True):
            m_val = float(means.get(sid, 0.0))
            act = g.sort_values("date")["sales"].to_numpy(dtype=float)
            n = min(len(test_dates), len(act))
            for i in range(n):
                mean_rows.append({
                    "origin": orig.index,
                    "series_id": sid,
                    "model": "historical_mean",
                    "date": test_dates[i],
                    "y_true": float(act[i]),
                    "y_pred": float(max(m_val, 0.0)),
                    "scale": float(scales.get(sid, 1.0)),
                    "item_id": sid.rsplit("_", 2)[0],
                    "store_id": sid.rsplit("_", 2)[1] + "_" + sid.rsplit("_", 2)[2],
                    "dept_id": sid.rsplit("_", 3)[0],
                    "cat_id": sid.split("_")[0],
                    "state_id": "CA",
                })
        preds_with_mean = pd.concat([preds, pd.DataFrame(mean_rows)], ignore_index=True)
        all_preds.append(preds_with_mean)

    df_preds = pd.concat(all_preds, ignore_index=True)
    df_preds["model"] = df_preds["model"].astype("category")
    df_preds["series_id"] = df_preds["series_id"].astype("category")
    df_preds["y_true"] = df_preds["y_true"].astype("float32")
    df_preds["y_pred"] = df_preds["y_pred"].astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(output_path, compression="zstd")
    log.info("saved_ca_predictions", path=str(output_path), rows=len(df_preds))
    return df_preds


def decision_panel_cutoff(panel: pd.DataFrame) -> pd.Timestamp:
    """The last day a decision may use: the day before the first scoring day."""
    panel_dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    return panel_dates[-(EVAL_WINDOW_DAYS + 1)]


def split_panel_for_decisions(
    panel: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the panel into (decision, scoring) rows and check the predictions are the scoring days.

    The scoring days are the last ``EVAL_WINDOW_DAYS`` panel days, which the backtest predictions
    must cover exactly; the decision rows are everything before them.  Segmentation is built from
    the decision rows only, so a class (which decides who gets 98% or 90%) is never chosen with
    knowledge of the sales it will be scored on.  Cached predictions that cover other days would
    break that split silently, so this raises instead.
    """
    pred_dates = pd.DatetimeIndex(sorted(pd.to_datetime(predictions["date"]).unique()))
    panel_dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    expected = panel_dates[-EVAL_WINDOW_DAYS:]
    if not pred_dates.equals(expected):
        raise ValueError(
            "Cached CA predictions cover "
            f"{pred_dates.min().date()}..{pred_dates.max().date()} ({len(pred_dates)} days) but the "
            f"evaluation window is the last {EVAL_WINDOW_DAYS} panel days "
            f"({expected.min().date()}..{expected.max().date()}). Re-run with --rerun-backtest."
        )
    scoring = panel[panel["date"].isin(expected)]
    decision = panel[panel["date"] <= decision_panel_cutoff(panel)]
    return decision, scoring


def demand_bases(
    panel: pd.DataFrame, scoring: pd.DataFrame, predictions: pd.DataFrame, champion: str
) -> dict[str, pd.DataFrame]:
    """Mean daily demand per SKU under three bases, for the working-capital comparison.

    ``forecast``        the champion's mean forecast on the scoring days (the planning basis)
    ``actual_backtest_days``  mean actual sales on the scoring days (look-ahead; reference only)
    ``full_history_mean``     mean over the whole panel, the figure the README used before M5-10
                              (it includes the scoring days, at about 4% weight)
    ``forecast_first_origin`` the champion's forecast over the first 28 scored days only. The
                              forecasts for the later two windows use data up to their own origins,
                              which a planner deciding at the first origin would not have had, so
                              ``forecast`` is a slightly generous planning basis; this row bounds that.
    """
    key = ["item_id", "store_id"]
    champ = predictions[predictions["model"] == champion].copy()
    sids = champ["series_id"].astype(str)
    champ["item_id"] = sids.str.rsplit("_", n=2).str[0]
    champ["store_id"] = sids.str.rsplit("_", n=2).str[1] + "_" + sids.str.rsplit("_", n=2).str[2]
    forecast = champ.groupby(key, observed=True)["y_pred"].mean().rename("demand").reset_index()
    actual = scoring.groupby(key, observed=True)["sales"].mean().rename("demand").reset_index()
    full = panel.groupby(key, observed=True)["sales"].mean().rename("demand").reset_index()
    first_window_end = champ["date"].min() + pd.Timedelta(days=HORIZON - 1)
    first = champ[champ["date"] <= first_window_end]
    forecast_first = first.groupby(key, observed=True)["y_pred"].mean().rename("demand").reset_index()
    return {
        "forecast": forecast,
        "forecast_first_origin": forecast_first,
        "actual_backtest_days": actual,
        "full_history_mean": full,
    }


def compute_safety_stock_by_model(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    lead_time: int = 7,
    csl: float = 0.95,
) -> pd.DataFrame:
    """Evaluate required safety stock units and working capital across models at fixed CSL.

    The dollar value of a better forecast: connects forecast error (sigma_e) directly to
    required safety stock capital at 95% CSL.

    sigma_e is the per-SKU RMSE of the residuals (error about zero), **not** their standard
    deviation: a standard deviation removes each model's mean error and so costs a
    consistently-too-low forecast as if it were accurate (M5-12b). The std and the mean error
    (``mean_bias``, actual - forecast; positive = under-forecast) are reported alongside for
    transparency, but only the RMSE sizes the stock.
    """
    df = predictions.copy()
    if "residual" not in df.columns:
        # float64: the cached parquet stores float32, which would round every residual.
        df["residual"] = df["y_true"].astype(float) - df["y_pred"].astype(float)
    if "item_id" not in df.columns or "store_id" not in df.columns:
        sids = df["series_id"].astype(str)
        df["item_id"] = sids.str.rsplit("_", n=2).str[0]
        df["store_id"] = sids.str.rsplit("_", n=2).str[1] + "_" + sids.str.rsplit("_", n=2).str[2]

    # Pooled WAPE per model on these CA series (sum|error| / sum(actual)). Reported next to
    # the RMSE so the champion choice can be checked against this panel, not only the main one.
    abs_err = df["residual"].abs().groupby(df["model"], observed=True).sum()
    actual = df["y_true"].astype(float).groupby(df["model"], observed=True).sum()
    wape_pct = (100.0 * abs_err / actual).rename("wape_pct").reset_index()

    # Load and aggregate sell prices for CA stores
    prices_ca = prices[prices["store_id"].isin(["CA_1", "CA_2"])]
    avg_prices = (
        prices_ca.groupby(["item_id", "store_id"], observed=True)["sell_price"]
        .mean()
        .reset_index()
        .rename(columns={"sell_price": "avg_sell_price"})
    )
    avg_prices["cat_id"] = avg_prices["item_id"].str.split("_").str[0]
    avg_prices["margin"] = avg_prices["cat_id"].map(DEFAULT_COST_MARGINS).fillna(0.50)
    avg_prices["unit_cost"] = avg_prices["avg_sell_price"] * (1.0 - avg_prices["margin"])

    # Forecast error per model and SKU over the full 84 test days (RMSE sizes the stock)
    grouped = summarise_forecast_errors(df, by=["model", "item_id", "store_id"])
    grouped = grouped.merge(avg_prices, on=["item_id", "store_id"], how="left")

    z = _z_score(csl)
    sqrt_lt = np.sqrt(lead_time)
    grouped["ss_units"] = z * grouped["forecast_error_rmse"] * sqrt_lt
    grouped["ss_capital"] = grouped["ss_units"] * grouped["unit_cost"].fillna(0.0)

    summary = (
        grouped.groupby("model", observed=True)
        .agg(
            mean_rmse=("forecast_error_rmse", "mean"),
            mean_std=("forecast_error_std", "mean"),
            mean_bias=("forecast_error_bias", "mean"),
            total_ss_units=("ss_units", "sum"),
            total_ss_capital=("ss_capital", "sum"),
            priced_skus=("unit_cost", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    summary = summary.merge(wape_pct, on="model", how="left")

    naive_row = summary.loc[summary["model"] == "naive"]
    naive_cap = float(naive_row["total_ss_capital"].values[0]) if len(naive_row) > 0 else 0.0

    summary["capital_reduction_vs_naive"] = naive_cap - summary["total_ss_capital"]
    summary["pct_reduction_vs_naive"] = np.where(
        naive_cap > 0,
        (summary["capital_reduction_vs_naive"] / naive_cap) * 100.0,
        0.0,
    )

    summary = summary.sort_values("total_ss_capital").reset_index(drop=True)
    for col in ("mean_rmse", "mean_std", "mean_bias"):
        summary[col] = summary[col].round(4)
    summary["wape_pct"] = summary["wape_pct"].round(2)
    summary["total_ss_units"] = summary["total_ss_units"].round(2)
    summary["total_ss_capital"] = summary["total_ss_capital"].round(2)
    summary["capital_reduction_vs_naive"] = summary["capital_reduction_vs_naive"].round(2)
    summary["pct_reduction_vs_naive"] = summary["pct_reduction_vs_naive"].round(2)

    return summary


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
    # Segmentation is a decision, so it only sees data before the scoring days: ABC revenue uses the
    # history up to the cutoff and XYZ / N use its last EVAL_WINDOW_DAYS days (ending the day
    # before the first test day).  The test days are for scoring only.
    cutoff = decision_panel_cutoff(panel)
    decision_panel = panel[panel["date"] <= cutoff]
    classifier = ABCXYZClassifier(cv_window_days=EVAL_WINDOW_DAYS)
    abc_xyz_df = classifier.classify(decision_panel, prices=prices)
    log.info("segmentation_window", decision_cutoff=str(cutoff.date()), xyz_days=EVAL_WINDOW_DAYS)
    _save_table(abc_xyz_df, output_dir / "abc_xyz_classification")
    log.info("abc_xyz_saved", rows=len(abc_xyz_df))

    # Plots
    if abc_xyz_df["abc_class"].notna().any():
        plot_abc_pareto(abc_xyz_df, output_dir / "abc_pareto_curve.png")
        plot_abc_xyz_heatmap(abc_xyz_df, output_dir / "abc_xyz_heatmap.png")

    # ---------------------------------------------------------------
    # 3. Safety stock from rolling-origin backtest forecasts (CA_1, CA_2)
    # ---------------------------------------------------------------
    log.info("running_safety_stock")
    ca_preds_path = Path(args.ca_predictions) if args.ca_predictions else output_dir / "ca_predictions.parquet"

    # In sample mode, force rerun on the sample panel; otherwise check cache/generate
    force_rerun = args.rerun_backtest or (args.sample_series is not None)
    ca_preds = ensure_ca_backtest_predictions(
        panel=panel,
        loader=loader,
        output_path=ca_preds_path,
        force_rerun=force_rerun,
    )
    decision_panel, scoring_panel = split_panel_for_decisions(panel, ca_preds)
    log.info(
        "decision_scoring_split",
        decision_end=str(decision_panel["date"].max().date()),
        scoring_start=str(scoring_panel["date"].min().date()),
        scoring_end=str(scoring_panel["date"].max().date()),
        scoring_days=EVAL_WINDOW_DAYS,
    )

    # 3a. Model comparison: "The dollar value of a better forecast"
    log.info("running_safety_stock_by_model")
    model_ss_df = compute_safety_stock_by_model(
        predictions=ca_preds,
        prices=prices,
        lead_time=args.lead_time,
        csl=0.95,
    )
    _save_table(model_ss_df, output_dir / "safety_stock_by_model")
    log.info("safety_stock_by_model_saved", models=len(model_ss_df))

    # 3b. Champion model (ADIDA) safety stock calculation
    calc = SafetyStockCalculator(
        lead_time_days=args.lead_time,
        service_levels=(0.90, 0.95, 0.98),
    )
    champion_preds = ca_preds[ca_preds["model"] == args.champion_model]
    if len(champion_preds) == 0:
        log.warning("champion_model_not_found", champion=args.champion_model, fallback="adida")
        champion_preds = ca_preds[ca_preds["model"] == "adida"]

    ss_df = calc.from_forecast_errors(
        champion_preds,
        panel=None,  # planning demand = the champion's mean forecast; actuals are scoring-only
        model=args.champion_model,
    )
    _save_table(ss_df, output_dir / "safety_stock")
    log.info("safety_stock_saved", rows=len(ss_df), model=args.champion_model)

    # ---------------------------------------------------------------
    # 4. Service-level scenario analysis
    # ---------------------------------------------------------------
    log.info("running_service_level_analysis")
    sla = ServiceLevelAnalyzer()
    scenarios = sla.analyze(ss_df, abc_xyz_df, prices=prices, panel=panel)
    _save_table(scenarios, output_dir / "service_level_scenarios")

    seg_summary = sla.segment_summary(scenarios)
    _save_table(seg_summary, output_dir / "segment_summary")

    marginals = sla.marginal_analysis(scenarios)
    _save_table(marginals, output_dir / "marginal_analysis")

    if len(seg_summary) > 0:
        plot_service_level_tradeoff(seg_summary, output_dir / "service_level_tradeoff.png")

    # ---------------------------------------------------------------
    # 5. Working capital analysis
    # ---------------------------------------------------------------
    log.info("running_working_capital")
    wca = WorkingCapitalAnalyzer()
    kpis = wca.compute_kpis(scenarios)
    _save_table(kpis, output_dir / "inventory_kpis")

    recs = wca.recommend_policy(kpis, current_sl=args.current_sl)
    _save_table(recs, output_dir / "policy_recommendations")

    portfolio = wca.portfolio_summary(recs)
    _save_table(portfolio, output_dir / "portfolio_summary")

    if len(portfolio) > 0:
        plot_working_capital_waterfall(portfolio, output_dir / "working_capital_waterfall.png")

    # 5a. Policies side by side: capital released is only meaningful next to annual cost (M5-10).
    policy_cmp = wca.compare_policies(kpis, current_sl=args.current_sl)
    _save_table(policy_cmp, output_dir / "policy_comparison")
    segment_opt = wca.segment_cost_optimum(kpis, current_sl=args.current_sl)
    _save_table(segment_opt, output_dir / "segment_cost_optimum")

    # 5a'. How far the 98%-everywhere conclusion moves with the assumed gross margin (M5-15).
    # Break-evens come from the same cost formula the analyzer uses, not from hand-typed numbers.
    _save_table(
        margin_sensitivity(sla.cost_margins, holding_cost_rate=sla.holding_cost_rate),
        output_dir / "margin_sensitivity",
    )
    _save_table(
        break_even_table(holding_cost_rate=sla.holding_cost_rate),
        output_dir / "margin_break_even",
    )

    # 5b. Current working capital under three mean-demand bases.  Only the forecast is a planning
    # input; actual sales on the scored days are look-ahead and the full-history mean still
    # contains those days (about 4% weight), so both are reference only.
    bases = demand_bases(panel, scoring_panel, ca_preds, args.champion_model)
    basis_table = wca.demand_basis_comparison(
        kpis, bases, current_sl=args.current_sl, planning_basis="forecast"
    )
    _save_table(basis_table, output_dir / "working_capital_demand_bases")
    for _, r in policy_cmp.iterrows():
        log.info(
            "policy_compared",
            policy=r["policy"],
            wc_change=r["wc_change_vs_current"],
            total_cost_change=r["total_cost_change"],
        )

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
