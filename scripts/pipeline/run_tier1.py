"""
Phase 5: evaluate Tier 1 statistical models through the existing harness.

Everything here is scored by `src.evaluation.point_metrics` on splits produced
by `src.evaluation.splits` -- no bespoke scoring path (spec constraint 4). The
only thing this script owns is fitting the models and arranging the results.

The validation target
---------------------
Priority 1 is AutoETS aggregated bottom-up, which reproduces the published
`ES_bu` benchmark. Makridakis et al. (2022), IJF 38:1346-1364, Table 2 gives
RMSSE 0.915 at level 12 (product-store). Landing near that independently
confirms the loader, splits and metrics are all correct simultaneously.

Read the comparison with the caveat this script prints alongside it: the
published number is dollar-weighted WRMSSE over all 30,490 series on the true
competition window, ours is unweighted RMSSE over a subsample on a shifted
window. Ours being somewhat worse is expected.

AutoARIMA
---------
Excluded from the level-12 run by default. Hyndman-Khandakar order selection
across thousands of 95%-zero series is expensive and the selected orders are
mostly degenerate -- differencing a series that is zero on 19 days in 20 does
not identify an AR or MA structure. `--with-arima` runs it on aggregated
levels, where series are continuous and the model is meaningful. This is a
documented methodological choice, not an omission (spec constraint 6).

Usage:
    python scripts/pipeline/run_tier1.py --sample-series 3000 --n-origins 3
    python scripts/pipeline/run_tier1.py --full --n-origins 5
    python scripts/pipeline/run_tier1.py --with-arima --sample-series 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.logging import get_logger  # noqa: E402
from src.config.settings import M5, settings  # noqa: E402
from src.data.loader import M5Loader  # noqa: E402
from src.evaluation.point_metrics import (  # noqa: E402
    metrics_by_group,
    naive_scale,
)
from src.evaluation.splits import assert_no_temporal_leakage, rolling_origins  # noqa: E402
from src.models.statistical.statsforecast_models import (  # noqa: E402
    DEFAULT_MODELS,
    LEVEL_KEYS,
    Tier1Runner,
    aggregate_bottom_up,
)

log = get_logger("tier1")

# Published M5 level-12 RMSSE reference points (Makridakis et al. 2022, Table 2).
PUBLISHED_LEVEL12 = {
    "Best of top-50 (wyzJack_STU)": 0.875,
    "Competition winner (YJ_STU)": 0.884,
    "ES_bu benchmark": 0.915,
    "Croston": 0.926,
}


def score_at_level(
    predictions: pd.DataFrame,
    meta: pd.DataFrame,
    level: str,
    scales: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Score every model at one aggregation level.

    At `item_store` the per-series RMSSE scale is carried through, so RMSSE is
    the M5-comparable quantity. Above that level, an aggregated series needs
    its OWN in-sample naive scale; it is recomputed from the aggregated
    actuals rather than reusing the bottom-level scale, which would be wrong
    by orders of magnitude.
    """
    if level == "item_store":
        df = predictions.copy()
        if scales and "scale" not in df.columns:
            df["scale"] = df["series_id"].map(scales)
        has_scale = "scale" in df.columns or bool(scales)
        metrics = metrics_by_group(df, ["model"], scale_col="scale" if has_scale else None)
        metrics["level"] = "item_store"
        return metrics

    agg = aggregate_bottom_up(predictions, meta, level)
    keys = [k for k in LEVEL_KEYS[level] if k in agg.columns]

    rows: list[dict[str, Any]] = []
    for model, g in agg.groupby("model", observed=True):
        # Per aggregated series: score, then average scaled errors across
        # them -- the same construction the M5 rules use one level down.
        series_rmsse: list[float] = []
        abs_err = 0.0
        actual_sum = 0.0

        grouper = g.groupby(keys, observed=True) if keys else [((), g)]
        for _, sub in grouper:
            sub = sub.sort_values("date")
            yt = sub["y_true"].to_numpy(float)
            yp = sub["y_pred"].to_numpy(float)
            abs_err += float(np.abs(yt - yp).sum())
            actual_sum += float(np.abs(yt).sum())
            scale = naive_scale(yt, seasonality=1)
            if np.isfinite(scale) and scale > 0:
                series_rmsse.append(float(np.sqrt(np.mean((yt - yp) ** 2) / scale)))

        rows.append(
            {
                "model": str(model),
                "level": level,
                "wape": 100.0 * abs_err / actual_sum if actual_sum > 0 else np.nan,
                "rmsse": float(np.mean(series_rmsse)) if series_rmsse else np.nan,
                "n_aggregated_series": len(series_rmsse) if series_rmsse else 1,
                "n_obs": len(g),
            }
        )
    return pd.DataFrame(rows).sort_values("wape").reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Tier 1 statistical model evaluation")
    p.add_argument("--sample-series", type=int, default=3000)
    p.add_argument("--full", action="store_true", help="all 30,490 series")
    p.add_argument("--n-origins", type=int, default=3)
    p.add_argument("--horizon", type=int, default=M5.HORIZON_DAYS)
    p.add_argument("--history-days", type=int, default=900)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument(
        "--with-arima", action="store_true", help="include AutoARIMA (slow; see module docstring)"
    )
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else settings.base_dir / "reports" / "tier1"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    if args.with_arima and "autoarima" not in models:
        models.append("autoarima")

    t_start = time.time()

    # --- load ---
    loader = M5Loader()
    panel = loader.load_panel(
        sample_series=None if args.full else args.sample_series,
        last_n_days=args.history_days,
        with_prices=False,
        random_state=args.seed,
    )
    panel["series_id"] = panel["item_id"].astype(str) + "_" + panel["store_id"].astype(str)

    meta = (
        panel.groupby("series_id", observed=True)
        .agg(
            item_id=("item_id", "first"),
            store_id=("store_id", "first"),
            dept_id=("dept_id", "first"),
            cat_id=("cat_id", "first"),
            state_id=("state_id", "first"),
        )
        .reset_index()
    )
    n_series = len(meta)

    origins = rolling_origins(
        n_origins=args.n_origins,
        horizon=args.horizon,
        last_date=panel["date"].max(),
        first_date=panel["date"].min(),
    )

    # --- run every origin ---
    all_preds: list[pd.DataFrame] = []
    all_timings: list[pd.DataFrame] = []
    scales: dict[str, float] = {}
    failures: dict[str, str] = {}

    for origin in origins:
        train, test = origin.split_frame(panel)
        assert_no_temporal_leakage(train, test)

        log.info("tier1_origin", detail=origin.describe(), n_train=len(train), n_test=len(test))

        # RMSSE scale from the TRAINING window, per M5 rules. Recomputed each
        # origin because the training window differs.
        origin_scales = {
            sid: naive_scale(g.sort_values("date")["sales"].to_numpy(float), seasonality=1)
            for sid, g in train.groupby("series_id", observed=True)
        }
        scales = origin_scales

        runner = Tier1Runner(models=models, n_jobs=args.n_jobs)
        result = runner.fit_predict(train, horizon=args.horizon)
        failures.update(result.models_failed)

        if result.predictions.empty:
            log.error("tier1_no_predictions", origin=origin.name)
            continue

        actuals = test[["series_id", "date", "sales"]].rename(columns={"sales": "y_true"})
        block = result.predictions.merge(actuals, on=["series_id", "date"], how="inner")
        block["origin"] = origin.name
        block["scale"] = block["series_id"].map(origin_scales)
        all_preds.append(block)

        timings = result.timings.copy()
        timings["origin"] = origin.name
        all_timings.append(timings)

    if not all_preds:
        print("\nNo predictions produced. Check the failures above.\n")
        return 1

    preds = pd.concat(all_preds, ignore_index=True)
    timings = pd.concat(all_timings, ignore_index=True)

    # --- score through the harness, at every level ---
    overall = metrics_by_group(preds, ["model"], scale_col="scale")
    by_origin = metrics_by_group(preds, ["model", "origin"], scale_col="scale")
    by_category = (
        metrics_by_group(preds, ["model", "cat_id"], scale_col="scale")
        if "cat_id" in preds.columns
        else pd.DataFrame()
    )

    # Attach segment metadata for the category breakdown.
    if by_category.empty:
        preds_meta = preds.merge(meta, on="series_id", how="left")
        by_category = metrics_by_group(preds_meta, ["model", "cat_id"], scale_col="scale")

    level_frames = [score_at_level(preds, meta, lvl, scales) for lvl in LEVEL_KEYS]
    by_level = pd.concat(level_frames, ignore_index=True)
    by_level["level"] = pd.Categorical(
        by_level["level"], categories=list(M5.HIERARCHY_LEVELS), ordered=True
    )
    by_level = by_level.sort_values(["level", "wape"]).reset_index(drop=True)

    mean_timings = (
        timings.groupby(["model", "group"], observed=True)["fit_predict_seconds"]
        .mean()
        .reset_index()
        .rename(columns={"fit_predict_seconds": "mean_seconds"})
    )
    overall = overall.merge(mean_timings, on="model", how="left")

    # --- persist ---
    for name, frame in {
        "tier1_overall": overall,
        "tier1_by_origin": by_origin,
        "tier1_by_category": by_category,
        "tier1_by_level": by_level,
        "tier1_timings": timings,
    }.items():
        if not frame.empty:
            frame.to_csv(out_dir / f"{name}.csv", index=False)
    preds.to_parquet(out_dir / "predictions.parquet", index=False)

    # --- MLflow ---
    mlflow_logged = False
    if not args.no_mlflow:
        mlflow_logged = _log_mlflow(overall, by_level, args, n_series)

    # --- the ES_bu validation check ---
    item_level = by_level[by_level["level"] == "item_store"]
    ets_row = item_level[item_level["model"] == "autoets"]
    ets_rmsse = float(ets_row["rmsse"].iloc[0]) if len(ets_row) else float("nan")

    manifest = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "n_series": int(n_series),
        "n_origins": args.n_origins,
        "horizon": args.horizon,
        "models_requested": models,
        "models_failed": failures,
        "origins": [o.to_dict() for o in origins],
        "autoets_rmsse_level12": ets_rmsse,
        "published_reference": PUBLISHED_LEVEL12,
        "comparability_caveat": (
            "Published level-12 figures are dollar-weighted WRMSSE over all "
            "30,490 series on the true competition test window. This run is an "
            "unweighted mean RMSSE over a subsample on a shifted window (the "
            "published window has no actuals in this data release). A somewhat "
            "worse figure here is expected, not a defect."
        ),
        "arima_note": (
            "AutoARIMA included"
            if args.with_arima
            else "AutoARIMA excluded at level 12: order selection across 95%-zero "
            "series is expensive and yields degenerate orders. Run with "
            "--with-arima, or at aggregated levels where series are continuous."
        ),
        "mlflow_logged": mlflow_logged,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    # --- console report ---
    pd.set_option("display.width", 200)
    print("\n" + "=" * 88)
    print("TIER 1 -- STATISTICAL MODELS")
    print("=" * 88)
    print(
        f"  series: {n_series:,}   origins: {args.n_origins}   horizon: {args.horizon}d   "
        f"models: {len(overall)}   elapsed: {manifest['elapsed_seconds']}s"
    )
    if failures:
        print(f"  FAILED: {failures}")

    cols = [
        c
        for c in ("model", "wape", "rmsse", "mase", "mae", "bias", "mean_seconds")
        if c in overall.columns
    ]
    print("\n--- OVERALL, item-store level (all origins pooled) " + "-" * 36)
    print(overall[cols].round(3).to_string(index=False))

    print("\n--- BY HIERARCHY LEVEL: RMSSE " + "-" * 57)
    piv = by_level.pivot_table(index="model", columns="level", values="rmsse", observed=True)
    print(piv.round(3).to_string())

    print("\n--- ES_bu VALIDATION CHECK " + "-" * 60)
    print(f"  AutoETS RMSSE at item-store (level 12): {ets_rmsse:.4f}")
    for label, value in PUBLISHED_LEVEL12.items():
        print(f"    published {label:<32} {value:.3f}")
    print(f"\n  {manifest['comparability_caveat']}")

    print(f"\n  Artefacts: {out_dir}")
    print("=" * 88 + "\n")
    return 0


def _log_mlflow(overall: pd.DataFrame, by_level: pd.DataFrame, args, n_series: int) -> bool:
    """
    Log every Tier 1 model to the shared MLflow experiment.

    Same experiment as Tiers 2 and 5 so the leaderboard is cross-tier
    comparable. Falls back to a local SQLite store when no server is up, and
    never raises -- losing tracking must not lose the run.
    """
    try:
        import socket
        from urllib.parse import urlparse

        import mlflow

        uri = settings.mlflow.tracking_uri
        if uri.startswith(("http://", "https://")):
            parsed = urlparse(uri)
            try:
                with socket.create_connection(
                    (parsed.hostname or "localhost", parsed.port or 80), timeout=1.5
                ):
                    pass
            except OSError:
                uri = f"sqlite:///{(settings.base_dir / 'mlflow.db').as_posix()}"

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(f"{settings.mlflow.experiment_prefix}/m5_model_zoo")

        for _, row in overall.iterrows():
            model = str(row["model"])
            with mlflow.start_run(run_name=f"tier1_{model}"):
                mlflow.set_tags(
                    {
                        "tier": "1_statistical",
                        "model": model,
                        "library": "statsforecast",
                        "n_series": str(n_series),
                        "n_origins": str(args.n_origins),
                        "horizon": str(args.horizon),
                    }
                )
                metrics = {
                    f"item_store_{k}": float(row[k])
                    for k in ("wape", "rmsse", "mase", "mae", "bias")
                    if k in row and np.isfinite(row[k])
                }
                if "mean_seconds" in row and np.isfinite(row["mean_seconds"]):
                    metrics["fit_predict_seconds"] = float(row["mean_seconds"])

                for _, lvl in by_level[by_level["model"] == model].iterrows():
                    level = str(lvl["level"])
                    for metric in ("wape", "rmsse"):
                        value = lvl.get(metric)
                        if value is not None and np.isfinite(value):
                            metrics[f"{level}_{metric}"] = float(value)

                mlflow.log_metrics(metrics)
        log.info("mlflow_logged", experiment="m5_model_zoo", n_models=len(overall))
        return True
    except Exception as exc:
        log.warning("mlflow_logging_failed", error=str(exc)[:200])
        return False


if __name__ == "__main__":
    raise SystemExit(main())
