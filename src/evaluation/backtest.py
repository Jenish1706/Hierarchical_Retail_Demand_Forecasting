"""
Walk-forward backtest: model x segment accuracy on real M5 data.

What this produces
------------------
A clean accuracy table -- WAPE and RMSSE broken down by model, by category, by
store, and by hierarchy level -- computed on genuine out-of-sample forecasts
over multiple rolling origins. Everything a reader needs to judge whether the
machine learning earned its complexity, in one artefact.

Protocol
--------
Rolling-origin evaluation (Tashman 2000). For each of `--n-origins` origins,
spaced `--horizon` days apart and ending at the last available date:

    |--------------- train ---------------|=== test (28d) ===|
    |------------- train -------------|=== test (28d) ===|
    |----------- train -----------|=== test (28d) ===|

The model is refit at every origin on data strictly before it. A single
train/test split would give one draw from a noisy distribution; several
origins show whether a ranking is stable or an artefact of one lucky window.

Why RMSSE is computed per series
--------------------------------
RMSSE scales squared error by each series' own in-sample naive error. That
scale is a property of the SERIES, and the M5 rules require it to come from
the training window. So it is computed once per (series, origin) from that
origin's training data and carried alongside each prediction row -- never
recomputed on the test window, which would leak, and never pooled across
series, which would destroy the scale-invariance that makes RMSSE worth using.

Usage:
    python scripts/backtest.py --sample-series 300 --n-origins 3
    python scripts/backtest.py --stores CA_1 CA_2 --categories FOODS
    python scripts/backtest.py --full            # all 30,490 series (slow)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
from src.features.base import FeaturePipeline  # noqa: E402
from src.models.router import SegmentRouter  # noqa: E402
from src.models.statistical.intermittent import (  # noqa: E402
    ADIDAForecaster,
    CrostonForecaster,
)
from src.models.statistical.naive import build_baseline  # noqa: E402

log = get_logger("backtest")

# Per-series baselines evaluated at every origin. Cheap, and they define the
# bar the global model has to clear.
BASELINE_MODELS = ("naive", "seasonal_naive_weekly", "seasonal_naive", "moving_average", "ses")


@dataclass
class Origin:
    """One rolling-origin split."""

    index: int
    train_end: pd.Timestamp  # last date the model may see
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> str:
        return (
            f"origin {self.index}: train <= {self.train_end.date()}, "
            f"test {self.test_start.date()}..{self.test_end.date()}"
        )


def build_origins(dates: pd.Series, horizon: int, n_origins: int) -> list[Origin]:
    """
    Construct non-overlapping rolling origins ending at the last date.

    Non-overlapping test windows keep the origins close to independent draws,
    so averaging across them is meaningful rather than double-counting the
    same days.
    """
    unique_dates = np.sort(pd.to_datetime(dates.unique()))
    if len(unique_dates) < horizon * (n_origins + 1):
        raise ValueError(
            f"need at least {horizon * (n_origins + 1)} days for {n_origins} "
            f"origins at horizon {horizon}; have {len(unique_dates)}"
        )

    origins: list[Origin] = []
    for i in range(n_origins):
        # Origin 0 is the most recent; each earlier one steps back a horizon.
        end_pos = len(unique_dates) - 1 - i * horizon
        start_pos = end_pos - horizon + 1
        origins.append(
            Origin(
                index=i,
                train_end=pd.Timestamp(unique_dates[start_pos - 1]),
                test_start=pd.Timestamp(unique_dates[start_pos]),
                test_end=pd.Timestamp(unique_dates[end_pos]),
            )
        )
    return list(reversed(origins))  # chronological order


# ---------------------------------------------------------------------------
# Per-series models
# ---------------------------------------------------------------------------


def run_series_models(
    panel: pd.DataFrame,
    origin: Origin,
    horizon: int,
    first_sale_dates: dict[str, pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, dict[str, float], SegmentRouter]:
    """
    Fit and score every per-series model at one origin.

    Returns:
        (long predictions frame, {series_id: rmsse_scale}, fitted router)
    """
    train = panel[panel["date"] <= origin.train_end]
    test = panel[(panel["date"] >= origin.test_start) & (panel["date"] <= origin.test_end)]

    history: dict[str, np.ndarray] = {}
    actuals: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    first_sale_indices: dict[str, int] = {}
    meta: dict[str, dict[str, Any]] = {}

    for sid, g in train.groupby("series_id", observed=True):
        g = g.sort_values("date")
        y = g["sales"].to_numpy(dtype=float)
        history[sid] = y

        fs_idx = None
        if first_sale_dates is not None and sid in first_sale_dates:
            fs_date = first_sale_dates[sid]
            if pd.isna(fs_date):
                fs_idx = len(y)
            else:
                train_dates = pd.to_datetime(g["date"].values)
                if fs_date <= train_dates[0]:
                    fs_idx = 0  # Launched at or before start of window -> keep in-window zeros
                elif fs_date <= train_dates[-1]:
                    fs_idx = int(np.searchsorted(train_dates, fs_date))  # Launched in window -> trim pre-launch
                else:
                    fs_idx = len(y)  # Not yet launched
            first_sale_indices[sid] = fs_idx

        # RMSSE denominator: in-sample naive MSE on the TRAINING window only.
        scales[sid] = naive_scale(y, seasonality=1, first_sale_idx=fs_idx)
        meta[sid] = {
            "item_id": g["item_id"].iloc[0],
            "store_id": g["store_id"].iloc[0],
            "dept_id": g["dept_id"].iloc[0],
            "cat_id": g["cat_id"].iloc[0],
            "state_id": g["state_id"].iloc[0],
        }

    for sid, g in test.groupby("series_id", observed=True):
        actuals[sid] = g.sort_values("date")["sales"].to_numpy(dtype=float)

    common = sorted(set(history) & set(actuals))
    test_dates = pd.to_datetime(sorted(test["date"].unique()))

    rows: list[dict[str, Any]] = []

    # --- baselines ---
    for model_name in BASELINE_MODELS:
        for sid in common:
            try:
                preds = build_baseline(model_name).fit_predict(history[sid], horizon)
            except Exception:
                preds = np.zeros(horizon)
            rows.extend(
                _emit(
                    sid, model_name, test_dates, actuals[sid], preds, scales[sid], meta[sid], origin
                )
            )

    # --- intermittent methods, applied to every series so the accuracy table
    #     shows what they cost on dense series, not only what they gain on
    #     sparse ones ---
    for model_name, factory in (
        ("croston_sba", lambda: CrostonForecaster(variant="sba")),
        ("adida", ADIDAForecaster),
    ):
        for sid in common:
            try:
                preds = factory().fit_predict(history[sid], horizon)
            except Exception:
                preds = np.zeros(horizon)
            rows.extend(
                _emit(
                    sid, model_name, test_dates, actuals[sid], preds, scales[sid], meta[sid], origin
                )
            )

    router_fs_indices = (
        {sid: first_sale_indices[sid] for sid in common if sid in first_sale_indices}
        if first_sale_indices
        else None
    )
    router = SegmentRouter().fit(
        {sid: history[sid] for sid in common},
        first_sale_indices=router_fs_indices,
    )
    return pd.DataFrame(rows), scales, router


def _emit(
    sid: str,
    model: str,
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    pred: np.ndarray,
    scale: float,
    meta: dict[str, Any],
    origin: Origin,
) -> list[dict[str, Any]]:
    """Flatten one series-model forecast into prediction rows."""
    n = min(len(dates), len(actual), len(pred))
    return [
        {
            "origin": origin.index,
            "series_id": sid,
            "model": model,
            "date": dates[i],
            "y_true": float(actual[i]),
            "y_pred": float(max(pred[i], 0.0)),
            "scale": scale,
            **meta,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Global ML model
# ---------------------------------------------------------------------------


def run_global_model(
    features: pd.DataFrame,
    feature_cols: list[str],
    origin: Origin,
    scales: dict[str, float],
    model_name: str = "lightgbm",
) -> pd.DataFrame:
    """
    Fit one global tree model across all series at this origin and score it.

    A single model over the pooled panel, not one model per series. That is the
    approach that won M5: cross-series pooling lets a sparse item borrow
    strength from the calendar and price behaviour of similar items, which a
    per-series fit cannot do.

    DIRECT strategy. `features` must have been built with
    `FeaturePipeline(horizon=h)`, so every lag and rolling column on a test row
    is shifted by the full horizon and is knowable at the origin. The model
    then predicts all h days in one pass with no feedback loop, so no forecast
    error compounds across the horizon.

    This matters more than it sounds. Built with horizon=1 features instead,
    the model would receive each test day's true previous-day sales, turning a
    28-day-ahead evaluation into 28 one-step-ahead evaluations and making the
    comparison against the from-origin baselines meaningless.
    """
    from lightgbm import LGBMRegressor

    train = features[features["date"] <= origin.train_end]
    test = features[(features["date"] >= origin.test_start) & (features["date"] <= origin.test_end)]
    if train.empty or test.empty:
        return pd.DataFrame()

    X_tr = train[feature_cols].astype("float32").fillna(0.0)
    y_tr = train["sales"].astype(float).clip(lower=0)
    X_te = test[feature_cols].astype("float32").fillna(0.0)

    common: dict[str, Any] = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
        "random_state": 42,
        "verbose": -1,
    }
    if model_name == "lightgbm_tweedie":
        model = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.1, **common)
    else:
        model = LGBMRegressor(**common)

    t0 = time.time()
    model.fit(X_tr, y_tr)
    preds = np.clip(model.predict(X_te), 0.0, None)
    log.info(
        "global_model_fitted",
        model=model_name,
        origin=origin.index,
        n_train=len(X_tr),
        n_test=len(X_te),
        seconds=round(time.time() - t0, 1),
    )

    out = test[
        ["series_id", "date", "sales", "item_id", "store_id", "dept_id", "cat_id", "state_id"]
    ].copy()
    out = out.rename(columns={"sales": "y_true"})
    out["y_pred"] = preds
    out["model"] = model_name
    out["origin"] = origin.index
    out["scale"] = out["series_id"].map(scales)
    out["y_true"] = out["y_true"].astype(float)
    return out


# ---------------------------------------------------------------------------
# Hierarchical evaluation
# ---------------------------------------------------------------------------


def evaluate_by_hierarchy_level(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Score every model at each of the six M5 aggregation levels.

    Forecasts are produced at item_store and then SUMMED upward. This measures
    something a bottom-level-only table cannot: errors that partially cancel
    when aggregated. A model can look mediocre per item yet be excellent at
    store level, which is precisely the level a replenishment planner buys at.

    RMSSE is not reported above the bottom level here, because an aggregated
    series needs its own in-sample naive scale, which the reconciler computes.
    Reporting the bottom-level scale against aggregated errors would be wrong,
    so WAPE (which aggregates cleanly) is used instead.
    """
    level_keys = {
        "item_store": ["series_id"],
        "dept_store": ["dept_id", "store_id"],
        "cat_store": ["cat_id", "store_id"],
        "store": ["store_id"],
        "state": ["state_id"],
        "total": [],
    }

    records: list[dict[str, Any]] = []
    for level, keys in level_keys.items():
        group_cols = ["model", "origin", "date"] + keys
        agg = (
            preds.groupby(group_cols, observed=True)
            .agg(y_true=("y_true", "sum"), y_pred=("y_pred", "sum"))
            .reset_index()
        )

        for model, g in agg.groupby("model", observed=True):
            denom = g["y_true"].abs().sum()
            records.append(
                {
                    "level": level,
                    "model": model,
                    "n_series_at_level": int(g[keys].drop_duplicates().shape[0] if keys else 1),
                    "wape": float(100.0 * (g["y_true"] - g["y_pred"]).abs().sum() / denom)
                    if denom > 0
                    else np.nan,
                    "bias": float((g["y_pred"] - g["y_true"]).mean()),
                    "n_obs": len(g),
                }
            )

    out = pd.DataFrame(records)
    # Order levels bottom-up rather than alphabetically.
    out["level"] = pd.Categorical(out["level"], categories=list(M5.HIERARCHY_LEVELS), ordered=True)
    return out.sort_values(["level", "wape"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Walk-forward M5 backtest")
    p.add_argument("--stores", nargs="*", default=None)
    p.add_argument("--states", nargs="*", default=None)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--departments", nargs="*", default=None)
    p.add_argument(
        "--sample-series",
        type=int,
        default=300,
        help="random sample of series (default 300); use --full for all",
    )
    p.add_argument("--full", action="store_true", help="use every series (slow)")
    p.add_argument("--horizon", type=int, default=M5.HORIZON_DAYS)
    p.add_argument("--n-origins", type=int, default=3)
    p.add_argument(
        "--history-days",
        type=int,
        default=730,
        help="days of history to load before the earliest origin",
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else settings.base_dir / "reports" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # --- load ---
    loader = M5Loader()
    panel = loader.load_panel(
        stores=args.stores,
        states=args.states,
        categories=args.categories,
        departments=args.departments,
        sample_series=None if args.full else args.sample_series,
        last_n_days=args.history_days,
        random_state=args.seed,
    )
    panel["series_id"] = panel["item_id"].astype(str) + "_" + panel["store_id"].astype(str)

    n_series = panel["series_id"].nunique()
    origins = build_origins(panel["date"], args.horizon, args.n_origins)
    for o in origins:
        log.info("origin_defined", detail=o.describe())

    # --- features for the global model ---
    # Built ONCE over the whole panel with HORIZON-SAFE shifts, so a test row's
    # features never reference an actual that falls inside the test window.
    # The train/test separation is then just a date filter on the same frame.
    pipe = FeaturePipeline(horizon=args.horizon)
    features = pipe.transform(panel)
    features["series_id"] = panel["series_id"].values

    # Price columns are dropped whenever any selected store lacks price data,
    # otherwise the model would learn "price is null" as a store identifier
    # rather than as a price signal. See the workbook truncation note.
    stores_present = set(panel["store_id"].astype(str).unique())
    price_covered = stores_present <= set(M5.STORES_WITH_PRICE_DATA)
    feature_cols = pipe.model_feature_columns(features, drop_price=not price_covered)
    log.info(
        "feature_matrix_ready",
        n_features=len(feature_cols),
        price_features_included=price_covered,
        stores=sorted(stores_present),
    )

    # Determine each series' first non-zero sale from full history (d_1 onward)
    first_sale_dates = (
        loader.load_first_sale_dates()
        if loader.available().get("sales_train", False)
        else None
    )

    # --- run every origin ---
    all_preds: list[pd.DataFrame] = []
    routing_reports: list[pd.DataFrame] = []
    routing_benefits: list[dict[str, Any]] = []

    for origin in origins:
        log.info("running_origin", **{"detail": origin.describe()})

        series_preds, scales, router = run_series_models(
            panel, origin, args.horizon, first_sale_dates=first_sale_dates
        )
        all_preds.append(series_preds)

        for gm in ("lightgbm", "lightgbm_tweedie"):
            gp = run_global_model(features, feature_cols, origin, scales, model_name=gm)
            if not gp.empty:
                all_preds.append(gp)

        # --- routed forecast + the honesty check ---
        lgbm = all_preds[-2] if len(all_preds) >= 2 else pd.DataFrame()
        lgbm = pd.concat([d for d in all_preds if not d.empty and (d["model"] == "lightgbm").any()])
        lgbm = lgbm[(lgbm["model"] == "lightgbm") & (lgbm["origin"] == origin.index)]

        if not lgbm.empty:
            global_fc = {
                sid: g.sort_values("date")["y_pred"].to_numpy()
                for sid, g in lgbm.groupby("series_id", observed=True)
            }
            actuals = {
                sid: g.sort_values("date")["y_true"].to_numpy()
                for sid, g in lgbm.groupby("series_id", observed=True)
            }
            try:
                benefit = router.evaluate_routing_benefit(actuals, global_fc)
                routing_benefits.append(
                    {
                        "origin": origin.index,
                        "routed_wape": benefit.routed_wape,
                        "global_wape": benefit.global_wape,
                        "improvement_pct": benefit.improvement_pct,
                        "improved": benefit.improved,
                        "n_rerouted": benefit.n_rerouted,
                    }
                )
                seg = benefit.by_segment.copy()
                seg["origin"] = origin.index
                routing_reports.append(seg)

                routed = router.predict(args.horizon, global_forecasts=global_fc)
                routed_rows = []
                for sid, fc in routed.items():
                    src = lgbm[lgbm["series_id"] == sid].sort_values("date")
                    n = min(len(src), len(fc))
                    if n == 0:
                        continue
                    r = src.head(n).copy()
                    r["y_pred"] = fc[:n]
                    r["model"] = "routed_ensemble"
                    routed_rows.append(r)
                if routed_rows:
                    all_preds.append(pd.concat(routed_rows, ignore_index=True))
            except Exception as exc:
                log.warning("routing_evaluation_failed", origin=origin.index, error=str(exc))

    preds = pd.concat([d for d in all_preds if not d.empty], ignore_index=True)
    preds = preds.drop_duplicates(subset=["origin", "model", "series_id", "date"])

    # --- accuracy tables ---
    overall = metrics_by_group(preds, ["model"], scale_col="scale")
    by_origin = metrics_by_group(preds, ["model", "origin"], scale_col="scale")
    by_category = metrics_by_group(preds, ["model", "cat_id"], scale_col="scale")
    by_store = metrics_by_group(preds, ["model", "store_id"], scale_col="scale")
    by_level = evaluate_by_hierarchy_level(preds)

    routing_df = (
        pd.concat(routing_reports, ignore_index=True) if routing_reports else pd.DataFrame()
    )
    benefit_df = pd.DataFrame(routing_benefits)

    # --- persist ---
    artefacts: dict[str, pd.DataFrame] = {
        "accuracy_overall": overall,
        "accuracy_by_origin": by_origin,
        "accuracy_by_category": by_category,
        "accuracy_by_store": by_store,
        "accuracy_by_hierarchy_level": by_level,
        "routing_by_segment": routing_df,
        "routing_benefit": benefit_df,
    }
    for name, frame in artefacts.items():
        if not frame.empty:
            frame.to_csv(out_dir / f"{name}.csv", index=False)

    preds.to_parquet(out_dir / "predictions.parquet", index=False)

    manifest = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "protocol": "rolling-origin walk-forward, non-overlapping test windows",
        "n_series": int(n_series),
        "n_origins": args.n_origins,
        "horizon_days": args.horizon,
        "history_days": args.history_days,
        "origins": [
            {
                "index": o.index,
                "train_end": str(o.train_end.date()),
                "test_start": str(o.test_start.date()),
                "test_end": str(o.test_end.date()),
            }
            for o in origins
        ],
        "filters": {
            "stores": args.stores,
            "states": args.states,
            "categories": args.categories,
            "departments": args.departments,
            "sample_series": None if args.full else args.sample_series,
        },
        "n_features": len(feature_cols),
        "price_features_included": bool(price_covered),
        "price_coverage_note": (
            f"sell_price present only for {list(M5.STORES_WITH_PRICE_DATA)} "
            f"({M5.PRICE_ROW_COVERAGE_PCT}% of the full M5 price table); "
            f"the source workbook was truncated at Excel's row limit"
        ),
        "champion": overall.iloc[0]["model"] if not overall.empty else None,
        "champion_wape": float(overall.iloc[0]["wape"]) if not overall.empty else None,
        "champion_rmsse": float(overall.iloc[0].get("rmsse", np.nan))
        if not overall.empty
        else None,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    # --- console report ---
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print("\n" + "=" * 86)
    print("M5 WALK-FORWARD BACKTEST")
    print("=" * 86)
    print(
        f"  series: {n_series:,}   origins: {args.n_origins}   horizon: {args.horizon}d   "
        f"features: {len(feature_cols)}   elapsed: {manifest['elapsed_seconds']}s"
    )
    for o in origins:
        print(f"    {o.describe()}")

    cols = ["model", "wape", "rmsse", "mae", "bias", "n_obs"]
    print("\n--- OVERALL (all origins pooled) " + "-" * 52)
    print(overall[cols].round(3).to_string(index=False))

    print("\n--- BY HIERARCHY LEVEL (forecasts summed upward) " + "-" * 36)
    piv = by_level.pivot_table(index="model", columns="level", values="wape", observed=True)
    print(piv.round(2).to_string())

    print("\n--- BY CATEGORY " + "-" * 69)
    pivc = by_category.pivot_table(index="model", columns="cat_id", values="wape", observed=True)
    print(pivc.round(2).to_string())

    if not benefit_df.empty:
        print("\n--- SEGMENT ROUTING BENEFIT " + "-" * 57)
        print(benefit_df.round(3).to_string(index=False))

    print(f"\n  Artefacts written to: {out_dir}")
    print("=" * 86 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
