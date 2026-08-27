"""
Tier 1 statistical forecasting via `statsforecast` (Nixtla).

Why statsforecast rather than statsmodels
------------------------------------------
Fitting AutoETS to 30,490 series with statsmodels means 30,490 Python-level
optimisation loops. statsforecast implements the same estimators in Numba-
compiled kernels and parallelises across series, turning hours into minutes.
The estimators are the standard ones -- this is an implementation choice, not
a modelling shortcut.

The validation target (why AutoETS is priority 1)
-------------------------------------------------
Makridakis, Spiliotis & Assimakopoulos (2022), IJF 38:1346-1364, Table 2
reports RMSSE at level 12 (product-store):

    Best of top-50 (wyzJack_STU)   0.875
    Competition winner (YJ_STU)    0.884
    ES_bu benchmark                0.915
    Croston                        0.926

`ES_bu` is exactly AutoETS fitted at the bottom level and aggregated upward.
Landing near 0.915 independently validates the loader, the splits, and the
metric implementations all at once -- no other tier offers an external
reference point like this.

COMPARABILITY CAVEAT, which must travel with any comparison to those numbers:
the published figures are WRMSSE -- dollar-weighted by each series' revenue
share, which favours fast-movers -- computed over all 30,490 series against the
true competition test window. Ours is an unweighted mean RMSSE over a
subsample, on a shifted window (the published test period has no actuals in
this data release). Ours being somewhat worse is expected and is not evidence
of a defect.

Seasonality choice
------------------
`season_length=7`. M5 demand has both weekly and annual cycles, but classical
ETS and ARIMA admit only ONE seasonal period, and the EDA measured the weekly
cycle as by far the stronger: ACF at lag 7 is 0.808 against 0.673 at lag 1.
Multi-seasonal structure is the specific job of MSTL and TBATS, which is why
they are separate entries rather than a replacement for ETS.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config.logging import get_logger
from src.config.settings import M5

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

log = get_logger(__name__)

# The dominant cycle in M5 daily demand (EDA §3.4).
WEEKLY_SEASON = 7


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def build_model_registry(season_length: int = WEEKLY_SEASON) -> dict[str, Callable[[], Any]]:
    """
    Name -> factory for every Tier 1 model.

    Factories rather than instances because statsforecast models carry fitted
    state; a fresh instance per run keeps origins independent.
    """
    from statsforecast import models as sfm

    registry: dict[str, Callable[[], Any]] = {}

    # -- 1a. Naive family ---------------------------------------------------
    # Included even though `models/statistical/naive.py` has its own
    # implementations: running both through the same harness cross-checks our
    # code against a third-party one. A disagreement would be a bug in ours.
    registry["sf_naive"] = lambda: sfm.Naive()
    registry["sf_seasonal_naive"] = lambda: sfm.SeasonalNaive(season_length=season_length)
    registry["sf_window_average"] = lambda: sfm.WindowAverage(window_size=28)
    registry["sf_seasonal_window_average"] = lambda: sfm.SeasonalWindowAverage(
        season_length=season_length, window_size=4
    )

    # -- 1b. Exponential smoothing ------------------------------------------
    # AutoETS searches the (Error, Trend, Season) taxonomy by AICc. "ZZZ" lets
    # it choose all three components; this is the ES_bu benchmark.
    registry["autoets"] = lambda: sfm.AutoETS(season_length=season_length, model="ZZZ")
    # Damped trend: phi^h prevents a linear trend extrapolating absurdly at
    # h=28. Gardner & McKenzie (1985) found damping improves accuracy on most
    # empirical series, and the M3/M4 competitions confirmed it.
    registry["ets_damped"] = lambda: sfm.AutoETS(
        season_length=season_length, model="ZZZ", damped=True
    )
    registry["ses"] = lambda: sfm.SimpleExponentialSmoothingOptimized()
    registry["holt_winters"] = lambda: sfm.HoltWinters(season_length=season_length)

    # -- 1c. ARIMA ----------------------------------------------------------
    # Hyndman-Khandakar stepwise order selection. Expensive -- see the
    # aggregation-level constraint in `run_tier1`.
    registry["autoarima"] = lambda: sfm.AutoARIMA(season_length=season_length)

    # -- 1d. Decomposition / complex seasonality ----------------------------
    registry["theta"] = lambda: sfm.Theta(season_length=season_length)
    registry["autotheta"] = lambda: sfm.AutoTheta(season_length=season_length)
    registry["autoces"] = lambda: sfm.AutoCES(season_length=season_length)
    # MSTL handles weekly AND annual simultaneously -- the one classical model
    # here that is not forced to pick a single period.
    registry["mstl"] = lambda: sfm.MSTL(season_length=[7, 365])

    # -- 1e. Intermittent ---------------------------------------------------
    # The dataset's defining property: 95.2% of series are INTERMITTENT or
    # LUMPY (EDA §3.3), so this family is the main event, not a footnote.
    registry["croston_classic"] = lambda: sfm.CrostonClassic()
    registry["croston_optimized"] = lambda: sfm.CrostonOptimized()
    registry["croston_sba"] = lambda: sfm.CrostonSBA()
    # TSB updates demand PROBABILITY every period, not only when a demand
    # occurs. That is what lets it decay toward zero for an item that has
    # stopped selling -- Croston cannot, because it never updates during a
    # dry spell. With 23,511 of 30,490 series introduced mid-panel and many
    # discontinued, obsolescence handling matters here.
    registry["tsb"] = lambda: sfm.TSB(alpha_d=0.2, alpha_p=0.2)
    registry["adida"] = lambda: sfm.ADIDA()
    # IMAPA aggregates at MULTIPLE temporal levels and combines, rather than
    # ADIDA's single level -- a forecast-combination view of the same idea.
    registry["imapa"] = lambda: sfm.IMAPA()

    return registry


# Grouped for reporting and for the priority order in spec Task 3.
MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "naive": ("sf_naive", "sf_seasonal_naive", "sf_window_average", "sf_seasonal_window_average"),
    "exponential_smoothing": ("autoets", "ets_damped", "ses", "holt_winters"),
    "arima": ("autoarima",),
    "decomposition": ("theta", "autotheta", "autoces", "mstl"),
    "intermittent": (
        "croston_classic",
        "croston_optimized",
        "croston_sba",
        "tsb",
        "adida",
        "imapa",
    ),
}

# Priority tiers from the execution spec. Priority 1 is the validation target.
PRIORITY_1 = ("autoets",)
PRIORITY_2 = ("croston_classic", "croston_optimized", "croston_sba", "tsb", "adida", "imapa")
PRIORITY_3 = ("theta", "autotheta", "autoces")
PRIORITY_4 = ("autoarima",)
PRIORITY_5 = ("mstl",)

DEFAULT_MODELS = (
    PRIORITY_1
    + PRIORITY_2
    + PRIORITY_3
    + ("sf_naive", "sf_seasonal_naive", "sf_window_average", "ses", "ets_damped")
)


# ---------------------------------------------------------------------------
# Panel <-> statsforecast format
# ---------------------------------------------------------------------------


def to_statsforecast_format(
    panel: pd.DataFrame,
    series_col: str = "series_id",
    date_col: str = "date",
    target_col: str = "sales",
) -> pd.DataFrame:
    """
    Convert the M5 panel to the (unique_id, ds, y) long format Nixtla expects.

    Sorted by (unique_id, ds) because statsforecast assumes contiguous,
    chronologically ordered blocks per series and does not re-sort.
    """
    out = panel[[series_col, date_col, target_col]].rename(
        columns={series_col: "unique_id", date_col: "ds", target_col: "y"}
    )
    out["ds"] = pd.to_datetime(out["ds"])
    out["y"] = out["y"].astype("float32")
    return out.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)


@dataclass
class Tier1Result:
    """Fit/predict outcome for one statsforecast run."""

    predictions: pd.DataFrame  # long: model, series_id, date, y_pred
    timings: pd.DataFrame  # per-model fit and predict seconds
    models_run: list[str] = field(default_factory=list)
    models_failed: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Tier1Runner:
    """
    Fits Tier 1 models across many series and returns harness-format predictions.

    Deliberately returns predictions ONLY. Scoring goes through the existing
    evaluation harness (`src.evaluation.point_metrics`) so Tier 1 numbers are
    computed by the same code as every other tier -- spec constraint 4.
    """

    def __init__(
        self,
        models: Sequence[str] | None = None,
        season_length: int = WEEKLY_SEASON,
        n_jobs: int = 1,
        freq: str = "D",
    ) -> None:
        """
        Args:
            n_jobs: process-level parallelism. Defaults to 1, which is NOT a
                performance concession -- it is the correct default here.

                statsforecast's estimators are already Numba-compiled and
                vectorised across series inside a single process, so the
                process pool only helps at very large series counts. Below
                that, pool spin-up dominates: a 40-series run measured ~12s
                per model with `n_jobs=-1`, essentially all of it setup.

                On Windows the pool is also fragile -- the `spawn` start
                method re-imports the module in each worker and needs a
                `__main__` guard, and several models here raised
                BrokenProcessPool without one. Raise `n_jobs` deliberately for
                full-panel runs launched from a guarded script, not by default.
        """
        self.model_names = list(models) if models else list(DEFAULT_MODELS)
        self.season_length = season_length
        self.n_jobs = n_jobs
        self.freq = freq
        self._registry = build_model_registry(season_length)
        self._log = get_logger("Tier1Runner")

        unknown = [m for m in self.model_names if m not in self._registry]
        if unknown:
            raise KeyError(f"unknown Tier 1 models {unknown}; available: {sorted(self._registry)}")

    def fit_predict(
        self,
        train_panel: pd.DataFrame,
        horizon: int = M5.HORIZON_DAYS,
        levels: Iterable[float] | None = None,
    ) -> Tier1Result:
        """
        Fit every configured model on `train_panel` and forecast `horizon` days.

        Each model is run in its own StatsForecast call rather than all at once.
        That costs a little repeated setup but buys two things worth more: a
        clean per-model timing (needed for the accuracy-vs-compute Pareto
        figure) and isolation, so one model raising does not lose the rest.

        Args:
            train_panel: long panel with series_id, date, sales -- TRAINING
                rows only. Anything dated inside the evaluation window here
                would leak.
            horizon: days ahead to forecast.
            levels: optional prediction-interval levels, e.g. [80, 95].
        """
        from statsforecast import StatsForecast

        sf_df = to_statsforecast_format(train_panel)
        n_series = sf_df["unique_id"].nunique()

        frames: list[pd.DataFrame] = []
        timings: list[dict[str, Any]] = []
        failed: dict[str, str] = {}
        ran: list[str] = []

        for name in self.model_names:
            model = self._registry[name]()
            self._log.info("tier1_fitting", model=name, n_series=n_series)

            try:
                t0 = time.time()
                sf = StatsForecast(models=[model], freq=self.freq, n_jobs=self.n_jobs)
                # `forecast` fits and predicts in one pass without retaining
                # state -- the memory-efficient path for wide panels.
                fcst = sf.forecast(df=sf_df, h=horizon, level=list(levels) if levels else None)
                elapsed = time.time() - t0

                # statsforecast names the output column after the model class,
                # which varies (AutoETS, CrostonSBA, ...). Take whatever column
                # is not an identifier and is not an interval bound.
                value_cols = [
                    c
                    for c in fcst.columns
                    if c not in ("unique_id", "ds") and "-lo-" not in c and "-hi-" not in c
                ]
                if not value_cols:
                    raise RuntimeError(f"no forecast column in output: {list(fcst.columns)}")

                point_col = value_cols[0]
                block = fcst[["unique_id", "ds", point_col]].rename(
                    columns={"unique_id": "series_id", "ds": "date", point_col: "y_pred"}
                )
                # Demand is a non-negative count; several classical models can
                # emit small negatives, which are never actionable.
                block["y_pred"] = block["y_pred"].astype(float).clip(lower=0.0)
                block["model"] = name

                # Carry interval bounds through when requested, for Tier 3
                # comparison against the probabilistic models.
                for c in fcst.columns:
                    if "-lo-" in c or "-hi-" in c:
                        suffix = c.split("-", 1)[1].replace("-", "_").lower()
                        block[f"pred_{suffix}"] = fcst[c].to_numpy(dtype=float).clip(min=0.0)

                frames.append(block)
                ran.append(name)
                timings.append(
                    {
                        "model": name,
                        "group": self._group_of(name),
                        "fit_predict_seconds": round(elapsed, 2),
                        "n_series": int(n_series),
                        "seconds_per_1k_series": round(1000 * elapsed / max(n_series, 1), 3),
                    }
                )
                self._log.info("tier1_complete", model=name, seconds=round(elapsed, 1))

            except Exception as exc:
                # A single model failing must not lose the whole run. Failures
                # are recorded and reported rather than silently skipped --
                # spec constraint 6.
                failed[name] = f"{type(exc).__name__}: {exc}"
                self._log.warning("tier1_model_failed", model=name, error=str(exc)[:200])

        predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return Tier1Result(
            predictions=predictions,
            timings=pd.DataFrame(timings),
            models_run=ran,
            models_failed=failed,
        )

    @staticmethod
    def _group_of(name: str) -> str:
        for group, members in MODEL_GROUPS.items():
            if name in members:
                return group
        return "other"


# ---------------------------------------------------------------------------
# Hierarchical aggregation (the "_bu" in ES_bu)
# ---------------------------------------------------------------------------

LEVEL_KEYS: dict[str, list[str]] = {
    "item_store": ["series_id"],
    "dept_store": ["dept_id", "store_id"],
    "cat_store": ["cat_id", "store_id"],
    "store": ["store_id"],
    "state": ["state_id"],
    "total": [],
}


def aggregate_bottom_up(
    predictions: pd.DataFrame,
    meta: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """
    Sum item-store forecasts upward to a hierarchy level.

    This is the "_bu" in the ES_bu benchmark: forecast once at the bottom, then
    add. Bottom-up is automatically coherent -- the parts sum to the whole by
    construction -- and it is the baseline every reconciliation method (MinT,
    top-down) is measured against.

    Summation is exact rather than approximate because the forecasts are unit
    counts at the (item, store, day) grain.

    Args:
        predictions: long frame with model, series_id, date, y_pred (and
            y_true when scoring).
        meta: series_id -> dept_id, cat_id, store_id, state_id.
        level: one of LEVEL_KEYS.
    """
    if level not in LEVEL_KEYS:
        raise ValueError(f"unknown level {level!r}; expected one of {sorted(LEVEL_KEYS)}")

    df = predictions.merge(meta, on="series_id", how="left", suffixes=("", "_meta"))
    keys = [k for k in LEVEL_KEYS[level] if k in df.columns]
    group_cols = ["model", "date"] + keys

    value_cols = [c for c in ("y_pred", "y_true") if c in df.columns]
    out = df.groupby(group_cols, observed=True)[value_cols].sum().reset_index()
    out["level"] = level
    return out
