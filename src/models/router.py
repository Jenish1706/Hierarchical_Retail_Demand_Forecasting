"""
Segment routing: send each series to the method suited to its demand pattern.

Motivation
----------
A single global LightGBM trained across all 30,490 M5 series is a strong
default, but it is not uniformly best. On series that sell nothing for weeks
at a time, a squared-error objective pulls the forecast toward the
unconditional mean and the model effectively predicts a small constant --
which is exactly what Croston does, only Croston does it with two parameters
instead of several hundred trees, and without pretending the daily variation
is learnable.

So the router classifies each series with the Syntetos-Boylan-Teunter scheme
and dispatches:

    ADI  < 1.32  and  CV^2 < 0.49  ->  SMOOTH        ->  global LightGBM
    ADI >= 1.32  and  CV^2 < 0.49  ->  INTERMITTENT  ->  Croston-SBA
    ADI  < 1.32  and  CV^2 >= 0.49 ->  ERRATIC       ->  global LightGBM
    ADI >= 1.32  and  CV^2 >= 0.49 ->  LUMPY         ->  ADIDA
    (no demand at all)             ->  NO_DEMAND     ->  zero forecast

The frequent quadrants go to the global model because that is where the
engineered lag and calendar features carry real signal. The sparse quadrants
go to dedicated estimators.

The important part
------------------
`evaluate_routing_benefit()` MEASURES whether this actually helps, by scoring
the routed forecast against the all-global forecast on the same holdout. The
routing rule above is a hypothesis drawn from the literature, not a result.
If the measurement says routing loses on this data, the honest response is to
report that and route everything to the global model -- and the method
returns the per-segment breakdown needed to make that call.

References
----------
Syntetos, Boylan & Croston (2005), JORS 56(5):495-503.
Petropoulos & Kourentzes (2015), "Forecast combinations for intermittent
    demand", JORS 66(6):914-924 -- evidence that per-segment method selection
    beats a single method across a heterogeneous catalogue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.models.statistical.intermittent import (
    ADIDAForecaster,
    CrostonForecaster,
    DemandClassification,
    DemandPattern,
    ZeroForecaster,
    classify_demand,
)
from src.evaluation.point_metrics import compute_metrics, naive_scale
from src.config.logging import get_logger

log = get_logger(__name__)


# Which method each SBT quadrant is routed to. Declared as data rather than
# an if/else chain so the mapping can be overridden wholesale in an
# experiment (e.g. to test "route ERRATIC to ADIDA too") without editing
# control flow.
DEFAULT_ROUTING: Mapping[DemandPattern, str] = {
    DemandPattern.SMOOTH: "global",
    DemandPattern.ERRATIC: "global",
    DemandPattern.INTERMITTENT: "croston_sba",
    DemandPattern.LUMPY: "adida",
    DemandPattern.NO_DEMAND: "zero",
}


@dataclass
class RoutingDecision:
    """Where one series was sent, and the statistics that decided it."""

    series_id: str
    pattern: DemandPattern
    method: str
    adi: float
    cv2: float
    zero_fraction: float
    mean_demand: float


@dataclass
class RoutingBenefit:
    """
    Outcome of comparing routed forecasts against all-global forecasts.

    `improved` is the headline: True only if the routed WAPE is strictly
    better. `by_segment` carries the per-quadrant detail, which is where the
    interesting finding usually lives -- routing can help a lot on LUMPY
    while losing slightly on INTERMITTENT, and a single pooled number hides
    that entirely.
    """

    routed_wape: float
    global_wape: float
    routed_rmsse: float
    global_rmsse: float
    improvement_pct: float
    improved: bool
    n_series: int
    n_rerouted: int
    by_segment: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        verdict = "ROUTING HELPS" if self.improved else "ROUTING DOES NOT HELP"
        return (
            f"{verdict}: routed WAPE {self.routed_wape:.2f}% vs "
            f"global-only {self.global_wape:.2f}% "
            f"({self.improvement_pct:+.2f}% relative) over {self.n_series:,} "
            f"series, {self.n_rerouted:,} of which were rerouted away from "
            f"the global model."
        )


class SegmentRouter:
    """
    Classify series and dispatch each to its method.

    Usage:
        router = SegmentRouter()
        router.fit(history)                      # {series_id: 1-D array}
        preds = router.predict(h=28, global_forecasts=lgbm_preds)

    The global model is injected as a dict of precomputed forecasts rather
    than as a fitted estimator. That keeps the router independent of the
    feature pipeline: it never needs to know how the LightGBM forecast was
    produced, only what it was.
    """

    def __init__(
        self,
        routing: Mapping[DemandPattern, str] | None = None,
        croston_alpha: float = 0.1,
        adida_alpha: float = 0.2,
    ) -> None:
        self.routing = dict(routing or DEFAULT_ROUTING)
        self.croston_alpha = croston_alpha
        self.adida_alpha = adida_alpha

        self.classifications_: dict[str, DemandClassification] = {}
        self.decisions_: dict[str, RoutingDecision] = {}
        self._local_models: dict[str, Any] = {}
        self._history: dict[str, np.ndarray] = {}
        self._fitted = False

    # -- construction of the local (non-global) estimators -------------------

    def _build(self, method: str) -> Any:
        factories: dict[str, Callable[[], Any]] = {
            "croston_sba": lambda: CrostonForecaster(alpha=self.croston_alpha, variant="sba"),
            "croston": lambda: CrostonForecaster(alpha=self.croston_alpha, variant="croston"),
            "adida": lambda: ADIDAForecaster(alpha=self.adida_alpha),
            "zero": ZeroForecaster,
        }
        if method not in factories:
            raise KeyError(f"no local estimator for method {method!r}")
        return factories[method]()

    # -- fit -----------------------------------------------------------------

    def fit(self, history: Mapping[str, np.ndarray]) -> "SegmentRouter":
        """
        Classify every series and fit the local models it needs.

        Args:
            history: series_id -> chronologically ordered unit sales.
                     Must contain TRAINING data only. Classifying on data
                     that includes the evaluation window would leak the very
                     information the holdout is meant to withhold.
        """
        if not history:
            raise ValueError("history is empty; nothing to route")

        self.classifications_.clear()
        self.decisions_.clear()
        self._local_models.clear()
        self._history = {k: np.asarray(v, dtype=float).ravel() for k, v in history.items()}

        for sid, y in self._history.items():
            cls = classify_demand(y)
            method = self.routing.get(cls.pattern, "global")

            self.classifications_[sid] = cls
            self.decisions_[sid] = RoutingDecision(
                series_id=sid,
                pattern=cls.pattern,
                method=method,
                adi=cls.adi,
                cv2=cls.cv2,
                zero_fraction=cls.zero_fraction,
                mean_demand=cls.mean_demand,
            )

            # "global" needs no local fit -- its forecast is supplied later.
            if method != "global":
                self._local_models[sid] = self._build(method).fit(y)

        self._fitted = True
        # Log the quadrant mix. `segment_counts` is indexed by
        # (pattern, method), so flatten to pattern only for the log line.
        by_pattern: dict[str, int] = {}
        for d in self.decisions_.values():
            key = d.pattern.value.lower()
            by_pattern[key] = by_pattern.get(key, 0) + 1
        log.info("segment_routing_fitted", n_series=len(self._history), **by_pattern)
        return self

    # -- predict -------------------------------------------------------------

    def predict(
        self,
        h: int,
        global_forecasts: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Produce one forecast per series, each from its routed method.

        Args:
            h: horizon in days.
            global_forecasts: series_id -> forecast, for series routed to the
                global model. A series routed to "global" with no forecast
                supplied falls back to its local mean and is logged -- silently
                emitting zeros would look like a confident prediction of no
                demand.
        """
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")

        global_forecasts = global_forecasts or {}
        out: dict[str, np.ndarray] = {}
        missing = 0

        for sid, decision in self.decisions_.items():
            if decision.method == "global":
                fc = global_forecasts.get(sid)
                if fc is None:
                    missing += 1
                    fc = np.full(h, float(self._history[sid].mean()))
                out[sid] = np.clip(np.asarray(fc, dtype=float)[:h], 0.0, None)
            else:
                out[sid] = self._local_models[sid].predict(h)

        if missing:
            log.warning(
                "router_missing_global_forecasts",
                n_missing=missing,
                fallback="series mean",
            )
        return out

    # -- reporting -----------------------------------------------------------

    def segment_counts(self) -> pd.DataFrame:
        """Series count and share of total demand per SBT quadrant."""
        rows = [
            {
                "pattern": d.pattern.value,
                "method": d.method,
                "mean_demand": d.mean_demand,
                "zero_fraction": d.zero_fraction,
                "adi": d.adi,
                "cv2": d.cv2,
            }
            for d in self.decisions_.values()
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        agg = df.groupby(["pattern", "method"]).agg(
            n_series=("mean_demand", "size"),
            total_mean_demand=("mean_demand", "sum"),
            avg_zero_fraction=("zero_fraction", "mean"),
            median_adi=("adi", "median"),
            median_cv2=("cv2", "median"),
        )
        agg["pct_of_series"] = 100.0 * agg["n_series"] / agg["n_series"].sum()
        # Volume share matters as much as series count: the sparse quadrants
        # are often a large share of the catalogue but a small share of units.
        agg["pct_of_volume"] = (
            100.0 * agg["total_mean_demand"] / agg["total_mean_demand"].sum()
        )
        return agg.sort_values("n_series", ascending=False)

    def routing_table(self) -> pd.DataFrame:
        """Per-series routing decisions, for audit and for joining to results."""
        return pd.DataFrame([vars(d) | {"pattern": d.pattern.value} for d in self.decisions_.values()])

    # -- the honesty check ---------------------------------------------------

    def evaluate_routing_benefit(
        self,
        actuals: Mapping[str, np.ndarray],
        global_forecasts: Mapping[str, np.ndarray],
    ) -> RoutingBenefit:
        """
        Measure whether routing beats using the global model everywhere.

        Both arms are scored on the SAME holdout with the SAME global
        forecasts, so the only difference is the substitution made on the
        sparse quadrants. That isolates the effect of the routing decision
        itself.

        Args:
            actuals:          series_id -> holdout actuals.
            global_forecasts: series_id -> global model forecast over the
                              same holdout window.

        Returns:
            RoutingBenefit, including a per-segment breakdown.
        """
        if not self._fitted:
            raise RuntimeError("call fit() before evaluate_routing_benefit()")

        common = sorted(set(actuals) & set(global_forecasts) & set(self.decisions_))
        if not common:
            raise ValueError(
                "no overlap between actuals, global_forecasts and routed series"
            )

        h = min(len(np.asarray(actuals[sid]).ravel()) for sid in common)
        routed = self.predict(h, global_forecasts=global_forecasts)

        rows: list[dict[str, Any]] = []
        for sid in common:
            yt = np.asarray(actuals[sid], dtype=float).ravel()[:h]
            yg = np.clip(np.asarray(global_forecasts[sid], dtype=float).ravel()[:h], 0, None)
            yr = routed[sid][:h]

            # RMSSE scale comes from the TRAINING history, per the M5 rules.
            scale = naive_scale(self._history[sid], seasonality=1)

            rows.append(
                {
                    "series_id": sid,
                    "pattern": self.decisions_[sid].pattern.value,
                    "method": self.decisions_[sid].method,
                    "rerouted": self.decisions_[sid].method != "global",
                    "abs_err_routed": float(np.abs(yt - yr).sum()),
                    "abs_err_global": float(np.abs(yt - yg).sum()),
                    "actual_sum": float(np.abs(yt).sum()),
                    "sq_err_routed": float(np.mean((yt - yr) ** 2)),
                    "sq_err_global": float(np.mean((yt - yg) ** 2)),
                    "scale": scale,
                }
            )

        df = pd.DataFrame(rows)

        def _pooled_wape(err_col: str) -> float:
            total = df["actual_sum"].sum()
            return float(100.0 * df[err_col].sum() / total) if total > 0 else float("nan")

        def _mean_rmsse(sq_col: str) -> float:
            ok = df["scale"].notna() & (df["scale"] > 0)
            if not ok.any():
                return float("nan")
            return float(np.sqrt(df.loc[ok, sq_col] / df.loc[ok, "scale"]).mean())

        routed_wape = _pooled_wape("abs_err_routed")
        global_wape = _pooled_wape("abs_err_global")
        improvement = (
            100.0 * (global_wape - routed_wape) / global_wape
            if global_wape and np.isfinite(global_wape) and global_wape > 0
            else float("nan")
        )

        # Per-segment breakdown -- the part worth reading.
        seg = df.groupby(["pattern", "method"]).apply(
            lambda g: pd.Series(
                {
                    "n_series": len(g),
                    "wape_routed": 100.0 * g["abs_err_routed"].sum() / g["actual_sum"].sum()
                    if g["actual_sum"].sum() > 0 else np.nan,
                    "wape_global": 100.0 * g["abs_err_global"].sum() / g["actual_sum"].sum()
                    if g["actual_sum"].sum() > 0 else np.nan,
                }
            ),
            include_groups=False,
        )
        if not seg.empty:
            seg["wape_delta"] = seg["wape_routed"] - seg["wape_global"]
            seg["routing_helps"] = seg["wape_delta"] < 0

        benefit = RoutingBenefit(
            routed_wape=routed_wape,
            global_wape=global_wape,
            routed_rmsse=_mean_rmsse("sq_err_routed"),
            global_rmsse=_mean_rmsse("sq_err_global"),
            improvement_pct=improvement,
            improved=bool(np.isfinite(improvement) and improvement > 0),
            n_series=len(df),
            n_rerouted=int(df["rerouted"].sum()),
            by_segment=seg.reset_index(),
        )
        log.info(
            "routing_benefit_evaluated",
            routed_wape=round(routed_wape, 3),
            global_wape=round(global_wape, 3),
            improved=benefit.improved,
        )
        return benefit
