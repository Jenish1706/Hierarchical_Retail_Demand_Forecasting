"""
Router versus the best single model (M5-13).

``SegmentRouter.evaluate_routing_benefit`` compares the routed forecast only
with the global LightGBM model.  That is the wrong yardstick: the question a
reader cares about is whether classifying series by demand pattern beats
simply using the best single method everywhere, and the best single method in
this benchmark is ADIDA, not LightGBM.

This module scores the router against every single model on the *same*
predictions, per origin and per demand segment.  It reads the long
predictions frame the backtest already writes plus the router's own
per-series assignments (``routing_assignments.csv``), so there is one scoring
path for the leaderboard and for this comparison, and WAPE is the same pooled
definition (``sum|y - yhat| / sum|y|``) used everywhere else.

Two honest limits are part of the definition, not footnotes:

* **Selection origin.**  The INTERMITTENT -> ADIDA assignment was chosen from
  this same backtest (see ``src/models/router.py``), so the router is scored on
  the origins that selected its rule.  A clean test needs an earlier selection
  origin.  The router still loses to ADIDA, so no false win can result.
* **Best single model is picked after the fact.**  For each row the best
  single model is the lowest-WAPE one *in that row*.  That is an oracle choice
  made with the test scores in hand, which favours the single model and makes
  the router's deficit a conservative one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROUTED_MODEL = "routed_ensemble"
ADIDA_MODEL = "adida"
GLOBAL_MODEL = "lightgbm"

#: ``pattern`` value for a row that pools every demand segment.
ALL_SEGMENTS = "ALL"
#: ``origin`` value for a row that pools every backtest origin.
ALL_ORIGINS = "all"

#: Columns of the table returned by :func:`compare_router`.
COMPARISON_COLUMNS = [
    "origin",
    "pattern",
    "routes_to",
    "n_series_origins",
    "wape_routed",
    "wape_adida",
    "wape_lightgbm",
    "best_single_model",
    "wape_best_single",
    "routed_minus_adida",
    "routed_minus_best_single",
]


def _wape(abs_err: float, actual: float) -> float:
    """Pooled WAPE in percent; NaN when the actuals sum to zero."""
    return float(100.0 * abs_err / actual) if actual > 0 else float("nan")


def _row(
    block: pd.DataFrame, origin: str, pattern: str, routes_to: str
) -> dict[str, object]:
    """Score every model on one block of (origin, series, model) error sums."""
    per_model = block.groupby("model", observed=True).agg(
        abs_err=("abs_err", "sum"), actual=("actual", "sum")
    )
    wape = {m: _wape(r.abs_err, r.actual) for m, r in per_model.iterrows()}

    singles = {m: w for m, w in wape.items() if m != ROUTED_MODEL and np.isfinite(w)}
    best_model = min(singles, key=lambda m: singles[m]) if singles else ""
    best_wape = singles.get(best_model, float("nan"))
    routed = wape.get(ROUTED_MODEL, float("nan"))
    adida = wape.get(ADIDA_MODEL, float("nan"))

    # One routed forecast exists per (origin, series), so counting the routed
    # model's units gives the number of (series, origin) pairs in the block.
    n_units = int(block.loc[block["model"] == ROUTED_MODEL, "n_units"].sum())

    return {
        "origin": origin,
        "pattern": pattern,
        "routes_to": routes_to,
        "n_series_origins": n_units,
        "wape_routed": routed,
        "wape_adida": adida,
        "wape_lightgbm": wape.get(GLOBAL_MODEL, float("nan")),
        "best_single_model": best_model,
        "wape_best_single": best_wape,
        "routed_minus_adida": routed - adida,
        "routed_minus_best_single": routed - best_wape,
    }


def compare_router(preds: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Score the routed ensemble against every single model, per origin and segment.

    Args:
        preds: long backtest predictions with ``origin``, ``series_id``,
            ``model``, ``y_true`` and ``y_pred`` (the ``predictions.parquet``
            frame).  Must contain the ``routed_ensemble`` model.
        assignments: one row per (origin, series) with ``pattern`` (the demand
            quadrant the router assigned from that origin's training window)
            and ``method`` (where it was sent).

    Returns:
        One row per scope, with ``origin`` in {each origin, ``"all"``} and
        ``pattern`` in {each demand pattern, ``"ALL"``}.  WAPE columns are in
        percent; ``routed_minus_*`` are percentage points, positive when the
        router is *worse*.  ``n_series_origins`` counts (series, origin) pairs.

    Raises:
        ValueError: if the router is missing from ``preds`` or an assignment is
            missing for some prediction row.  A silent inner join would drop
            rows and report a WAPE over a different set of series.
    """
    if ROUTED_MODEL not in set(preds["model"].unique()):
        raise ValueError(f"predictions contain no {ROUTED_MODEL!r} rows")

    need = ["origin", "series_id", "pattern", "method"]
    missing = [c for c in need if c not in assignments.columns]
    if missing:
        raise ValueError(f"assignments lack columns {missing}")
    assign = assignments[need].drop_duplicates(["origin", "series_id"])

    frame = preds[["origin", "series_id", "model", "y_true", "y_pred"]].copy()
    frame["abs_err"] = (frame["y_true"] - frame["y_pred"]).abs()
    merged = frame.merge(assign, on=["origin", "series_id"], how="left", validate="m:1")
    if merged["pattern"].isna().any():
        raise ValueError(
            f"{int(merged['pattern'].isna().sum())} prediction rows have no routing assignment"
        )

    # Collapse to one row per (origin, pattern, method, model).  Error and
    # actual sums are additive, and (origin, series) pairs are distinct across
    # origins, so every pooled scope below is a plain sum of these rows.
    per_unit = (
        merged.groupby(["origin", "pattern", "method", "model"], observed=True)
        .agg(abs_err=("abs_err", "sum"), actual=("y_true", "sum"), n_units=("series_id", "nunique"))
        .reset_index()
    )

    origins = sorted(per_unit["origin"].unique())
    patterns = sorted(per_unit["pattern"].unique())
    method_of = (
        per_unit.drop_duplicates("pattern").set_index("pattern")["method"].to_dict()
    )

    rows: list[dict[str, object]] = []
    for origin_scope in [*origins, ALL_ORIGINS]:
        by_origin = (
            per_unit
            if origin_scope == ALL_ORIGINS
            else per_unit[per_unit["origin"] == origin_scope]
        )
        for pattern_scope in [*patterns, ALL_SEGMENTS]:
            block = (
                by_origin
                if pattern_scope == ALL_SEGMENTS
                else by_origin[by_origin["pattern"] == pattern_scope]
            )
            if block.empty:
                continue
            routes_to = "mixed" if pattern_scope == ALL_SEGMENTS else str(method_of[pattern_scope])
            rows.append(_row(block, str(origin_scope), str(pattern_scope), routes_to))

    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def per_origin_columns(comparison: pd.DataFrame) -> pd.DataFrame:
    """
    The five per-origin columns appended to ``routing_benefit.csv``.

    Returns a frame indexed by integer ``origin`` with ``adida_wape``,
    ``best_single_model``, ``best_single_wape``, ``routed_minus_adida`` and
    ``routed_minus_best_single`` for the all-segment rows of each origin.
    """
    sel = comparison[
        (comparison["pattern"] == ALL_SEGMENTS) & (comparison["origin"] != ALL_ORIGINS)
    ].copy()
    sel["origin"] = sel["origin"].astype(int)
    return sel.set_index("origin").rename(
        columns={
            "wape_adida": "adida_wape",
            "wape_best_single": "best_single_wape",
        }
    )[
        [
            "adida_wape",
            "best_single_model",
            "best_single_wape",
            "routed_minus_adida",
            "routed_minus_best_single",
        ]
    ]
