"""
Forecast accuracy metrics for the M5 platform.

Metric selection rationale
--------------------------
M5 series are sparse -- far more so than commonly assumed. Measured over the
full panel (reports/eda): 68.2% of the 58.3M (item, store, day) observations
are zero, and 95.2% of series fall in the INTERMITTENT or LUMPY quadrants.
That single property disqualifies the metrics most people reach for first.

  * MAPE is undefined at y=0 and explodes as y -> 0. Averaging it over M5
    produces a number dominated by the handful of days a slow item sold one
    unit. Computed here only for continuity with older reports, always over
    the non-zero subset, and never used for model selection.

  * WAPE = sum|y - yhat| / sum(y) is well defined whenever the denominator
    window has any sales at all, and weights errors by volume -- which is the
    business-relevant weighting, since being 2 units wrong on a 200-unit item
    matters less than on a 2-unit item. Primary business metric.

  * RMSSE is the official M5 accuracy metric. It scales squared forecast
    error by the in-sample one-step naive squared error, which makes it
    comparable across series of wildly different volume and gives it an
    absolute interpretation: RMSSE < 1 beats a random walk, RMSSE > 1 loses
    to it. Primary model-selection metric.

  * Bias (signed mean error) is reported alongside because WAPE and RMSSE are
    magnitude-only. A model can post an excellent WAPE while systematically
    under-forecasting every series, which in a replenishment setting means
    persistent stockouts. Only a signed metric surfaces that.

  * Pinball loss and empirical coverage evaluate the quantile forecasts.
    Coverage answers "does the nominal 80% interval actually contain 80% of
    outcomes"; pinball loss is the proper scoring rule quantile regression
    directly minimises.

References
----------
Makridakis, Spiliotis & Assimakopoulos (2022), "M5 accuracy competition:
    Results, findings and conclusions", IJF 38(4):1346-1364.  [RMSSE]
Hyndman & Koehler (2006), "Another look at measures of forecast accuracy",
    IJF 22(4):679-688.  [scaled errors]
Gneiting & Raftery (2007), "Strictly proper scoring rules, prediction, and
    estimation", JASA 102(477):359-378.  [pinball loss]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Guard for the scaling denominator. If a series never changes value in
# sample (e.g. all zeros), the naive MSE is 0 and RMSSE would be infinite.
# Returning NaN and excluding the series is honest; substituting 1.0 would
# silently invent a scale.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Point-forecast metrics
# ---------------------------------------------------------------------------


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error, expressed in percent.

        WAPE = 100 * sum|y_t - yhat_t| / sum|y_t|

    Also known as the MAD/Mean ratio. Returns NaN when the actuals sum to
    zero, since the quantity is genuinely undefined there rather than
    conveniently 0 or 100.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.abs(y_true).sum()
    if denom < _EPS:
        return float("nan")
    return float(100.0 * np.abs(y_true - y_pred).sum() / denom)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error, in units."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error, in units."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Signed mean error: positive means the model over-forecasts.

    Reported separately from WAPE/RMSSE because those are magnitude-only and
    hide systematic directional error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def bias_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Bias as a percentage of mean actual demand -- comparable across series."""
    y_true = np.asarray(y_true, dtype=float)
    mean_actual = np.abs(y_true).mean()
    if mean_actual < _EPS:
        return float("nan")
    return float(100.0 * bias(y_true, y_pred) / mean_actual)


def mape_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE over the non-zero subset only, in percent.

    Retained for continuity with legacy reporting. NOT used for model
    selection: restricting to non-zero actuals changes the evaluation set
    between models and over-weights low-volume days.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > _EPS
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


# ---------------------------------------------------------------------------
# Scaled error -- the official M5 metric
# ---------------------------------------------------------------------------


def naive_scale(
    y_insample: np.ndarray,
    seasonality: int = 1,
    trim_pre_launch: bool = True,
) -> float:
    """
    Denominator of the scaled error: in-sample MSE of a seasonal naive forecast.

        scale = (1 / (n - m)) * sum_{t=m+1..n} (y_t - y_{t-m})^2

    with m = `seasonality` (m=1 is the random walk used by the M5 rules).

    The M5 guidelines specify this be computed on the TRAINING window only.
    When `trim_pre_launch=True` (canonical M5 rule), the evaluation window starts
    from the first non-zero sale rather than day 1, preventing pre-introduction
    zeros from diluting the scale.
    """
    y = np.asarray(y_insample, dtype=float).ravel()
    if trim_pre_launch:
        nonzero_indices = np.where(y > _EPS)[0]
        if len(nonzero_indices) > 0:
            y = y[nonzero_indices[0]:]
        else:
            return float("nan")

    if y.size <= seasonality:
        return float("nan")

    diffs = y[seasonality:] - y[:-seasonality]
    return float(np.mean(diffs**2))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler 2006).

        MASE = mean|y_t - yhat_t| / mean|y_t - y_{t-m}|

    The absolute-error sibling of RMSSE: same scaling idea, but scaled by the
    in-sample naive MEAN ABSOLUTE error rather than the mean squared error.
    Reported alongside RMSSE because the two can disagree, and where they do,
    the disagreement is informative -- MASE is less sensitive to a handful of
    large misses, so a model that beats another on MASE but loses on RMSSE is
    winning on typical days and losing on spikes.

    Returns NaN when the series is constant in sample.
    """
    y = np.asarray(y_insample, dtype=float)
    if y.size <= seasonality:
        return float("nan")

    denom = float(np.mean(np.abs(y[seasonality:] - y[:-seasonality])))
    if denom < _EPS:
        return float("nan")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Root Mean Squared Scaled Error -- the official M5 accuracy metric.

        RMSSE = sqrt( mean((y_t - yhat_t)^2) / scale )

    where `scale` is the in-sample naive MSE from `naive_scale`.

    Interpretation is absolute, which is what makes it useful across a
    hierarchy spanning single-unit hobby items and thousand-unit food staples:

        RMSSE < 1  -- beats the in-sample random walk
        RMSSE = 1  -- indistinguishable from it
        RMSSE > 1  -- worse than doing nothing

    Returns NaN if the series is constant in sample (scale == 0), because no
    meaningful scaling exists. Callers aggregate with nan-aware means so such
    series are excluded rather than defaulted.
    """
    scale = naive_scale(y_insample, seasonality=seasonality)
    if not np.isfinite(scale) or scale < _EPS:
        return float("nan")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse_forecast = float(np.mean((y_true - y_pred) ** 2))
    return float(np.sqrt(mse_forecast / scale))


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """
    Pinball (quantile) loss at level q -- the proper scoring rule for quantiles.

        L_q(y, f) = max( q * (y - f), (q - 1) * (y - f) )

    Under-predicting is penalised by q, over-predicting by (1 - q), so high
    quantiles are punished harder for being too low. Minimised in expectation
    exactly at the true q-th quantile.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {q}")

    y_true = np.asarray(y_true, dtype=float)
    y_pred_q = np.asarray(y_pred_q, dtype=float)
    delta = y_true - y_pred_q
    return float(np.mean(np.maximum(q * delta, (q - 1.0) * delta)))


def empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Fraction of actuals falling inside [lower, upper], as a percentage.

    Compared against the nominal level to judge calibration. A nominal 80%
    interval covering 62% of outcomes is over-confident and will under-serve
    safety-stock decisions built on it; covering 95% is wasteful.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(100.0 * inside.mean())


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Mean interval width, in units.

    Read together with coverage: an interval can always reach nominal
    coverage by becoming arbitrarily wide, so coverage alone is not a
    sufficient calibration report.
    """
    return float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


# ---------------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: np.ndarray | None = None,
    seasonality: int = 1,
    clip_negative: bool = True,
) -> dict[str, float]:
    """
    Standard metric bundle for a single evaluation window.

    Args:
        y_true:     Actual unit sales over the evaluation window.
        y_pred:     Forecast unit sales, same length.
        y_insample: Training-window actuals, used only for the RMSSE scale.
                    When omitted, `rmsse` is NaN and is reported as such
                    rather than silently dropped from the bundle.
        seasonality: Lag for the naive scale. 1 = random walk (M5 default).
        clip_negative: Demand is a non-negative count, so negative model
                    output is clipped to 0 before scoring. This is a genuine
                    improvement, not a metric trick -- a negative order
                    quantity is not actionable.

    Returns:
        Metric name -> value. Values may be NaN where undefined; callers
        aggregate with nan-aware reducers.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    if clip_negative:
        y_pred = np.clip(y_pred, 0.0, None)

    out: dict[str, float] = {
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "bias_pct": bias_pct(y_true, y_pred),
        "mape_nonzero": mape_nonzero(y_true, y_pred),
        "n_obs": float(y_true.size),
        "zero_pct": float(100.0 * np.mean(np.abs(y_true) <= _EPS)),
    }

    if y_insample is not None:
        out["rmsse"] = rmsse(y_true, y_pred, y_insample, seasonality=seasonality)
        out["mase"] = mase(y_true, y_pred, y_insample, seasonality=seasonality)
    else:
        out["rmsse"] = float("nan")
        out["mase"] = float("nan")
    return out


def compute_quantile_metrics(
    y_true: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> dict[str, float]:
    """
    Calibration bundle for a set of quantile forecasts.

    Reports per-quantile pinball loss, plus coverage and mean width for the
    80% and 90% central intervals when the constituent quantiles are present.
    """
    y_true = np.asarray(y_true, dtype=float)
    out: dict[str, float] = {}

    for q, preds in sorted(quantile_preds.items()):
        out[f"pinball_q{int(round(q * 100)):02d}"] = pinball_loss(y_true, preds, q)

    out["pinball_mean"] = float(np.mean([v for k, v in out.items() if k.startswith("pinball_q")]))

    # Central intervals, when both tails were forecast.
    for lo_q, hi_q, label in ((0.10, 0.90, "80"), (0.05, 0.95, "90")):
        if lo_q in quantile_preds and hi_q in quantile_preds:
            lo, hi = quantile_preds[lo_q], quantile_preds[hi_q]
            out[f"coverage_{label}"] = empirical_coverage(y_true, lo, hi)
            out[f"width_{label}"] = interval_width(lo, hi)

    return out


def metrics_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    scale_col: str | None = None,
    weight_col: str | None = None,
    series_id_col: str = "series_id",
) -> pd.DataFrame:
    """
    Per-segment accuracy table -- the shape wanted for a backtest report.

    Args:
        df:          Long-format predictions, one row per (series, date).
        group_cols:  Segmentation, e.g. ["model", "cat_id"] or ["model", "store_id"].
        scale_col:   Optional per-row RMSSE scale (in-sample naive MSE for that
                     series). Supplying it lets RMSSE be aggregated correctly:
                     the scale is a property of the SERIES, so it must be
                     carried per row rather than recomputed on the group.

    Returns:
        One row per group with the standard metric bundle plus observation
        counts, sorted by WAPE ascending.
    """
    records: list[dict[str, Any]] = []

    for keys, g in df.groupby(group_cols, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(group_cols, key_tuple))

        yt = g[y_true_col].to_numpy(dtype=float)
        yp = np.clip(g[y_pred_col].to_numpy(dtype=float), 0.0, None)

        row.update(compute_metrics(yt, yp))

        if scale_col is not None and scale_col in g.columns and series_id_col in g.columns:
            # Aggregate RMSSE per series, then take the mean (or weighted mean) across series.
            series_rmsse = []
            series_weights = []

            # Using numpy operations is faster, but for exactness we group by series
            for _sid, s_group in g.groupby(series_id_col):
                s_yt = s_group[y_true_col].to_numpy(dtype=float)
                s_yp = np.clip(s_group[y_pred_col].to_numpy(dtype=float), 0.0, None)
                s_scale = s_group[scale_col].iloc[0]

                if np.isfinite(s_scale) and s_scale > _EPS:
                    mse_forecast = float(np.mean((s_yt - s_yp) ** 2))
                    series_rmsse.append(np.sqrt(mse_forecast / s_scale))
                    if weight_col is not None and weight_col in s_group.columns:
                        series_weights.append(s_group[weight_col].iloc[0])
                    else:
                        series_weights.append(1.0)

            if series_rmsse:
                series_rmsse = np.array(series_rmsse)
                series_weights = np.array(series_weights)

                w_sum = series_weights.sum()
                if w_sum > 0:
                    series_weights = series_weights / w_sum
                    row["rmsse"] = float(np.sum(series_rmsse * series_weights))
                else:
                    row["rmsse"] = float(np.mean(series_rmsse))

        records.append(row)

    result = pd.DataFrame(records)
    if "wape" in result.columns:
        result = result.sort_values("wape", na_position="last").reset_index(drop=True)
    return result
