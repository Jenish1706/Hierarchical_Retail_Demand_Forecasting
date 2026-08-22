"""
Probabilistic forecast evaluation.

The central rule
----------------
Never judge a probabilistic forecast by a point metric, and never judge an
interval by coverage alone.

Coverage alone is trivially gameable: predicting [0, 10^9] achieves 100%
coverage on every series and is useless. Sharpness alone is equally gameable
in the other direction. Every function here that reports coverage also reports
width, and `interval_report()` refuses to return one without the other.

What each metric answers
------------------------
  pinball loss  -- is this specific quantile in the right place?
  CRPS          -- is the whole predictive distribution right? Reduces to MAE
                   when the forecast is a point mass, which makes it directly
                   comparable across probabilistic and deterministic models.
  coverage      -- do nominal 80% intervals actually contain 80% of outcomes?
  MIS / Winkler -- one number combining coverage and width, so a model cannot
                   buy coverage with width.
  PIT           -- is the forecast calibrated at EVERY quantile simultaneously?
                   A flat PIT histogram is the strongest calibration evidence
                   available; U-shaped means over-confident, hump-shaped means
                   under-confident.

References
----------
Gneiting & Raftery (2007), "Strictly proper scoring rules, prediction, and
    estimation", JASA 102(477):359-378.  [CRPS, propriety]
Gneiting & Katzfuss (2014), "Probabilistic forecasting", Annual Review of
    Statistics 1:125-151.  [sharpness subject to calibration]
Winkler (1972), "A decision-theoretic approach to interval estimation", JASA
    67(337):187-191.
Dawid (1984), "Statistical theory: the prequential approach", JRSS-A
    147(2):278-292.  [PIT]
Makridakis et al. (2022) -- the M5 Uncertainty track used weighted pinball
    loss over nine quantile levels.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

_EPS = 1e-12

# The nine levels used by the M5 Uncertainty competition. Retained as the
# default so results are comparable with the published leaderboard.
M5_QUANTILES: tuple[float, ...] = (
    0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995,
)


# ---------------------------------------------------------------------------
# Quantile losses
# ---------------------------------------------------------------------------

def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """
    Pinball (quantile) loss at level q.

        L_q(y, f) = q * (y - f)        if y >= f
                    (1 - q) * (f - y)  otherwise

    Asymmetric on purpose. At q=0.9, under-predicting costs 9x what
    over-predicting costs, which is what forces the estimator up into the
    upper tail. Minimised in expectation exactly at the true q-th quantile,
    making it a proper scoring rule for that quantile.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {q}")

    y_true = np.asarray(y_true, dtype=float)
    y_pred_q = np.asarray(y_pred_q, dtype=float)
    delta = y_true - y_pred_q
    return float(np.mean(np.maximum(q * delta, (q - 1.0) * delta)))


def mean_pinball_loss(
    y_true: np.ndarray,
    quantile_preds: Mapping[float, np.ndarray],
) -> float:
    """
    Average pinball loss across all supplied quantile levels.

    This is the M5 Uncertainty track's headline metric (before its
    series-level scaling). Averaging across levels stops a model from winning
    by nailing the median while getting both tails badly wrong.
    """
    if not quantile_preds:
        return float("nan")
    return float(np.mean([pinball_loss(y_true, p, q) for q, p in quantile_preds.items()]))


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

def crps_from_quantiles(
    y_true: np.ndarray,
    quantile_preds: Mapping[float, np.ndarray],
) -> float:
    """
    CRPS approximated from a discrete set of predicted quantiles.

    The continuous ranked probability score is

        CRPS(F, y) = integral (F(x) - 1{x >= y})^2 dx

    which is exactly twice the integral of the pinball loss over q in (0,1):

        CRPS(F, y) = 2 * integral_0^1 L_q(y, F^-1(q)) dq

    With quantiles available only at discrete levels, that integral is
    approximated by the trapezoidal rule over the sorted levels. Accuracy
    therefore depends on how densely the levels are spaced -- nine M5 levels
    give a good approximation, three do not, so the level count is worth
    reporting alongside the score.

    Lower is better. For a deterministic forecast CRPS collapses to MAE, which
    is what makes it the one metric that can rank a point model and a
    distributional model on the same axis.
    """
    if len(quantile_preds) < 2:
        return float("nan")

    levels = np.array(sorted(quantile_preds), dtype=float)
    losses = np.array([pinball_loss(y_true, quantile_preds[q], q) for q in levels])
    return float(2.0 * np.trapezoid(losses, levels))


def crps_from_samples(y_true: np.ndarray, samples: np.ndarray) -> float:
    """
    CRPS from Monte Carlo samples, via the energy form

        CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent draws from the predictive distribution. Used
    for models that emit samples rather than quantiles -- bootstrap ensembles,
    MC dropout, deep ensembles, DeepAR.

    Args:
        y_true:  (n,) actuals.
        samples: (n, m) m samples per observation.
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != y_true.shape[0]:
        raise ValueError(f"samples must be (n, m) matching y_true; got {samples.shape}")

    term1 = np.abs(samples - y_true[:, None]).mean(axis=1)

    # E|X - X'| via the sorted-sample identity, which is O(m log m) instead of
    # the O(m^2) all-pairs computation:
    #   E|X-X'| = (2 / m^2) * sum_i (2i - m + 1) * x_(i)
    m = samples.shape[1]
    ordered = np.sort(samples, axis=1)
    weights = 2.0 * np.arange(m) - m + 1.0
    term2 = (2.0 / (m * m)) * (ordered * weights).sum(axis=1)

    return float(np.mean(term1 - 0.5 * term2))


# ---------------------------------------------------------------------------
# Interval metrics -- coverage ALWAYS reported with sharpness
# ---------------------------------------------------------------------------

def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Percentage of actuals inside [lower, upper]."""
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower, dtype=float)) & (y_true <= np.asarray(upper, dtype=float))
    return float(100.0 * inside.mean())


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width, in units. The sharpness half of the calibration picture."""
    return float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """
    Winkler (1972) interval score for a nominal (1 - alpha) interval.

        W = (u - l)                          if l <= y <= u
          = (u - l) + (2/alpha) * (l - y)    if y < l
          = (u - l) + (2/alpha) * (y - u)    if y > u

    Width is always charged; misses are charged extra in proportion to how far
    outside they fall, scaled by 2/alpha so a 99% interval is punished far
    harder for a miss than an 80% one. This is what makes it un-gameable: you
    cannot buy coverage with width, because width is itself the base penalty.

    Lower is better. Also known as the interval score.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    width = upper - lower
    below = np.maximum(lower - y_true, 0.0)
    above = np.maximum(y_true - upper, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def mean_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """MIS is the Winkler score; aliased because both names appear in the literature."""
    return winkler_score(y_true, lower, upper, alpha)


def scaled_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    y_insample: np.ndarray,
    alpha: float,
    seasonality: int = 1,
) -> float:
    """
    MIS scaled by the in-sample naive MAE, so it is comparable across series.

    Same motivation as MASE and RMSSE: an unscaled interval score on a
    500-unit-per-day staple dwarfs one on a 0.3-unit-per-day hobby item, and
    averaging them would let the staples decide the ranking.
    """
    y = np.asarray(y_insample, dtype=float)
    if y.size <= seasonality:
        return float("nan")
    denom = float(np.mean(np.abs(y[seasonality:] - y[:-seasonality])))
    if denom < _EPS:
        return float("nan")
    return winkler_score(y_true, lower, upper, alpha) / denom


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------

def pit_values(
    y_true: np.ndarray,
    quantile_preds: Mapping[float, np.ndarray],
) -> np.ndarray:
    """
    Probability Integral Transform values, one per observation.

    PIT_i = F_i(y_i), the predictive CDF evaluated at the actual. If the
    forecasts are calibrated, these are Uniform(0,1) -- that is Dawid's (1984)
    result and it holds regardless of the distribution family.

    With only discrete quantile levels available, F_i(y_i) is estimated as the
    fraction of predicted quantiles at or below the actual. The resolution of
    the histogram is therefore capped by the number of levels.

    NOTE on count data: M5 targets are integers with a large point mass at
    zero, and PIT is only exactly uniform for continuous predictive
    distributions. For a discrete target the histogram shows spurious spikes
    at the mass points. `randomised_pit` addresses this properly.
    """
    y_true = np.asarray(y_true, dtype=float)
    levels = sorted(quantile_preds)
    matrix = np.column_stack([quantile_preds[q] for q in levels])
    return (matrix <= y_true[:, None]).mean(axis=1)


def randomised_pit(
    y_true: np.ndarray,
    quantile_preds: Mapping[float, np.ndarray],
    random_state: int = 42,
) -> np.ndarray:
    """
    Randomised PIT for discrete targets (Czado, Gneiting & Held 2009).

    For a discrete predictive distribution the standard PIT cannot be uniform
    even under perfect calibration, because the CDF jumps. The fix is to draw

        u ~ Uniform(F(y-), F(y))

    which restores exact uniformity under calibration. Since M5 sales are
    integer counts with roughly 30% zeros, this is the correct PIT to plot;
    the plain version produces a huge artefactual spike at the zero mass that
    is easily misread as miscalibration.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true, dtype=float)
    levels = sorted(quantile_preds)
    matrix = np.column_stack([quantile_preds[q] for q in levels])

    f_at = (matrix <= y_true[:, None]).mean(axis=1)      # F(y)
    f_below = (matrix < y_true[:, None]).mean(axis=1)    # F(y-)
    return f_below + rng.random(len(y_true)) * (f_at - f_below)


def pit_histogram(
    y_true: np.ndarray,
    quantile_preds: Mapping[float, np.ndarray],
    n_bins: int = 10,
    randomised: bool = True,
) -> pd.DataFrame:
    """
    Binned PIT with the uniform reference, ready for Fig 16.

    `deviation` is the signed departure from the uniform expectation, so a
    reader can see the SHAPE at a glance: positive at both ends = U-shaped =
    intervals too narrow; positive in the middle = too wide.
    """
    pit = randomised_pit(y_true, quantile_preds) if randomised else pit_values(y_true, quantile_preds)
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    freq = counts / max(counts.sum(), 1)
    expected = 1.0 / n_bins

    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts,
            "frequency": freq,
            "expected_frequency": expected,
            "deviation": freq - expected,
        }
    )


def reliability_curve(
    y_true: np.ndarray,
    quantile_preds: Mapping[float, np.ndarray],
) -> pd.DataFrame:
    """
    Nominal vs empirical coverage at every quantile level -- Fig 15.

    For each level q, the fraction of actuals at or below the predicted q-th
    quantile should be q. Points on the 45-degree line are calibrated; below
    the line means the quantile sits too high, above means too low.
    """
    y_true = np.asarray(y_true, dtype=float)
    rows = []
    for q in sorted(quantile_preds):
        pred = np.asarray(quantile_preds[q], dtype=float)
        empirical = float((y_true <= pred).mean())
        rows.append(
            {
                "nominal_quantile": q,
                "empirical_coverage": empirical,
                "calibration_error": empirical - q,
                "pinball_loss": pinball_loss(y_true, pred, q),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bundled reports
# ---------------------------------------------------------------------------

def interval_report(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_level: float,
    y_insample: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Coverage AND sharpness for one interval. Never returns one alone.

    `coverage_gap` is the number to read first: positive means the interval is
    wider than it needs to be, negative means it is over-confident and will
    under-serve any decision built on it.
    """
    alpha = 1.0 - nominal_level
    coverage = empirical_coverage(y_true, lower, upper)
    width = mean_interval_width(lower, upper)

    out = {
        "nominal_level": nominal_level * 100.0,
        "empirical_coverage": coverage,
        "coverage_gap": coverage - nominal_level * 100.0,
        "mean_width": width,
        "median_width": float(np.median(np.asarray(upper, float) - np.asarray(lower, float))),
        "winkler_score": winkler_score(y_true, lower, upper, alpha),
    }
    if y_insample is not None:
        out["scaled_interval_score"] = scaled_interval_score(
            y_true, lower, upper, y_insample, alpha
        )
    return out


def compute_probabilistic_metrics(
    y_true: np.ndarray,
    quantile_preds: Mapping[float, np.ndarray],
    y_insample: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Full probabilistic bundle for one evaluation window.

    Returns per-level pinball losses, the mean, CRPS, and a coverage/sharpness
    pair for every central interval whose two tail quantiles are present.
    """
    y_true = np.asarray(y_true, dtype=float)
    out: dict[str, float] = {}

    for q in sorted(quantile_preds):
        out[f"pinball_q{q:g}"] = pinball_loss(y_true, quantile_preds[q], q)

    out["pinball_mean"] = mean_pinball_loss(y_true, quantile_preds)
    out["crps"] = crps_from_quantiles(y_true, quantile_preds)
    out["n_quantile_levels"] = float(len(quantile_preds))

    # Central intervals: pair each low quantile with its mirror image.
    for lo in sorted(q for q in quantile_preds if q < 0.5):
        hi = round(1.0 - lo, 6)
        if hi not in quantile_preds:
            continue
        level = hi - lo
        label = f"{level * 100:.0f}"
        report = interval_report(
            y_true, quantile_preds[lo], quantile_preds[hi], level, y_insample
        )
        out[f"coverage_{label}"] = report["empirical_coverage"]
        out[f"coverage_gap_{label}"] = report["coverage_gap"]
        out[f"width_{label}"] = report["mean_width"]
        out[f"winkler_{label}"] = report["winkler_score"]

    # A single scalar summary of calibration across ALL levels, so models can
    # be ranked on calibration without reading the whole reliability curve.
    curve = reliability_curve(y_true, quantile_preds)
    out["mean_abs_calibration_error"] = float(curve["calibration_error"].abs().mean())
    return out


def enforce_monotonic_quantiles(
    quantile_preds: Mapping[float, np.ndarray],
) -> tuple[dict[float, np.ndarray], float]:
    """
    Monotonic rearrangement of independently fitted quantile forecasts.

    Separately trained quantile models share no constraint, so on some rows
    they cross: q90 lands below q50, which is not a distribution and cannot
    support a service-level decision.

    Chernozhukov, Fernandez-Val & Galichon (2010, Econometrica 78(3)) show
    that simply SORTING each row's predicted quantiles fixes this and weakly
    IMPROVES the estimator -- the rearranged curve is never further from the
    true quantile function than the original, in any L^p norm. Distribution-
    free, and costs one sort per row.

    Returns:
        (rearranged predictions, violation rate before the repair)
    """
    levels = sorted(quantile_preds)
    matrix = np.column_stack([np.asarray(quantile_preds[q], dtype=float) for q in levels])

    violations = float(np.mean(np.any(np.diff(matrix, axis=1) < 0, axis=1)))
    matrix = np.sort(matrix, axis=1)
    return {q: matrix[:, i] for i, q in enumerate(levels)}, violations
