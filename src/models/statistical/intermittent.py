"""
Intermittent demand forecasting: SBT classification, Croston-SBA and ADIDA.

The problem
-----------
Most M5 series are not smooth. A large share of the 30,490 (item, store)
series sell nothing on the majority of days and then sell one or two units.
Fitting a squared-error model to that history produces a forecast that is
"correct on average" -- a flat 0.3 units/day -- while being wrong on every
single day, and it systematically under-forecasts the non-zero days that
actually drive replenishment.

Croston (1972) reframes the problem: instead of forecasting demand directly,
separately forecast

    z -- the size of a demand when one occurs, and
    p -- the interval between demands,

then combine them as z / p. Because each component is smoothed only on the
periods where it is observed, neither is dragged toward zero by the gaps.

The bias correction
-------------------
Croston's original estimator is biased. Syntetos & Boylan (2005) showed the
expectation of z/p over-estimates mean demand per period, because E[1/p] is
not 1/E[p] (Jensen's inequality), and derived the correction factor
(1 - alpha/2). That variant -- Croston-SBA -- is the default here; plain
Croston is retained only as a reference implementation.

ADIDA
-----
Nikolopoulos et al. (2011) take a different route: aggregate the series into
non-overlapping buckets of length k until it stops being intermittent,
forecast the aggregated series with a simple method, then disaggregate back.
Temporal aggregation raises the signal-to-noise ratio and reduces the zero
fraction directly. It performs best on "lumpy" series where both the interval
and the size are erratic, which is why the router sends that quadrant here.

References
----------
Croston (1972), "Forecasting and stock control for intermittent demands",
    Operational Research Quarterly 23(3):289-303.
Syntetos & Boylan (2005), "The accuracy of intermittent demand estimates",
    IJF 21(2):303-314.  [the (1 - alpha/2) correction]
Syntetos, Boylan & Croston (2005), "On the categorization of demand
    patterns", JORS 56(5):495-503.  [the ADI / CV^2 quadrants]
Nikolopoulos, Syntetos, Boylan, Petropoulos & Assimakopoulos (2011), "An
    aggregate-disaggregate intermittent demand approach (ADIDA)", JORS
    62(3):544-554.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

_EPS = 1e-12

# ---------------------------------------------------------------------------
# Syntetos-Boylan-Teunter (SBT) classification cut-offs
#
# These two constants are not tuning knobs. They are the analytically derived
# boundaries at which the expected mean-squared error of Croston's method
# crosses that of the alternatives (SBC 2005, section 4). Changing them
# invalidates the citation, so they are module-level constants rather than
# configuration.
# ---------------------------------------------------------------------------
ADI_CUTOFF = 1.32  # average inter-demand interval, in periods
CV2_CUTOFF = 0.49  # squared coefficient of variation of non-zero sizes


class DemandPattern(str, Enum):  # noqa: UP042
    """The four SBT quadrants, plus a guard for series with no demand at all."""

    SMOOTH = "SMOOTH"  # frequent, stable    -> global ML
    ERRATIC = "ERRATIC"  # frequent, variable  -> global ML
    INTERMITTENT = "INTERMITTENT"  # sparse,   stable    -> Croston-SBA
    LUMPY = "LUMPY"  # sparse,   variable  -> ADIDA
    NO_DEMAND = "NO_DEMAND"  # zero throughout the training window


@dataclass(frozen=True)
class DemandClassification:
    """Result of classifying one series, with the statistics behind it."""

    pattern: DemandPattern
    adi: float  # average inter-demand interval
    cv2: float  # squared CV of non-zero demand sizes
    n_periods: int
    n_nonzero: int
    zero_fraction: float
    mean_demand: float  # over all periods, including zeros

    @property
    def is_intermittent(self) -> bool:
        """True for the two sparse quadrants that need a dedicated method."""
        return self.pattern in (DemandPattern.INTERMITTENT, DemandPattern.LUMPY)


def classify_demand(y: np.ndarray) -> DemandClassification:
    """
    Assign a series to an SBT quadrant.

        ADI  = n_periods / n_nonzero          (mean gap between demands)
        CV^2 = (std(sizes) / mean(sizes))^2   (variability of demand size)

    Note CV^2 is computed over NON-ZERO observations only. Including the zeros
    would conflate the two axes -- a sparse series would automatically look
    size-variable, and the quadrants would collapse onto one diagonal.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = int(y.size)

    if n == 0:
        raise ValueError("cannot classify an empty series")

    nonzero = y[y > _EPS]
    n_nz = int(nonzero.size)

    if n_nz == 0:
        # Never sold in the training window. Not a modelling problem -- there
        # is nothing to learn -- so it is flagged and routed to a zero
        # forecast rather than being forced into a quadrant.
        return DemandClassification(
            pattern=DemandPattern.NO_DEMAND,
            adi=float("inf"),
            cv2=0.0,
            n_periods=n,
            n_nonzero=0,
            zero_fraction=1.0,
            mean_demand=0.0,
        )

    adi = n / n_nz
    mean_size = float(nonzero.mean())
    # ddof=1: the sample standard deviation. With ddof=0 a series with a
    # single demand event would report CV^2 = 0 and be called "stable" on the
    # strength of one observation.
    std_size = float(nonzero.std(ddof=1)) if n_nz > 1 else 0.0
    cv2 = (std_size / mean_size) ** 2 if mean_size > _EPS else 0.0

    if adi < ADI_CUTOFF:
        pattern = DemandPattern.SMOOTH if cv2 < CV2_CUTOFF else DemandPattern.ERRATIC
    else:
        pattern = DemandPattern.INTERMITTENT if cv2 < CV2_CUTOFF else DemandPattern.LUMPY

    return DemandClassification(
        pattern=pattern,
        adi=float(adi),
        cv2=float(cv2),
        n_periods=n,
        n_nonzero=n_nz,
        zero_fraction=float(1.0 - n_nz / n),
        mean_demand=float(y.mean()),
    )


# ---------------------------------------------------------------------------
# Croston family
# ---------------------------------------------------------------------------


class CrostonForecaster:
    """
    Croston's method with the Syntetos-Boylan bias correction (SBA).

    Two exponentially smoothed states are maintained, both updated ONLY on
    periods with non-zero demand:

        z_t = alpha * y_t     + (1 - alpha) * z_{t-1}    (demand size)
        p_t = alpha * q_t     + (1 - alpha) * p_{t-1}    (interval)

    where q_t is the number of periods since the previous demand. The
    per-period forecast is flat:

        yhat = c * z_t / p_t

    with c = 1 for `variant="croston"` and c = (1 - alpha/2) for
    `variant="sba"` (the default, and the one supported by the evidence).

    The forecast is intentionally a constant rate, not a spiky series. Croston
    estimates the mean demand *per period*; it does not attempt to predict
    which specific days will have a demand, and pretending otherwise would be
    over-claiming.
    """

    name = "croston_sba"

    def __init__(self, alpha: float = 0.1, variant: str = "sba") -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if variant not in ("croston", "sba"):
            raise ValueError(f"variant must be 'croston' or 'sba', got {variant!r}")

        self.alpha = alpha
        self.variant = variant
        self.name = "croston" if variant == "croston" else "croston_sba"

        self.z_: float | None = None  # smoothed demand size
        self.p_: float | None = None  # smoothed inter-demand interval
        self.rate_: float | None = None  # bias-corrected per-period rate
        self._fitted = False

    @property
    def _correction(self) -> float:
        """SBA multiplier (1 - alpha/2); 1.0 for classical Croston."""
        return 1.0 - self.alpha / 2.0 if self.variant == "sba" else 1.0

    def fit(self, y: np.ndarray) -> CrostonForecaster:
        y = np.asarray(y, dtype=float).ravel()
        if y.size == 0:
            raise ValueError("cannot fit on an empty series")

        nz_idx = np.flatnonzero(y > _EPS)

        if nz_idx.size == 0:
            # No demand ever observed -> forecast zero. Honest, and avoids a
            # division by an undefined interval.
            self.z_, self.p_, self.rate_ = 0.0, float(y.size), 0.0
            self._fitted = True
            return self

        # Initialise on the first observed demand. Initialising the interval
        # at the position of the first demand (1-indexed) rather than at 1
        # avoids over-estimating the rate for a series that starts with a long
        # dry spell -- common in M5 for items introduced mid-panel.
        z = float(y[nz_idx[0]])
        p = float(nz_idx[0] + 1)

        # Subsequent demands update both states.
        for prev, cur in zip(nz_idx[:-1], nz_idx[1:]):
            gap = float(cur - prev)
            z = self.alpha * float(y[cur]) + (1.0 - self.alpha) * z
            p = self.alpha * gap + (1.0 - self.alpha) * p

        self.z_ = z
        self.p_ = max(p, 1.0)  # an interval below one period is not meaningful
        self.rate_ = self._correction * self.z_ / self.p_
        self._fitted = True
        return self

    def predict(self, h: int) -> np.ndarray:
        """Flat per-period demand rate over the horizon."""
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")
        if h < 1:
            raise ValueError(f"horizon must be >= 1, got {h}")
        return np.full(h, max(self.rate_, 0.0), dtype=float)

    def fit_predict(self, y: np.ndarray, h: int) -> np.ndarray:
        return self.fit(y).predict(h)


class ADIDAForecaster:
    """
    Aggregate-Disaggregate Intermittent Demand Approach.

    Three steps:

      1. AGGREGATE the series into non-overlapping buckets of length k, summing
         within each bucket. Summation raises the mean well above zero and
         collapses the zero fraction, so the aggregated series is no longer
         intermittent.
      2. FORECAST one bucket ahead with simple exponential smoothing.
      3. DISAGGREGATE by dividing the bucket forecast equally across its k
         periods, giving a flat per-period rate.

    Choosing k
    ----------
    With `aggregation_level=None` the bucket size is set to ceil(ADI), i.e.
    just large enough that a typical bucket contains about one demand event.
    That is the self-tuning rule from the original paper and it adapts per
    series, which matters across a catalogue spanning near-daily staples and
    items selling a few times a year.

    Buckets are built from the END of the series backwards, so the most recent
    observations always align to a bucket boundary. Aggregating forwards would
    leave a ragged final bucket containing partial recent data, biasing the
    level estimate downward exactly where it matters most.
    """

    name = "adida"

    def __init__(
        self,
        aggregation_level: int | None = None,
        alpha: float = 0.2,
    ) -> None:
        if aggregation_level is not None and aggregation_level < 1:
            raise ValueError(f"aggregation_level must be >= 1, got {aggregation_level}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self.aggregation_level = aggregation_level
        self.alpha = alpha
        self.k_: int | None = None  # bucket size actually used
        self.bucket_forecast_: float | None = None
        self.rate_: float | None = None
        self._fitted = False

    @staticmethod
    def _aggregate(y: np.ndarray, k: int) -> np.ndarray:
        """Sum into non-overlapping buckets of size k, anchored at the end."""
        n_buckets = y.size // k
        if n_buckets == 0:
            return np.array([y.sum()], dtype=float)
        trimmed = y[-(n_buckets * k) :]  # drop the oldest ragged remainder
        return trimmed.reshape(n_buckets, k).sum(axis=1).astype(float)

    def fit(self, y: np.ndarray) -> ADIDAForecaster:
        y = np.asarray(y, dtype=float).ravel()
        if y.size == 0:
            raise ValueError("cannot fit on an empty series")

        if self.aggregation_level is None:
            cls = classify_demand(y)
            # ADI is inf for an all-zero series; fall back to weekly buckets.
            self.k_ = 7 if not np.isfinite(cls.adi) else max(1, int(np.ceil(cls.adi)))
        else:
            self.k_ = self.aggregation_level

        agg = self._aggregate(y, self.k_)

        # SES on the aggregated series.
        level = float(agg[0])
        for value in agg[1:]:
            level = self.alpha * float(value) + (1.0 - self.alpha) * level

        self.bucket_forecast_ = level
        self.rate_ = max(level / self.k_, 0.0)  # equal-weight disaggregation
        self._fitted = True
        return self

    def predict(self, h: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")
        if h < 1:
            raise ValueError(f"horizon must be >= 1, got {h}")
        return np.full(h, self.rate_, dtype=float)

    def fit_predict(self, y: np.ndarray, h: int) -> np.ndarray:
        return self.fit(y).predict(h)


class ZeroForecaster:
    """
    Forecast identically zero.

    Used for NO_DEMAND series. Keeping it as an explicit model rather than a
    special case means such series still appear in accuracy tables with a
    named method, instead of vanishing from the evaluation.
    """

    name = "zero"

    def fit(self, y: np.ndarray) -> ZeroForecaster:
        return self

    def predict(self, h: int) -> np.ndarray:
        return np.zeros(h, dtype=float)

    def fit_predict(self, y: np.ndarray, h: int) -> np.ndarray:
        return self.predict(h)


INTERMITTENT_REGISTRY = {
    "croston": lambda: CrostonForecaster(variant="croston"),
    "croston_sba": lambda: CrostonForecaster(variant="sba"),
    "adida": ADIDAForecaster,
    "zero": ZeroForecaster,
}
