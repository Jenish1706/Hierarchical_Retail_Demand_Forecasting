"""
Advanced feature engineering: target encoding, Fourier terms, interactions.

Everything in this module is a HYPOTHESIS until the ablation harness at the
bottom measures it. The base pipeline in `pipeline.py` supplies lag, rolling,
calendar and price features; the transforms here are candidate additions, and
`FeatureAblationStudy` exists to decide which of them survive.

That ordering matters. Each of these techniques is standard, well-cited, and
routinely added to forecasting pipelines without ever being measured. Target
encoding in particular is capable of *reducing* accuracy while appearing to
help, because the leakage it introduces flatters validation exactly as much as
it flatters training.

Leakage discipline
------------------
Two rules govern every transform here:

  1. No transform may use information from a row's own target.
  2. No transform may use information from any row dated on or after the row
     being encoded.

Rule 1 is handled by out-of-fold construction; rule 2 by using a
TimeSeriesSplit rather than a random KFold for those folds. Both are needed:
out-of-fold encoding with random folds still leaks the future.

References
----------
Micci-Barreca (2001), "A preprocessing scheme for high-cardinality categorical
    attributes in classification and prediction problems", SIGKDD Explorations
    3(1):27-32.  [target encoding with smoothing]
Harrell (2015), "Regression Modeling Strategies", 2nd ed., ch. 2 -- harmonic
    regression for periodic effects.
Prokhorenkova et al. (2018), "CatBoost: unbiased boosting with categorical
    features", NeurIPS -- on target-leakage in encoding schemes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config.logging import get_logger
from src.evaluation.point_metrics import compute_metrics

log = get_logger(__name__)

_EPS = 1e-12


# ===========================================================================
# Target encoding
# ===========================================================================


class TargetEncoder:
    """
    Out-of-fold target encoding with Bayesian (empirical-Bayes) smoothing.

    The naive encoding -- replace a category with the mean target for that
    category -- has two failure modes:

      * LEAKAGE. A category's mean includes the row being encoded. With a
        high-cardinality key such as item_id (3,049 levels), the encoding of a
        rare item is dominated by its own target and the model learns to read
        the answer off the feature.

      * VARIANCE. A category seen three times gets a mean estimated from three
        observations, which is mostly noise, yet is presented to the model with
        the same apparent authority as a category seen 30,000 times.

    Both are addressed here.

    Leakage: encodings are built out-of-fold over a TimeSeriesSplit. The value
    assigned to a row is computed only from folds strictly earlier in time.
    Rows in the first fold have no earlier data and receive the global prior.

    Variance: each category mean is shrunk toward the global mean in proportion
    to how little evidence supports it,

        encoded_c = (n_c * mean_c + m * prior) / (n_c + m)

    where m (`smoothing`) is the weight given to the prior, expressed in
    pseudo-observations. A category needs roughly m observations before its own
    mean outweighs the prior. m=20 is the default: enough to suppress
    single-sighting noise, small enough that a genuinely distinctive item is
    not flattened.
    """

    def __init__(
        self,
        cols: Sequence[str],
        target_col: str = "sales",
        smoothing: float = 20.0,
        n_splits: int = 5,
        prior: float | None = None,
    ) -> None:
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")
        self.cols = list(cols)
        self.target_col = target_col
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.prior_ = prior

        # Full-data mappings, fitted once and applied at inference time. These
        # are safe to use for scoring FUTURE rows because those rows'
        # targets are not in the training data by construction.
        self.mappings_: dict[str, pd.Series] = {}
        self._fitted = False

    @staticmethod
    def _key(df: pd.DataFrame, col: str) -> pd.Series:
        """Composite keys are given as 'a+b'; resolve to a single joined key."""
        if "+" in col:
            parts = col.split("+")
            return df[parts].astype(str).agg("|".join, axis=1)
        return df[col].astype(str)

    def _smoothed_map(self, keys: pd.Series, y: pd.Series, prior: float) -> pd.Series:
        """Empirical-Bayes shrunk mean per category."""
        stats = y.groupby(keys).agg(["count", "mean"])
        n, mean = stats["count"], stats["mean"]
        return (n * mean + self.smoothing * prior) / (n + self.smoothing)

    def fit_transform(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """
        Produce out-of-fold encodings for training data.

        The frame is sorted by date first so the TimeSeriesSplit folds are
        genuine time blocks. Encoding a row using a fold that comes after it
        would be a future leak dressed up as cross-validation.
        """
        if self.target_col not in df.columns:
            raise KeyError(f"target column {self.target_col!r} not in frame")

        out = df.copy()
        if date_col in out.columns:
            out = out.sort_values(date_col, kind="mergesort")

        y = out[self.target_col].astype(float)
        prior = float(y.mean()) if self.prior_ is None else self.prior_
        self.prior_ = prior

        splitter = TimeSeriesSplit(n_splits=self.n_splits)
        splits = list(splitter.split(out))

        for col in self.cols:
            keys = self._key(out, col)
            encoded = pd.Series(np.nan, index=out.index, dtype=float)

            for train_idx, valid_idx in splits:
                mapping = self._smoothed_map(keys.iloc[train_idx], y.iloc[train_idx], prior)
                encoded.iloc[valid_idx] = keys.iloc[valid_idx].map(mapping).to_numpy()

            # Rows never in a validation fold (the first block) and categories
            # unseen in their training fold both fall back to the prior --
            # the correct answer when there is no evidence.
            out[f"te_{col.replace('+', '_')}"] = encoded.fillna(prior)

            # Fit the full-data mapping for inference-time use.
            self.mappings_[col] = self._smoothed_map(keys, y, prior)

        self._fitted = True
        log.info(
            "target_encoding_fitted",
            cols=self.cols,
            prior=round(prior, 4),
            smoothing=self.smoothing,
        )
        return out.loc[df.index] if out.index.equals(df.index) else out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted mappings to new (future) rows."""
        if not self._fitted:
            raise RuntimeError("call fit_transform() before transform()")

        out = df.copy()
        for col in self.cols:
            keys = self._key(out, col)
            out[f"te_{col.replace('+', '_')}"] = (
                keys.map(self.mappings_[col]).fillna(self.prior_).astype(float)
            )
        return out


# ===========================================================================
# Fourier / harmonic terms
# ===========================================================================


def add_fourier_terms(
    df: pd.DataFrame,
    date_col: str = "date",
    periods: Iterable[tuple[float, int]] = ((7.0, 3), (365.25, 5)),
) -> pd.DataFrame:
    """
    Add sine/cosine harmonics for each (period, n_harmonics) pair.

        For k = 1..K:
            sin_k = sin(2*pi*k*t / P)
            cos_k = cos(2*pi*k*t / P)

    Why harmonics rather than dummies
    ---------------------------------
    Day-of-week could be represented as six dummy variables, and day-of-year as
    364. Harmonics represent the same periodic structure in 2K columns while
    enforcing smoothness and continuity -- crucially, they make 31 December and
    1 January adjacent, which dummies do not.

    Defaults: 3 harmonics on the weekly cycle (captures a weekend spike plus
    a mid-week shape), 5 on the annual (captures broad seasonal swing without
    chasing week-level noise). Period 365.25 rather than 365 to stay aligned
    across leap years over M5's 5.25-year span.

    `t` is measured in days since the series start, so the phase is anchored to
    a fixed origin and remains consistent between training and inference.
    """
    if date_col not in df.columns:
        raise KeyError(f"{date_col!r} not in frame")

    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    # Fixed origin so train and score share a phase reference.
    t = (dates - pd.Timestamp("2011-01-29")).dt.days.to_numpy(dtype=float)

    for period, n_harmonics in periods:
        label = int(round(period))
        for k in range(1, int(n_harmonics) + 1):
            angle = 2.0 * np.pi * k * t / period
            out[f"fourier_sin_{label}_{k}"] = np.sin(angle)
            out[f"fourier_cos_{label}_{k}"] = np.cos(angle)

    log.info("fourier_terms_added", periods=list(periods))
    return out


# ===========================================================================
# Interaction terms
# ===========================================================================

# Interactions worth testing, each with a stated mechanism. Products are only
# useful to a tree model when the interaction is multiplicative in a way axis-
# aligned splits approximate poorly; listing the reason keeps this from
# becoming a combinatorial dump of every column pair.
DEFAULT_INTERACTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "snap_flag",
        "price_ratio_to_28d_mean",
        "SNAP benefit days concentrate purchasing among price-sensitive "
        "households, so a discount landing on a SNAP day should act more "
        "strongly than the same discount off-SNAP.",
    ),
    (
        "is_weekend",
        "roll_mean_7",
        "Weekend uplift is roughly proportional to a series' base rate rather "
        "than a constant number of units, so the effect is multiplicative.",
    ),
    (
        "is_event_day",
        "roll_mean_28",
        "Same argument for calendar events: a holiday lifts a high-volume "
        "staple by far more units than a slow item.",
    ),
    (
        "price_ratio_to_28d_mean",
        "roll_mean_28",
        "Absolute unit response to a price change scales with base volume; "
        "the ratio alone carries no volume information.",
    ),
)


def add_interaction_terms(
    df: pd.DataFrame,
    interactions: Sequence[tuple[str, str, str]] = DEFAULT_INTERACTIONS,
    skip_missing: bool = True,
) -> pd.DataFrame:
    """
    Add pairwise products for the listed column pairs.

    Missing columns are skipped with a warning rather than raising, so the
    ablation harness can run against feature sets that lack some inputs.
    """
    out = df.copy()
    added, skipped = [], []

    for left, right, _rationale in interactions:
        if left not in out.columns or right not in out.columns:
            skipped.append(f"{left}*{right}")
            if not skip_missing:
                raise KeyError(f"interaction inputs missing: {left!r}, {right!r}")
            continue
        name = f"inter_{left}__{right}"
        out[name] = out[left].astype(float) * out[right].astype(float)
        added.append(name)

    log.info("interaction_terms_added", n_added=len(added), skipped=skipped)
    return out


# ===========================================================================
# Ablation harness -- the part that decides what survives
# ===========================================================================


@dataclass
class AblationResult:
    """One arm of the ablation."""

    name: str
    description: str
    n_features: int
    wape: float
    rmsse: float
    mae: float
    bias: float
    delta_wape_vs_base: float = float("nan")
    relative_improvement_pct: float = float("nan")
    verdict: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "arm": self.name,
            "n_features": self.n_features,
            "wape": round(self.wape, 4),
            "rmsse": round(self.rmsse, 4) if np.isfinite(self.rmsse) else None,
            "mae": round(self.mae, 4),
            "bias": round(self.bias, 4),
            "delta_wape": round(self.delta_wape_vs_base, 4),
            "rel_improvement_pct": round(self.relative_improvement_pct, 3),
            "verdict": self.verdict,
        }


@dataclass
class AblationStudy:
    """Full ablation outcome, with an explicit keep/drop recommendation."""

    results: list[AblationResult] = field(default_factory=list)
    baseline_wape: float = float("nan")
    keep: list[str] = field(default_factory=list)
    drop: list[str] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_row() for r in self.results])

    def summary(self) -> str:
        lines = [
            "FEATURE ABLATION STUDY",
            f"  Baseline WAPE: {self.baseline_wape:.3f}%",
            f"  KEEP: {', '.join(self.keep) if self.keep else '(none earned inclusion)'}",
            f"  DROP: {', '.join(self.drop) if self.drop else '(none)'}",
        ]
        return "\n".join(lines)


class FeatureAblationStudy:
    """
    Measure whether each advanced feature group actually improves accuracy.

    Design
    ------
    Each arm trains the SAME model on the SAME temporal split, varying only
    the feature set. The evaluation window is the final `holdout_days` of the
    data, held out from every arm, so arms are directly comparable.

    A LightGBM with fixed hyperparameters is used rather than a tuned one:
    re-tuning per arm would confound "this feature helps" with "this arm got a
    luckier hyperparameter draw". Fixing the model isolates the feature effect,
    which is the question being asked.

    The decision rule
    -----------------
    A group is KEPT only if it improves WAPE by more than `min_improvement_pct`
    relative to baseline. Anything smaller is inside run-to-run noise, and
    keeping it means paying compute and complexity for a coin flip. Features
    that make accuracy WORSE are reported as such rather than quietly retained,
    which is the outcome target encoding not infrequently earns.
    """

    def __init__(
        self,
        target_col: str = "sales",
        date_col: str = "date",
        holdout_days: int = 28,
        min_improvement_pct: float = 0.5,
        model_factory: Any = None,
    ) -> None:
        self.target_col = target_col
        self.date_col = date_col
        self.holdout_days = holdout_days
        self.min_improvement_pct = min_improvement_pct
        self._model_factory = model_factory
        self._log = get_logger("FeatureAblationStudy")

    def _make_model(self) -> Any:
        if self._model_factory is not None:
            return self._model_factory()
        from lightgbm import LGBMRegressor

        # Fixed, deliberately middle-of-the-road settings. Identical across
        # every arm.
        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )

    def _numeric_features(self, df: pd.DataFrame) -> list[str]:
        exclude = {
            self.target_col,
            self.date_col,
            "id",
            "d",
            "wm_yr_wk",
            "item_id",
            "store_id",
            "dept_id",
            "cat_id",
            "state_id",
            "feature_version",
            "event_name_1",
            "event_name_2",
            "event_type_1",
            "event_type_2",
        }
        return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    def _evaluate(
        self, df: pd.DataFrame, feature_cols: list[str], name: str
    ) -> tuple[dict[str, float], int]:
        """Train on everything before the holdout window, score inside it."""
        d = df.copy()
        d[self.date_col] = pd.to_datetime(d[self.date_col])
        d = d.sort_values(self.date_col, kind="mergesort")

        cutoff = d[self.date_col].max() - pd.Timedelta(days=self.holdout_days)
        train, test = d[d[self.date_col] <= cutoff], d[d[self.date_col] > cutoff]

        if train.empty or test.empty:
            raise ValueError(
                f"arm {name!r}: empty split (train={len(train)}, test={len(test)}); "
                f"holdout_days={self.holdout_days} may exceed the data range"
            )

        X_tr = train[feature_cols].astype(float).fillna(0.0)
        y_tr = train[self.target_col].astype(float).clip(lower=0)
        X_te = test[feature_cols].astype(float).fillna(0.0)
        y_te = test[self.target_col].astype(float).clip(lower=0)

        model = self._make_model()
        model.fit(X_tr, y_tr)
        preds = np.clip(model.predict(X_te), 0.0, None)

        metrics = compute_metrics(y_te.to_numpy(), preds, y_insample=y_tr.to_numpy())
        return metrics, len(feature_cols)

    def run(
        self,
        df_base: pd.DataFrame,
        arms: dict[str, tuple[pd.DataFrame, str]] | None = None,
    ) -> AblationStudy:
        """
        Execute the study.

        Args:
            df_base: feature frame from the base pipeline -- the control arm.
            arms:    name -> (frame, description). When omitted, the standard
                     four arms are constructed from `df_base`: target encoding,
                     Fourier terms, interactions, and all three combined.

        Returns:
            AblationStudy with per-arm metrics and a keep/drop recommendation.
        """
        if arms is None:
            arms = self._default_arms(df_base)

        base_metrics, base_n = self._evaluate(df_base, self._numeric_features(df_base), "baseline")
        baseline_wape = base_metrics["wape"]

        results = [
            AblationResult(
                name="baseline",
                description="Base feature pipeline only (control arm)",
                n_features=base_n,
                wape=base_metrics["wape"],
                rmsse=base_metrics["rmsse"],
                mae=base_metrics["mae"],
                bias=base_metrics["bias"],
                delta_wape_vs_base=0.0,
                relative_improvement_pct=0.0,
                verdict="control",
            )
        ]
        keep: list[str] = []
        drop: list[str] = []

        for name, (frame, description) in arms.items():
            metrics, n_feat = self._evaluate(frame, self._numeric_features(frame), name)
            delta = metrics["wape"] - baseline_wape
            rel = (
                100.0 * (baseline_wape - metrics["wape"]) / baseline_wape
                if baseline_wape and np.isfinite(baseline_wape)
                else float("nan")
            )

            if not np.isfinite(rel):
                verdict = "INCONCLUSIVE"
            elif rel >= self.min_improvement_pct:
                verdict = "KEEP"
                keep.append(name)
            elif rel <= -self.min_improvement_pct:
                verdict = "HARMFUL - DROP"
                drop.append(name)
            else:
                verdict = "NO EFFECT - DROP"
                drop.append(name)

            results.append(
                AblationResult(
                    name=name,
                    description=description,
                    n_features=n_feat,
                    wape=metrics["wape"],
                    rmsse=metrics["rmsse"],
                    mae=metrics["mae"],
                    bias=metrics["bias"],
                    delta_wape_vs_base=delta,
                    relative_improvement_pct=rel,
                    verdict=verdict,
                )
            )
            self._log.info(
                "ablation_arm_complete",
                arm=name,
                wape=round(metrics["wape"], 4),
                rel_improvement_pct=round(rel, 3) if np.isfinite(rel) else None,
                verdict=verdict,
            )

        return AblationStudy(results=results, baseline_wape=baseline_wape, keep=keep, drop=drop)

    def _default_arms(self, df_base: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, str]]:
        """Construct the four standard arms from the base frame."""
        arms: dict[str, tuple[pd.DataFrame, str]] = {}

        te_cols = [c for c in ("item_id", "store_id", "dept_id") if c in df_base.columns]
        if te_cols:
            encoder = TargetEncoder(cols=te_cols, target_col=self.target_col)
            arms["target_encoding"] = (
                encoder.fit_transform(df_base, date_col=self.date_col),
                f"Base + out-of-fold smoothed target encoding on {te_cols}",
            )

        arms["fourier"] = (
            add_fourier_terms(df_base, date_col=self.date_col),
            "Base + Fourier harmonics (weekly K=3, annual K=5)",
        )
        arms["interactions"] = (
            add_interaction_terms(df_base),
            "Base + hypothesis-driven pairwise interaction terms",
        )

        combined = df_base
        if te_cols:
            combined = TargetEncoder(cols=te_cols, target_col=self.target_col).fit_transform(
                combined, date_col=self.date_col
            )
        combined = add_interaction_terms(add_fourier_terms(combined, date_col=self.date_col))
        arms["all_advanced"] = (combined, "Base + target encoding + Fourier + interactions")

        return arms
