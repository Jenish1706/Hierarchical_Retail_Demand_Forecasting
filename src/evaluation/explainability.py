"""
Model explainability: SHAP values and permutation importance.

Why gain importance is not enough
---------------------------------
LightGBM's built-in `feature_importances_` reports total split gain. It has
three properties that make it misleading on this feature set specifically:

  1. IT SPLITS CREDIT AMONG CORRELATED FEATURES. `roll_mean_7`, `roll_mean_28`
     and `roll_mean_90` are highly correlated by construction. The tree picks
     one at each split largely arbitrarily, so their individual gains are
     diluted and none looks important -- even though the rolling-mean GROUP is
     doing most of the work. Read naively, the table suggests dropping all
     three.

  2. IT IS BIASED TOWARD HIGH-CARDINALITY FEATURES. Continuous features offer
     more candidate split points than binary flags, so they accumulate gain
     simply by having more chances to be chosen. `snap_flag` competes for
     attention against `time_idx` on unequal terms.

  3. IT IS TRAINING-SET ONLY. Gain measures how much a feature helped fit the
     training data, not how much it helps predict unseen data. A feature the
     model overfits on scores highly.

This module adds two complementary views:

  * SHAP (Lundberg & Lee, 2017) attributes each individual prediction to its
    features with a game-theoretic guarantee: contributions sum exactly to the
    prediction minus the base value. TreeSHAP computes this exactly for tree
    ensembles in polynomial time. It answers "why THIS forecast" as well as
    "what matters overall".

  * PERMUTATION IMPORTANCE (Breiman 2001; Fisher, Rudin & Dominici 2019)
    measures the drop in HOLDOUT accuracy when one feature's values are
    shuffled. It is model-agnostic, measured on data the model did not train
    on, and directly answers the operational question: what does this feature
    buy me in production?

Reading them together
---------------------
The two disagree in an informative way. Permutation importance also suffers
under correlation -- shuffling `roll_mean_7` barely hurts if `roll_mean_28`
carries the same information -- so both can understate a correlated GROUP.
`permutation_importance_grouped()` addresses that directly by shuffling whole
correlated blocks together, which is the measurement that actually answers
"could I drop the rolling features".

References
----------
Lundberg & Lee (2017), "A unified approach to interpreting model predictions",
    NeurIPS 30.
Lundberg et al. (2020), "From local explanations to global understanding with
    explainable AI for trees", Nature Machine Intelligence 2:56-67.
Breiman (2001), "Random forests", Machine Learning 45(1):5-32.
Fisher, Rudin & Dominici (2019), "All models are wrong but many are useful",
    JMLR 20(177):1-81.
Hooker, Mentch & Zhou (2021) -- on the extrapolation problem in permutation
    methods under correlated features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.point_metrics import wape
from src.config.logging import get_logger

log = get_logger(__name__)

try:
    import shap

    _HAS_SHAP = True
except ImportError:                                        # pragma: no cover
    shap, _HAS_SHAP = None, False


# Feature groups used for grouped permutation importance. Members are
# correlated by construction, so shuffling them individually understates the
# group. Prefixes are matched against column names.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "lags": ("lag_",),
    "rolling_mean": ("roll_mean_",),
    "rolling_dispersion": ("roll_std_", "roll_max_", "cv_"),
    "trend": ("trend_",),
    "intermittency": ("zero_rate_", "days_since_sale"),
    "calendar_cyclical": ("dow_", "month_sin", "month_cos", "week_of_year", "quarter"),
    "calendar_flags": ("is_weekend", "is_month_start", "is_month_end", "is_event_day",
                       "event_type_"),
    "snap": ("snap_flag",),
    "price": ("price", "is_discounted"),
    "time": ("time_idx", "year_num", "day_of_month", "month_num"),
}


@dataclass
class ImportanceReport:
    """Combined importance view from all three methods."""

    gain: pd.DataFrame = field(default_factory=pd.DataFrame)
    permutation: pd.DataFrame = field(default_factory=pd.DataFrame)
    permutation_grouped: pd.DataFrame = field(default_factory=pd.DataFrame)
    shap_global: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_score: float = float("nan")

    def combined(self, top_n: int = 25) -> pd.DataFrame:
        """
        Side-by-side ranking from each method.

        Where the ranks disagree is the interesting part: a feature ranked high
        by gain but low by permutation is one the model leaned on to fit
        training data without it generalising.
        """
        frames = []
        if not self.gain.empty:
            frames.append(self.gain.set_index("feature")["gain_importance"].rename("gain"))
        if not self.permutation.empty:
            frames.append(
                self.permutation.set_index("feature")["importance_mean"].rename("permutation")
            )
        if not self.shap_global.empty:
            frames.append(
                self.shap_global.set_index("feature")["mean_abs_shap"].rename("shap")
            )
        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, axis=1)
        for col in out.columns:
            out[f"rank_{col}"] = out[col].rank(ascending=False, method="min")
        rank_cols = [c for c in out.columns if c.startswith("rank_")]
        out["rank_mean"] = out[rank_cols].mean(axis=1)
        # Disagreement between methods -- large values deserve a look.
        out["rank_spread"] = out[rank_cols].max(axis=1) - out[rank_cols].min(axis=1)
        return out.sort_values("rank_mean").head(top_n).reset_index()


class ModelExplainer:
    """
    Explainability for a fitted tree model.

    Usage:
        ex = ModelExplainer(model, X_holdout, y_holdout, feature_names)
        report = ex.full_report()
        ex.save_plots(Path("reports/explainability"))
    """

    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        max_shap_rows: int = 20_000,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.X = X
        self.y = None if y is None else np.asarray(y, dtype=float)
        self.feature_names = list(feature_names or X.columns)
        self.max_shap_rows = max_shap_rows
        self.random_state = random_state
        self._log = get_logger("ModelExplainer")

        self._shap_values: np.ndarray | None = None
        self._shap_sample: pd.DataFrame | None = None

    # -- gain ---------------------------------------------------------------

    def gain_importance(self) -> pd.DataFrame:
        """Built-in split-gain importance. Reported for contrast, not for decisions."""
        if not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "gain_importance": np.asarray(self.model.feature_importances_, dtype=float),
                }
            )
            .sort_values("gain_importance", ascending=False)
            .reset_index(drop=True)
        )

    # -- SHAP ---------------------------------------------------------------

    def compute_shap(self, sample_rows: int | None = None) -> np.ndarray:
        """
        Exact TreeSHAP values for a sample of rows.

        Sampling is for runtime only: TreeSHAP is polynomial but not free, and
        global summaries converge quickly. The sample is random rather than
        head/tail so it is not a single time slice.
        """
        if not _HAS_SHAP:
            raise ImportError("shap is not installed; `pip install shap`")

        n = sample_rows or min(self.max_shap_rows, len(self.X))
        sample = self.X.sample(n=min(n, len(self.X)), random_state=self.random_state)

        explainer = shap.TreeExplainer(self.model)
        values = explainer.shap_values(sample)
        if isinstance(values, list):          # multi-output; regression -> [0]
            values = values[0]

        self._shap_values = np.asarray(values)
        self._shap_sample = sample
        self._log.info("shap_computed", rows=len(sample), features=self._shap_values.shape[1])
        return self._shap_values

    def shap_global_importance(self) -> pd.DataFrame:
        """
        Mean |SHAP| per feature -- the standard global summary.

        Mean ABSOLUTE value, because positive and negative contributions would
        otherwise cancel: a feature that pushes forecasts up on weekends and
        down on weekdays is important, not neutral.
        """
        if self._shap_values is None:
            self.compute_shap()

        mean_abs = np.abs(self._shap_values).mean(axis=0)
        mean_signed = self._shap_values.mean(axis=0)
        return (
            pd.DataFrame(
                {
                    "feature": self._shap_sample.columns.tolist(),
                    "mean_abs_shap": mean_abs,
                    "mean_signed_shap": mean_signed,
                    "shap_std": self._shap_values.std(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def shap_by_segment(self, segment: pd.Series, top_n: int = 10) -> pd.DataFrame:
        """
        Top features per segment (e.g. per category).

        Reveals heterogeneity a single global ranking hides -- for instance
        SNAP mattering for FOODS but not for HOBBIES, which is a substantive
        finding about the business rather than about the model.
        """
        if self._shap_values is None:
            self.compute_shap()

        seg = segment.loc[self._shap_sample.index]
        records: list[dict[str, Any]] = []

        for value, idx in seg.groupby(seg, observed=True).groups.items():
            positions = self._shap_sample.index.get_indexer(idx)
            positions = positions[positions >= 0]
            if positions.size == 0:
                continue
            mean_abs = np.abs(self._shap_values[positions]).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:top_n]
            for rank, j in enumerate(order, start=1):
                records.append(
                    {
                        "segment": value,
                        "rank": rank,
                        "feature": self._shap_sample.columns[j],
                        "mean_abs_shap": float(mean_abs[j]),
                        "n_rows": int(positions.size),
                    }
                )
        return pd.DataFrame(records)

    def explain_prediction(self, row_index: int, top_n: int = 10) -> dict[str, Any]:
        """
        Waterfall decomposition of a single forecast.

        Contributions sum exactly to (prediction - base_value); that additivity
        is SHAP's defining guarantee and is what makes a per-forecast
        explanation defensible to a planner asking why a number moved.
        """
        if not _HAS_SHAP:
            raise ImportError("shap is not installed")

        row = self.X.iloc[[row_index]]
        explainer = shap.TreeExplainer(self.model)
        values = explainer.shap_values(row)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values).ravel()

        base = explainer.expected_value
        base = float(np.ravel(base)[0]) if np.ndim(base) else float(base)
        prediction = float(self.model.predict(row)[0])

        contributions = (
            pd.DataFrame(
                {
                    "feature": self.X.columns,
                    "value": row.iloc[0].to_numpy(),
                    "shap_contribution": values,
                }
            )
            .assign(abs_contribution=lambda d: d["shap_contribution"].abs())
            .sort_values("abs_contribution", ascending=False)
            .head(top_n)
            .drop(columns="abs_contribution")
            .reset_index(drop=True)
        )

        return {
            "row_index": int(row_index),
            "base_value": base,
            "prediction": prediction,
            "sum_of_contributions": float(values.sum()),
            "reconciliation_error": float(prediction - base - values.sum()),
            "top_contributions": contributions,
        }

    # -- permutation importance ---------------------------------------------

    def permutation_importance(
        self,
        n_repeats: int = 5,
        scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> pd.DataFrame:
        """
        Holdout permutation importance, scored by WAPE.

        For each feature: shuffle its column, re-score, and record the
        DEGRADATION relative to the unshuffled baseline. Repeated `n_repeats`
        times because a single shuffle is one noisy draw; the standard
        deviation across repeats is reported so a reader can tell a real effect
        from noise.

        Positive importance = shuffling made the model worse = the feature was
        being used. Near-zero or negative = the model was not relying on it.
        """
        if self.y is None:
            raise ValueError("permutation importance requires holdout targets `y`")

        score_fn = scorer or wape
        rng = np.random.default_rng(self.random_state)

        baseline_pred = np.clip(self.model.predict(self.X), 0.0, None)
        baseline = score_fn(self.y, baseline_pred)

        records: list[dict[str, Any]] = []
        for col in self.feature_names:
            if col not in self.X.columns:
                continue
            original = self.X[col].to_numpy(copy=True)
            deltas: list[float] = []

            for _ in range(n_repeats):
                shuffled = original.copy()
                rng.shuffle(shuffled)
                self.X[col] = shuffled
                pred = np.clip(self.model.predict(self.X), 0.0, None)
                deltas.append(score_fn(self.y, pred) - baseline)

            self.X[col] = original           # always restore

            records.append(
                {
                    "feature": col,
                    "importance_mean": float(np.mean(deltas)),
                    "importance_std": float(np.std(deltas)),
                    "baseline_wape": baseline,
                }
            )

        out = (
            pd.DataFrame(records)
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        self._log.info(
            "permutation_importance_computed",
            baseline_wape=round(baseline, 4),
            n_features=len(out),
            n_repeats=n_repeats,
        )
        return out

    def permutation_importance_grouped(
        self,
        groups: Mapping[str, Sequence[str]] | None = None,
        n_repeats: int = 5,
    ) -> pd.DataFrame:
        """
        Permute whole correlated blocks together.

        This is the measurement that answers "can I drop the rolling features".
        Shuffling `roll_mean_7` alone barely hurts, because `roll_mean_28`
        still carries most of the signal -- so the individual scores understate
        the group. Shuffling the entire block at once removes the information
        rather than just one of its several copies.
        """
        if self.y is None:
            raise ValueError("permutation importance requires holdout targets `y`")

        groups = groups or FEATURE_GROUPS
        rng = np.random.default_rng(self.random_state)

        baseline = wape(self.y, np.clip(self.model.predict(self.X), 0.0, None))
        records: list[dict[str, Any]] = []

        for group_name, prefixes in groups.items():
            members = [
                c for c in self.feature_names
                if c in self.X.columns and any(c.startswith(p) for p in prefixes)
            ]
            if not members:
                continue

            originals = {c: self.X[c].to_numpy(copy=True) for c in members}
            deltas: list[float] = []

            for _ in range(n_repeats):
                # One shared permutation across the block preserves the
                # within-group correlation structure while destroying the
                # relationship to the target -- which isolates the group's
                # contribution rather than also breaking its internal
                # consistency.
                order = rng.permutation(len(self.X))
                for c in members:
                    self.X[c] = originals[c][order]
                pred = np.clip(self.model.predict(self.X), 0.0, None)
                deltas.append(wape(self.y, pred) - baseline)

            for c, values in originals.items():
                self.X[c] = values

            records.append(
                {
                    "group": group_name,
                    "n_features": len(members),
                    "importance_mean": float(np.mean(deltas)),
                    "importance_std": float(np.std(deltas)),
                    "members": ", ".join(sorted(members)[:6]) + ("..." if len(members) > 6 else ""),
                }
            )

        return (
            pd.DataFrame(records)
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

    # -- orchestration ------------------------------------------------------

    def full_report(self, n_repeats: int = 5, shap_rows: int | None = None) -> ImportanceReport:
        """Run every available method and bundle the results."""
        report = ImportanceReport(gain=self.gain_importance())

        if _HAS_SHAP:
            try:
                self.compute_shap(sample_rows=shap_rows)
                report.shap_global = self.shap_global_importance()
            except Exception as exc:                       # pragma: no cover
                self._log.warning("shap_failed", error=str(exc))
        else:
            self._log.warning("shap_unavailable", reason="package not installed")

        if self.y is not None:
            report.permutation = self.permutation_importance(n_repeats=n_repeats)
            report.permutation_grouped = self.permutation_importance_grouped(n_repeats=n_repeats)
            if not report.permutation.empty:
                report.baseline_score = float(report.permutation["baseline_wape"].iloc[0])

        return report

    def save_plots(self, out_dir: Path | str) -> list[Path]:
        """
        Write the standard SHAP plots (bar + beeswarm) as PNGs.

        Beeswarm is the more informative of the two: it shows not just which
        features matter but the DIRECTION and spread of their effect, so you
        can see e.g. that high `roll_mean_7` pushes forecasts up while low
        values push them down, and how much variation there is.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        if not _HAS_SHAP or self._shap_values is None:
            self._log.warning("plots_skipped", reason="SHAP values not computed")
            return written

        import matplotlib
        matplotlib.use("Agg")                 # headless; no display needed
        import matplotlib.pyplot as plt

        for kind, filename in (("bar", "shap_bar.png"), ("dot", "shap_beeswarm.png")):
            try:
                plt.figure()
                shap.summary_plot(
                    self._shap_values, self._shap_sample,
                    plot_type=kind, show=False, max_display=25,
                )
                path = out_dir / filename
                plt.tight_layout()
                plt.savefig(path, dpi=130, bbox_inches="tight")
                plt.close()
                written.append(path)
            except Exception as exc:                       # pragma: no cover
                self._log.warning("plot_failed", kind=kind, error=str(exc))

        self._log.info("shap_plots_saved", files=[str(p) for p in written])
        return written
