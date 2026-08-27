"""Two-part hurdle model for zero-inflated demand.

Motivation
----------
The distribution screen (§3.8 of EDA) shows 47.2% of M5 series prefer an
explicitly zero-inflated process by AIC.  A hurdle model maps directly onto
what the screen found: it separates the zero-generating process from the
positive-demand process, making each diagnosable independently.

Architecture
~~~~~~~~~~~~
.. code-block:: text

    P(y > 0 | x)      →  LightGBM binary classifier
    E[y | y > 0, x]    →  LightGBM regressor on positive-only subset
    prediction         →  P(y > 0 | x) × E[y | y > 0, x]

Three advantages over patching NGBoost with a NB distribution:

1. **Uses the existing harness.** Both components are standard LightGBM fits
   with no custom distribution plumbing.
2. **Separately diagnosable.** A bad forecast is either a probability error or
   a magnitude error, and you can see which — this is not possible with a single
   distributional model.
3. **Produces a full predictive distribution** when the positive part uses a
   quantile or distributional loss.

Statistical justification for the quadrant inversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a Poisson with rate λ, P(X = 0) = e^(−λ).  At λ = 0.1, that is 90% zeros
from the sampling process alone.  A zero-inflated model must then separate
structural zeros from sampling zeros, but when λ is tiny the likelihood surface
is nearly flat in the mixing parameter π — both explanations fit equally well.
AIC/BIC penalise the extra parameter without compensating gain.

For a SMOOTH series with λ = 5, Poisson predicts P(X = 0) ≈ 0.7%.  Any material
excess of zeros is strong evidence of a distinct zero-generating process, and π
is sharply identified.

So the inversion is not "sparse series lack density" — it is that zero-inflation
is only *identifiable* when the baseline count process predicts few zeros.  This
is a statistical property of the model class, not an artifact of sample size,
and it will replicate on any dataset with this sparsity profile.

Test this model provides
~~~~~~~~~~~~~~~~~~~~~~~~
Whether modelling the zero process separately helps *conditionally* (given
features), not just marginally (in the univariate distribution screen).

References
----------
Mullahy (1986), "Specification and testing of some modified count data models",
    Journal of Econometrics 33(3):341-365.  [the hurdle model]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from src.config.logging import get_logger

log = get_logger(__name__)


def _clean_features(
    X: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return finite numeric features; align to fitted columns if provided."""
    out = X.copy() if columns is None else X.reindex(columns=columns)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class HurdleForecaster:
    """LightGBM hurdle model: binary gate × positive-conditional regressor.

    The two components are fitted independently and combined at prediction
    time.  The binary gate learns ``P(y > 0 | x)`` on all training rows.
    The regressor learns ``E[y | y > 0, x]`` on the positive-demand subset
    only, so its tree splits never see the zeros that dominate M5.

    Quantile prediction is supported: fit the positive part with
    ``objective="quantile"`` and combine with the gate probability.
    """

    def __init__(
        self,
        *,
        gate_params: dict[str, Any] | None = None,
        regressor_params: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        _gate_defaults = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "n_jobs": 1,
            "random_state": random_state,
            "verbosity": -1,
        }
        _reg_defaults = {
            "n_estimators": 400,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "n_jobs": 1,
            "random_state": random_state,
            "verbosity": -1,
        }
        if gate_params:
            _gate_defaults.update(gate_params)
        if regressor_params:
            _reg_defaults.update(regressor_params)

        self.gate = LGBMClassifier(**_gate_defaults)
        self.regressor = LGBMRegressor(**_reg_defaults)
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> HurdleForecaster:
        """Fit gate on all rows, regressor on positive-demand rows only."""
        target = np.asarray(y, dtype=float).ravel()
        if len(X) != len(target):
            raise ValueError("X and y must have the same number of rows")
        if not np.isfinite(target).all() or (target < 0).any():
            raise ValueError("y must be finite non-negative values")

        X_clean = _clean_features(X)
        self.feature_names_ = X_clean.columns.tolist()

        # Gate: P(y > 0 | x)
        binary = (target > 0).astype(int)
        self.gate.fit(X_clean, binary)

        # Regressor: E[y | y > 0, x]
        positive_mask = target > 0
        n_positive = int(positive_mask.sum())
        if n_positive < 10:
            log.warning(
                "hurdle_few_positive_rows",
                n_positive=n_positive,
                n_total=len(target),
                consequence="regressor may underfit",
            )
        self.regressor.fit(X_clean[positive_mask], target[positive_mask])
        self._fitted = True

        zero_frac = 1.0 - positive_mask.mean()
        log.info(
            "hurdle_fitted",
            n_total=len(target),
            n_positive=n_positive,
            zero_fraction=round(zero_frac, 4),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return E[y | x] = P(y > 0 | x) × E[y | y > 0, x], clipped to ≥ 0."""
        if not self._fitted:
            raise RuntimeError("call fit before predict")
        X_clean = _clean_features(X, self.feature_names_)

        # Gate probability
        prob_positive = self.gate.predict_proba(X_clean)[:, 1]

        # Conditional positive forecast
        positive_pred = np.clip(self.regressor.predict(X_clean), 0.0, None)

        return np.clip(prob_positive * positive_pred, 0.0, None)

    def predict_components(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return both components for diagnostics: gate probability and
        conditional forecast separately, so error decomposition is possible.

        A bad forecast is either:
        - a **probability error** (gate mis-predicts whether demand occurs), or
        - a **magnitude error** (regressor mis-predicts the conditional level).

        This decomposition is not available from a single distributional model.
        """
        if not self._fitted:
            raise RuntimeError("call fit before predict_components")
        X_clean = _clean_features(X, self.feature_names_)
        prob_positive = self.gate.predict_proba(X_clean)[:, 1]
        cond_pred = np.clip(self.regressor.predict(X_clean), 0.0, None)
        combined = np.clip(prob_positive * cond_pred, 0.0, None)
        return pd.DataFrame(
            {
                "prob_positive": prob_positive,
                "cond_positive_pred": cond_pred,
                "hurdle_pred": combined,
            }
        )

    def gate_feature_importance(self) -> pd.Series:
        """Feature importance from the zero/non-zero gate model."""
        if not self._fitted:
            raise RuntimeError("call fit before gate_feature_importance")
        return pd.Series(
            self.gate.feature_importances_,
            index=self.feature_names_,
            name="gate_importance",
        ).sort_values(ascending=False)

    def regressor_feature_importance(self) -> pd.Series:
        """Feature importance from the positive-conditional regressor."""
        if not self._fitted:
            raise RuntimeError("call fit before regressor_feature_importance")
        return pd.Series(
            self.regressor.feature_importances_,
            index=self.feature_names_,
            name="regressor_importance",
        ).sort_values(ascending=False)
