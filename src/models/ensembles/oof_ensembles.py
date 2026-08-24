"""Forecast combinations fitted strictly on rolling-origin out-of-fold output.

An ensemble can look spectacular when its meta-learner sees in-sample base
predictions and then collapse on the next origin.  Every fit entry point here
therefore accepts an explicit ``origin`` column and constructs its design
matrix from predictions that were generated before the corresponding actuals
were observed.

The useful ensemble diagnostic is residual correlation, not a long list of
model names.  Two strong models with nearly identical residuals offer little
diversification; two strong models that miss at different times can improve a
blend even when neither wins every series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

OOF_REQUIRED_COLUMNS = {"series_id", "date", "origin", "model", "y_true", "y_pred"}


@dataclass(frozen=True)
class OOFMatrix:
    """Aligned out-of-fold target and base-prediction matrix."""

    y_true: np.ndarray
    predictions: pd.DataFrame
    keys: pd.DataFrame

    @property
    def model_names(self) -> list[str]:
        return self.predictions.columns.tolist()


@dataclass(frozen=True)
class EnsembleDiagnostics:
    """Error-correlation evidence used to justify a blend's composition."""

    residual_correlation: pd.DataFrame
    base_mse: pd.Series
    base_mae: pd.Series


def oof_prediction_matrix(
    predictions: pd.DataFrame,
    model_names: list[str] | None = None,
) -> OOFMatrix:
    """Pivot a long rolling-origin prediction frame into an aligned OOF matrix.

    ``origin`` is required deliberately: a bare prediction frame gives no
    evidence that a stacking learner is being trained out of sample.  The
    shared backtest harness emits one row per ``(series, date, origin, model)``
    which is the unit this function validates.
    """
    missing = OOF_REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise KeyError(
            "OOF ensemble fitting requires rolling-origin predictions with columns "
            f"{sorted(OOF_REQUIRED_COLUMNS)}; missing {sorted(missing)}"
        )

    cols = ["series_id", "date", "origin", "model", "y_true", "y_pred"]
    frame = predictions[cols].copy()
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame = frame.dropna(subset=["y_true", "y_pred"])
    if model_names is not None:
        frame = frame[frame["model"].isin(model_names)]

    if frame.empty:
        raise ValueError("no finite OOF prediction rows remain after filtering")

    key_cols = ["series_id", "date", "origin"]
    if frame.duplicated(key_cols + ["model"]).any():
        raise ValueError("OOF frame has duplicate predictions for one key and model")
    target_counts = frame.groupby(key_cols, observed=True)["y_true"].nunique()
    if (target_counts > 1).any():
        raise ValueError("OOF frame has conflicting actuals for the same key")

    wide = frame.pivot(index=key_cols, columns="model", values="y_pred").sort_index()
    # A combination must be judged on the same observations for every base
    # model.  Partial rows make a model look better simply because it skipped
    # difficult forecasts, so they are excluded rather than imputed.
    wide = wide.dropna(axis=0, how="any")
    if wide.shape[1] < 2:
        raise ValueError("an ensemble needs predictions from at least two models")
    if wide.empty:
        raise ValueError("no OOF rows have predictions from every selected model")

    targets = frame.drop_duplicates(key_cols).set_index(key_cols)["y_true"].reindex(wide.index)
    return OOFMatrix(
        y_true=targets.to_numpy(dtype=float),
        predictions=wide.reset_index(drop=True),
        keys=wide.index.to_frame(index=False).reset_index(drop=True),
    )


def ensemble_diagnostics(matrix: OOFMatrix) -> EnsembleDiagnostics:
    """Compute residual correlation and base accuracy on exactly the OOF rows."""
    residuals = matrix.predictions.sub(matrix.y_true, axis=0)
    return EnsembleDiagnostics(
        residual_correlation=residuals.corr(),
        base_mse=(residuals**2).mean().sort_values(),
        base_mae=residuals.abs().mean().sort_values(),
    )


class SimpleAverageEnsemble:
    """Equal-weight average baseline; difficult to beat when bases are correlated."""

    def __init__(self, model_names: list[str] | None = None) -> None:
        self.model_names = model_names
        self.model_names_: list[str] = []

    def fit_oof(self, predictions: pd.DataFrame) -> SimpleAverageEnsemble:
        matrix = oof_prediction_matrix(predictions, self.model_names)
        self.model_names_ = matrix.model_names
        return self

    def predict(self, prediction_matrix: pd.DataFrame) -> np.ndarray:
        cols = self._validate_prediction_columns(prediction_matrix)
        return np.clip(prediction_matrix[cols].mean(axis=1).to_numpy(float), 0.0, None)

    def _validate_prediction_columns(self, prediction_matrix: pd.DataFrame) -> list[str]:
        if not self.model_names_:
            raise RuntimeError("call fit_oof before predict")
        missing = set(self.model_names_) - set(prediction_matrix.columns)
        if missing:
            raise KeyError(f"prediction matrix is missing base models: {sorted(missing)}")
        return self.model_names_


class ConstrainedWeightedBlend(SimpleAverageEnsemble):
    """Non-negative, sum-to-one blend optimized on rolling-origin OOF output."""

    def __init__(self, model_names: list[str] | None = None, l2_penalty: float = 1e-8) -> None:
        super().__init__(model_names)
        self.l2_penalty = float(l2_penalty)
        self.weights_: np.ndarray | None = None
        self.optimization_result_: Any | None = None

    def fit_oof(self, predictions: pd.DataFrame) -> ConstrainedWeightedBlend:
        matrix = oof_prediction_matrix(predictions, self.model_names)
        X, y = matrix.predictions.to_numpy(float), matrix.y_true
        n_models = X.shape[1]

        # MSE is smooth enough for SLSQP's gradient approximation.  On a fixed
        # OOF set, minimizing MAE and WAPE uses the same absolute-error numerator
        # but non-smooth MAE can make SLSQP unstable; validation still reports
        # the full point-metric bundle through the shared harness.
        def objective(weights: np.ndarray) -> float:
            residual = y - X @ weights
            return float(np.mean(residual**2) + self.l2_penalty * np.sum(weights**2))

        result = minimize(
            objective,
            x0=np.full(n_models, 1.0 / n_models),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success:
            raise RuntimeError(f"SLSQP blend fitting failed: {result.message}")

        self.model_names_ = matrix.model_names
        self.weights_ = np.asarray(result.x, dtype=float)
        self.optimization_result_ = result
        return self

    def predict(self, prediction_matrix: pd.DataFrame) -> np.ndarray:
        cols = self._validate_prediction_columns(prediction_matrix)
        if self.weights_ is None:
            raise RuntimeError("call fit_oof before predict")
        values = prediction_matrix[cols].to_numpy(float) @ self.weights_
        return np.clip(values, 0.0, None)

    def weight_table(self) -> pd.DataFrame:
        if self.weights_ is None:
            raise RuntimeError("call fit_oof before requesting weights")
        return pd.DataFrame({"model": self.model_names_, "weight": self.weights_}).sort_values(
            "weight", ascending=False
        )


class RidgeStackingEnsemble(SimpleAverageEnsemble):
    """Ridge meta-learner fitted only on rolling-origin OOF prediction rows."""

    def __init__(self, model_names: list[str] | None = None, alpha: float = 1.0) -> None:
        super().__init__(model_names)
        self.alpha = float(alpha)
        self.meta_model_: Ridge | None = None

    def fit_oof(self, predictions: pd.DataFrame) -> RidgeStackingEnsemble:
        matrix = oof_prediction_matrix(predictions, self.model_names)
        # An intercept can correct shared level bias; Ridge regularization keeps
        # collinear base forecasts from yielding extreme unstable coefficients.
        model = Ridge(alpha=self.alpha, fit_intercept=True)
        model.fit(matrix.predictions.to_numpy(float), matrix.y_true)
        self.model_names_ = matrix.model_names
        self.meta_model_ = model
        return self

    def predict(self, prediction_matrix: pd.DataFrame) -> np.ndarray:
        cols = self._validate_prediction_columns(prediction_matrix)
        if self.meta_model_ is None:
            raise RuntimeError("call fit_oof before predict")
        return np.clip(self.meta_model_.predict(prediction_matrix[cols].to_numpy(float)), 0.0, None)


def best_cross_family_blend(
    predictions: pd.DataFrame,
    model_families: dict[str, list[str]],
) -> tuple[ConstrainedWeightedBlend, EnsembleDiagnostics]:
    """Choose the best OOF base in each family, then fit a diversified blend.

    This avoids silently giving an overrepresented family five votes in a
    simple average.  Family selection and blend fitting happen on the same OOF
    data, so the returned model must still be evaluated on a later origin; it
    is a candidate generator, not a final unbiased score.
    """
    full = oof_prediction_matrix(predictions)
    diagnostics = ensemble_diagnostics(full)
    available = set(full.model_names)
    selected: list[str] = []
    for _family, candidates in model_families.items():
        usable = [name for name in candidates if name in available]
        if not usable:
            continue
        selected.append(str(diagnostics.base_mse[usable].idxmin()))
    selected = list(dict.fromkeys(selected))
    if len(selected) < 2:
        raise ValueError("cross-family blend needs at least two available model families")

    blend = ConstrainedWeightedBlend(selected).fit_oof(predictions)
    return blend, ensemble_diagnostics(oof_prediction_matrix(predictions, selected))
