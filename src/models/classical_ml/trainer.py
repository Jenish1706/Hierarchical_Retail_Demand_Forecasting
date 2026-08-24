"""
Demand forecasting trainer: nested CV, ensembling, and the MLflow promotion gate.

What changed from the previous version, and why it mattered
------------------------------------------------------------
The earlier trainer tuned hyperparameters on the LAST CV fold and then
reported cross-validated performance using those same hyperparameters:

    *_, (train_idx, val_idx) = tscv.split(X)     # tune here
    ...
    for train_idx, val_idx in tscv.split(X):     # then report here
        ...

Fold 3's validation rows selected the hyperparameters and were then scored
with them. The reported CV WAPE was therefore optimistically biased, and the
bias grows with the size of the search space -- precisely the regime Optuna
operates in. This is the classic selection-bias-in-model-evaluation failure
documented by Cawley & Talbot (2010).

The fix is NESTED cross-validation:

    OUTER loop (TimeSeriesSplit, k=3)  -- reports unbiased performance
      |
      +-- INNER loop (TimeSeriesSplit, k=3) on the outer TRAINING rows only
            -- Optuna searches hyperparameters here
      +-- refit on all outer-training rows with the winning params
      +-- score once on the outer-test fold, which tuning never saw

The cost is real: k_outer x n_trials x k_inner fits instead of n_trials. The
benefit is that the number reported is the number you get in production.

Temporal ordering
-----------------
Every split is a TimeSeriesSplit, never KFold. Rows are sorted by date before
splitting so a fold boundary is a point in time. Shuffling would let the
model train on next Tuesday to predict last Monday, which inflates every
metric and is the single most common leak in forecasting code.

References
----------
Cawley & Talbot (2010), "On over-fitting in model selection and subsequent
    selection bias in performance evaluation", JMLR 11:2079-2107.
Makridakis et al. (2022), IJF 38(4):1346-1364 -- gradient boosting dominance
    on M5; the reason no neural model appears here.
Bergstra et al. (2011) -- TPE, the sampler Optuna uses by default.
"""
from __future__ import annotations

import json
import tempfile
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.logging import get_logger
from src.evaluation.point_metrics import compute_metrics, naive_scale
from src.mlops.registry import PromotionCandidate, PromotionGate
from src.mlops.tracking import ExperimentTracker

warnings.filterwarnings("ignore", category=FutureWarning)
log = get_logger(__name__)

# Optional heavy dependencies. Import failures are tolerated so the module can
# be imported (and unit-tested) without the full stack installed; the affected
# model simply drops out of the registry.
try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:                                        # pragma: no cover
    mlflow, _HAS_MLFLOW = None, False

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:                                        # pragma: no cover
    optuna, _HAS_OPTUNA = None, False

from lightgbm import LGBMRegressor

try:
    from xgboost import XGBRegressor
except ImportError:                                        # pragma: no cover
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:                                        # pragma: no cover
    CatBoostRegressor = None

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# ===========================================================================
# Model registry
# ===========================================================================

def _lgbm_tweedie(**overrides: Any) -> LGBMRegressor:
    """
    LightGBM with a Tweedie objective.

    68.2% of M5 (item, store, day) observations are zero (measured over the
    full panel, reports/eda) and the remainder are small positive counts --
    a compound Poisson-gamma shape. The EDA also found 99.6% of series
    overdispersed and Negative Binomial beating Poisson on 91.5% of sampled
    series, so an equidispersed Poisson objective is empirically wrong here.
    Tweedie with 1 < p < 2 is the natural likelihood for exactly that: it has
    a point mass at zero and a continuous positive part. An L2 objective
    instead assumes symmetric Gaussian noise around the mean, which on a
    zero-inflated count target pushes predictions upward and produces the
    familiar "forecasts everything as 0.4 units" behaviour.

    variance_power=1.1 sits near the Poisson end, appropriate when the zero
    mass is large. It is tunable, but the default is deliberately conservative.
    """
    params: dict[str, Any] = dict(
        objective="tweedie",
        tweedie_variance_power=1.1,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    params.update(overrides)
    return LGBMRegressor(**params)


MODEL_REGISTRY: dict[str, Callable[..., Any]] = {
    # Linear reference. Scaled because Ridge penalises coefficients on the raw
    # scale, and the feature matrix mixes counts with 0/1 flags.
    "ridge": lambda: Pipeline(
        [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))]
    ),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        n_jobs=-1, random_state=42,
    ),
    "extra_trees": lambda: ExtraTreesRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=10,
        n_jobs=-1, random_state=42,
    ),
    "lightgbm": lambda: LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbose=-1,
    ),
    # The zero-inflation-aware variant. Kept as a separate entry rather than
    # replacing the L2 version so the accuracy table shows what the objective
    # is worth on this data instead of asserting it.
    "lightgbm_tweedie": _lgbm_tweedie,
}

if XGBRegressor is not None:
    MODEL_REGISTRY["xgboost"] = lambda: XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, random_state=42,
        tree_method="hist",
    )

if CatBoostRegressor is not None:
    MODEL_REGISTRY["catboost"] = lambda: CatBoostRegressor(
        iterations=500, depth=7, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=42, verbose=0, allow_writing_files=False,
    )


# ===========================================================================
# Hyperparameter search spaces
# ===========================================================================

def _lgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _xgb_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "lightgbm": _lgbm_space,
    "lightgbm_tweedie": _lgbm_space,
    "xgboost": _xgb_space,
}


def _rebuild(model_name: str, params: dict[str, Any]) -> Any:
    """Reconstruct a tuned estimator from a parameter dict."""
    if model_name == "lightgbm":
        return LGBMRegressor(**params, n_jobs=-1, random_state=42, verbose=-1)
    if model_name == "lightgbm_tweedie":
        return _lgbm_tweedie(**params)
    if model_name == "xgboost":
        return XGBRegressor(**params, n_jobs=-1, random_state=42, tree_method="hist")
    return MODEL_REGISTRY[model_name]()


# ===========================================================================
# Ensemble estimators
#
# These are real sklearn-compatible estimators rather than a dict of weights
# held inside the trainer. The reason is operational: the champion is very
# often an ensemble, and a champion that cannot be logged as a model artefact
# cannot be registered, promoted, or loaded by batch inference. Wrapping the
# combination in an object with `.predict()` makes an ensemble exactly as
# servable as a single LightGBM.
# ===========================================================================

from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402


class _BaseEnsemble(BaseEstimator, RegressorMixin):
    """Shared plumbing: hold fitted base models, stack their predictions."""

    def __init__(self, base_models: dict[str, Any] | None = None) -> None:
        self.base_models = base_models or {}

    @property
    def base_names(self) -> list[str]:
        return list(self.base_models)

    def _base_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Column-stack base model predictions in a stable, named order."""
        return np.column_stack(
            [np.clip(self.base_models[n].predict(X), 0.0, None) for n in self.base_names]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _BaseEnsemble:
        # Base models arrive already fitted; the meta-parameters are set by
        # the trainer from out-of-fold predictions. Present only to satisfy
        # the sklearn estimator contract.
        return self


class BlendEnsemble(_BaseEnsemble):
    """Convex combination of base model predictions with fixed weights."""

    def __init__(
        self,
        base_models: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(base_models)
        self.weights = weights or {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        w = np.array([self.weights[n] for n in self.base_names], dtype=float)
        return np.clip(self._base_matrix(X) @ w, 0.0, None)


class StackEnsemble(_BaseEnsemble):
    """Ridge meta-learner over base model predictions."""

    def __init__(
        self,
        base_models: dict[str, Any] | None = None,
        meta_model: Any = None,
    ) -> None:
        super().__init__(base_models)
        self.meta_model = meta_model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.meta_model.predict(self._base_matrix(X)), 0.0, None)


# ===========================================================================
# Configuration and results
# ===========================================================================

@dataclass
class TrainingConfig:
    """Configuration for one training run."""

    target_col: str = "sales"
    date_col: str = "date"
    series_cols: tuple[str, ...] = ("item_id", "store_id")

    forecast_horizon: int = 28

    # Nested CV. Both loops are TimeSeriesSplit.
    outer_splits: int = 3
    inner_splits: int = 3
    tune_trials: int = 30
    tune_models: tuple[str, ...] = ("lightgbm", "lightgbm_tweedie")

    models: tuple[str, ...] | None = None      # None -> whole registry
    feature_cols: list[str] | None = None

    # Ensembling
    build_stack: bool = True
    build_blend: bool = True

    # Recency weighting. Older M5 history is less representative of the
    # forecast period (assortment turnover, price drift), so samples are
    # down-weighted by exp(-lambda * age_in_days / 365). 0.0 disables it.
    recency_half_life_days: float = 0.0

    experiment_name: str = "m5_demand_forecasting"
    register_champion: bool = True

    def resolved_models(self) -> list[str]:
        names = list(self.models) if self.models else list(MODEL_REGISTRY)
        unknown = [n for n in names if n not in MODEL_REGISTRY]
        if unknown:
            raise KeyError(f"unknown models {unknown}; available: {sorted(MODEL_REGISTRY)}")
        return names


@dataclass
class ModelResult:
    """Outer-CV outcome for one model."""

    model_name: str
    cv_wape: float
    cv_rmsse: float
    cv_mae: float
    cv_bias: float
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    tuned: bool = False
    train_seconds: float = 0.0
    run_id: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "cv_wape": self.cv_wape,
            "cv_rmsse": self.cv_rmsse,
            "cv_mae": self.cv_mae,
            "cv_bias": self.cv_bias,
            "tuned": self.tuned,
            "train_seconds": round(self.train_seconds, 1),
            "run_id": self.run_id,
        }


# ===========================================================================
# Trainer
# ===========================================================================

class DemandTrainer:
    """
    Trains the model family, ensembles, and gates promotion to Production.

    Usage:
        trainer = DemandTrainer(TrainingConfig())
        leaderboard = trainer.fit(df_features)
        decision    = trainer.promote_champion()
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self._log = get_logger("DemandTrainer")

        self.results_: dict[str, ModelResult] = {}
        self.leaderboard_: pd.DataFrame = pd.DataFrame()
        self.champion_: str | None = None
        self.fitted_models_: dict[str, Any] = {}
        self.feature_names_: list[str] = []
        self._rmsse_scale: float = float("nan")
        self._tracker: ExperimentTracker | None = None

        self._mlflow_ready = self._init_mlflow() if _HAS_MLFLOW else False

    def _init_mlflow(self) -> bool:
        """
        Configure the shared tracking adapter.

        Tracking is used by every tier, so the fallback and reachability policy
        lives in ``src.mlops.tracking`` rather than being reimplemented in each
        trainer.  A failure to configure MLflow remains non-fatal.
        """
        self._tracker = ExperimentTracker(self.config.experiment_name)
        return self._tracker.available

    # -- data preparation ---------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Build (X, y, sample_weight) with rows in chronological order.

        Sorting by date first is what makes TimeSeriesSplit meaningful: the
        splitter cuts on row position, so position must encode time.
        """
        cfg = self.config
        if cfg.target_col not in df.columns:
            raise KeyError(f"target column {cfg.target_col!r} not in frame")

        if cfg.date_col in df.columns:
            df = df.copy()
            df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
            df = df.sort_values(cfg.date_col, kind="mergesort").reset_index(drop=True)

        if cfg.feature_cols:
            feature_cols = [c for c in cfg.feature_cols if c in df.columns]
        else:
            # Exclude identifiers, the target, and anything non-numeric. Tree
            # models could take encoded categoricals, but the feature pipeline
            # already emits encoded versions; passing the raw IDs as well
            # would let the model memorise series identity.
            exclude = {
                cfg.target_col, cfg.date_col, "id", "d", "wm_yr_wk",
                "item_id", "store_id", "dept_id", "cat_id", "state_id",
                "feature_version", "event_name_1", "event_name_2",
                "event_type_1", "event_type_2",
            }
            feature_cols = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]

        if not feature_cols:
            raise ValueError("no usable numeric feature columns found")

        keep = df[cfg.target_col].notna()
        df = df.loc[keep]

        X = df[feature_cols].astype(float).fillna(0.0)
        y = df[cfg.target_col].astype(float).clip(lower=0.0)

        # Recency weights, if enabled.
        if cfg.recency_half_life_days > 0 and cfg.date_col in df.columns:
            age_days = (df[cfg.date_col].max() - df[cfg.date_col]).dt.days.to_numpy()
            # w = 0.5 ** (age / half_life): weight halves every half_life days.
            weights = np.power(0.5, age_days / cfg.recency_half_life_days)
        else:
            weights = np.ones(len(X), dtype=float)

        self.feature_names_ = feature_cols

        # RMSSE scale from the training target as a whole. A per-series scale
        # is used in the backtest; here a single pooled scale keeps the CV
        # metric comparable across folds.
        self._rmsse_scale = naive_scale(y.to_numpy(), seasonality=1)

        self._log.info(
            "data_prepared",
            rows=len(X),
            features=len(feature_cols),
            zero_pct=round(100.0 * float((y == 0).mean()), 1),
            weighted=bool(cfg.recency_half_life_days > 0),
        )
        return X, y, weights

    # -- fitting helpers ----------------------------------------------------

    @staticmethod
    def _fit(model: Any, X: pd.DataFrame, y: pd.Series, w: np.ndarray | None) -> Any:
        """Fit, passing sample weights only when they are non-uniform."""
        if w is not None and not np.allclose(w, w[0]):
            try:
                model.fit(X, y, sample_weight=w)
                return model
            except TypeError:
                # Not every estimator in the registry accepts sample_weight.
                pass
        model.fit(X, y)
        return model

    def _score(self, model: Any, X: pd.DataFrame, y: pd.Series, y_in: np.ndarray) -> dict[str, float]:
        preds = np.clip(model.predict(X), 0.0, None)
        return compute_metrics(y.to_numpy(), preds, y_insample=y_in)

    # -- inner loop: hyperparameter search ----------------------------------

    def _tune_inner(
        self,
        model_name: str,
        X_outer_train: pd.DataFrame,
        y_outer_train: pd.Series,
        w_outer_train: np.ndarray,
    ) -> dict[str, Any]:
        """
        Search hyperparameters using ONLY the outer-training rows.

        The inner TimeSeriesSplit is built over `X_outer_train`, so the outer
        test fold is not merely excluded from fitting -- it is not present in
        the frame the sampler can see at all. That containment is what makes
        the outer score unbiased.
        """
        if not _HAS_OPTUNA or model_name not in SEARCH_SPACES:
            return {}

        inner_cv = TimeSeriesSplit(n_splits=self.config.inner_splits)
        space_fn = SEARCH_SPACES[model_name]

        def objective(trial: optuna.Trial) -> float:
            params = space_fn(trial)
            fold_scores: list[float] = []

            for tr_idx, va_idx in inner_cv.split(X_outer_train):
                model = _rebuild(model_name, params)
                self._fit(
                    model,
                    X_outer_train.iloc[tr_idx],
                    y_outer_train.iloc[tr_idx],
                    w_outer_train[tr_idx],
                )
                preds = np.clip(model.predict(X_outer_train.iloc[va_idx]), 0.0, None)
                score = compute_metrics(
                    y_outer_train.iloc[va_idx].to_numpy(), preds
                )["wape"]
                # A NaN WAPE means the inner validation window had no sales.
                # Treat it as uninformative rather than as a perfect score.
                fold_scores.append(score if np.isfinite(score) else 1e6)

            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(objective, n_trials=self.config.tune_trials, show_progress_bar=False)

        self._log.info(
            "inner_tuning_complete",
            model=model_name,
            best_inner_wape=round(study.best_value, 3),
            n_trials=len(study.trials),
        )
        return dict(study.best_params)

    # -- outer loop: unbiased evaluation ------------------------------------

    def _nested_cv(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        w: np.ndarray,
    ) -> ModelResult:
        """
        Run the full nested cross-validation for one model.

        Returns outer-fold-averaged metrics. These are the numbers reported in
        the leaderboard and written to MLflow.
        """
        cfg = self.config
        outer_cv = TimeSeriesSplit(n_splits=cfg.outer_splits)
        will_tune = model_name in cfg.tune_models and model_name in SEARCH_SPACES

        fold_metrics: list[dict[str, float]] = []
        fold_params: list[dict[str, Any]] = []
        t0 = time.time()

        for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            w_tr = w[tr_idx]

            # --- inner: select hyperparameters on outer-train only ---
            params = self._tune_inner(model_name, X_tr, y_tr, w_tr) if will_tune else {}
            fold_params.append(params)

            # --- refit on all outer-train rows with the winning params ---
            model = _rebuild(model_name, params) if params else MODEL_REGISTRY[model_name]()
            self._fit(model, X_tr, y_tr, w_tr)

            # --- score once on the untouched outer-test fold ---
            metrics = self._score(model, X_te, y_te, y_in=y_tr.to_numpy())
            metrics["fold"] = float(fold)
            fold_metrics.append(metrics)

            self._log.info(
                "outer_fold_scored",
                model=model_name,
                fold=fold,
                wape=round(metrics["wape"], 3),
                rmsse=round(metrics["rmsse"], 3) if np.isfinite(metrics["rmsse"]) else None,
                n_train=len(tr_idx),
                n_test=len(te_idx),
            )

        elapsed = time.time() - t0

        def _avg(key: str) -> float:
            vals = [m[key] for m in fold_metrics if np.isfinite(m.get(key, np.nan))]
            return float(np.mean(vals)) if vals else float("nan")

        # The parameters carried forward are those from the LAST outer fold,
        # i.e. selected on the largest and most recent training window. The
        # reported metrics remain the outer-fold average -- the params are for
        # the final refit, not for the score.
        best_params = fold_params[-1] if fold_params else {}

        return ModelResult(
            model_name=model_name,
            cv_wape=_avg("wape"),
            cv_rmsse=_avg("rmsse"),
            cv_mae=_avg("mae"),
            cv_bias=_avg("bias"),
            fold_metrics=fold_metrics,
            best_params=best_params,
            tuned=will_tune,
            train_seconds=elapsed,
        )

    # -- ensembles ----------------------------------------------------------

    def _build_ensembles(
        self, X: pd.DataFrame, y: pd.Series, w: np.ndarray
    ) -> dict[str, ModelResult]:
        """
        Stacked and weighted-blend ensembles over the base models.

        Both are built on out-of-fold predictions generated by a TimeSeriesSplit,
        never on in-sample predictions. Stacking on in-sample predictions is a
        well-known leak: the meta-learner sees base models that have already
        memorised the rows it is being trained on, and assigns them weights
        that collapse on unseen data.
        """
        from scipy.optimize import minimize

        base_names = [n for n in self.results_ if not n.startswith("ensemble")]
        if len(base_names) < 2:
            return {}

        cv = TimeSeriesSplit(n_splits=self.config.outer_splits)
        splits = list(cv.split(X))

        # Only rows appearing in some validation fold get OOF predictions.
        # TimeSeriesSplit never validates on the first block, so those rows
        # are excluded rather than filled in.
        oof = pd.DataFrame(index=X.index, columns=base_names, dtype=float)

        for name in base_names:
            params = self.results_[name].best_params
            for tr_idx, va_idx in splits:
                model = _rebuild(name, params) if params else MODEL_REGISTRY[name]()
                self._fit(model, X.iloc[tr_idx], y.iloc[tr_idx], w[tr_idx])
                oof.iloc[va_idx, oof.columns.get_loc(name)] = np.clip(
                    model.predict(X.iloc[va_idx]), 0.0, None
                )

        mask = oof.notna().all(axis=1)
        if mask.sum() < 100:
            self._log.warning("ensemble_skipped", reason="insufficient OOF rows")
            return {}

        oof_valid = oof.loc[mask]
        y_valid = y.loc[mask]

        # Score the ensembles on the final third of the OOF rows, held out
        # from the meta-fit, so ensemble metrics are comparable to base
        # models' outer-CV metrics rather than being in-sample.
        cut = int(len(oof_valid) * 2 / 3)
        Xm_tr, Xm_te = oof_valid.iloc[:cut], oof_valid.iloc[cut:]
        ym_tr, ym_te = y_valid.iloc[:cut], y_valid.iloc[cut:]

        out: dict[str, ModelResult] = {}

        # Fitted base models, in the same column order as the OOF matrix, so
        # the ensemble objects index them consistently at inference time.
        fitted_bases = {n: self.fitted_models_[n] for n in base_names}

        # --- stacked: Ridge meta-learner over base predictions ---
        if self.config.build_stack:
            # positive=True keeps every base contribution non-negative, which
            # makes the learned coefficients readable as "how much this model
            # is trusted" and stops the meta-learner from fitting noise by
            # subtracting one base model from another.
            meta = Ridge(alpha=1.0, positive=True, random_state=42)
            meta.fit(Xm_tr, ym_tr)
            preds = np.clip(meta.predict(Xm_te), 0.0, None)
            m = compute_metrics(ym_te.to_numpy(), preds, y_insample=ym_tr.to_numpy())
            self.fitted_models_["ensemble_stack"] = StackEnsemble(
                base_models=fitted_bases, meta_model=meta
            )
            out["ensemble_stack"] = ModelResult(
                model_name="ensemble_stack",
                cv_wape=m["wape"], cv_rmsse=m["rmsse"],
                cv_mae=m["mae"], cv_bias=m["bias"],
                best_params={"base_models": base_names,
                             "coef": dict(zip(base_names, np.round(meta.coef_, 4).tolist()))},
            )

        # --- weighted blend: SLSQP over the simplex ---
        if self.config.build_blend:
            A = Xm_tr.to_numpy()
            target = ym_tr.to_numpy()

            def loss(weights: np.ndarray) -> float:
                blended = A @ weights
                denom = np.abs(target).sum()
                return float(np.abs(target - blended).sum() / denom) if denom else 1e9

            n = len(base_names)
            res = minimize(
                loss,
                x0=np.full(n, 1.0 / n),
                method="SLSQP",
                # Simplex constraints: weights are non-negative and sum to 1.
                # Non-negativity keeps the blend interpretable (no model is
                # subtracted) and acts as regularisation against a single
                # noisy base model dominating.
                bounds=[(0.0, 1.0)] * n,
                constraints=[{"type": "eq", "fun": lambda ws: ws.sum() - 1.0}],
                options={"maxiter": 300, "ftol": 1e-9},
            )
            weights = res.x / res.x.sum()
            preds = np.clip(Xm_te.to_numpy() @ weights, 0.0, None)
            m = compute_metrics(ym_te.to_numpy(), preds, y_insample=ym_tr.to_numpy())
            self.fitted_models_["ensemble_blend"] = BlendEnsemble(
                base_models=fitted_bases,
                weights={n: float(w) for n, w in zip(base_names, weights)},
            )
            out["ensemble_blend"] = ModelResult(
                model_name="ensemble_blend",
                cv_wape=m["wape"], cv_rmsse=m["rmsse"],
                cv_mae=m["mae"], cv_bias=m["bias"],
                best_params={"weights": {k: round(float(v), 4)
                                         for k, v in zip(base_names, weights)}},
            )
            self._log.info(
                "blend_weights",
                **{k: round(float(v), 3) for k, v in zip(base_names, weights)},
            )

        return out

    # -- public API ---------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train every configured model under nested CV and return the leaderboard.

        Returns:
            DataFrame sorted by CV WAPE ascending, one row per model.
        """
        X, y, w = self._prepare(df)
        self.results_.clear()
        self.fitted_models_.clear()

        for name in self.config.resolved_models():
            self._log.info("training_model", model=name)
            result = self._nested_cv(name, X, y, w)

            # Final refit on ALL data with the selected hyperparameters. This
            # is the artefact that gets served; its performance estimate comes
            # from the outer CV above, not from this fit.
            final = _rebuild(name, result.best_params) if result.best_params else MODEL_REGISTRY[name]()
            self._fit(final, X, y, w)
            self.fitted_models_[name] = final

            result.run_id = self._log_to_mlflow(name, result, final, X)
            self.results_[name] = result

            self._log.info(
                "model_complete",
                model=name,
                cv_wape=round(result.cv_wape, 3),
                cv_rmsse=round(result.cv_rmsse, 3) if np.isfinite(result.cv_rmsse) else None,
            )

        # Ensembles are built after all base models exist. The fitted ensemble
        # object is passed through so it is logged as a real model artefact --
        # without that the promotion gate cannot register an ensemble champion.
        for name, result in self._build_ensembles(X, y, w).items():
            result.run_id = self._log_to_mlflow(name, result, self.fitted_models_.get(name), X)
            self.results_[name] = result

        self.leaderboard_ = (
            pd.DataFrame([r.to_row() for r in self.results_.values()])
            .sort_values("cv_wape", na_position="last")
            .reset_index(drop=True)
        )
        self.champion_ = self._select_champion()

        self._log.info(
            "training_complete",
            champion=self.champion_,
            champion_wape=round(self.results_[self.champion_].cv_wape, 3),
            n_models=len(self.results_),
        )
        return self.leaderboard_

    def _select_champion(self) -> str:
        """
        Pick the champion on CV WAPE, ignoring models that failed to score.

        WAPE rather than RMSSE for selection because the pooled RMSSE scale
        used during CV is a coarser signal than the volume-weighted WAPE;
        RMSSE comes into its own in the per-series backtest, where each series
        carries its own scale.
        """
        valid = {
            k: v for k, v in self.results_.items()
            if np.isfinite(v.cv_wape) and v.cv_wape > 0
        }
        if not valid:
            raise RuntimeError("no model produced a finite CV WAPE")
        return min(valid, key=lambda k: valid[k].cv_wape)

    # -- MLflow -------------------------------------------------------------

    def _log_to_mlflow(
        self, name: str, result: ModelResult, model: Any, X: pd.DataFrame
    ) -> str | None:
        """Log params, outer-fold metrics and the model artefact. Never fatal."""
        if not (_HAS_MLFLOW and self._mlflow_ready and self._tracker is not None):
            return None

        try:
            with self._tracker.run(
                run_name=name,
                tags={
                    "model_name": name,
                    "tuned": str(result.tuned),
                    "cv_scheme": f"nested({self.config.outer_splits}x{self.config.inner_splits})",
                    "dataset": "M5",
                    "n_features": str(len(self.feature_names_)),
                },
            ) as run:
                if run is None:
                    return None
                if result.best_params:
                    self._tracker.log_params(result.best_params)
                self._tracker.log_metrics(
                    {
                        "cv_wape": result.cv_wape,
                        "cv_mae": result.cv_mae,
                        "cv_bias": result.cv_bias,
                        **({"cv_rmsse": result.cv_rmsse} if np.isfinite(result.cv_rmsse) else {}),
                        "train_seconds": result.train_seconds,
                    }
                )
                # Per-fold metrics as steps, so the MLflow UI shows fold-to-fold
                # variance -- a champion that wins on average but loses badly on
                # one fold is worth seeing before promotion.
                for m in result.fold_metrics:
                    step = int(m.get("fold", 0))
                    self._tracker.log_metrics({"fold_wape": m["wape"]}, step=step)
                    if np.isfinite(m.get("rmsse", np.nan)):
                        self._tracker.log_metrics({"fold_rmsse": m["rmsse"]}, step=step)

                if model is not None:
                    self._log_model_artifact(name, model, X)

                return run.info.run_id
        except Exception as exc:                            # pragma: no cover
            self._log.warning("mlflow_logging_failed", model=name, error=str(exc))
            return None

    def _log_model_artifact(self, name: str, model: Any, X: pd.DataFrame) -> None:
        """Log the fitted model plus a gain-importance CSV."""
        from mlflow.models.signature import infer_signature

        sample = X.head(5)
        signature = infer_signature(sample, np.zeros(len(sample)))

        # Native lightgbm / xgboost flavours for single models: they reload
        # without this training module on the path.
        #
        # Everything else goes through the sklearn flavour pinned to
        # CLOUDPICKLE. MLflow 3.x defaults that flavour to `skops`, which
        # refuses to serialise types it does not recognise -- including the
        # ensemble wrappers here and the LightGBM boosters nested inside them.
        # Cloudpickle handles them, at the cost of requiring the same library
        # versions at load time (recorded in the run's environment spec).
        if name.startswith("lightgbm"):
            mlflow.lightgbm.log_model(model, name="model", signature=signature)
        elif name == "xgboost":
            mlflow.xgboost.log_model(model, name="model", signature=signature)
        else:
            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame(
                {"feature": self.feature_names_, "gain_importance": model.feature_importances_}
            ).sort_values("gain_importance", ascending=False)
            # tempfile, not a hardcoded /tmp path -- the previous version wrote
            # to "/tmp/..." which does not exist on Windows and silently
            # swallowed the failure.
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / f"gain_importance_{name}.csv"
                imp.to_csv(path, index=False)
                mlflow.log_artifact(str(path), "feature_importance")

    # -- promotion registry --------------------------------------------------

    def promote_champion(self, force: bool = False) -> dict[str, Any]:
        """
        Delegate promotion to the reusable registry gate.

        The trainer remains responsible for choosing a champion from nested CV;
        registering and staging its artifact belongs to ``src.mlops.registry``.
        Keeping those responsibilities separate lets Tier 1, Tier 3, and Tier
        4 experiments use the same auditable promotion policy.
        """
        if not (_HAS_MLFLOW and self._mlflow_ready):
            return {"promoted": False, "reason": "mlflow_unavailable"}
        if self.champion_ is None:
            return {"promoted": False, "reason": "no_champion_selected"}

        champion = self.results_[self.champion_]
        if champion.run_id is None:
            return {"promoted": False, "reason": "champion_has_no_mlflow_run"}

        candidate = PromotionCandidate(
            run_id=champion.run_id,
            algorithm=self.champion_,
            cv_wape=champion.cv_wape,
            cv_rmsse=champion.cv_rmsse if np.isfinite(champion.cv_rmsse) else None,
            cv_scheme=f"nested({self.config.outer_splits}x{self.config.inner_splits})",
        )
        decision = PromotionGate().promote(candidate, force=force)
        return decision.to_dict()

    # -- persistence --------------------------------------------------------

    def save_leaderboard(self, path: str | Path) -> Path:
        """Write the leaderboard to CSV, plus a JSON sidecar of fold detail."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.leaderboard_.to_csv(path, index=False)

        detail = {
            name: {
                "cv_wape": r.cv_wape,
                "cv_rmsse": r.cv_rmsse,
                "fold_metrics": r.fold_metrics,
                "best_params": {k: v for k, v in r.best_params.items()},
                "tuned": r.tuned,
            }
            for name, r in self.results_.items()
        }
        sidecar = path.with_suffix(".folds.json")
        sidecar.write_text(json.dumps(detail, indent=2, default=str), encoding="utf-8")

        self._log.info("leaderboard_saved", path=str(path), detail=str(sidecar))
        return path
