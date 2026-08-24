"""Distribution checks and conditional count models for sparse M5 demand.

Distribution choice is measured, not assumed.  The univariate fitting helper
compares Poisson, negative-binomial, and zero-inflated variants by likelihood
on an observed series.  Tweedie is reported with a mean deviance diagnostic,
not placed in the AIC ranking, because its power parameter is fixed rather than
jointly maximum-likelihood fitted in this lightweight screen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import nbinom, poisson
from sklearn.metrics import mean_tweedie_deviance


@dataclass(frozen=True)
class DistributionComparison:
    """Likelihood evidence for one non-negative integer demand history."""

    table: pd.DataFrame
    best_aic_distribution: str | None
    n_observations: int
    zero_fraction: float
    mean: float
    variance: float


def _validate_counts(y: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(y, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot fit a distribution to an empty series")
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.rint(values)):
        raise ValueError("distribution fitting expects finite non-negative integer unit counts")
    return values.astype(int)


def _aic_bic(log_likelihood: float, n_parameters: int, n: int) -> tuple[float, float]:
    return -2.0 * log_likelihood + 2.0 * n_parameters, -2.0 * log_likelihood + n_parameters * np.log(n)


def _fit_zip(y: np.ndarray) -> tuple[float, float, float]:
    """Maximum-likelihood zero-inflated Poisson fit: (pi, lambda, loglik)."""
    positive_mean = float(y[y > 0].mean()) if (y > 0).any() else 1e-6
    initial = np.array([0.0, np.log(max(positive_mean, 1e-6))])

    def negative_log_likelihood(theta: np.ndarray) -> float:
        pi, rate = expit(theta[0]), np.exp(theta[1])
        zero_prob = pi + (1.0 - pi) * np.exp(-rate)
        logp = np.where(y == 0, np.log(np.clip(zero_prob, 1e-300, None)), np.log1p(-pi) + poisson.logpmf(y, rate))
        return float(-np.sum(logp))

    result = minimize(negative_log_likelihood, initial, method="L-BFGS-B")
    pi, rate = expit(result.x[0]), np.exp(result.x[1])
    return float(pi), float(rate), float(-result.fun)


def _fit_nb(y: np.ndarray) -> tuple[float, float, float]:
    """Maximum-likelihood NB2 fit: (mean, dispersion, loglik)."""
    mean = max(float(np.mean(y)), 1e-6)
    variance = max(float(np.var(y, ddof=1)) if y.size > 1 else mean, mean + 1e-6)
    initial_alpha = max((variance - mean) / (mean**2), 1e-6)

    def negative_log_likelihood(theta: np.ndarray) -> float:
        mu, alpha = np.exp(theta[0]), np.exp(theta[1])
        size = 1.0 / alpha
        probability = size / (size + mu)
        return float(-np.sum(nbinom.logpmf(y, size, probability)))

    result = minimize(
        negative_log_likelihood,
        np.array([np.log(mean), np.log(initial_alpha)]),
        method="L-BFGS-B",
    )
    mu, alpha = np.exp(result.x[0]), np.exp(result.x[1])
    return float(mu), float(alpha), float(-result.fun)


def _fit_zinb(y: np.ndarray) -> tuple[float, float, float, float]:
    """Maximum-likelihood ZINB2 fit: (pi, mean, dispersion, loglik)."""
    mean = max(float(np.mean(y)), 1e-6)
    variance = max(float(np.var(y, ddof=1)) if y.size > 1 else mean, mean + 1e-6)
    initial_dispersion = max((variance - mean) / (mean**2), 1e-6)
    initial = np.array([0.0, np.log(mean), np.log(initial_dispersion)])

    def negative_log_likelihood(theta: np.ndarray) -> float:
        pi, mu, alpha = expit(theta[0]), np.exp(theta[1]), np.exp(theta[2])
        size = 1.0 / alpha
        probability = size / (size + mu)
        zero_prob = pi + (1.0 - pi) * np.exp(nbinom.logpmf(0, size, probability))
        logp = np.where(
            y == 0,
            np.log(np.clip(zero_prob, 1e-300, None)),
            np.log1p(-pi) + nbinom.logpmf(y, size, probability),
        )
        return float(-np.sum(logp))

    result = minimize(negative_log_likelihood, initial, method="L-BFGS-B")
    pi, mu, alpha = expit(result.x[0]), np.exp(result.x[1]), np.exp(result.x[2])
    return float(pi), float(mu), float(alpha), float(-result.fun)


def fit_count_distributions(y: np.ndarray | pd.Series, tweedie_power: float = 1.5) -> DistributionComparison:
    """Fit candidate count distributions and select only among comparable AICs.

    Negative-binomial uses the NB2 variance law ``Var(Y)=mu+alpha*mu^2``.  A
    zero-inflated model earns its extra parameter only if it improves AIC; a
    high zero share alone is not evidence for a separate zero-generating
    process.
    """
    values = _validate_counts(y)
    n = len(values)
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1)) if n > 1 else 0.0
    zero_fraction = float(np.mean(values == 0))
    if mean == 0.0:
        table = pd.DataFrame(
            [{"distribution": "degenerate_zero", "log_likelihood": 0.0, "aic": 0.0, "bic": 0.0, "selection_eligible": True}]
        )
        return DistributionComparison(table, "degenerate_zero", n, zero_fraction, mean, variance)

    rows: list[dict[str, Any]] = []
    poisson_ll = float(np.sum(poisson.logpmf(values, mean)))
    aic, bic = _aic_bic(poisson_ll, 1, n)
    rows.append({"distribution": "poisson", "log_likelihood": poisson_ll, "aic": aic, "bic": bic, "selection_eligible": True, "mean": mean})

    nb_mean, nb_alpha, nb_ll = _fit_nb(values)
    aic, bic = _aic_bic(nb_ll, 2, n)
    rows.append({"distribution": "negative_binomial", "log_likelihood": nb_ll, "aic": aic, "bic": bic, "selection_eligible": True, "mean": nb_mean, "dispersion": nb_alpha})

    zip_pi, zip_rate, zip_ll = _fit_zip(values)
    aic, bic = _aic_bic(zip_ll, 2, n)
    rows.append({"distribution": "zero_inflated_poisson", "log_likelihood": zip_ll, "aic": aic, "bic": bic, "selection_eligible": True, "mean": zip_rate, "zero_inflation": zip_pi})

    zinb_pi, zinb_mean, zinb_alpha, zinb_ll = _fit_zinb(values)
    aic, bic = _aic_bic(zinb_ll, 3, n)
    rows.append({"distribution": "zero_inflated_negative_binomial", "log_likelihood": zinb_ll, "aic": aic, "bic": bic, "selection_eligible": True, "mean": zinb_mean, "dispersion": zinb_alpha, "zero_inflation": zinb_pi})

    if not 1.0 < tweedie_power < 2.0:
        raise ValueError("Tweedie power must lie in (1, 2) for compound Poisson-Gamma demand")
    tweedie_deviance = mean_tweedie_deviance(values, np.full(n, mean), power=tweedie_power)
    rows.append({"distribution": "tweedie", "mean_tweedie_deviance": tweedie_deviance, "power": tweedie_power, "selection_eligible": False})

    table = pd.DataFrame(rows).sort_values("aic", na_position="last").reset_index(drop=True)
    eligible = table[table["selection_eligible"]]
    best = str(eligible.iloc[0]["distribution"]) if not eligible.empty else None
    return DistributionComparison(table, best, n, zero_fraction, mean, variance)


class NGBoostPoissonForecaster:
    """Conditional Poisson distributional baseline; compare it, do not presume it wins."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.03, random_state: int = 42) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.model_: Any | None = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> NGBoostPoissonForecaster:
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import Poisson
        except ImportError as exc:  # pragma: no cover - declared project dependency
            raise ImportError("install ngboost to use NGBoostPoissonForecaster") from exc
        target = _validate_counts(y)
        features = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.feature_names_ = features.columns.tolist()
        self.model_ = NGBRegressor(
            Dist=Poisson,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=False,
        ).fit(features, target)
        return self

    def predict_quantiles(self, X: pd.DataFrame, quantiles: list[float]) -> dict[float, np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("call fit before predict_quantiles")
        levels = [float(q) for q in quantiles]
        if any(not 0.0 < q < 1.0 for q in levels):
            raise ValueError("quantiles must lie in (0, 1)")
        features = X.reindex(columns=self.feature_names_).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        distribution = self.model_.pred_dist(features)
        raw = {q: np.clip(np.asarray(distribution.ppf(q), dtype=float), 0.0, None) for q in levels}
        matrix = np.sort(np.column_stack([raw[q] for q in sorted(raw)]), axis=1)
        return {q: matrix[:, i] for i, q in enumerate(sorted(raw))}
