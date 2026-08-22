"""
Statistical significance testing for forecast comparison.

The problem
-----------
A leaderboard showing model A at 81.9% WAPE and model B at 82.4% invites the
conclusion that A is better. Across 30,490 heterogeneous series with fat-tailed
per-series errors, a 0.5-point gap is frequently noise. Declaring a winner from
the raw difference is the single most common overclaim in forecasting write-ups.

This module supplies the tests that turn "A scored lower" into "A is better,
and here is the evidence" -- or, just as valuably, "these models are
statistically indistinguishable".

Which test for which question
-----------------------------
  Diebold-Mariano  -- two models, one series: is the loss differential
                      significantly non-zero, accounting for the fact that
                      forecast errors are autocorrelated across the horizon?
  Friedman         -- many models, many series: is ANY model different? An
                      omnibus test that must reject before pairwise testing is
                      justified.
  Nemenyi          -- which specific pairs differ, given Friedman rejected.
                      Produces the critical difference that Fig 4 draws.
  Holm / BH        -- multiple-comparison correction. With 12 models there are
                      66 pairs; at alpha=0.05 roughly three will look
                      significant by chance alone.

The rank-based approach (Friedman/Nemenyi) is preferred over averaging raw
metrics because per-series WAPE is heavily skewed -- a handful of near-zero
series produce enormous percentage errors and dominate any mean. Ranks are
immune to that.

References
----------
Diebold & Mariano (1995), "Comparing predictive accuracy", JBES 13(3):253-263.
Harvey, Leybourne & Newbold (1997), "Testing the equality of prediction mean
    squared errors", IJF 13(2):281-291.  [the small-sample correction]
Demsar (2006), "Statistical comparisons of classifiers over multiple data
    sets", JMLR 7:1-30.  [Friedman + Nemenyi + the CD diagram]
Koning et al. (2005), "The M3 competition: Statistical tests of the results",
    IJF 21(3):397-409.  [MCB plots, as used in the M5 paper]
Benjamini & Hochberg (1995), JRSS-B 57(1):289-300.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.config.logging import get_logger

log = get_logger(__name__)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------

@dataclass
class DMResult:
    """Outcome of one Diebold-Mariano test."""

    model_a: str
    model_b: str
    dm_statistic: float
    p_value: float
    mean_loss_diff: float
    horizon: int
    n_obs: int
    better: str          # model_a, model_b, or "indistinguishable"
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < self.alpha)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "dm_statistic": self.dm_statistic,
            "p_value": self.p_value,
            "mean_loss_diff": self.mean_loss_diff,
            "significant": self.significant,
            "better": self.better,
            "n_obs": self.n_obs,
        }


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    horizon: int = 1,
    loss: Literal["squared", "absolute"] = "absolute",
    model_a: str = "A",
    model_b: str = "B",
    alpha: float = 0.05,
    harvey_correction: bool = True,
) -> DMResult:
    """
    Test whether two models' forecast accuracy differs significantly.

    The statistic is built on the loss differential

        d_t = L(e_{a,t}) - L(e_{b,t})

    under H0: E[d_t] = 0. The subtlety is the variance of d-bar. Multi-step
    forecast errors are serially correlated -- a model that is wrong on
    Monday is usually wrong on Tuesday -- so the naive variance understates
    the true one and inflates the statistic. The standard fix uses a
    Newey-West-style long-run variance truncated at (horizon - 1) lags:

        DM = d_bar / sqrt( [gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k] / n )

    Harvey, Leybourne & Newbold (1997) further showed DM over-rejects in small
    samples and proposed the correction factor

        HLN = sqrt( [n + 1 - 2h + h(h-1)/n] / n )

    with the statistic then compared against t_{n-1} rather than N(0,1).
    Applied by default; M5-style evaluation windows (28 days) are exactly the
    small-sample regime it was designed for.

    Args:
        errors_a/errors_b: forecast errors (y_true - y_pred) for each model,
            on the SAME observations in the SAME order.
        horizon: forecast horizon; sets the autocovariance truncation lag.
        loss: "absolute" pairs naturally with WAPE/MAE reporting; "squared"
            with RMSE/RMSSE. Pick the one matching the headline metric,
            because DM tests the loss you give it, not the one you report.
    """
    errors_a = np.asarray(errors_a, dtype=float).ravel()
    errors_b = np.asarray(errors_b, dtype=float).ravel()

    if errors_a.shape != errors_b.shape:
        raise ValueError(
            f"error arrays must align: {errors_a.shape} vs {errors_b.shape}"
        )

    mask = np.isfinite(errors_a) & np.isfinite(errors_b)
    errors_a, errors_b = errors_a[mask], errors_b[mask]
    n = errors_a.size

    if n < 3:
        return DMResult(model_a, model_b, float("nan"), float("nan"),
                        float("nan"), horizon, n, "indistinguishable", alpha)

    if loss == "squared":
        d = errors_a ** 2 - errors_b ** 2
    else:
        d = np.abs(errors_a) - np.abs(errors_b)

    d_bar = float(d.mean())

    # Long-run variance with autocovariances up to lag h-1.
    d_centred = d - d_bar
    gamma0 = float(np.mean(d_centred ** 2))
    lrv = gamma0
    for k in range(1, min(horizon, n)):
        gamma_k = float(np.mean(d_centred[k:] * d_centred[:-k]))
        lrv += 2.0 * gamma_k

    # A negative long-run variance estimate is possible with raw truncation.
    # Fall back to gamma0, which is always non-negative, rather than emitting
    # a NaN that would silently drop the pair from the results table.
    if lrv <= _EPS:
        lrv = gamma0
    if lrv <= _EPS:
        return DMResult(model_a, model_b, float("nan"), float("nan"),
                        d_bar, horizon, n, "indistinguishable", alpha)

    dm = d_bar / np.sqrt(lrv / n)

    if harvey_correction and n > 1:
        h = horizon
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        if factor > 0:
            dm *= np.sqrt(factor)
        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1)))
    else:
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm))))

    if p_value < alpha:
        # d = loss(a) - loss(b); negative means a has the smaller loss.
        better = model_a if d_bar < 0 else model_b
    else:
        better = "indistinguishable"

    return DMResult(
        model_a=model_a, model_b=model_b,
        dm_statistic=float(dm), p_value=p_value,
        mean_loss_diff=d_bar, horizon=horizon, n_obs=n,
        better=better, alpha=alpha,
    )


def pairwise_dm(
    errors: Mapping[str, np.ndarray],
    horizon: int = 1,
    loss: Literal["squared", "absolute"] = "absolute",
    alpha: float = 0.05,
    correction: Literal["holm", "bh", "none"] = "holm",
) -> pd.DataFrame:
    """
    Diebold-Mariano across every model pair, with multiple-comparison control.

    With m models there are m(m-1)/2 tests. At alpha=0.05 and 12 models, ~3 of
    the 66 will appear significant purely by chance, so an uncorrected table
    manufactures findings. Holm is the default: it is uniformly more powerful
    than Bonferroni and controls the family-wise error rate without extra
    assumptions.
    """
    names = sorted(errors)
    results = [
        diebold_mariano(errors[a], errors[b], horizon=horizon, loss=loss,
                        model_a=a, model_b=b, alpha=alpha)
        for a, b in combinations(names, 2)
    ]

    df = pd.DataFrame([r.to_dict() for r in results])
    if df.empty:
        return df

    df["p_adjusted"] = adjust_pvalues(df["p_value"].to_numpy(), method=correction)
    df["significant_adjusted"] = df["p_adjusted"] < alpha
    df.loc[~df["significant_adjusted"], "better"] = "indistinguishable"
    df["correction"] = correction

    log.info(
        "pairwise_dm_complete",
        n_pairs=len(df),
        n_significant_raw=int(df["significant"].sum()),
        n_significant_adjusted=int(df["significant_adjusted"].sum()),
        correction=correction,
    )
    return df.sort_values("p_adjusted").reset_index(drop=True)


def adjust_pvalues(
    p: np.ndarray,
    method: Literal["holm", "bh", "none"] = "holm",
) -> np.ndarray:
    """
    Multiple-comparison correction.

    holm: step-down family-wise error control. Sort ascending, multiply the
        k-th smallest by (m - k), enforce monotonicity. Controls the
        probability of ANY false positive -- the right choice when a single
        wrong "significant" claim would be embarrassing.

    bh: Benjamini-Hochberg false discovery rate. Less conservative; controls
        the EXPECTED PROPORTION of false positives among rejections. Better
        for exploratory screening of many pairs.
    """
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0 or method == "none":
        return p

    order = np.argsort(p)
    sorted_p = p[order]
    adjusted = np.empty(m, dtype=float)

    if method == "holm":
        running = 0.0
        for i, value in enumerate(sorted_p):
            running = max(running, (m - i) * value)
            adjusted[i] = min(running, 1.0)
    elif method == "bh":
        running = 1.0
        for i in range(m - 1, -1, -1):
            running = min(running, sorted_p[i] * m / (i + 1))
            adjusted[i] = min(running, 1.0)
    else:
        raise ValueError(f"unknown correction method {method!r}")

    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return out


# ---------------------------------------------------------------------------
# Friedman + Nemenyi (the CD diagram, Fig 4)
# ---------------------------------------------------------------------------

@dataclass
class FriedmanResult:
    """Omnibus test across models and series."""

    statistic: float
    p_value: float
    n_models: int
    n_series: int
    average_ranks: pd.Series = field(default_factory=pd.Series)
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < self.alpha)

    def summary(self) -> str:
        verdict = (
            "models DIFFER -- post-hoc testing is justified"
            if self.significant
            else "no detectable difference -- do NOT run post-hoc tests"
        )
        return (
            f"Friedman chi2={self.statistic:.3f}, p={self.p_value:.3g} "
            f"({self.n_models} models x {self.n_series} series): {verdict}"
        )


def friedman_test(scores: pd.DataFrame, alpha: float = 0.05) -> FriedmanResult:
    """
    Omnibus test: is any model different from the others?

    Args:
        scores: rows = series (blocks), columns = models, values = a metric
            where LOWER IS BETTER (WAPE, RMSSE, ...). Rows with any NaN are
            dropped, because the test requires complete blocks -- a model
            missing on some series would otherwise be ranked against a
            different subset than its competitors.

    The Friedman statistic ranks models WITHIN each series and tests whether
    the average ranks depart from uniformity. Ranking within-series is what
    makes it robust: it never compares a WAPE from a high-volume staple
    against one from a near-dead item.

    This must reject before Nemenyi is meaningful. Running post-hoc tests after
    a non-significant omnibus is a well-known way to manufacture findings.
    """
    clean = scores.dropna(axis=0, how="any")
    n_series, n_models = clean.shape

    if n_models < 3:
        raise ValueError(f"Friedman needs >= 3 models, got {n_models}")
    if n_series < 2:
        raise ValueError(f"Friedman needs >= 2 series, got {n_series}")

    statistic, p_value = stats.friedmanchisquare(*[clean[c].to_numpy() for c in clean.columns])
    ranks = clean.rank(axis=1, ascending=True).mean(axis=0).sort_values()

    result = FriedmanResult(
        statistic=float(statistic), p_value=float(p_value),
        n_models=n_models, n_series=n_series,
        average_ranks=ranks, alpha=alpha,
    )
    log.info("friedman_test", statistic=round(float(statistic), 3),
             p_value=float(p_value), significant=result.significant)
    return result


# Studentised range statistic q_alpha at alpha=0.05, divided by sqrt(2), for
# k = 2..20 models (Demsar 2006, Table 5). Tabulated because scipy has no
# direct inverse for the studentised range with infinite degrees of freedom.
_NEMENYI_Q05: dict[int, float] = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
    15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}
_NEMENYI_Q10: dict[int, float] = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780,
    9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120,
    15: 3.159, 16: 3.196, 17: 3.230, 18: 3.261, 19: 3.291, 20: 3.319,
}


def nemenyi_critical_difference(
    n_models: int,
    n_series: int,
    alpha: float = 0.05,
) -> float:
    """
    Critical difference for the Nemenyi post-hoc test.

        CD = q_alpha * sqrt( k(k+1) / (6N) )

    where k = number of models, N = number of series, and q_alpha is the
    studentised range statistic divided by sqrt(2).

    Two models whose average ranks differ by less than CD are statistically
    indistinguishable -- that is exactly the horizontal connecting bar drawn
    in a critical-difference diagram (Fig 4).

    Note CD shrinks as sqrt(1/N): with 30,490 series almost any rank gap
    becomes "significant", which is a real limitation of the test at this
    scale. Report the effect size alongside, because statistical significance
    on 30k series is not the same as a difference worth acting on.
    """
    table = _NEMENYI_Q05 if alpha == 0.05 else _NEMENYI_Q10
    if n_models not in table:
        # Normal approximation beyond the tabulated range.
        q = stats.norm.ppf(1.0 - alpha / (n_models * (n_models - 1)))
    else:
        q = table[n_models]
    return float(q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_series)))


def nemenyi_test(scores: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """
    Post-hoc pairwise comparison following a significant Friedman test.

    Returns average ranks, the critical difference, a pairwise significance
    matrix, and the "cliques" of mutually indistinguishable models that a CD
    diagram connects with bars.
    """
    clean = scores.dropna(axis=0, how="any")
    n_series, n_models = clean.shape

    ranks = clean.rank(axis=1, ascending=True)
    avg_ranks = ranks.mean(axis=0).sort_values()
    cd = nemenyi_critical_difference(n_models, n_series, alpha=alpha)

    names = list(avg_ranks.index)
    diff = pd.DataFrame(index=names, columns=names, dtype=float)
    sig = pd.DataFrame(index=names, columns=names, dtype=bool)
    for a in names:
        for b in names:
            delta = abs(avg_ranks[a] - avg_ranks[b])
            diff.loc[a, b] = delta
            sig.loc[a, b] = bool(delta > cd)

    # Cliques: maximal runs of adjacent models (by rank) spanning less than CD.
    cliques: list[list[str]] = []
    for i, a in enumerate(names):
        group = [b for b in names[i:] if avg_ranks[b] - avg_ranks[a] <= cd]
        if len(group) > 1 and not any(set(group) <= set(existing) for existing in cliques):
            cliques.append(group)

    log.info("nemenyi_test", n_models=n_models, n_series=n_series,
             critical_difference=round(cd, 4), n_cliques=len(cliques))

    return {
        "average_ranks": avg_ranks,
        "critical_difference": cd,
        "rank_differences": diff,
        "significant_pairs": sig,
        "cliques": cliques,
        "n_models": n_models,
        "n_series": n_series,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Multiple Comparisons with the Best (Fig 5)
# ---------------------------------------------------------------------------

def mcb_test(scores: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Multiple Comparisons with the Best -- the M5 paper's comparison plot.

    Rather than testing all pairs, MCB compares every model against the BEST
    one, which is the question actually being asked. Each model gets an
    average rank and a confidence interval of width CD/2; a model whose
    interval does not overlap the best model's is significantly worse.

    Reference: Koning, Franses, Hibon & Stekler (2005), IJF 21(3):397-409.
    """
    clean = scores.dropna(axis=0, how="any")
    n_series, n_models = clean.shape

    avg_ranks = clean.rank(axis=1, ascending=True).mean(axis=0).sort_values()
    cd = nemenyi_critical_difference(n_models, n_series, alpha=alpha)
    half = cd / 2.0

    best_name = avg_ranks.index[0]
    best_upper = avg_ranks.iloc[0] + half

    out = pd.DataFrame(
        {
            "model": avg_ranks.index,
            "average_rank": avg_ranks.to_numpy(),
            "ci_lower": avg_ranks.to_numpy() - half,
            "ci_upper": avg_ranks.to_numpy() + half,
        }
    )
    # A model is not distinguishable from the best if its lower bound sits
    # below the best model's upper bound.
    out["indistinguishable_from_best"] = out["ci_lower"] <= best_upper
    out["is_best"] = out["model"] == best_name
    out["critical_difference"] = cd
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: build the per-series score matrix the tests consume
# ---------------------------------------------------------------------------

def per_series_scores(
    predictions: pd.DataFrame,
    metric: str = "wape",
    model_col: str = "model",
    series_col: str = "series_id",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    scale_col: str | None = "scale",
) -> pd.DataFrame:
    """
    Pivot long predictions into the (series x model) matrix the tests need.

    Series where every model scores NaN -- typically zero actuals across the
    whole window, which makes WAPE undefined -- are dropped, since they carry
    no ranking information and would otherwise shrink the complete-block set
    for everyone.
    """
    rows: list[dict[str, Any]] = []

    for (sid, model), g in predictions.groupby([series_col, model_col], observed=True):
        yt = g[y_true_col].to_numpy(dtype=float)
        yp = np.clip(g[y_pred_col].to_numpy(dtype=float), 0.0, None)

        if metric == "wape":
            denom = np.abs(yt).sum()
            value = float(100.0 * np.abs(yt - yp).sum() / denom) if denom > _EPS else np.nan
        elif metric == "mae":
            value = float(np.abs(yt - yp).mean())
        elif metric == "rmse":
            value = float(np.sqrt(np.mean((yt - yp) ** 2)))
        elif metric == "rmsse":
            scale = float(g[scale_col].iloc[0]) if scale_col in g.columns else np.nan
            value = (
                float(np.sqrt(np.mean((yt - yp) ** 2) / scale))
                if np.isfinite(scale) and scale > _EPS else np.nan
            )
        else:
            raise ValueError(f"unsupported metric {metric!r}")

        rows.append({series_col: sid, model_col: model, "value": value})

    matrix = pd.DataFrame(rows).pivot(index=series_col, columns=model_col, values="value")
    return matrix.dropna(axis=0, how="all")


def full_significance_report(
    predictions: pd.DataFrame,
    metric: str = "wape",
    horizon: int = 28,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Run the whole battery in the correct order and bundle the results.

    Order matters: Friedman gates the post-hoc tests. If the omnibus does not
    reject, `nemenyi` and `mcb` are returned as None and the report says the
    models are indistinguishable, rather than reporting pairwise results that
    the omnibus does not license.
    """
    matrix = per_series_scores(predictions, metric=metric)
    out: dict[str, Any] = {"metric": metric, "score_matrix": matrix}

    friedman = friedman_test(matrix, alpha=alpha)
    out["friedman"] = friedman

    if friedman.significant:
        out["nemenyi"] = nemenyi_test(matrix, alpha=alpha)
        out["mcb"] = mcb_test(matrix, alpha=alpha)
    else:
        out["nemenyi"] = None
        out["mcb"] = None
        log.warning(
            "friedman_not_significant",
            consequence="post-hoc tests skipped; models are indistinguishable",
        )

    # Pairwise DM on pooled per-observation errors.
    errors: dict[str, np.ndarray] = {}
    for model, g in predictions.groupby("model", observed=True):
        g = g.sort_values(["series_id", "date"])
        errors[str(model)] = (g["y_true"].to_numpy(float) - g["y_pred"].to_numpy(float))
    out["diebold_mariano"] = pairwise_dm(errors, horizon=horizon, alpha=alpha)

    return out
