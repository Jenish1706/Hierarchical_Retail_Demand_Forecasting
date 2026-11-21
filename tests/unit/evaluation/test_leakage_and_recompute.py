"""
Leakage and recomputation guards for the evaluation harness.

Changing sales *inside the test window* must not change any forecast; the critical-difference formula matches
the textbook value; the leaderboard and the Nemenyi average ranks are recomputed from the saved predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import Origin, run_global_model, run_series_models
from src.evaluation.significance import nemenyi_critical_difference
from src.features.base import FeaturePipeline
from tests.unit.features.test_polars_parity import _synthetic_panel

ROOT = Path(__file__).resolve().parents[3]
BT = ROOT / "reports" / "backtest_main"
INV = ROOT / "reports" / "inventory"
README = (ROOT / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Leakage perturbation: the test window must not influence any forecast
# ---------------------------------------------------------------------------


def _panel_and_origin() -> tuple[pd.DataFrame, Origin]:
    panel = _synthetic_panel(n_series=8, n_days=200)
    panel["series_id"] = panel["item_id"].astype(str) + "_" + panel["store_id"].astype(str)
    dates = sorted(panel["date"].unique())
    origin = Origin(
        index=0,
        train_end=pd.Timestamp(dates[139]),
        test_start=pd.Timestamp(dates[140]),
        test_end=pd.Timestamp(dates[167]),
    )
    return panel, origin


def _perturb_test_window(panel: pd.DataFrame, origin: Origin) -> pd.DataFrame:
    out = panel.copy()
    mask = out["date"] >= origin.test_start
    out.loc[mask, "sales"] = (out.loc[mask, "sales"].astype(int) + 100).astype("int16")
    return out


def test_series_models_ignore_the_test_window() -> None:
    """Naive, seasonal naive, moving average, SES, Croston SBA and ADIDA are fitted before the origin."""
    panel, origin = _panel_and_origin()
    base, _, _ = run_series_models(panel, origin, 28)
    shifted, _, _ = run_series_models(_perturb_test_window(panel, origin), origin, 28)
    keys = ["model", "series_id", "date"]
    a = base.sort_values(keys).reset_index(drop=True)
    b = shifted.sort_values(keys).reset_index(drop=True)
    assert len(a) == len(b) > 0
    assert (a["y_true"] != b["y_true"]).any(), "the perturbation did not reach the actuals"
    np.testing.assert_allclose(a["y_pred"].to_numpy(), b["y_pred"].to_numpy())
    np.testing.assert_allclose(a["scale"].to_numpy(), b["scale"].to_numpy())


def test_global_model_ignores_the_test_window() -> None:
    """LightGBM's training rows end at the origin and its test features are horizon-shifted."""
    panel, origin = _panel_and_origin()
    preds = []
    for p in (panel, _perturb_test_window(panel, origin)):
        pipe = FeaturePipeline(horizon=28, use_polars=True)
        feats = pipe.transform(p, validate=False)
        feats["series_id"] = p["series_id"].to_numpy()
        cols = pipe.model_feature_columns(feats, drop_price=True)
        scales = dict.fromkeys(p["series_id"].unique(), 1.0)
        preds.append(run_global_model(feats, cols, origin, scales, model_name="lightgbm"))
    a = preds[0].sort_values(["series_id", "date"]).reset_index(drop=True)
    b = preds[1].sort_values(["series_id", "date"]).reset_index(drop=True)
    assert len(a) == len(b) > 0
    assert (a["y_true"] != b["y_true"]).any()
    np.testing.assert_allclose(a["y_pred"].to_numpy(), b["y_pred"].to_numpy())


# ---------------------------------------------------------------------------
# Critical difference and the leaderboard, recomputed from the saved predictions
# ---------------------------------------------------------------------------


def test_nemenyi_critical_difference_matches_the_textbook_value() -> None:
    # Demsar (2006), Table 5: q_0.05 for k = 3 is 2.343, so CD = 2.343 * sqrt(k(k+1) / (6 N)).
    assert nemenyi_critical_difference(3, 10, alpha=0.05) == pytest.approx(2.343 * np.sqrt(3 * 4 / (6 * 10)), rel=2e-3)
    # k = 10 models, N = 2,950 series: the README's 0.2494 (q_0.05 = 3.164).
    assert nemenyi_critical_difference(10, 2950, alpha=0.05) == pytest.approx(3.164 * np.sqrt(10 * 11 / (6 * 2950)), rel=2e-3)


@pytest.fixture(scope="module")
def predictions() -> pd.DataFrame:
    return pd.read_parquet(BT / "predictions.parquet", columns=["origin", "series_id", "model", "date", "y_true", "y_pred", "scale"])


def test_accuracy_overall_is_recomputed_from_the_saved_predictions(predictions: pd.DataFrame) -> None:
    acc = pd.read_csv(BT / "accuracy_overall.csv").set_index("model")
    p = predictions
    wape = p.assign(ae=(p.y_true - p.y_pred).abs()).groupby("model", observed=True).apply(
        lambda g: 100 * g["ae"].sum() / g["y_true"].sum(), include_groups=False
    )
    for model, value in wape.items():
        assert acc.loc[str(model), "wape"] == pytest.approx(value, rel=1e-9)
    # RMSSE: per (series, origin) unit with that origin's own scale, then the mean over units.
    q = p[p["scale"] > 0].assign(se=lambda d: (d.y_true - d.y_pred) ** 2)
    unit = q.groupby(["model", "series_id", "origin"], observed=True).agg(mse=("se", "mean"), scale=("scale", "first")).reset_index()
    unit["rmsse"] = np.sqrt(unit["mse"] / unit["scale"])
    rmsse = unit.groupby("model", observed=True)["rmsse"].mean()
    for model, value in rmsse.items():
        assert acc.loc[str(model), "rmsse"] == pytest.approx(value, rel=1e-6)


def test_nemenyi_average_ranks_are_recomputed_from_the_saved_predictions(predictions: pd.DataFrame) -> None:
    sig = json.loads((BT / "significance" / "significance_report.json").read_text(encoding="utf-8"))
    # Per-series WAPE for every (series, model), summed in row order exactly as a reader would with numpy.
    # (A pandas groupby sums in a different order, which moves scores by about 1e-12 and breaks the exact
    # ties between identical forecasts, so the ranks would differ in the third decimal.)
    scores: dict[tuple[str, str], float] = {}
    for (sid, model), g in predictions.groupby(["series_id", "model"], observed=True):
        y, f = g["y_true"].to_numpy(float), np.clip(g["y_pred"].to_numpy(float), 0.0, None)
        if np.abs(y).sum() > 1e-9:
            scores[(str(sid), str(model))] = 100.0 * np.abs(y - f).sum() / np.abs(y).sum()
    matrix = pd.Series(scores).unstack().dropna(axis=0, how="any")
    assert len(matrix) == sig["n_series_complete"]
    ranks = matrix.rank(axis=1, method="average").mean()
    for model, value in ranks.items():
        assert sig["nemenyi"]["average_ranks"][str(model)] == pytest.approx(value, abs=5e-4)
    cd = nemenyi_critical_difference(len(ranks), len(matrix), alpha=sig["alpha"])
    assert sig["nemenyi"]["critical_difference"] == pytest.approx(cd, abs=5e-5)


def test_results_leaderboard_uses_the_canonical_rmsse() -> None:
    """reports/results/leaderboard.csv once held a pooled RMSSE that ranked LightGBM first and contradicted the README."""
    lb = pd.read_csv(ROOT / "reports" / "results" / "leaderboard.csv").set_index("model")
    acc = pd.read_csv(BT / "accuracy_overall.csv").set_index("model")
    for model in lb.index:
        assert lb.loc[model, "rmsse"] == pytest.approx(acc.loc[model, "rmsse"], rel=1e-9)
    assert lb["rmsse"].idxmin() == acc["rmsse"].idxmin() == "adida"


# ---------------------------------------------------------------------------
# README: reproduction commands, windows, sample, and the prose numbers that were untied
# ---------------------------------------------------------------------------


