"""
G4b round 2: regression tests for what the independent review showed the suite would not catch.

The reviewer mutated the code and re-ran the suite: baselines or LightGBM trained through the test
window, a wrong Nemenyi critical difference, and untied README sentences all passed.  These tests
close those gaps:

* leakage perturbation: changing sales *inside the test window* must not change any forecast;
* the critical-difference formula against a hand value;
* the leaderboard and the Nemenyi average ranks recomputed from the saved predictions;
* the README quickstart equals the DVC stage commands, and the prose numbers the reviewer found
  untied are tied to their CSVs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import Origin, run_global_model, run_series_models
from src.evaluation.significance import nemenyi_critical_difference
from src.features.base import FeaturePipeline
from tests.unit.test_polars_parity import _synthetic_panel

ROOT = Path(__file__).resolve().parents[2]
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


def test_readme_quickstart_runs_the_dvc_stage_commands() -> None:
    dvc = (ROOT / "dvc.yaml").read_text(encoding="utf-8")
    for stage in ("backtest", "significance", "inventory"):
        m = re.search(rf"^\s*{stage}:\s*\n\s*cmd:\s*(.+)$", dvc, flags=re.M)
        assert m is not None, stage
        assert m.group(1).strip() in README, f"README quickstart does not run the {stage} stage command"
    assert "python scripts/run_backtest.py --sample-series 3000 --n-origins 3\n" not in README


def test_runbook_uses_the_dvc_backtest_command() -> None:
    dvc = (ROOT / "dvc.yaml").read_text(encoding="utf-8")
    cmd = re.search(r"^\s*backtest:\s*\n\s*cmd:\s*(.+)$", dvc, flags=re.M).group(1).strip()
    runbook = (ROOT / "docs" / "guide" / "06_RUNBOOK.md").read_text(encoding="utf-8")
    assert cmd in runbook.replace("& .venv\\Scripts\\python.exe", "python").replace("\\", "/") or cmd in runbook


def test_readme_states_the_real_training_windows_and_sample() -> None:
    man = json.loads((BT / "manifest.json").read_text(encoding="utf-8"))
    end = pd.Timestamp(max(o["test_end"] for o in man["origins"]))
    start = end - pd.Timedelta(days=man["history_days"] - 1)
    days = [(pd.Timestamp(o["train_end"]) - start).days + 1 for o in sorted(man["origins"], key=lambda o: o["test_start"])]
    assert f"{days[0]}, {days[1]} and {days[2]} days" in README
    p = pd.read_parquet(BT / "predictions.parquet", columns=["series_id", "store_id"]).drop_duplicates("series_id")
    assert p["store_id"].nunique() == 10  # the sample spans every store, so "California stores" would be wrong
    assert "across California stores rather than" not in README


def test_readme_safety_stock_coverage_is_stated_correctly() -> None:
    ss = pd.read_csv(INV / "safety_stock.csv")
    a = ss[(ss["model"] == "adida") & (ss["service_level"] == 0.95)]
    by_model = pd.read_csv(INV / "safety_stock_by_model.csv").set_index("model")
    assert len(a) == 6098
    assert a["safety_stock_units"].sum() == pytest.approx(by_model.loc["adida", "total_ss_units"], rel=1e-5)  # units: all SKUs
    assert a["forecast_error_rmse"].mean() == pytest.approx(by_model.loc["adida", "mean_rmse"], abs=5e-5)  # RMSE: all SKUs
    assert int(by_model.loc["adida", "priced_skus"]) == 4591  # capital: priced SKUs only
    assert "(6,098 SKUs, of which 4,591 are priced)" in README
    assert "all CA SKUs" in README
    assert "priced SKUs only" in README


def test_untied_readme_prose_numbers_match_the_csvs() -> None:
    """The five sentences the reviewer could change without any test noticing."""
    by_model = pd.read_csv(INV / "safety_stock_by_model.csv").set_index("model")
    naive, adida = by_model.loc["naive"], by_model.loc["adida"]
    assert f"{100 * (1 - adida['mean_std'] / naive['mean_std']):.0f}%" == "22%"
    assert "the gap is about 22%" in README
    assert f"mean error {naive['mean_bias']:.2f} per day" in README  # -0.49
    assert f"needs about {naive['mean_rmse']:.2f} units per day" in README  # 2.13
    assert f"about {adida['mean_rmse']:.2f} for" in README  # 1.49
    bases = pd.read_csv(INV / "working_capital_demand_bases.csv").set_index("basis")
    ratio = bases.loc["actual_backtest_days", "mean_daily_demand_total"] / bases.loc["forecast", "mean_daily_demand_total"] - 1
    assert round(100 * abs(ratio), 1) == 0.1
    assert "within about 0.1% of the actuals" in README
    assert int(by_model.loc["naive", "priced_skus"]) == 4591


def test_readme_planning_demand_row_and_look_ahead_disclosure() -> None:
    bases = pd.read_csv(INV / "working_capital_demand_bases.csv").set_index("basis")
    first, forecast = bases.loc["forecast_first_origin", "working_capital"], bases.loc["forecast", "working_capital"]
    assert f"working capital ${first:,.0f} instead of ${forecast:,.0f}" in README
    assert "a planner deciding at 2016-01-31 would not have had" in README
    assert bases.loc["forecast_first_origin", "used_for"] == "reference only"
    assert first < forecast  # the later windows' forecasts see more recent demand


def test_policy_service_levels_in_the_readme_come_from_the_recommendations() -> None:
    recs = pd.read_csv(INV / "policy_recommendations.csv")
    mode = recs.groupby("abc_xyz_class")["recommended_service_level"].agg(lambda s: s.mode().iloc[0])
    for cell in ("AX", "BZ", "CY", "CZ", "AY"):
        assert f"| **{cell}** |" in README
        row = next(ln for ln in README.splitlines() if ln.startswith(f"| **{cell}** |"))
        assert f"**{mode[cell]:.0%}**" in row


def test_readme_discloses_the_tier1_models_and_does_not_oversell_adida() -> None:
    t1 = pd.read_csv(ROOT / "reports" / "tier1" / "tier1_overall.csv")
    acc = pd.read_csv(BT / "accuracy_overall.csv").set_index("model")
    best = t1.loc[t1["wape"].idxmin()]
    assert best["wape"] < acc.loc["adida", "wape"]  # the fact that makes "ADIDA leads" a statement about the 10 models
    assert f"{float(best['wape']):.2f}%" in README
    assert "not about every model scored on this data" in README


def test_readme_no_longer_explains_the_croston_adida_split_by_series_volume() -> None:
    assert "favoring rank metrics on sparse items" not in README
    p = pd.read_parquet(BT / "predictions.parquet", columns=["series_id", "model", "y_true", "y_pred"])
    p = p[p["model"].isin(["adida", "croston_sba"])]
    g = p.assign(ae=(p.y_true - p.y_pred).abs()).groupby(["series_id", "model"], observed=True).agg(ae=("ae", "sum"), v=("y_true", "sum")).reset_index()
    w = g.pivot(index="series_id", columns="model", values="ae")
    w["v"] = g[g["model"] == "adida"].set_index("series_id")["v"]
    w = w[w["v"] > 0]
    w["wins"] = w["croston_sba"] < w["adida"]
    w["q"] = pd.qcut(w["v"].rank(method="first"), 5, labels=False)
    by_q = w.groupby("q")["wins"].mean()
    # The old story needed Croston to win MORE on low-volume series; it wins less there.
    assert by_q.iloc[0] < by_q.iloc[-1]
    assert "does not mean \"equal\"" in README


# ---------------------------------------------------------------------------
# Guide corrections the review found (so they cannot drift back)
# ---------------------------------------------------------------------------

GUIDE = ROOT / "docs" / "guide"


def _slug(heading: str) -> str:
    s = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return s.replace(" ", "-")


def test_guide_05_signatures_match_the_code() -> None:
    """Every `def name(args): ...` quoted in the code walkthrough has the real parameter names."""
    import ast

    real: dict[str, list[list[str]]] = {}
    for f in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef):
                a = n.args
                real.setdefault(n.name, []).append([x.arg for x in a.posonlyargs + a.args + a.kwonlyargs if x.arg not in ("self", "cls")])

    def split_params(args: str) -> list[str]:
        out, depth, cur = [], 0, ""
        for ch in args:
            depth += ch in "([{"
            depth -= ch in ")]}"
            if ch == "," and depth == 0:
                out.append(cur)
                cur = ""
            else:
                cur += ch
        out.append(cur)
        names = [p.split(":")[0].split("=")[0].strip().lstrip("*") for p in out if p.strip()]
        return [n for n in names if n not in ("self", "cls", "")]

    doc = (GUIDE / "05_CODE_WALKTHROUGH.md").read_text(encoding="utf-8")
    bad = []
    for m in re.finditer(r"^\s+def (\w+)\((.*)\)\s*(?:->.*)?:\s*\.\.\.\s*$", doc, flags=re.M):
        name, params = m.group(1), split_params(m.group(2))
        if name not in real or params not in real[name]:
            bad.append(f"{name}({', '.join(params)})")
    assert not bad, f"guide 05 quotes signatures that do not exist: {bad}"


def test_guides_do_not_name_functions_that_do_not_exist() -> None:
    text = "\n".join(p.read_text(encoding="utf-8") for p in GUIDE.glob("*.md"))
    for gone in ("calculate_safety_stock", "classify_abc_xyz", "SafetyStockCalculator.calculate", "src/data/schema.py", "polars_engine.py:assert_no_temporal_leakage"):
        assert gone not in text, gone
    assert "#L" not in "\n".join(re.findall(r"\]\([^)]*\)", text)), "line-number anchors drift; link the file"


def test_guide_links_to_readme_anchors_resolve() -> None:
    headings = {_slug(m.group(1)) for m in re.finditer(r"^#{1,6} (.+)$", README, flags=re.M)}
    broken = []
    for p in GUIDE.glob("*.md"):
        for anchor in re.findall(r"README\.md#([^)\s]+)", p.read_text(encoding="utf-8")):
            if anchor not in headings:
                broken.append(f"{p.name}#{anchor}")
    assert not broken, broken
    # and the README table title the guides name still exists
    assert "6. Does demand classification help?" in README
    assert "Does Demand-Pattern Routing Help?" not in "\n".join(p.read_text(encoding="utf-8") for p in GUIDE.glob("*.md"))


def test_guides_state_the_real_windows_sample_and_tweedie_power() -> None:
    methodology = (GUIDE / "04_METHODOLOGY.md").read_text(encoding="utf-8")
    man = json.loads((BT / "manifest.json").read_text(encoding="utf-8"))
    end = pd.Timestamp(max(o["test_end"] for o in man["origins"]))
    start = end - pd.Timedelta(days=man["history_days"] - 1)
    for o in man["origins"]:
        days = (pd.Timestamp(o["train_end"]) - start).days + 1
        assert f"Train History ({days} Days)" in methodology
    assert "stratified" not in methodology.lower().replace("not a stratified", "")
    src = (ROOT / "src" / "evaluation" / "backtest.py").read_text(encoding="utf-8")
    power = re.search(r"tweedie_variance_power=([\d.]+)", src).group(1)
    for name in ("05_CODE_WALKTHROUGH.md", "09_RESULTS_AND_LIMITATIONS.md"):
        text = (GUIDE / name).read_text(encoding="utf-8")
        assert f"{power}" in text
        assert "p=1.5" not in text
        assert "variance_power=1.5" not in text
    assert "Walmart only record" not in (GUIDE / "06_RUNBOOK.md").read_text(encoding="utf-8")


def test_architecture_worked_example_numbers_are_the_artifact_numbers() -> None:
    arch = (GUIDE / "02_ARCHITECTURE.md").read_text(encoding="utf-8")
    sid = re.search(r"`([A-Z]+_\d_\d{3}_[A-Z]{2}_\d)`, the first INTERMITTENT series", arch).group(1)
    asg = pd.read_csv(BT / "routing_assignments.csv")
    row = asg[(asg["origin"] == 0) & (asg["series_id"] == sid)].iloc[0]
    assert f"$ADI = {row['adi']:.2f}$" in arch
    assert f"$CV^2 = {row['cv2']:.2f}$" in arch
    p = pd.read_parquet(BT / "predictions.parquet")
    g = p[(p["series_id"] == sid) & (p["origin"] == 0)]
    mean = g.groupby("model", observed=True)["y_pred"].mean()
    assert f"**ADIDA**: {mean['adida']:.3f}" in arch
    assert f"**Croston SBA**: {mean['croston_sba']:.3f}" in arch
    assert f"**Router**: {mean['routed_ensemble']:.3f}" in arch
    a = g[g["model"] == "adida"]
    scale = float(a["scale"].iloc[0])
    assert f"$s_i = {scale:.3f}$" in arch
    assert f"RMSSE for this (series, origin) is {np.sqrt(np.mean((a.y_true - a.y_pred) ** 2) / scale):.3f}" in arch
    # The first-sale claim and the invented forecasts of the earlier version are gone.
    assert "recorded its first non-zero sale on **Day 160**" not in arch
    assert "0.814" not in arch
    assert "0.805" not in arch


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "sell_prices.parquet").exists(), reason="raw M5 parquet files not present")
def test_data_guide_price_range_and_christmas_claims_match_the_raw_data() -> None:
    data = (GUIDE / "03_DATA.md").read_text(encoding="utf-8")
    pr = pd.read_parquet(ROOT / "data" / "raw" / "sell_prices.parquet", columns=["sell_price"])["sell_price"]
    assert f"`${pr.min():.2f}` to `${pr.max():.2f}`" in data
    cal = pd.read_parquet(ROOT / "data" / "raw" / "calendar.parquet")
    xmas = cal[pd.to_datetime(cal["date"]).dt.strftime("%m-%d") == "12-25"]["d"].tolist()
    sales = pd.read_parquet(ROOT / "data" / "raw" / "sales_train_validation.parquet", columns=["item_id", *xmas])
    counts = [int((sales[d] > 0).sum()) for d in xmas]
    assert ", ".join(f"d_{{{d.split('_')[1]}}}" for d in xmas) in data.replace("$", "")
    assert f"only {min(counts)} to {max(counts)} of the 30,490 series" in data
