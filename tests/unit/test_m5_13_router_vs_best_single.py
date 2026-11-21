"""
M5-13: report the router honestly and use the right classification name.

Two parts:

* ``compare_router`` scores the routed ensemble against every single model,
  per origin and per demand segment, with the same pooled WAPE as the rest of
  the benchmark.  Tested on small hand-built frames first, then against the
  committed backtest artifacts.
* The quadrant classification with cut-offs ADI = 1.32 / CV^2 = 0.49 is the
  Syntetos-Boylan-Croston (SBC, 2005) scheme.  The earlier three-letter name with a
  "T" is not a standard name, so no tracked text file may use it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import render_readme_tables as rrt
from src.evaluation.point_metrics import wape
from src.evaluation.routing_comparison import (
    ADIDA_MODEL,
    ALL_ORIGINS,
    ALL_SEGMENTS,
    COMPARISON_COLUMNS,
    GLOBAL_MODEL,
    ROUTED_MODEL,
    compare_router,
    per_origin_columns,
)

ROOT = Path(__file__).resolve().parents[2]
BACKTEST = ROOT / "reports" / "backtest_main"
ASSIGNMENTS_CSV = BACKTEST / "routing_assignments.csv"
COMPARISON_CSV = BACKTEST / "routing_vs_best_single.csv"
BENEFIT_CSV = BACKTEST / "routing_benefit.csv"
PREDICTIONS = BACKTEST / "predictions.parquet"


# ---------------------------------------------------------------------------
# compare_router on hand-built data
# ---------------------------------------------------------------------------


def _frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two origins, two segments (SMOOTH -> global, INTERMITTENT -> adida), 4 models."""
    rows = []
    # (origin, series, pattern, method, actual, {model: forecast})
    spec = [
        (0, "s1", "SMOOTH", "global", 10.0, {"routed_ensemble": 9.0, "adida": 7.0, "lightgbm": 9.0, "naive": 14.0}),
        (0, "s2", "INTERMITTENT", "adida", 2.0, {"routed_ensemble": 1.0, "adida": 1.0, "lightgbm": 4.0, "naive": 0.0}),
        (1, "s1", "SMOOTH", "global", 10.0, {"routed_ensemble": 12.0, "adida": 8.0, "lightgbm": 12.0, "naive": 10.0}),
        (1, "s2", "INTERMITTENT", "adida", 2.0, {"routed_ensemble": 3.0, "adida": 3.0, "lightgbm": 0.0, "naive": 5.0}),
    ]
    assign = []
    for origin, sid, pattern, method, actual, fc in spec:
        assign.append({"origin": origin, "series_id": sid, "pattern": pattern, "method": method})
        for model, f in fc.items():
            for d in range(2):  # two days per (series, origin)
                rows.append(
                    {"origin": origin, "series_id": sid, "model": model, "date": d, "y_true": actual, "y_pred": f}
                )
    return pd.DataFrame(rows), pd.DataFrame(assign)


def test_pooled_wape_matches_the_benchmark_definition() -> None:
    preds, assign = _frame()
    cmp = compare_router(preds, assign)
    row = cmp[(cmp["origin"] == ALL_ORIGINS) & (cmp["pattern"] == ALL_SEGMENTS)].iloc[0]
    sub = preds[preds["model"] == "adida"]
    assert row["wape_adida"] == pytest.approx(wape(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()))
    sub = preds[preds["model"] == ROUTED_MODEL]
    assert row["wape_routed"] == pytest.approx(wape(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()))


def test_segment_rows_by_hand() -> None:
    preds, assign = _frame()
    cmp = compare_router(preds, assign)
    smooth = cmp[(cmp["origin"] == ALL_ORIGINS) & (cmp["pattern"] == "SMOOTH")].iloc[0]
    # SMOOTH actual 10 on 2 origins x 2 days = 40. Routed errors |1| and |2| -> (1+2)*2 = 6 -> 15%.
    assert smooth["wape_routed"] == pytest.approx(100 * 6 / 40)
    # ADIDA errors |3| and |2| -> 10 -> 25%.
    assert smooth["wape_adida"] == pytest.approx(100 * 10 / 40)
    assert smooth["routes_to"] == "global"
    assert smooth["n_series_origins"] == 2
    # In this block the router beats ADIDA, so the delta is negative (router better).
    assert smooth["routed_minus_adida"] == pytest.approx(15 - 25)


def test_best_single_excludes_the_router_and_is_chosen_per_row() -> None:
    preds, assign = _frame()
    cmp = compare_router(preds, assign)
    for _, r in cmp.iterrows():
        assert r["best_single_model"] != ROUTED_MODEL
        assert r["wape_best_single"] <= r["wape_adida"] + 1e-12
        assert r["wape_best_single"] <= r["wape_lightgbm"] + 1e-12
        assert r["routed_minus_best_single"] == pytest.approx(r["wape_routed"] - r["wape_best_single"])
    # origin 0, INTERMITTENT: ADIDA and the router forecast the same (|1| error) and beat LightGBM/naive.
    inter = cmp[(cmp["origin"] == "0") & (cmp["pattern"] == "INTERMITTENT")].iloc[0]
    assert inter["wape_routed"] == pytest.approx(inter["wape_adida"])
    assert inter["routed_minus_adida"] == pytest.approx(0.0)


def test_router_worse_than_the_best_single_model_gives_a_positive_delta() -> None:
    """A router that routes badly must show up as a positive (worse) delta."""
    preds, assign = _frame()
    # Make the router forecast far from the actuals everywhere.
    preds.loc[preds["model"] == ROUTED_MODEL, "y_pred"] = 50.0
    cmp = compare_router(preds, assign)
    allrow = cmp[(cmp["origin"] == ALL_ORIGINS) & (cmp["pattern"] == ALL_SEGMENTS)].iloc[0]
    assert allrow["routed_minus_adida"] > 0
    assert allrow["routed_minus_best_single"] > 0


def test_every_scope_is_present() -> None:
    preds, assign = _frame()
    cmp = compare_router(preds, assign)
    assert list(cmp.columns) == COMPARISON_COLUMNS
    assert set(cmp["origin"]) == {"0", "1", ALL_ORIGINS}
    assert set(cmp["pattern"]) == {"SMOOTH", "INTERMITTENT", ALL_SEGMENTS}
    assert len(cmp) == 3 * 3


def test_missing_assignment_raises_instead_of_dropping_rows() -> None:
    preds, assign = _frame()
    with pytest.raises(ValueError, match="no routing assignment"):
        compare_router(preds, assign.iloc[:-1])


def test_no_routed_model_raises() -> None:
    preds, assign = _frame()
    with pytest.raises(ValueError, match="routed_ensemble"):
        compare_router(preds[preds["model"] != ROUTED_MODEL], assign)


def test_per_origin_columns_are_one_row_per_origin() -> None:
    preds, assign = _frame()
    cols = per_origin_columns(compare_router(preds, assign))
    assert list(cols.index) == [0, 1]
    assert set(cols.columns) == {
        "adida_wape",
        "best_single_model",
        "best_single_wape",
        "routed_minus_adida",
        "routed_minus_best_single",
    }


# ---------------------------------------------------------------------------
# Committed artifacts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed() -> dict[str, pd.DataFrame]:
    for p in (ASSIGNMENTS_CSV, COMPARISON_CSV, BENEFIT_CSV, PREDICTIONS):
        assert p.exists(), f"missing committed artifact {p}"
    return {
        "assign": pd.read_csv(ASSIGNMENTS_CSV),
        "cmp": pd.read_csv(COMPARISON_CSV, dtype={"origin": str}),
        "benefit": pd.read_csv(BENEFIT_CSV),
    }


def test_committed_comparison_is_recomputed_from_predictions(committed: dict[str, pd.DataFrame]) -> None:
    preds = pd.read_parquet(PREDICTIONS, columns=["origin", "series_id", "model", "y_true", "y_pred"])
    fresh = compare_router(preds, committed["assign"])
    saved = committed["cmp"]
    assert list(saved.columns) == COMPARISON_COLUMNS
    pd.testing.assert_frame_equal(
        fresh.reset_index(drop=True),
        saved.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-9,
    )


def test_assignments_agree_with_the_routed_predictions(committed: dict[str, pd.DataFrame]) -> None:
    """The saved assignments must describe the forecasts the router actually produced."""
    preds = pd.read_parquet(
        PREDICTIONS, columns=["origin", "series_id", "model", "date", "y_pred"]
    )
    keys = ["origin", "series_id", "date"]
    routed = preds[preds["model"] == ROUTED_MODEL].rename(columns={"y_pred": "routed"})
    wide = routed[[*keys, "routed"]]
    for model, col in ((ADIDA_MODEL, "adida"), (GLOBAL_MODEL, "global")):
        other = preds[preds["model"] == model].rename(columns={"y_pred": col})
        wide = wide.merge(other[[*keys, col]], on=keys, how="left")
    wide = wide.merge(committed["assign"][["origin", "series_id", "method"]], on=["origin", "series_id"])

    adida = wide[wide["method"] == "adida"]
    glob = wide[wide["method"] == "global"]
    zero = wide[wide["method"] == "zero"]
    assert len(adida) > 0
    assert len(glob) > 0
    assert len(zero) > 0
    # Global and zero routes are exact.
    assert np.allclose(glob["routed"], glob["global"])
    assert np.allclose(zero["routed"], 0.0)
    # The router fits ADIDA after trimming pre-launch zeros, the leaderboard's ADIDA fits the
    # whole window, so a small share of routed ADIDA forecasts differ, and only slightly.
    same = np.isclose(adida["routed"], adida["adida"])
    assert same.mean() > 0.95
    assert (adida["routed"] - adida["adida"]).abs().max() < 0.1
    assert set(wide["method"]) <= {"adida", "global", "zero"}


def test_routing_benefit_gains_the_per_origin_comparison(committed: dict[str, pd.DataFrame]) -> None:
    benefit = committed["benefit"]
    for col in ("adida_wape", "best_single_model", "best_single_wape", "routed_minus_adida", "routed_minus_best_single"):
        assert col in benefit.columns, col
    cmp = committed["cmp"]
    allrows = cmp[(cmp["pattern"] == ALL_SEGMENTS) & (cmp["origin"] != ALL_ORIGINS)].copy()
    allrows["origin"] = allrows["origin"].astype(int)
    merged = benefit.merge(allrows, on="origin")
    assert len(merged) == len(benefit) == 3
    # The router's own evaluation and the recomputation agree on its WAPE.
    assert np.allclose(merged["routed_wape"], merged["wape_routed"], atol=1e-6)
    assert np.allclose(merged["global_wape"], merged["wape_lightgbm"], atol=1e-6)
    assert np.allclose(merged["adida_wape"], merged["wape_adida"])


def test_router_does_not_beat_adida_at_any_origin(committed: dict[str, pd.DataFrame]) -> None:
    """The finding the README states; if a rerun changes it, the prose must change too."""
    benefit = committed["benefit"]
    assert (benefit["routed_minus_adida"] > 0).all()


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def test_readme_renders_the_router_comparison() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert rrt.ROUTER_START in readme
    assert rrt.ROUTER_END in readme
    body = readme.split(rrt.ROUTER_START)[1].split(rrt.ROUTER_END)[0]
    assert body.strip() == rrt.render_router_comparison().strip()
    assert "ADIDA everywhere" in body
    assert "Router" in body


def test_readme_verdict_matches_the_numbers(committed: dict[str, pd.DataFrame]) -> None:
    cmp = committed["cmp"]
    pooled = cmp[(cmp["origin"] == ALL_ORIGINS) & (cmp["pattern"] == ALL_SEGMENTS)].iloc[0]
    body = rrt.render_router_comparison()
    if pooled["routed_minus_adida"] > 0:
        assert "did not beat" in body
    else:
        assert "beat applying ADIDA everywhere" in body
    # Every segment's LightGBM-vs-ADIDA direction is named on the right side of the sentence.
    seg = cmp[(cmp["origin"] == ALL_ORIGINS) & (cmp["routes_to"] == "global")]
    better = seg.loc[seg["wape_lightgbm"] < seg["wape_adida"], "pattern"].tolist()
    worse = seg.loc[seg["wape_lightgbm"] >= seg["wape_adida"], "pattern"].tolist()
    won = body.split("LightGBM beat ADIDA on ")[1].split(" and lost to it on ")[0]
    lost = body.split(" and lost to it on ")[1].split(" (LightGBM vs ADIDA")[0]
    for name in better:
        assert name in won
        assert name not in lost
    for name in worse:
        assert name in lost
        assert name not in won


def test_readme_states_the_selection_origin_caveat() -> None:
    body = rrt.render_router_comparison()
    assert "earlier selection origin" in body
    assert "best single model" in body.lower()


def test_results_guide_q6_gives_the_honest_answer() -> None:
    """Guide 09 Q6 used to say "Yes" on the strength of router-vs-LightGBM alone."""
    text = (ROOT / "docs" / "guide" / "09_RESULTS_AND_LIMITATIONS.md").read_text(encoding="utf-8")
    q6 = text.split("### Q6:")[1].split("### Q7:")[0]
    assert "**No.**" in q6
    assert "best single model is ADIDA" in q6
    assert "earlier selection origin" in q6
    for stale in ("77.87", "78.81", "75.51", "+2.63%", "+1.42%", "Yes"):
        assert stale not in q6
    assert "routing_vs_best_single.csv" in q6


# ---------------------------------------------------------------------------
# Naming: SBC
# ---------------------------------------------------------------------------


def _tracked_text_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")
    files = []
    for rel in out.decode("utf-8").split("\0"):
        if not rel:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        head = path.read_bytes()[:8192]
        if b"\0" in head:  # binary: parquet, png, ...
            continue
        files.append(path)
    return files


def test_no_tracked_text_file_uses_the_retired_classification_name() -> None:
    retired = "".join(["s", "b", "t"])  # built at run time so this file does not contain it
    offenders = []
    for path in _tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for no, line in enumerate(text.splitlines(), 1):
            if retired in line.lower():
                offenders.append(f"{path.relative_to(ROOT)}:{no}")
    assert not offenders, "retired classification name still present:\n" + "\n".join(offenders[:20])


def test_classification_is_credited_to_syntetos_boylan_croston() -> None:
    src = (ROOT / "src" / "models" / "statistical" / "intermittent.py").read_text(encoding="utf-8")
    assert "Syntetos-Boylan-Croston (SBC)" in src
    router = (ROOT / "src" / "models" / "router.py").read_text(encoding="utf-8")
    assert "Syntetos-Boylan-Croston (SBC)" in router
