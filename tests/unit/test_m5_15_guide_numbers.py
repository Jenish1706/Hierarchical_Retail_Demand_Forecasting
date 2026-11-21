"""
M5-15: result numbers quoted in the guide are tied to the committed artifacts.

The README tables are rendered; the guide quotes a handful of headline results in prose
and tables (00 headline table, 04 statistical battery, 09 findings).  Each one is derived
here from the artifact it names, so a rerun that changes a result fails this test instead
of leaving a stale number in the guide.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
BT = ROOT / "reports" / "backtest_main"
GUIDE = ROOT / "docs" / "guide"


@pytest.fixture(scope="module")
def art() -> dict:
    sig = json.loads((BT / "significance" / "significance_report.json").read_text(encoding="utf-8"))
    acc = pd.read_csv(BT / "accuracy_overall.csv").set_index("model")
    boot = pd.read_csv(BT / "significance" / "bootstrap_intervals.csv").set_index("model")
    eda = json.loads((ROOT / "reports" / "eda" / "eda_findings.json").read_text(encoding="utf-8"))
    return {"sig": sig, "acc": acc, "boot": boot, "eda": eda}


def _text(name: str) -> str:
    return (GUIDE / name).read_text(encoding="utf-8")


def test_guide_00_headline_table(art: dict) -> None:
    text = _text("00_START_HERE.md")
    acc, boot, sig = art["acc"], art["boot"], art["sig"]
    ranks = sig["nemenyi"]["average_ranks"]
    b = boot.loc["adida"]
    assert f"**ADIDA** leads at **{b['wape']:.2f}%** (95% bootstrap CI: [{b['wape_ci_lower']:.2f}%, {b['wape_ci_upper']:.2f}%])" in text
    assert f"**ADIDA** leads at **{acc.loc['adida', 'rmsse']:.3f}** (95% bootstrap CI: [{b['rmsse_ci_lower']:.3f}, {b['rmsse_ci_upper']:.3f}])" in text
    assert f"average rank **{ranks['croston_sba']:.2f}**" in text
    assert f"$CD = {sig['nemenyi']['critical_difference']:.4f}" in text
    r = boot.loc["routed_ensemble"]
    assert f"$\\Delta\\text{{WAPE}} = {r['delta_wape_point']:+.2f}\\%$, 95% CI [{r['delta_wape_ci_lower']:+.2f}%, {r['delta_wape_ci_upper']:+.2f}%]" in text
    m = boot.loc["moving_average"]
    assert f"$\\Delta\\text{{RMSSE}} = {m['delta_rmsse_point']:+.4f}$, 95% CI [{m['delta_rmsse_ci_lower']:+.4f}, {m['delta_rmsse_ci_upper']:+.4f}]" in text
    assert f"scores {acc.loc['lightgbm_tweedie', 'wape']:.2f}% WAPE (rank {ranks['lightgbm_tweedie']:.2f})" in text
    # Section 1 describes the dataset: 30,490 store-item SERIES, which are 3,049 items in each of 10 stores.
    s31 = art["eda"]["3.1_structure"]
    card = s31["hierarchy_cardinality"]
    assert f"({s31['n_series']:,} store-item series: {card['item']:,} items in each of {card['store']} stores" in text
    assert f"spanning {s31['n_days']:,} days" in text
    assert card["item"] * card["store"] == s31["n_series"]


def test_guide_04_statistical_battery(art: dict) -> None:
    text = _text("04_METHODOLOGY.md")
    sig, boot = art["sig"], art["boot"]
    ranks = sig["nemenyi"]["average_ranks"]
    cd = sig["nemenyi"]["critical_difference"]
    n = sig["n_series_complete"]
    assert f"{n:,}" in text
    assert f"$\\chi_F^2 = {sig['friedman']['statistic']:.2f}$" in text
    assert f"= {cd:.4f}$" in text
    best = min(ranks.values())
    assert f"`croston_sba`: **{ranks['croston_sba']:.3f}** (Rank Leader)" in text
    for model, label in (
        ("adida", "Same Clique as Best"),
        ("routed_ensemble", "Same Clique as Best"),
        ("ses", "Significantly worse"),
        ("moving_average", "Significantly worse"),
        ("lightgbm_tweedie", "Significantly worse"),
        ("lightgbm", "Significantly worse"),
    ):
        gap = round(ranks[model], 3) - round(best, 3)  # the guide subtracts the rounded ranks
        assert f"`{model}`: **{ranks[model]:.3f}**" in text or f"`{model}`: {ranks[model]:.3f}" in text
        assert f"{gap:+.3f} {'<' if gap < cd else '>'} CD" in text, model
        assert label in text
    r = boot.loc["routed_ensemble"]
    assert f"$\\Delta\\text{{WAPE}} = {r['delta_wape_point']:+.2f}\\%$, 95% CI: **[{r['delta_wape_ci_lower']:+.2f}%, {r['delta_wape_ci_upper']:+.2f}%]**" in text
    m = boot.loc["moving_average"]
    assert f"$\\Delta\\text{{RMSSE}} = {m['delta_rmsse_point']:+.4f}$, 95% CI: **[{m['delta_rmsse_ci_lower']:+.4f}, {m['delta_rmsse_ci_upper']:+.4f}]**" in text
    t = boot.loc["lightgbm_tweedie"]
    assert f"$\\Delta\\text{{WAPE}} = {t['delta_wape_point']:+.2f}\\%$, 95% CI: [{t['delta_wape_ci_lower']:+.2f}%, {t['delta_wape_ci_upper']:+.2f}%]" in text
    h2h = next(p for p in sig["pairwise_per_series"] if {p["model_a"], p["model_b"]} == {"adida", "croston_sba"})
    csba_win = h2h["b_win_pct"] if h2h["model_b"] == "croston_sba" else h2h["a_win_pct"]
    adida_win = h2h["a_win_pct"] if h2h["model_a"] == "adida" else h2h["b_win_pct"]
    assert f"wins on **{csba_win:.2f}%** of series, while ADIDA wins on **{adida_win:.2f}%** of series ({h2h['tie_pct']:.2f}% ties)" in text
    assert f"**{h2h['dm_indistinguishable_pct']:.2f}% of series**" in text
    assert f"{abs(ranks['croston_sba'] - ranks['adida']):.3f} \\le {cd:.4f}" in text


def test_guide_09_findings(art: dict) -> None:
    text = _text("09_RESULTS_AND_LIMITATIONS.md")
    acc, sig, boot = art["acc"], art["sig"], art["boot"]
    ranks = sig["nemenyi"]["average_ranks"]
    q1 = text.split("### Q1:")[1].split("### Q2:")[0]
    assert f"`adida` achieves the lowest aggregate WAPE (**{acc.loc['adida', 'wape']:.2f}%**)" in q1
    assert f"`moving_average` ({acc.loc['moving_average', 'wape']:.2f}%)" in q1
    assert f"`routed_ensemble` ({acc.loc['routed_ensemble', 'wape']:.2f}%)" in q1
    assert f"(**{ranks['croston_sba']:.2f}**)" in q1
    assert f"`adida` ({ranks['adida']:.2f})" in q1
    assert f"`routed_ensemble` ({ranks['routed_ensemble']:.2f})" in q1
    assert f"$\\chi^2 = {sig['friedman']['statistic']:.2f}" in q1
    assert f"$CD = {sig['nemenyi']['critical_difference']:.4f}$" in q1
    r = boot.loc["routed_ensemble"]
    assert f"$\\Delta\\text{{WAPE}} = {r['delta_wape_point']:+.2f}\\%$ (95% CI $[{r['delta_wape_ci_lower']:+.2f}\\%, {r['delta_wape_ci_upper']:+.2f}\\%]$)" in q1

    q2 = text.split("### Q2:")[1].split("### Q3:")[0]
    for model in ("lightgbm", "lightgbm_tweedie"):
        assert f"`{model}` at {acc.loc[model, 'wape']:.2f}% WAPE / {acc.loc[model, 'rmsse']:.4f} RMSSE" in q2
    assert f"(`adida` at {acc.loc['adida', 'wape']:.2f}% WAPE)" in q2
    assert f"({acc.loc['moving_average', 'wape']:.2f}% WAPE / {acc.loc['moving_average', 'rmsse']:.4f} RMSSE)" in q2

    # Q5: the share of the catalogue that is intermittent or lumpy, and the naive baselines.
    q5 = text.split("### Q5:")[1].split("### Q6:")[0]
    share = art["eda"]["3.3_intermittency"]["intermittent_or_lumpy_pct"]
    assert f"{share:.1f}% of the catalog" in q5
    assert "72.6%" not in q5  # that is the INTERMITTENT quadrant alone
    assert f"`seasonal_naive_28d` WAPE {acc.loc['seasonal_naive', 'wape']:.2f}%" in q5
    assert f"`naive` WAPE {acc.loc['naive', 'wape']:.2f}%" in q5
