"""
G4b: the dashboard shows artifact numbers, not typed ones.

The EDA, overview and significance views used to carry typed results that had gone stale or were
wrong (95.2% sparse series where the artifact says 91.0%, a router "best Nemenyi rank (4.54)" that
the significance report contradicts, a What-If simulator with a made-up $10,000 baseline).  These
tests render the views against a stub Streamlit and tie what they show to the committed artifacts,
and forbid typed result numbers in dashboard string literals.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.dashboard.views import eda, inventory, overview, significance

ROOT = Path(__file__).resolve().parents[2]


def _stub_streamlit() -> MagicMock:
    st = MagicMock()
    st.columns.side_effect = lambda n, *a, **k: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    st.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
    st.selectbox.return_value = None
    return st


def _text_shown(st: MagicMock) -> str:
    """Everything the page wrote as markdown, info, success or metric text."""
    chunks = []
    for name in ("markdown", "info", "success", "warning", "caption"):
        for call in getattr(st, name).call_args_list:
            chunks.extend(str(a) for a in call.args)
    for call in st.metric.call_args_list:
        chunks.extend(str(a) for a in call.args)
        chunks.extend(str(v) for v in call.kwargs.values())
    return "\n".join(chunks)


def test_eda_view_shows_the_eda_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _stub_streamlit()
    monkeypatch.setattr(eda, "st", st)
    eda.render()
    shown = _text_shown(st)
    f = json.loads((ROOT / "reports" / "eda" / "eda_findings.json").read_text(encoding="utf-8"))
    s, d, mix = f["3.1_structure"], f["3.2_demand"], f["3.3_intermittency"]
    assert f"{d['overall_zero_rate']:.1%}" in shown
    assert f"{mix['intermittent_or_lumpy_pct']:.1f}%" in shown
    assert f"{mix['pct_of_series']['SMOOTH']:.1f}%" in shown
    assert f"{s['n_series']:,}" in shown
    # The stale literals the view used to carry.
    for stale in ("95.2%", "3.2%", "75.6", "19.5"):
        assert stale not in shown


def test_overview_view_derives_its_scale_from_the_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _stub_streamlit()
    monkeypatch.setattr(overview, "st", st)
    overview.render()
    shown = _text_shown(st)
    s = json.loads((ROOT / "reports" / "eda" / "eda_findings.json").read_text(encoding="utf-8"))["3.1_structure"]
    assert f"{s['n_series']:,} store-item series" in shown
    assert f"{s['long_format_rows'] / 1e6:.1f}M Records" in shown
    assert "controlled stockout risk" not in shown
    assert "without starving" not in shown


def test_significance_view_does_not_claim_the_router_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _stub_streamlit()
    monkeypatch.setattr(significance, "st", st)
    significance.render()
    shown = _text_shown(st)
    assert "4.54" not in shown  # the router's "best average Nemenyi rank": Croston SBA leads, not the router
    assert "57.5%" not in shown
    assert "2,461" not in shown  # a typed Friedman statistic that no artifact supports
    assert "35 of 45" not in shown  # the retired panel-pooled Diebold-Mariano headline
    sig = json.loads((ROOT / "reports" / "backtest_main" / "significance" / "significance_report.json").read_text(encoding="utf-8"))
    assert f"chi-sq = {sig['friedman']['statistic']:,.2f}" in shown
    assert f"{sig['nemenyi']['critical_difference']:.4f}" in shown
    r = pd.read_csv(ROOT / "reports" / "backtest_main" / "routing_vs_best_single.csv", dtype={"origin": str})
    pooled = r[(r["origin"] == "all") & (r["pattern"] == "ALL")].iloc[0]
    if "did not beat the best single model" in shown:
        assert pooled["routed_minus_adida"] > 0
        assert f"{pooled['wape_routed']:.2f}%" in shown
        assert f"{pooled['wape_adida']:.2f}%" in shown


def test_inventory_view_replaced_the_toy_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    src = (ROOT / "src" / "dashboard" / "views" / "inventory.py").read_text(encoding="utf-8")
    assert "baseline_capital" not in src
    assert "10000.0" not in src
    assert "without starving" not in src
    st = _stub_streamlit()
    monkeypatch.setattr(inventory, "st", st)
    inventory.render()
    shown = _text_shown(st)
    assert "Policy Comparison" in shown
    assert "policy_comparison.csv" in shown
    # Tab 1 pivots SKU counts from the classification table; the old code pivoted segment_summary
    # on columns it does not have and raised KeyError('abc_class').
    assert st.plotly_chart.call_count >= 1


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


@pytest.mark.parametrize("view", sorted((ROOT / "src" / "dashboard" / "views").glob("*.py")) + [ROOT / "streamlit_app.py"])
def test_dashboard_strings_carry_no_typed_result_numbers(view: Path) -> None:
    """No percentage with decimals, dollar amount or thousands-separated count inside a string literal."""
    offenders = []
    for s in _string_constants(view):
        for m in re.finditer(r"\d+\.\d+%|\$\s?\d|\b\d{1,3}(?:,\d{3})+\b", s):
            offenders.append(s[max(0, m.start() - 20) : m.end() + 20].replace("\n", " "))
    assert not offenders, f"{view.name}: typed result numbers {offenders[:5]}"
