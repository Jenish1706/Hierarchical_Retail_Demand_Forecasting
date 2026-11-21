"""
M5-15: one logical chain in the README and guide 00.

The README is ordered as eight questions, each raised by the answer to the one
before.  These tests pin that order, check that the chain links resolve, forbid
hand-typed numbers in the chain's own prose (numbers live in rendered blocks),
and tie the margin-sensitivity figures quoted in guide 09 to the committed CSVs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from scripts import render_readme_tables as rrt

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "guide"
INVENTORY = ROOT / "reports" / "inventory"

QUESTIONS = [
    "What does the demand look like?",
    "Which metric is fair for it?",
    "How do we evaluate without leakage?",
    "Which models, and what does each rung test?",
    "Are the differences real?",
    "Does demand classification help?",
    "What is a better forecast worth?",
    "Which service level should each segment get?",
]


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_has_the_eight_steps_in_order(readme: str) -> None:
    found = re.findall(r"^### (\d)\. (.+)$", readme, flags=re.M)
    assert [int(n) for n, _ in found] == list(range(1, 9))
    assert [q for _, q in found] == QUESTIONS


def test_chain_overview_links_resolve_to_the_step_headings(readme: str) -> None:
    anchors = {_slug(f"{i}. {q}") for i, q in enumerate(QUESTIONS, 1)}
    overview = readme.split("### The chain at a glance")[1].split("\n---")[0]
    links = re.findall(r"\]\(#([^)]+)\)", overview)
    assert len(links) == 8
    assert set(links) == anchors


def test_each_forecast_step_hands_over_to_the_next_question(readme: str) -> None:
    """Steps 1 to 6 each end by raising the question the next step answers."""
    for i in range(1, 7):
        start = readme.index(f"### {i}. {QUESTIONS[i - 1]}")
        end = readme.index(f"### {i + 1}. {QUESTIONS[i]}") if i < 6 else readme.index("## Inventory Decision Layer")
        assert "*Raises the next question*" in readme[start:end], f"step {i} does not hand over"


def test_the_inventory_steps_still_sit_under_the_inventory_heading(readme: str) -> None:
    inv = readme[readme.index("## Inventory Decision Layer") :]
    assert "### 7. What is a better forecast worth?" in inv
    assert "### 8. Which service level should each segment get?" in inv


def test_chain_prose_has_no_hand_typed_numbers(readme: str) -> None:
    """Steps 1 to 6 and the overview: every number sits in a rendered block (checked by the render check)."""
    start = readme.index("### The chain at a glance")
    end = readme.index("## Inventory Decision Layer")
    text = readme[start:end]
    # Drop rendered blocks and the generated model/leaderboard/statistics tables.
    for s, e in (
        (rrt.DEMAND_PROFILE_START, rrt.DEMAND_PROFILE_END),
        (rrt.EVAL_PROTOCOL_START, rrt.EVAL_PROTOCOL_END),
        (rrt.MODEL_LADDER_START, rrt.MODEL_LADDER_END),
        (rrt.LEADERBOARD_START, rrt.LEADERBOARD_END),
        (rrt.STATISTICS_START, rrt.STATISTICS_END),
        (rrt.ROUTER_START, rrt.ROUTER_END),
    ):
        text = re.sub(re.escape(s) + r".*?" + re.escape(e), "", text, flags=re.S)
    text = re.sub(r"\]\(#[^)]*\)", "]", text)  # anchor links carry the step numbers
    text = re.sub(r"^(#+ )?\d\. ", "", text, flags=re.M)  # step numbers in headings and the overview list
    text = re.sub(r"\bsteps? \d( to \d)?\b", "step", text)  # cross-references between steps
    text = re.sub(r"\bM5\b", "M-five", text)  # the dataset's name, not a number
    text = re.sub(r"\(Syntetos-Boylan-Croston, 2005\)", "", text)  # a citation year, not a result
    digits = re.findall(r"\d[\d.,]*", text)
    assert not digits, f"hand-typed numbers in the chain prose: {digits}"


def test_guide_00_chain_matches_the_readme_questions() -> None:
    text = (GUIDE / "00_START_HERE.md").read_text(encoding="utf-8")
    section = text.split("## 2. The Eight-Step Chain")[1].split("\n---")[0]
    rows = re.findall(r"^\| (\d) \| ([^|]+) \|", section, flags=re.M)
    assert [int(n) for n, _ in rows] == list(range(1, 9))
    assert [q.strip() for _, q in rows] == QUESTIONS
    assert not re.search(r"\d[\d.,]*%|\$\d", section), "the chain table must not carry result numbers"


def test_guide_00_links_in_the_chain_resolve() -> None:
    text = (GUIDE / "00_START_HERE.md").read_text(encoding="utf-8")
    section = text.split("## 2. The Eight-Step Chain")[1].split("\n---")[0]
    for target in re.findall(r"\]\(([^)#]+)\)", section):
        assert (GUIDE / target).resolve().exists(), target


def test_guide_09_margin_numbers_match_the_csvs() -> None:
    be = pd.read_csv(INVENTORY / "margin_break_even.csv").set_index("threshold")
    sens = pd.read_csv(INVENTORY / "margin_sensitivity.csv")
    text = (GUIDE / "09_RESULTS_AND_LIMITATIONS.md").read_text(encoding="utf-8")
    q10 = text.split("### Q10:")[1]
    b98 = be.loc["cheapest_98_over_95", "gross_margin"]
    b95 = be.loc["cheapest_95_over_90", "gross_margin"]
    x98 = be.loc["sl_star_reaches_98", "gross_margin"]
    assert f"98% is the cheapest above a {b98:.1%} margin" in q10
    assert f"95% between {b95:.1%} and {b98:.1%}" in q10
    assert f"90% below {b95:.1%}" in q10
    assert f"needs {x98:.1%}" in q10
    cats = sens[sens["scenario"] != "grid"]
    lo, hi = cats["sl_star"].min(), cats["sl_star"].max()
    assert f"{lo:.1%} to {hi:.1%} by category" in q10
    assert "99.5% to 99.7%" not in text  # the old range ignored HOBBIES at 45%
    grocery = sens[(sens["scenario"] == "grid") & sens["gross_margin"].between(0.25, 0.30)]
    assert (grocery["best_menu_level"] == 0.98).all()
    assert "At 25% to 30% grocery margins 98% is still the cheapest" in q10


def test_readme_margin_block_states_the_break_evens_and_the_range() -> None:
    body = rrt.render_margin_sensitivity()
    be = pd.read_csv(INVENTORY / "margin_break_even.csv").set_index("threshold")
    b98 = be.loc["cheapest_98_over_95", "gross_margin"]
    assert f"98% costs less than 95% above a {b98:.1%} gross margin" in body
    # The corrected range: HOBBIES at a 45% margin is the low end, not 99.5%.
    assert "99.4% (HOBBIES" in body
    assert "99.7% (FOODS" in body
    assert "Differentiating service levels by ABC/XYZ cannot lower cost" in body
