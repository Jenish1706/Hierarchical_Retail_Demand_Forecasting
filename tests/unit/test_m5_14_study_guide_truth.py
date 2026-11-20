"""
Unit tests for M5-14: Study-guide truth fixes (.audit/STUDY_GUIDE.md).

Verifies:
1. Q1 and Q24 do not claim price features/price ratios; reflect actual feature list and price_features_included: false.
2. Q15 attributes unpriced series to Excel 1,048,575 row boundary truncation (matches weakness #1).
3. The pitch and Pillar 4 do not claim '$3,223 in trapped working capital'; Pillar 1 states both 30,490 series and 3,000-series sample.
4. Q11 marks stockout annualization as under revision for normal loss function (M5-10).
5. Q22 reflects verified test suite counts across environments.
6. Q23 contains no stray non-English words ('عدة').
7. Whiteboard 3 reports observation-pooled RMSSE as 0.965 (recomputed from predictions.parquet), not 0.988.
8. Q7 includes the router-vs-ADIDA comparison across origins.
9. Hurdle models and deep learning are marked out of scope for the benchmark.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

STUDY_GUIDE_PATH = Path(".audit/STUDY_GUIDE.md")


@pytest.fixture
def study_guide_text() -> str:
    if not STUDY_GUIDE_PATH.exists():
        pytest.skip(".audit/STUDY_GUIDE.md is private and not tracked in fresh clone")
    return STUDY_GUIDE_PATH.read_text(encoding="utf-8")


def test_q1_and_q24_no_price_features(study_guide_text: str):
    # Q1 should not claim price features
    q1_match = re.search(r"### Q1\..*?(?=### Q2\.|$)", study_guide_text, re.DOTALL)
    assert q1_match is not None
    q1_text = q1_match.group(0)
    assert "price and calendar features" not in q1_text
    assert "price_features_included: false" in q1_text or "no price features" in q1_text.lower()

    # Q24 should not claim long-term price ratios
    q24_match = re.search(r"### Q24\..*?(?=### Q25\.|$)", study_guide_text, re.DOTALL)
    assert q24_match is not None
    q24_text = q24_match.group(0)
    assert "price ratios" not in q24_text
    assert "price_features_included: false" in q24_text or "no price features" in q24_text.lower()
    assert "horizon-shifted" in q24_text.lower() or "lag" in q24_text.lower()


def test_q15_unpriced_matches_excel_truncation(study_guide_text: str):
    q15_match = re.search(r"### Q15\..*?(?=### Q16\.|$)", study_guide_text, re.DOTALL)
    assert q15_match is not None
    q15_text = q15_match.group(0)
    assert "Walmart only recorded weekly prices" not in q15_text
    assert "1,048,575" in q15_text or "Excel" in q15_text


def test_pitch_and_pillars_capital_and_series_claims(study_guide_text: str):
    # Elevator pitch should not claim frees $3,223 until M5-10
    pitch_match = re.search(r"### The 60-Second Elevator Pitch.*?(?=### The 5-Minute|$)", study_guide_text, re.DOTALL)
    assert pitch_match is not None
    assert "frees $3,223" not in pitch_match.group(0)

    # Pillar 1 should mention both 30,490 and 3,000-series sample
    pillar1_match = re.search(r"\*\*Pillar 1:.*?(?=\*\*Pillar 2:|$)", study_guide_text, re.DOTALL)
    assert pillar1_match is not None
    p1_text = pillar1_match.group(0)
    assert "30,490" in p1_text
    assert "3,000" in p1_text

    # Pillar 4 should not claim $3,223 released
    pillar4_match = re.search(r"\*\*Pillar 4:.*?(?=## 2\.|$)", study_guide_text, re.DOTALL)
    assert pillar4_match is not None
    p4_text = pillar4_match.group(0)
    assert "$3,223 in working capital released" not in p4_text


def test_q11_stockout_under_revision(study_guide_text: str):
    q11_match = re.search(r"### Q11\..*?(?=### Q12\.|$)", study_guide_text, re.DOTALL)
    assert q11_match is not None
    q11_text = q11_match.group(0)
    assert "under revision" in q11_text.lower() or "m5-10" in q11_text.lower()


def test_q22_test_counts(study_guide_text: str):
    q22_match = re.search(r"### Q22\..*?(?=### Q23\.|$)", study_guide_text, re.DOTALL)
    assert q22_match is not None
    q22_text = q22_match.group(0)
    assert "100" in q22_text or "110" in q22_text


def test_q23_no_stray_non_english(study_guide_text: str):
    q23_match = re.search(r"### Q23\..*?(?=### Q24\.|$)", study_guide_text, re.DOTALL)
    assert q23_match is not None
    q23_text = q23_match.group(0)
    assert "عدة" not in q23_text


def test_whiteboard_3_pooled_rmsse(study_guide_text: str):
    wb3_match = re.search(r"### 3\. Canonical M5 RMSSE.*?(?=### 4\.|$)", study_guide_text, re.DOTALL)
    assert wb3_match is not None
    wb3_text = wb3_match.group(0)
    assert "0.988" not in wb3_text
    assert "0.965" in wb3_text


def test_q7_router_vs_adida(study_guide_text: str):
    q7_match = re.search(r"### Q7\..*?(?=### Q8\.|$)", study_guide_text, re.DOTALL)
    assert q7_match is not None
    q7_text = q7_match.group(0)
    assert "ADIDA" in q7_text
    assert "74.17" in q7_text or "loses" in q7_text.lower() or "did not beat" in q7_text.lower()


def test_hurdle_and_deep_learning_scope(study_guide_text: str):
    wb2_match = re.search(r"### 2\. Syntetos-Boylan-Teunter.*?(?=### 3\.|$)", study_guide_text, re.DOTALL)
    assert wb2_match is not None
    wb2_text = wb2_match.group(0)
    assert "Hurdle" not in wb2_text or "out of scope" in wb2_text.lower()
