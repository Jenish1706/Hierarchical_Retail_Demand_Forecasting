"""
Unit tests for M5-11: Price-feature caveat and global model handicaps.

Verifies:
1. README Limitation 1 states the true cause (Excel row-limit truncation) and the three handicaps:
   - no price features (price_features_included: false)
   - 3,000-series training sample
   - 900-day history
2. README leaderboard footnote discloses the global model handicaps.
3. docs/guide/09_RESULTS_AND_LIMITATIONS.md documents the three handicaps.
4. .audit/STUDY_GUIDE.md documents the three handicaps in the weaknesses section.
5. .audit/INTERVIEW_QA.md includes the interview question 'Why does LightGBM lose here when it won M5?'.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

README_PATH = Path("README.md")
RESULTS_LIMITATIONS_PATH = Path("docs/guide/09_RESULTS_AND_LIMITATIONS.md")
STUDY_GUIDE_PATH = Path(".audit/STUDY_GUIDE.md")
INTERVIEW_QA_PATH = Path(".audit/INTERVIEW_QA.md")


def test_readme_limitation_1_discloses_handicaps():
    assert README_PATH.exists()
    content = README_PATH.read_text(encoding="utf-8")

    # Limitation 1 check
    limitation_match = re.search(r"## Limitations.*?(?=## License|$)", content, re.DOTALL)
    assert limitation_match is not None, "Limitations section not found in README"
    lim_text = limitation_match.group(0)

    # Must state Excel row-limit truncation, no price features, 3,000 series, 900 days
    assert "Excel" in lim_text or "1,048,575" in lim_text
    assert "no price features" in lim_text.lower() or "price_features_included" in lim_text
    assert "3,000" in lim_text
    assert "900" in lim_text


def test_readme_leaderboard_footnote_discloses_handicaps():
    content = README_PATH.read_text(encoding="utf-8")
    table_match = re.search(r"<!-- BEGIN_LEADERBOARD_TABLE -->.*?<!-- END_LEADERBOARD_TABLE -->", content, re.DOTALL)
    assert table_match is not None
    table_text = table_match.group(0)

    assert "no price" in table_text.lower() or "price_features_included: false" in table_text.lower()
    assert "3,000" in table_text
    assert "900" in table_text


def test_guide_09_discloses_handicaps():
    assert RESULTS_LIMITATIONS_PATH.exists()
    content = RESULTS_LIMITATIONS_PATH.read_text(encoding="utf-8")
    assert "Excel" in content or "1,048,575" in content
    assert "no price" in content.lower() or "price_features_included" in content
    assert "3,000" in content
    assert "900" in content


def test_study_guide_weaknesses_disclose_handicaps():
    if not STUDY_GUIDE_PATH.exists():
        pytest.skip(".audit/STUDY_GUIDE.md is private and not tracked in fresh clone")
    content = STUDY_GUIDE_PATH.read_text(encoding="utf-8")
    weakness_match = re.search(r"## 4\. Remaining Weaknesses.*", content, re.DOTALL)
    assert weakness_match is not None
    w_text = weakness_match.group(0)
    assert "no price" in w_text.lower() or "price_features_included" in w_text
    assert "3,000" in w_text
    assert "900" in w_text


def test_interview_qa_why_lightgbm_loses():
    if not INTERVIEW_QA_PATH.exists():
        pytest.skip(".audit/INTERVIEW_QA.md is private and not tracked in fresh clone")
    content = INTERVIEW_QA_PATH.read_text(encoding="utf-8")
    assert "Why does LightGBM lose here when it won M5?" in content or "why did lightgbm lose here when it won m5" in content.lower()
