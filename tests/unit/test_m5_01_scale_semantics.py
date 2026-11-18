"""
Regression tests for M5-01: Scale semantics.
Verifies:
1. Multi-origin RMSSE aggregation scores each (series, origin) group with its own scale.
2. A series sold on day 1 keeps its leading in-window zeros (does not trim mid-life zeros).
3. A series launching on day 500 has days 1-499 trimmed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.point_metrics import metrics_by_group, naive_scale
from src.models.statistical.intermittent import classify_demand


def test_multi_origin_rmsse_aggregation():
    """
    Test 1: A multi-origin aggregation test that fails on the old code.
    Series s1 is evaluated across 2 rolling origins.
    Origin 0: scale = 1.0, errors = 1.0 -> MSE = 1.0, RMSSE = 1.0.
    Origin 1: scale = 4.0, errors = 2.0 -> MSE = 4.0, RMSSE = 1.0.
    True per-(series, origin) mean RMSSE = (1.0 + 1.0) / 2 = 1.0.
    Old code took scale from origin 0 (1.0) and pooled errors across origins (MSE = 2.5),
    yielding sqrt(2.5 / 1.0) = 1.5811.
    """
    df = pd.DataFrame({
        "series_id": ["s1", "s1", "s1", "s1"],
        "origin": [0, 0, 1, 1],
        "model": ["test_model"] * 4,
        "y_true": [0.0, 0.0, 0.0, 0.0],
        "y_pred": [1.0, 1.0, 2.0, 2.0],
        "scale": [1.0, 1.0, 4.0, 4.0],
    })
    res = metrics_by_group(df, ["model"], scale_col="scale")
    assert res.iloc[0]["rmsse"] == pytest.approx(1.0)


def test_series_sold_on_day_1_keeps_leading_in_window_zeros():
    """
    Test 2: A series sold on day 1 keeps its leading in-window zeros.
    The series launched on day 1 (so first_sale_idx <= 0 relative to this window).
    The window has leading zeros [0, 0, 1, 3, 6].
    Because the item was already launched long ago, these leading zeros are real zero-demand days,
    not pre-launch inactive days, and must NOT be trimmed.
    """
    window = np.array([0.0, 0.0, 1.0, 3.0, 6.0])
    # diffs with zeros kept: [0, 1, 2, 3] -> squared: [0, 1, 4, 9] -> mean = 14 / 4 = 3.5
    scale_kept = naive_scale(window, first_sale_idx=0)
    assert scale_kept == pytest.approx(3.5)

    cls_kept = classify_demand(window, first_sale_idx=0)
    assert cls_kept.n_periods == 5
    assert cls_kept.zero_fraction == pytest.approx(2.0 / 5.0)


def test_series_launching_on_day_500_has_days_1_to_499_trimmed():
    """
    Test 3: A series launching on day 500 has days 1-499 trimmed.
    Days 1 to 499 (indices 0 to 498) are pre-launch inactive zeros.
    First sale is at day 500 (index 499).
    """
    y = np.zeros(1000)
    # Day 500 (index 499) onward has sales
    y[499] = 1.0
    y[500] = 3.0
    y[501] = 6.0
    y[502:] = 6.0

    # first_sale_idx=499 trims days 0..498 (days 1 to 499)
    scale_trimmed = naive_scale(y, first_sale_idx=499)
    expected_scale = naive_scale(y[499:], trim_pre_launch=False)
    assert scale_trimmed == pytest.approx(expected_scale)

    cls_trimmed = classify_demand(y, first_sale_idx=499)
    assert cls_trimmed.n_periods == 501
