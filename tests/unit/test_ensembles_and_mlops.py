"""Tests that enforce OOF-only ensemble fitting and promotion policy semantics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mlops.registry import PromotionGate
from src.models.ensembles import (
    ConstrainedWeightedBlend,
    RidgeStackingEnsemble,
    ensemble_diagnostics,
    oof_prediction_matrix,
)


def _oof_predictions() -> pd.DataFrame:
    rows = []
    for origin in range(2):
        for series in ("a", "b"):
            for day, actual in enumerate((2.0, 4.0, 6.0), start=1):
                rows.append(
                    {"origin": origin, "series_id": series, "date": f"2016-01-{origin * 3 + day:02d}", "model": "strong", "y_true": actual, "y_pred": actual + 0.2}
                )
                rows.append(
                    {"origin": origin, "series_id": series, "date": f"2016-01-{origin * 3 + day:02d}", "model": "diverse", "y_true": actual, "y_pred": actual + (-1.0 if day % 2 else 0.7)}
                )
    return pd.DataFrame(rows)


class TestOOFEnsembles:
    def test_oof_matrix_rejects_missing_origin_evidence(self):
        with pytest.raises(KeyError, match="OOF ensemble"):
            oof_prediction_matrix(_oof_predictions().drop(columns="origin"))

    def test_weighted_blend_has_valid_simplex_weights(self):
        predictions = _oof_predictions()
        blend = ConstrainedWeightedBlend().fit_oof(predictions)
        weights = blend.weight_table()

        assert weights["weight"].between(0.0, 1.0).all()
        assert weights["weight"].sum() == pytest.approx(1.0)
        assert len(blend.predict(pd.DataFrame({"strong": [2.0], "diverse": [3.0]}))) == 1

    def test_stacking_and_diagnostics_use_the_same_oof_rows(self):
        predictions = _oof_predictions()
        matrix = oof_prediction_matrix(predictions)
        diagnostics = ensemble_diagnostics(matrix)
        stacked = RidgeStackingEnsemble().fit_oof(predictions)

        assert diagnostics.residual_correlation.shape == (2, 2)
        out = stacked.predict(pd.DataFrame({"strong": [3.0], "diverse": [4.0]}))
        assert out.shape == (1,)
        assert out[0] >= 0.0


class TestPromotionGate:
    def test_first_candidate_is_promotable(self):
        promoted, improvement, reason = PromotionGate.assess(10.0, None, 2.0)
        assert promoted
        assert improvement is None
        assert reason == "no_incumbent_production_model"

    def test_small_gain_stays_in_staging_unless_forced(self):
        promoted, improvement, _ = PromotionGate.assess(9.9, 10.0, 2.0)
        assert not promoted
        assert improvement == pytest.approx(1.0)

        forced, _, reason = PromotionGate.assess(9.9, 10.0, 2.0, force=True)
        assert forced
        assert reason == "forced"
