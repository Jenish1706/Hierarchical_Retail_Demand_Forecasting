"""OOF-only forecast combinations and residual-correlation diagnostics."""

from src.models.ensembles.oof_ensembles import (
    ConstrainedWeightedBlend,
    EnsembleDiagnostics,
    OOFMatrix,
    RidgeStackingEnsemble,
    SimpleAverageEnsemble,
    best_cross_family_blend,
    ensemble_diagnostics,
    oof_prediction_matrix,
)

__all__ = [
    "ConstrainedWeightedBlend",
    "EnsembleDiagnostics",
    "OOFMatrix",
    "RidgeStackingEnsemble",
    "SimpleAverageEnsemble",
    "best_cross_family_blend",
    "ensemble_diagnostics",
    "oof_prediction_matrix",
]
