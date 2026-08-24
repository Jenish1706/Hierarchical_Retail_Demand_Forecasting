"""Adapter for post-M5 neuralforecast architectures under one data contract.

N-BEATS, N-HiTS, DeepAR, TFT, PatchTST, DLinear, and TiDE use the same long
panel contract: ``unique_id``, ``ds``, and ``y``.  The adapter deliberately
does not choose a winner or tune each model to a different budget; callers
pass the shared horizon, input window, and maximum training steps from the
experiment configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


SUPPORTED_NEURALFORECAST_MODELS = {
    "nbeats": "NBEATS",
    "nhits": "NHITS",
    "deepar": "DeepAR",
    "tft": "TFT",
    "patchtst": "PatchTST",
    "dlinear": "DLinear",
    "tide": "TiDE",
}


@dataclass(frozen=True)
class LibraryModelSpec:
    """The fixed common budget that makes the library-model comparison fair."""

    name: str
    horizon: int
    input_size: int
    max_steps: int
    batch_size: int
    learning_rate: float


class NeuralForecastAdapter:
    """Fit and forecast one neuralforecast model through a stable local API."""

    def __init__(
        self,
        name: str,
        horizon: int,
        input_size: int,
        max_steps: int = 500,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_seed: int = 42,
        **model_kwargs: Any,
    ) -> None:
        normalized = name.lower()
        if normalized not in SUPPORTED_NEURALFORECAST_MODELS:
            raise KeyError(f"unknown library model {name!r}; choose {sorted(SUPPORTED_NEURALFORECAST_MODELS)}")
        self.spec = LibraryModelSpec(
            name=normalized,
            horizon=int(horizon),
            input_size=int(input_size),
            max_steps=int(max_steps),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
        )
        self.random_seed = int(random_seed)
        self.model_kwargs = model_kwargs
        self.forecaster_: Any | None = None
        self.output_column_: str | None = None

    def fit(self, panel: pd.DataFrame, validation_size: int = 0) -> NeuralForecastAdapter:
        """Fit on a chronologically sorted M5-format panel.

        ``validation_size`` denotes the final chronological tail of every
        series. It is passed through to neuralforecast rather than selected at
        random, preserving the project's chronological-split rule.
        """
        required = {"unique_id", "ds", "y"}
        missing = required - set(panel.columns)
        if missing:
            raise KeyError(f"panel is missing neuralforecast columns: {sorted(missing)}")
        if validation_size < 0:
            raise ValueError("validation_size must be non-negative")
        frame = panel[["unique_id", "ds", "y"]].copy()
        frame["ds"] = pd.to_datetime(frame["ds"])
        frame = frame.sort_values(["unique_id", "ds"], kind="mergesort")
        if frame.duplicated(["unique_id", "ds"]).any():
            raise ValueError("panel contains duplicate series/date rows")

        try:
            from neuralforecast import NeuralForecast
            import neuralforecast.models as models
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ImportError("install neuralforecast to use library deep-learning models") from exc

        class_name = SUPPORTED_NEURALFORECAST_MODELS[self.spec.name]
        model_class = getattr(models, class_name)
        model = model_class(
            h=self.spec.horizon,
            input_size=self.spec.input_size,
            max_steps=self.spec.max_steps,
            batch_size=self.spec.batch_size,
            learning_rate=self.spec.learning_rate,
            random_seed=self.random_seed,
            **self.model_kwargs,
        )
        self.forecaster_ = NeuralForecast(models=[model], freq="D")
        self.forecaster_.fit(df=frame, val_size=validation_size)
        self.output_column_ = class_name
        return self

    def predict(self, future_exogenous: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return long ``unique_id, ds, y_pred`` forecasts ready for the harness."""
        if self.forecaster_ is None or self.output_column_ is None:
            raise RuntimeError("call fit before predict")
        raw = self.forecaster_.predict(futr_df=future_exogenous)
        if self.output_column_ not in raw.columns:
            candidates = [c for c in raw.columns if c not in {"unique_id", "ds"}]
            if len(candidates) != 1:
                raise RuntimeError(f"cannot identify model prediction column from {list(raw.columns)}")
            self.output_column_ = candidates[0]
        return raw[["unique_id", "ds", self.output_column_]].rename(
            columns={self.output_column_: "y_pred"}
        )
