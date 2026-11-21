"""Shape-level tests for native deep forecasters; skipped when Torch is absent."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.deep_learning.pytorch_native import (  # noqa: E402
    WindowedTorchForecaster,
    build_native_model,
    make_windows,
)


def test_window_builder_keeps_direct_horizon_order():
    X, y = make_windows(np.arange(12, dtype=float), context_length=4, horizon=3)
    assert X.shape == (6, 4, 1)
    assert y.shape == (6, 3)
    assert y[0].tolist() == [4.0, 5.0, 6.0]


@pytest.mark.parametrize("architecture", ["mlp", "rnn", "lstm", "gru", "tcn", "seq2seq_attention"])
def test_native_architectures_emit_one_value_per_horizon(architecture):
    model = build_native_model(architecture, context_length=5, n_features=2, horizon=3, hidden_size=8)
    output = model(torch.zeros((4, 5, 2)))
    assert tuple(output.shape) == (4, 3)


def test_windowed_trainer_and_mc_dropout_produce_expected_shapes():
    values = np.sin(np.arange(50) / 3.0).astype(float)
    X, y = make_windows(values, context_length=6, horizon=2)
    model = WindowedTorchForecaster(
        "mlp", context_length=6, horizon=2, hidden_size=8, max_epochs=2, batch_size=16
    ).fit(X[:30], y[:30], X[30:], y[30:])

    assert model.predict(X[:3]).shape == (3, 2)
    assert model.predict_mc_dropout(X[:2], n_samples=3).shape == (3, 2, 2)
