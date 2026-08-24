"""PyTorch-native direct multi-horizon forecasters for the learning tier.

These models deliberately share one windowing, chronological-validation, and
training loop.  Comparing architectures with a different split or different
loss budget would answer a tuning-budget question rather than an architecture
question.  They predict non-negative unit demand over the whole horizon in one
forward pass; no validation path uses future actuals recursively.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


def make_windows(
    values: np.ndarray,
    context_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a chronological 1-D or 2-D sequence into direct forecast windows.

    When ``values`` has multiple columns, column zero is the target and the
    remaining columns are known inputs at each historical time step.  The
    caller must pass only information available at each origin; this utility
    never shuffles the windows.
    """
    sequence = np.asarray(values, dtype=np.float32)
    if sequence.ndim == 1:
        sequence = sequence[:, None]
    if sequence.ndim != 2:
        raise ValueError("values must have shape (time,) or (time, n_features)")
    if context_length < 1 or horizon < 1:
        raise ValueError("context_length and horizon must be positive")
    n_windows = len(sequence) - context_length - horizon + 1
    if n_windows <= 0:
        raise ValueError("sequence is too short for the requested context and horizon")

    X = np.stack([sequence[i : i + context_length] for i in range(n_windows)])
    y = np.stack(
        [sequence[i + context_length : i + context_length + horizon, 0] for i in range(n_windows)]
    )
    return X, y


class MLPForecastNet(nn.Module):
    def __init__(self, context_length: int, n_features: int, horizon: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(context_length * n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class RecurrentForecastNet(nn.Module):
    def __init__(
        self,
        cell: str,
        n_features: int,
        horizon: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        recurrent_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.recurrent = recurrent_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
            # torch's `dropout` argument applies BETWEEN stacked recurrent
            # layers, so with num_layers=1 it does nothing and torch emits a
            # UserWarning saying so. Passing 0.0 here is the honest value --
            # it documents that the recurrent stack is not regularised -- and
            # keeps the warning out of every test run.
            #
            # This does NOT disable MC dropout. The stochasticity that
            # `predict_mc_dropout` samples comes from `self.dropout` below,
            # which is a standalone module applied to the final hidden state
            # and stays active whenever the net is in train mode, regardless
            # of layer count.
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, horizon)

    def forward(self, x: Tensor) -> Tensor:
        sequence, _ = self.recurrent(x)
        return self.output(self.dropout(sequence[:, -1]))


class _TemporalBlock(nn.Module):
    """Causal residual TCN block; trailing crop removes right-padding leakage."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        kernel_size = 3
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.padding = padding

    def _causal(self, layer: nn.Conv1d, x: Tensor) -> Tensor:
        output = layer(x)
        return output[:, :, :-self.padding] if self.padding else output

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)
        out = torch.relu(self._causal(self.conv1, x))
        out = self.dropout(out)
        out = torch.relu(self._causal(self.conv2, out))
        out = self.dropout(out)
        return torch.relu(out + residual)


class TCNForecastNet(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            _TemporalBlock(n_features, hidden_size, dilation=1, dropout=dropout),
            _TemporalBlock(hidden_size, hidden_size, dilation=2, dropout=dropout),
            _TemporalBlock(hidden_size, hidden_size, dilation=4, dropout=dropout),
        )
        self.output = nn.Linear(hidden_size, horizon)

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.blocks(x.transpose(1, 2))
        return self.output(encoded[:, :, -1])


class Seq2SeqAttentionForecastNet(nn.Module):
    """Autoregressive decoder with dot-product attention over encoder states."""

    def __init__(self, n_features: int, horizon: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.GRU(n_features, hidden_size, batch_first=True)
        self.decoder = nn.GRU(1, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        encoded, hidden = self.encoder(x)
        decoder_input = torch.zeros((x.shape[0], 1, 1), device=x.device, dtype=x.dtype)
        forecasts: list[Tensor] = []
        for _ in range(self.horizon):
            decoded, hidden = self.decoder(decoder_input, hidden)
            # Score each encoder time step against the current decoder state.
            scores = torch.bmm(encoded, decoded.transpose(1, 2)).squeeze(-1)
            attention = torch.softmax(scores, dim=1).unsqueeze(1)
            context = torch.bmm(attention, encoded).squeeze(1)
            forecast = self.output(torch.cat([self.dropout(decoded.squeeze(1)), context], dim=1))
            forecasts.append(forecast.squeeze(1))
            decoder_input = forecast.unsqueeze(1)
        return torch.stack(forecasts, dim=1)


def build_native_model(
    architecture: str,
    context_length: int,
    n_features: int,
    horizon: int,
    hidden_size: int = 64,
    dropout: float = 0.10,
) -> nn.Module:
    """Create one of the six native architectures requested by the lab brief."""
    name = architecture.lower().replace("+", "_").replace(" ", "_")
    if name == "mlp":
        return MLPForecastNet(context_length, n_features, horizon, hidden_size, dropout)
    if name in {"rnn", "lstm", "gru"}:
        return RecurrentForecastNet(name, n_features, horizon, hidden_size, dropout)
    if name == "tcn":
        return TCNForecastNet(n_features, horizon, hidden_size, dropout)
    if name in {"seq2seq_attention", "seq2seq"}:
        return Seq2SeqAttentionForecastNet(n_features, horizon, hidden_size, dropout)
    raise KeyError("unknown native architecture; choose mlp, rnn, lstm, gru, tcn, or seq2seq_attention")


@dataclass
class TrainingHistory:
    """Per-epoch losses for the supplementary DL learning-curve figure."""

    rows: list[dict[str, float]] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class WindowedTorchForecaster:
    """Chronological trainer shared by all native MLP/RNN/TCN/Seq2Seq models."""

    def __init__(
        self,
        architecture: str,
        context_length: int,
        horizon: int,
        hidden_size: int = 64,
        dropout: float = 0.10,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        early_stopping_patience: int = 20,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.architecture = architecture
        self.context_length = int(context_length)
        self.horizon = int(horizon)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.early_stopping_patience = int(early_stopping_patience)
        self.random_state = int(random_state)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_: nn.Module | None = None
        self.n_features_: int | None = None
        self.history_ = TrainingHistory()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> WindowedTorchForecaster:
        """Fit using caller-supplied chronological train/validation windows.

        No random holdout is created internally. If validation data is supplied,
        it must be a later chronological block than training data.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        Xtr, ytr = self._validate_arrays(X_train, y_train)
        if Xtr.shape[1] != self.context_length or ytr.shape[1] != self.horizon:
            raise ValueError("window shapes do not match configured context_length and horizon")
        self.n_features_ = Xtr.shape[2]
        self.model_ = build_native_model(
            self.architecture,
            self.context_length,
            self.n_features_,
            self.horizon,
            self.hidden_size,
            self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
            batch_size=min(self.batch_size, len(Xtr)),
            shuffle=False,
        )

        validation = None
        if X_val is not None or y_val is not None:
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val must be supplied together")
            validation = self._validate_arrays(X_val, y_val)

        best_state: dict[str, Tensor] | None = None
        best_loss = float("inf")
        wait = 0
        self.history_ = TrainingHistory()
        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            train_losses: list[float] = []
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            row = {"epoch": float(epoch), "train_mse": float(np.mean(train_losses))}

            monitored = row["train_mse"]
            if validation is not None:
                self.model_.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(validation[0]).to(self.device)
                    yv = torch.from_numpy(validation[1]).to(self.device)
                    monitored = float(loss_fn(self.model_(xv), yv).cpu())
                row["validation_mse"] = monitored
            self.history_.rows.append(row)

            if monitored < best_loss - 1e-9:
                best_loss = monitored
                best_state = copy.deepcopy(self.model_.state_dict())
                wait = 0
            else:
                wait += 1
                if validation is not None and wait >= self.early_stopping_patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit before predict")
        values = self._validate_features(X)
        self.model_.eval()
        with torch.no_grad():
            result = self.model_(torch.from_numpy(values).to(self.device)).cpu().numpy()
        return np.clip(result, 0.0, None)

    def predict_mc_dropout(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Sample model uncertainty by retaining dropout at inference time."""
        if self.model_ is None:
            raise RuntimeError("call fit before predict_mc_dropout")
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        values = torch.from_numpy(self._validate_features(X)).to(self.device)
        was_training = self.model_.training
        self.model_.train()  # Only dropout is present; no BatchNorm statistics change.
        with torch.no_grad():
            samples = np.stack([self.model_(values).cpu().numpy() for _ in range(n_samples)])
        self.model_.train(was_training)
        return np.clip(samples, 0.0, None)

    def _validate_arrays(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = self._validate_features(X)
        target = np.asarray(y, dtype=np.float32)
        if target.ndim == 1:
            target = target[:, None]
        if target.ndim != 2 or len(target) != len(features):
            raise ValueError("y must have shape (n_windows, horizon)")
        if not np.isfinite(target).all():
            raise ValueError("target contains NaN or infinite values")
        return features, target

    def _validate_features(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=np.float32)
        if features.ndim != 3:
            raise ValueError("X must have shape (n_windows, context_length, n_features)")
        if not np.isfinite(features).all():
            raise ValueError("features contain NaN or infinite values")
        if self.n_features_ is not None and features.shape[2] != self.n_features_:
            raise ValueError("feature count differs from the fitted model")
        return features


class DeepEnsembleForecaster:
    """Independent native models for epistemic uncertainty via ensemble spread."""

    def __init__(self, factory: callable, n_members: int = 5) -> None:
        if n_members < 2:
            raise ValueError("n_members must be at least 2")
        self.factory = factory
        self.n_members = int(n_members)
        self.members_: list[WindowedTorchForecaster] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> DeepEnsembleForecaster:
        self.members_ = []
        for index in range(self.n_members):
            member = self.factory(index)
            member.fit(X_train, y_train, X_val, y_val)
            self.members_.append(member)
        return self

    def predict_samples(self, X: np.ndarray) -> np.ndarray:
        if not self.members_:
            raise RuntimeError("call fit before predict_samples")
        return np.stack([member.predict(X) for member in self.members_])
