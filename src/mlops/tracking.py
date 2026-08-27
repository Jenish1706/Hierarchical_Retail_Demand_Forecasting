"""Small, failure-tolerant experiment tracking adapter for research runs.

Tracking is deliberately a thin layer around MLflow rather than a prerequisite
for a forecasting experiment.  A local checkout should still be able to run a
backtest when no tracking server is available; in that case this module falls
back to a local SQLite store and records the effective URI in the log.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from src.config.logging import get_logger
from src.config.settings import settings


@dataclass(frozen=True)
class TrackingStatus:
    """Resolved tracking state, suitable for a run manifest."""

    available: bool
    tracking_uri: str | None
    experiment_name: str
    reason: str | None = None


class ExperimentTracker:
    """Configure MLflow once and provide safe, no-op-aware logging helpers.

    The adapter owns the server reachability check.  Without it, MLflow can
    retry an unavailable HTTP endpoint for minutes, which is unacceptable in a
    reproducible notebook or DVC stage where tracking is ancillary to the
    experiment itself.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self._configured_uri = tracking_uri or settings.mlflow.tracking_uri
        self._base_dir = base_dir or settings.base_dir
        self._log = get_logger("ExperimentTracker")
        self._mlflow: Any | None = None
        self.status = self._configure()

    @property
    def available(self) -> bool:
        return self.status.available

    @property
    def mlflow(self) -> Any:
        """Configured MLflow module, raising clearly when tracking is disabled."""
        if self._mlflow is None:
            raise RuntimeError(f"MLflow is unavailable: {self.status.reason or 'unknown reason'}")
        return self._mlflow

    def _resolve_uri(self) -> str:
        """Use SQLite when a configured HTTP tracking service is unreachable."""
        uri = self._configured_uri
        if not uri.startswith(("http://", "https://")):
            return uri

        parsed = urlparse(uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return uri
        except OSError:
            db_path = self._base_dir / "mlflow.db"
            fallback = f"sqlite:///{db_path.as_posix()}"
            self._log.warning(
                "mlflow_server_unreachable",
                configured_uri=uri,
                falling_back_to=fallback,
                consequence="experiment remains locally tracked",
            )
            return fallback

    def _configure(self) -> TrackingStatus:
        try:
            import mlflow

            uri = self._resolve_uri()
            mlflow.set_tracking_uri(uri)
            full_name = f"{settings.mlflow.experiment_prefix}/{self.experiment_name}"
            mlflow.set_experiment(full_name)
            self._mlflow = mlflow
            self._log.info("mlflow_ready", tracking_uri=uri, experiment=full_name)
            return TrackingStatus(True, uri, full_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            self._log.warning("mlflow_unavailable", error=str(exc))
            return TrackingStatus(False, None, self.experiment_name, str(exc))

    @contextmanager
    def run(
        self,
        run_name: str,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[Any | None]:
        """Yield an active MLflow run, or ``None`` when tracking is unavailable."""
        if not self.available:
            yield None
            return

        try:
            active_context = self.mlflow.start_run(run_name=run_name)
        except Exception as exc:  # pragma: no cover - external tracking state
            self._log.warning("mlflow_run_failed", run_name=run_name, error=str(exc))
            yield None
            return

        # Do not catch exceptions raised by the caller's experiment body here:
        # yielding a second time from a context manager masks the real failure
        # with ``RuntimeError: generator didn't stop``.
        with active_context as active_run:
            if tags:
                self.mlflow.set_tags({key: str(value) for key, value in tags.items()})
            yield active_run

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log only scalar parameters: MLflow parameters must be serialisable."""
        if not self.available:
            return
        safe = {
            key: value
            for key, value in params.items()
            if isinstance(value, (str, int, float, bool))
        }
        if safe:
            self.mlflow.log_params(safe)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log finite metric values only; NaN is a valid forecast outcome, not a metric."""
        if not self.available:
            return
        for name, value in metrics.items():
            if value is None or not np.isfinite(float(value)):
                continue
            self.mlflow.log_metric(name, float(value), step=step)
