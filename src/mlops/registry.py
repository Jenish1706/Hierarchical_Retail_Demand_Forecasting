"""Evidence-gated MLflow model promotion for the M5 research laboratory.

The registry does not decide that a model is scientifically better merely
because it tops a leaderboard.  It promotes only a completed, tracked run and
requires a configurable relative improvement over the incumbent.  Statistical
significance belongs in the results analysis; the operational gate protects
against churn caused by small nested-CV differences.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.config.logging import get_logger
from src.config.settings import settings


@dataclass(frozen=True)
class PromotionCandidate:
    """Minimal immutable evidence required to register a trained model."""

    run_id: str
    algorithm: str
    cv_wape: float
    cv_rmsse: float | None
    cv_scheme: str


@dataclass(frozen=True)
class PromotionDecision:
    """Auditable outcome of a registry decision."""

    promoted: bool
    reason: str
    model_name: str | None = None
    version: str | None = None
    stage: str | None = None
    algorithm: str | None = None
    challenger_wape: float | None = None
    incumbent_wape: float | None = None
    improvement_pct: float | None = None
    required_improvement_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromotionGate:
    """Register a challenger and promote only when its CV evidence clears a gate."""

    def __init__(
        self,
        model_name: str | None = None,
        min_improvement_pct: float | None = None,
        mlflow_module: Any | None = None,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name or settings.mlflow.registered_model_name
        self.min_improvement_pct = (
            settings.model.promotion_min_improvement_pct
            if min_improvement_pct is None
            else float(min_improvement_pct)
        )
        self._mlflow = mlflow_module
        self._client = client
        self._log = get_logger("PromotionGate")

    @staticmethod
    def assess(
        challenger_wape: float,
        incumbent_wape: float | None,
        min_improvement_pct: float,
        force: bool = False,
    ) -> tuple[bool, float | None, str]:
        """Return the pure, unit-testable promotion rule before any registry I/O."""
        if incumbent_wape is None:
            return True, None, "no_incumbent_production_model"
        if incumbent_wape <= 0:
            return False, None, "incumbent_wape_invalid_manual_review_required"

        improvement = 100.0 * (incumbent_wape - challenger_wape) / incumbent_wape
        if improvement >= min_improvement_pct:
            return True, improvement, "beat_incumbent"
        if force:
            return True, improvement, "forced"
        return False, improvement, (
            f"improvement {improvement:.2f}% below required {min_improvement_pct:.2f}%"
        )

    def promote(self, candidate: PromotionCandidate, force: bool = False) -> PromotionDecision:
        """Register ``candidate`` and return a decision record; registry outages do not raise."""
        try:
            mlflow, client = self._resolve_clients()
            version = mlflow.register_model(
                model_uri=f"runs:/{candidate.run_id}/model",
                name=self.model_name,
                tags={
                    "algorithm": candidate.algorithm,
                    "cv_wape": f"{candidate.cv_wape:.6f}",
                    "cv_rmsse": "" if candidate.cv_rmsse is None else f"{candidate.cv_rmsse:.6f}",
                    "cv_scheme": candidate.cv_scheme,
                },
            )
            client.set_model_version_tag(
                self.model_name, version.version, "cv_wape", f"{candidate.cv_wape:.6f}"
            )
            incumbent = self._current_production_version(client)
            incumbent_wape = self._version_wape(client, incumbent) if incumbent is not None else None
            if incumbent is not None and incumbent_wape is None:
                return self._transition(
                    client,
                    version.version,
                    "Staging",
                    candidate,
                    "incumbent_wape_unrecorded_manual_review_required",
                    incumbent_wape=None,
                )

            passes, improvement, reason = self.assess(
                candidate.cv_wape,
                incumbent_wape,
                self.min_improvement_pct,
                force,
            )
            stage = "Production" if passes else "Staging"
            decision = self._transition(
                client,
                version.version,
                stage,
                candidate,
                reason,
                incumbent_wape=incumbent_wape,
                improvement_pct=improvement,
            )
            if passes and incumbent is not None:
                # Keep exactly one live version without deleting reproducible history.
                client.transition_model_version_stage(
                    name=self.model_name,
                    version=incumbent.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
            self._log.info("promotion_decision", **decision.to_dict())
            return decision
        except Exception as exc:  # pragma: no cover - external registry state
            self._log.warning("promotion_gate_failed", error=str(exc))
            return PromotionDecision(False, f"registry_error: {exc}", algorithm=candidate.algorithm)

    def _resolve_clients(self) -> tuple[Any, Any]:
        mlflow = self._mlflow
        client = self._client
        if mlflow is None:
            import mlflow

        if client is None:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
        return mlflow, client

    def _current_production_version(self, client: Any) -> Any | None:
        for version in client.search_model_versions(f"name='{self.model_name}'"):
            if getattr(version, "current_stage", None) == "Production":
                return version
        return None

    @staticmethod
    def _version_wape(client: Any, version: Any) -> float | None:
        tags = getattr(version, "tags", {}) or {}
        if "cv_wape" in tags:
            try:
                return float(tags["cv_wape"])
            except (TypeError, ValueError):
                pass
        try:
            value = client.get_run(version.run_id).data.metrics.get("cv_wape")
            return None if value is None else float(value)
        except Exception:
            return None

    def _transition(
        self,
        client: Any,
        version: str,
        stage: str,
        candidate: PromotionCandidate,
        reason: str,
        incumbent_wape: float | None,
        improvement_pct: float | None = None,
    ) -> PromotionDecision:
        client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage,
            archive_existing_versions=False,
        )
        return PromotionDecision(
            promoted=stage == "Production",
            reason=reason,
            model_name=self.model_name,
            version=str(version),
            stage=stage,
            algorithm=candidate.algorithm,
            challenger_wape=round(candidate.cv_wape, 6),
            incumbent_wape=None if incumbent_wape is None else round(incumbent_wape, 6),
            improvement_pct=None if improvement_pct is None else round(improvement_pct, 6),
            required_improvement_pct=self.min_improvement_pct,
        )
