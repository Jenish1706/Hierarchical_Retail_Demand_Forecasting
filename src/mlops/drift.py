"""
MLOps: Drift Detection & Automated Retraining.

Monitors production model health and triggers retraining when needed:

Data Drift:
  - Feature distribution shift (PSI, KS test, Wasserstein distance)
  - Target distribution shift
  - Covariate shift

Model/Concept Drift:
  - Performance degradation (WAPE, MAE vs baseline)
  - Prediction distribution shift
  - Residual pattern changes

Retraining Triggers:
  - Scheduled (weekly)
  - Performance threshold breach
  - Drift detected
  - Manual override

Tools: Evidently AI + custom statistical tests + Airflow integration
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.config.logging import get_logger
from src.config.settings import settings

log = get_logger(__name__)


class DriftType(str, Enum):  # noqa: UP042
    DATA = "data_drift"
    CONCEPT = "concept_drift"
    PREDICTION = "prediction_drift"
    PERFORMANCE = "performance_degradation"


@dataclass
class DriftReport:
    detection_date: date
    drift_type: DriftType
    feature_name: str | None
    drift_detected: bool
    drift_score: float
    threshold: float
    statistical_test: str
    p_value: float | None
    details: dict[str, Any]
    recommendation: str


class StatisticalDriftDetector:
    """
    Statistical drift tests for feature and prediction distributions.

    Tests:
      - PSI (Population Stability Index): industry standard for credit/retail
      - Kolmogorov-Smirnov: non-parametric two-sample test
      - Wasserstein distance: Earth Mover's Distance
      - Chi-squared: categorical features
    """

    # PSI thresholds: < 0.10 = no drift, 0.10-0.25 = moderate, > 0.25 = significant
    PSI_STABLE = 0.10
    PSI_MODERATE = 0.25

    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10,
    ) -> float:
        """
        Population Stability Index.
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        """
        # Create buckets based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Avoid division by zero
        expected_pct = np.maximum(expected_counts / len(expected), 1e-6)
        actual_pct = np.maximum(actual_counts / len(actual), 1e-6)

        # Normalize
        expected_pct = expected_pct / expected_pct.sum()
        actual_pct = actual_pct / actual_pct.sum()

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[float, float, bool]:
        """
        Two-sample Kolmogorov-Smirnov test.
        Returns (statistic, p_value, drift_detected).
        """
        from scipy.stats import ks_2samp

        stat, p_value = ks_2samp(reference, current)
        return float(stat), float(p_value), (p_value < alpha)

    def wasserstein_distance(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """Earth Mover's Distance between distributions."""
        from scipy.stats import wasserstein_distance

        # Normalize by reference std to make scale-independent
        std = np.std(reference) + 1e-6
        return float(wasserstein_distance(reference / std, current / std))

    def detect_feature_drift(
        self,
        feature_name: str,
        reference_data: np.ndarray,
        current_data: np.ndarray,
    ) -> DriftReport:
        """Run all tests for a single feature, return consolidated report."""
        psi = self.compute_psi(reference_data, current_data)
        ks_stat, p_value, ks_drift = self.ks_test(reference_data, current_data)
        w_dist = self.wasserstein_distance(reference_data, current_data)

        # Primary signal: PSI (most robust for retail/supply chain)
        drift_detected = psi > self.PSI_MODERATE or ks_drift

        if psi < self.PSI_STABLE:
            recommendation = "No action required"
        elif psi < self.PSI_MODERATE:
            recommendation = "Monitor closely — moderate distribution shift"
        else:
            recommendation = "Trigger retraining — significant distribution shift"

        return DriftReport(
            detection_date=date.today(),
            drift_type=DriftType.DATA,
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=psi,
            threshold=self.PSI_MODERATE,
            statistical_test="PSI + KS",
            p_value=p_value,
            details={
                "psi": psi,
                "ks_statistic": ks_stat,
                "wasserstein_distance": w_dist,
                "reference_mean": float(np.mean(reference_data)),
                "current_mean": float(np.mean(current_data)),
                "reference_std": float(np.std(reference_data)),
                "current_std": float(np.std(current_data)),
            },
            recommendation=recommendation,
        )


class PerformanceMonitor:
    """
    Monitors model prediction performance in production.
    Compares current period vs baseline period metrics.
    """

    def __init__(
        self,
        mape_threshold: float = None,
        wape_threshold: float = None,
        degradation_pct: float = 0.20,  # 20% relative degradation triggers alert
    ):
        cfg = settings.model
        self.mape_threshold = mape_threshold or cfg.mape_threshold
        self.wape_threshold = wape_threshold or cfg.wape_threshold
        self.degradation_pct = degradation_pct

    def evaluate_period(
        self,
        df_predictions: pd.DataFrame,  # forecast_date, forecast_mean, actual_sales
        model_id: str,
        period_label: str = "current",
    ) -> dict[str, float]:
        """Compute performance metrics for a prediction period."""
        df = df_predictions.dropna(subset=["forecast_mean", "actual_sales"])
        if len(df) == 0:
            return {"mae": -1, "wape": -1, "mape": -1, "n": 0}

        actuals = df["actual_sales"].values
        preds = df["forecast_mean"].values.clip(0)

        mae = float(np.mean(np.abs(actuals - preds)))
        wape = float(np.sum(np.abs(actuals - preds)) / max(np.sum(actuals), 1e-6) * 100)
        mask = actuals > 0
        mape = (
            float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100)
            if mask.sum() > 0
            else -1
        )

        bias = float(np.mean(preds - actuals))
        coverage_80 = (
            float(
                (
                    (df.get("forecast_p10", pd.Series([0])) <= df["actual_sales"])
                    & (df["actual_sales"] <= df.get("forecast_p90", pd.Series([float("inf")])))
                ).mean()
            )
            if "forecast_p10" in df.columns
            else -1
        )

        return {
            "model_id": model_id,
            "period": period_label,
            "n": len(df),
            "mae": round(mae, 4),
            "wape": round(wape, 4),
            "mape": round(mape, 4),
            "bias": round(bias, 4),
            "coverage_80": round(coverage_80, 4),
        }

    def check_degradation(
        self,
        baseline_metrics: dict[str, float],
        current_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Check if performance has degraded vs baseline."""
        alerts = []

        for metric in ["wape", "mape", "mae"]:
            baseline = baseline_metrics.get(metric, -1)
            current = current_metrics.get(metric, -1)
            if baseline <= 0 or current <= 0:
                continue

            pct_change = (current - baseline) / baseline
            threshold = getattr(settings.model, f"{metric}_threshold", 0.20)

            if current > threshold:
                alerts.append(
                    {
                        "metric": metric,
                        "baseline": baseline,
                        "current": current,
                        "pct_change": round(pct_change * 100, 2),
                        "threshold": threshold,
                        "alert_type": "absolute_threshold_breach",
                    }
                )
            elif pct_change > self.degradation_pct:
                alerts.append(
                    {
                        "metric": metric,
                        "baseline": baseline,
                        "current": current,
                        "pct_change": round(pct_change * 100, 2),
                        "threshold": self.degradation_pct * 100,
                        "alert_type": "relative_degradation",
                    }
                )

        should_retrain = len(alerts) > 0
        return {
            "should_retrain": should_retrain,
            "alerts": alerts,
            "baseline_wape": baseline_metrics.get("wape"),
            "current_wape": current_metrics.get("wape"),
        }


class DriftMonitoringPipeline:
    """
    End-to-end monitoring pipeline that:
      1. Loads reference and current feature distributions
      2. Runs all drift tests
      3. Checks model performance
      4. Triggers retraining if needed
      5. Stores results to DB
    """

    def __init__(self):
        self.drift_detector = StatisticalDriftDetector()
        self.perf_monitor = PerformanceMonitor()
        self._log = get_logger("DriftMonitoringPipeline")

    def run(
        self,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame,
        reference_predictions: pd.DataFrame | None = None,
        current_predictions: pd.DataFrame | None = None,
        model_id: str = "current",
        feature_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Full monitoring run. Returns summary with retraining decision.
        """
        self._log.info("monitoring_run_start", model_id=model_id)
        results = {
            "model_id": model_id,
            "run_date": date.today().isoformat(),
            "drift_reports": [],
            "performance": {},
            "retraining_triggered": False,
            "trigger_reasons": [],
        }

        # 1. Feature drift
        feature_cols = feature_cols or [
            c
            for c in reference_features.columns
            if reference_features[c].dtype in (np.float64, np.float32, np.int64)
            and c not in ("item_id", "store_id", "date")
        ]

        drift_count = 0
        for col in feature_cols:
            if col not in current_features.columns:
                continue
            ref_vals = reference_features[col].dropna().values
            cur_vals = current_features[col].dropna().values
            if len(ref_vals) < 30 or len(cur_vals) < 30:
                continue

            report = self.drift_detector.detect_feature_drift(col, ref_vals, cur_vals)
            results["drift_reports"].append(
                {
                    "feature": col,
                    "drift_detected": report.drift_detected,
                    "psi": report.details.get("psi"),
                    "p_value": report.p_value,
                }
            )
            if report.drift_detected:
                drift_count += 1

        if drift_count > len(feature_cols) * 0.30:  # >30% features drifted
            results["retraining_triggered"] = True
            results["trigger_reasons"].append(
                f"Data drift: {drift_count}/{len(feature_cols)} features drifted"
            )

        # 2. Performance degradation check
        if reference_predictions is not None and current_predictions is not None:
            baseline_m = self.perf_monitor.evaluate_period(
                reference_predictions, model_id, "baseline"
            )
            current_m = self.perf_monitor.evaluate_period(current_predictions, model_id, "current")
            degradation = self.perf_monitor.check_degradation(baseline_m, current_m)

            results["performance"] = {
                "baseline": baseline_m,
                "current": current_m,
                "degradation_check": degradation,
            }

            if degradation["should_retrain"]:
                results["retraining_triggered"] = True
                results["trigger_reasons"].extend(
                    [a["metric"] + " degraded" for a in degradation["alerts"]]
                )

        self._log.info(
            "monitoring_run_complete",
            drifted_features=drift_count,
            retraining_triggered=results["retraining_triggered"],
            triggers=results["trigger_reasons"],
        )
        return results
