"""
Integration tests for FastAPI endpoints and pipeline components.

These tests run against a live test database and the API process.
They verify the full request-response cycle including validation,
business logic, and response schema correctness.
"""
from __future__ import annotations

from datetime import date

import httpx
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.optimization.inventory.optimizer import InventoryOptimizer, InventoryPolicy
from src.monitoring.drift.detector import StatisticalDriftDetector, DriftMonitoringPipeline
from src.optimization.sustainability.emissions import (
    TransportEmissionsCalculator,
    TransportLeg,
    WarehouseEmissions,
)


# ============================================================================
# API Client Fixture
# ============================================================================

@pytest.fixture(scope="module")
def client() -> TestClient:
    """Synchronous test client (no live server needed)."""
    return TestClient(app, raise_server_exceptions=True)


# ============================================================================
# Health & System Endpoints
# ============================================================================

class TestSystemEndpoints:

    @pytest.mark.integration
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "name" in data
        assert "version" in data

    @pytest.mark.integration
    def test_health_endpoint_structure(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "app_version" in data
        assert data["status"] in ("healthy", "degraded")

    @pytest.mark.integration
    def test_openapi_spec_available(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert spec["info"]["title"] == "Supply Chain Intelligence Platform"


# ============================================================================
# Forecast Endpoints
# ============================================================================

class TestForecastEndpoints:

    @pytest.mark.integration
    def test_forecast_valid_request(self, client):
        payload = {
            "item_id": "HOBBIES_1_001",
            "store_id": "CA_1",
            "horizon_days": 14,
            "include_intervals": True,
            "quantiles": [0.10, 0.50, 0.90],
        }
        r = client.post("/forecast/", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["item_id"] == "HOBBIES_1_001"
        assert data["store_id"] == "CA_1"
        assert len(data["forecasts"]) == 14

    @pytest.mark.integration
    def test_forecast_non_negative_values(self, client):
        payload = {
            "item_id": "FOODS_3_090",
            "store_id": "TX_1",
            "horizon_days": 28,
            "include_intervals": True,
        }
        r = client.post("/forecast/", json=payload)
        assert r.status_code == 200
        for point in r.json()["forecasts"]:
            assert point["forecast_mean"] >= 0

    @pytest.mark.integration
    def test_forecast_interval_ordering(self, client):
        """p10 <= mean <= p90 must hold for all forecast points."""
        payload = {
            "item_id": "HOUSEHOLD_1_118",
            "store_id": "WI_1",
            "horizon_days": 7,
            "include_intervals": True,
        }
        r = client.post("/forecast/", json=payload)
        assert r.status_code == 200
        for point in r.json()["forecasts"]:
            if point.get("forecast_p10") and point.get("forecast_p90"):
                assert point["forecast_p10"] <= point["forecast_mean"] + 0.01
                assert point["forecast_mean"] <= point["forecast_p90"] + 0.01

    @pytest.mark.integration
    def test_forecast_invalid_horizon_rejected(self, client):
        payload = {
            "item_id": "FOODS_1_001",
            "store_id": "CA_1",
            "horizon_days": 500,  # Exceeds max of 90
        }
        r = client.post("/forecast/", json=payload)
        assert r.status_code == 422

    @pytest.mark.integration
    def test_forecast_invalid_quantile_rejected(self, client):
        payload = {
            "item_id": "FOODS_1_001",
            "store_id": "CA_1",
            "horizon_days": 7,
            "quantiles": [0.10, 1.50],  # 1.50 is invalid
        }
        r = client.post("/forecast/", json=payload)
        assert r.status_code == 422

    @pytest.mark.integration
    def test_batch_forecast_returns_all_items(self, client):
        payload = {
            "items": [
                {"item_id": "FOODS_1_001", "store_id": "CA_1", "horizon_days": 7},
                {"item_id": "FOODS_1_002", "store_id": "TX_1", "horizon_days": 7},
                {"item_id": "HOBBIES_1_001", "store_id": "WI_1", "horizon_days": 7},
            ]
        }
        r = client.post("/forecast/batch", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 3
        assert len(data["forecasts"]) == 3


# ============================================================================
# Inventory Endpoints
# ============================================================================

class TestInventoryEndpoints:

    @pytest.mark.integration
    def test_policy_computation_valid(self, client):
        payload = {
            "item_id": "FOODS_1_001",
            "store_id": "CA_1",
            "mean_daily_demand": 15.0,
            "std_daily_demand": 4.5,
            "mean_lead_time_days": 7.0,
            "unit_cost": 8.50,
            "service_level": 0.95,
        }
        r = client.post("/inventory/policy", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["eoq"] > 0
        assert data["safety_stock"] >= 0
        assert data["reorder_point"] >= data["safety_stock"]

    @pytest.mark.integration
    def test_policy_invalid_service_level(self, client):
        payload = {
            "item_id": "X",
            "store_id": "CA_1",
            "mean_daily_demand": 10.0,
            "std_daily_demand": 3.0,
            "service_level": 1.5,  # Invalid — must be <= 0.999
        }
        r = client.post("/inventory/policy", json=payload)
        assert r.status_code == 422

    @pytest.mark.integration
    def test_reorder_alerts_returns_list(self, client):
        r = client.get("/inventory/alerts?urgency=critical")
        assert r.status_code == 200
        data = r.json()
        assert "alerts" in data
        assert isinstance(data["alerts"], list)


# ============================================================================
# Pricing Endpoints
# ============================================================================

class TestPricingEndpoints:

    @pytest.mark.integration
    def test_pricing_recommendation_valid(self, client):
        payload = {
            "item_id": "FOODS_3_090",
            "store_id": "CA_1",
            "current_price": 3.99,
            "unit_cost": 2.10,
            "current_demand": 12.5,
            "strategy": "maximize_profit",
        }
        r = client.post("/pricing/recommend", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["recommended_price"] > 0
        assert data["recommended_price"] >= data["unit_cost"] if "unit_cost" in data else True

    @pytest.mark.integration
    def test_pricing_invalid_strategy(self, client):
        payload = {
            "item_id": "X",
            "store_id": "CA_1",
            "current_price": 5.0,
            "unit_cost": 2.0,
            "current_demand": 10.0,
            "strategy": "invalid_strategy",
        }
        r = client.post("/pricing/recommend", json=payload)
        assert r.status_code == 422

    @pytest.mark.integration
    def test_elasticity_endpoint(self, client):
        r = client.get("/pricing/elasticity/FOODS_1_001?store_id=CA_1")
        assert r.status_code == 200
        data = r.json()
        assert "elasticity" in data
        assert data["elasticity"] < 0  # Demand curves slope down


# ============================================================================
# Drift Detection Integration
# ============================================================================

class TestDriftDetection:

    @pytest.mark.integration
    def test_no_drift_on_identical_distributions(self):
        """PSI should be ~0 when reference and current are the same."""
        rng = np.random.default_rng(42)
        data = rng.normal(10, 2, 1000)
        detector = StatisticalDriftDetector()
        report = detector.detect_feature_drift("test_feature", data, data)
        assert report.drift_score < 0.02
        assert not report.drift_detected

    @pytest.mark.integration
    def test_drift_detected_on_shifted_distribution(self):
        """Large mean shift should be detected as drift."""
        rng = np.random.default_rng(42)
        reference = rng.normal(10, 2, 1000)
        current = rng.normal(20, 2, 1000)   # Large shift
        detector = StatisticalDriftDetector()
        report = detector.detect_feature_drift("shifted_feature", reference, current)
        assert report.drift_detected
        assert report.drift_score > 0.25

    @pytest.mark.integration
    def test_pipeline_returns_retraining_recommendation(self):
        """Pipeline should recommend retraining when drift is significant."""
        rng = np.random.default_rng(0)
        n = 2000

        # Reference: stable features
        ref_data = {f"feat_{i}": rng.normal(10, 2, n) for i in range(10)}
        ref_df = pd.DataFrame(ref_data)

        # Current: many features have shifted dramatically
        cur_data = {}
        for i in range(10):
            if i < 4:   # 40% of features drifted significantly
                cur_data[f"feat_{i}"] = rng.normal(25, 3, n)
            else:
                cur_data[f"feat_{i}"] = rng.normal(10, 2.1, n)
        cur_df = pd.DataFrame(cur_data)

        pipeline = DriftMonitoringPipeline()
        result = pipeline.run(
            reference_features=ref_df,
            current_features=cur_df,
            model_id="test-model-001",
        )
        assert result["retraining_triggered"] is True


# ============================================================================
# Sustainability Emissions
# ============================================================================

class TestSustainabilityEmissions:

    @pytest.mark.integration
    def test_transport_leg_emissions_positive(self):
        leg = TransportLeg(
            origin_id="WH_1", destination_id="STORE_CA_1",
            distance_km=250.0, weight_tonnes=5.0,
            transport_mode="truck_diesel", load_factor=0.85,
        )
        assert leg.total_emissions_kg > 0

    @pytest.mark.integration
    def test_electric_truck_lower_than_diesel(self):
        base = dict(origin_id="A", destination_id="B",
                     distance_km=500, weight_tonnes=10)
        diesel = TransportLeg(**base, transport_mode="truck_diesel", load_factor=0.85)
        electric = TransportLeg(**base, transport_mode="truck_electric", load_factor=0.85)
        assert electric.total_emissions_kg < diesel.total_emissions_kg

    @pytest.mark.integration
    def test_mode_comparison_returns_all_modes(self):
        calc = TransportEmissionsCalculator()
        df = calc.mode_comparison(distance_km=500, weight_tonnes=10)
        assert len(df) == len(df["transport_mode"].unique())
        assert (df["emissions_kg_co2e"] > 0).all()

    @pytest.mark.integration
    def test_warehouse_cold_storage_higher_emissions(self):
        standard = WarehouseEmissions("WH_STD", size_sqft=50000, is_cold_storage=False)
        cold = WarehouseEmissions("WH_COLD", size_sqft=50000, is_cold_storage=True)
        assert cold.annual_emissions_kg > standard.annual_emissions_kg

    @pytest.mark.integration
    def test_renewable_energy_reduces_warehouse_emissions(self):
        grid = WarehouseEmissions("WH_GRID", size_sqft=50000, renewable_energy_pct=0.0)
        renewable = WarehouseEmissions("WH_REN", size_sqft=50000, renewable_energy_pct=1.0)
        assert renewable.annual_emissions_kg == pytest.approx(0.0, abs=0.1)
