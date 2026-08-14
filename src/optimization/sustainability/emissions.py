"""
Sustainability Analytics Module.

Estimates and optimizes the environmental footprint of supply chain operations:
  - Transportation emissions (by mode: truck, rail, air, sea)
  - Warehouse energy consumption
  - Packaging waste estimation
  - Carbon-cost joint optimization (Pareto: cost vs emissions)

Emission factors sourced from:
  - EPA Greenhouse Gas Emission Factors Hub (2023)
  - DEFRA Conversion Factors (UK government)
  - IPCC AR6 transport emission coefficients

Units: kg CO₂-equivalent per tonne-km (kgCO2e/t-km)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Emission Factors (kg CO₂e per tonne-km)
# ---------------------------------------------------------------------------

EMISSION_FACTORS: dict[str, float] = {
    "truck_diesel":    0.107,   # Heavy-goods vehicle, diesel
    "truck_electric":  0.027,   # HGV, electric (grid average US)
    "rail_diesel":     0.028,   # Freight rail, diesel
    "rail_electric":   0.009,   # Freight rail, electric
    "air_cargo":       0.602,   # Air freight
    "sea_container":   0.011,   # Container ship
    "sea_bulk":        0.007,   # Bulk carrier
    "van_diesel":      0.210,   # Last-mile delivery van
    "van_electric":    0.065,   # Last-mile EV van
}

# Warehouse energy: kg CO₂e per sq-ft per year (US grid average)
WAREHOUSE_ENERGY_FACTOR = 0.022   # Standard warehouse
WAREHOUSE_COLD_FACTOR   = 0.068   # Cold storage (refrigerated)

# Packaging: kg CO₂e per unit (average retail packaging)
PACKAGING_FACTOR = 0.045


@dataclass
class TransportLeg:
    """One segment of a shipment's journey."""
    origin_id: str
    destination_id: str
    distance_km: float
    weight_tonnes: float
    transport_mode: str = "truck_diesel"
    is_full_load: bool = False      # FTL vs LTL (affects emissions per unit)
    load_factor: float = 0.85       # Actual fill % of vehicle

    @property
    def emission_factor(self) -> float:
        base = EMISSION_FACTORS.get(self.transport_mode, 0.107)
        # Partially loaded vehicles have higher per-unit emissions
        return base / max(self.load_factor, 0.1)

    @property
    def total_emissions_kg(self) -> float:
        return self.emission_factor * self.weight_tonnes * self.distance_km


@dataclass
class WarehouseEmissions:
    """Annual emissions from a warehouse facility."""
    warehouse_id: str
    size_sqft: float
    is_cold_storage: bool = False
    renewable_energy_pct: float = 0.0   # % of energy from renewables

    @property
    def annual_emissions_kg(self) -> float:
        factor = WAREHOUSE_COLD_FACTOR if self.is_cold_storage else WAREHOUSE_ENERGY_FACTOR
        gross = factor * self.size_sqft
        return gross * (1 - self.renewable_energy_pct)

    @property
    def annual_emissions_tonnes(self) -> float:
        return self.annual_emissions_kg / 1000


@dataclass
class EmissionsSummary:
    """Consolidated emissions report for a supply chain network."""
    period_label: str
    transport_kg: float = 0.0
    warehouse_kg: float = 0.0
    packaging_kg: float = 0.0
    total_kg: float = field(init=False)

    def __post_init__(self):
        self.total_kg = self.transport_kg + self.warehouse_kg + self.packaging_kg

    @property
    def total_tonnes(self) -> float:
        return self.total_kg / 1000

    @property
    def breakdown_pct(self) -> dict[str, float]:
        t = max(self.total_kg, 1e-6)
        return {
            "transport": self.transport_kg / t * 100,
            "warehouse": self.warehouse_kg / t * 100,
            "packaging": self.packaging_kg / t * 100,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period_label,
            "transport_kg_co2e": round(self.transport_kg, 2),
            "warehouse_kg_co2e": round(self.warehouse_kg, 2),
            "packaging_kg_co2e": round(self.packaging_kg, 2),
            "total_kg_co2e": round(self.total_kg, 2),
            "total_tonnes_co2e": round(self.total_tonnes, 4),
            "breakdown_pct": {k: round(v, 1) for k, v in self.breakdown_pct.items()},
        }


class TransportEmissionsCalculator:
    """
    Calculates transport-related CO₂ emissions for supply chain routes.
    """

    def calculate_route(self, legs: list[TransportLeg]) -> float:
        """Total emissions for a multi-leg shipment (kg CO₂e)."""
        return sum(leg.total_emissions_kg for leg in legs)

    def calculate_batch(self, df_shipments: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized emissions calculation for a batch of shipments.

        Required columns: distance_km, weight_tonnes, transport_mode,
                          load_factor (optional, default 0.85)
        """
        df = df_shipments.copy()
        df["load_factor"] = df.get("load_factor", pd.Series(0.85, index=df.index))

        df["emission_factor"] = df["transport_mode"].map(EMISSION_FACTORS).fillna(0.107)
        df["effective_ef"] = df["emission_factor"] / df["load_factor"].clip(lower=0.1)
        df["emissions_kg"] = df["effective_ef"] * df["weight_tonnes"] * df["distance_km"]

        return df

    def mode_comparison(
        self,
        distance_km: float,
        weight_tonnes: float,
    ) -> pd.DataFrame:
        """
        Compare emissions and indicative cost across all transport modes
        for a given shipment. Used for mode-switching decisions.
        """
        # Indicative cost rates (USD per tonne-km) — approximate 2024 rates
        cost_rates = {
            "truck_diesel": 0.15,
            "truck_electric": 0.18,
            "rail_diesel": 0.05,
            "rail_electric": 0.04,
            "air_cargo": 1.50,
            "sea_container": 0.02,
            "van_diesel": 0.35,
            "van_electric": 0.30,
        }

        rows = []
        for mode, ef in EMISSION_FACTORS.items():
            emissions = ef * weight_tonnes * distance_km
            cost = cost_rates.get(mode, 0.10) * weight_tonnes * distance_km
            rows.append({
                "transport_mode": mode,
                "emissions_kg_co2e": round(emissions, 2),
                "indicative_cost_usd": round(cost, 2),
                "emissions_per_dollar": round(emissions / max(cost, 0.01), 3),
            })

        return pd.DataFrame(rows).sort_values("emissions_kg_co2e")


class SustainabilityOptimizer:
    """
    Joint cost-emissions optimization for transport mode selection.

    For each origin-destination pair, finds the Pareto-optimal set of
    transport mode configurations considering both cost and emissions.
    """

    def __init__(self, calculator: TransportEmissionsCalculator):
        self.calculator = calculator

    def optimize_mode(
        self,
        distance_km: float,
        weight_tonnes: float,
        max_transit_days: int = 7,
        carbon_price_per_tonne: float = 50.0,   # USD social cost of carbon
    ) -> dict[str, Any]:
        """
        Select optimal transport mode given time and cost constraints.

        carbon_price_per_tonne: Internal carbon price for shadow pricing.
        Higher value shifts recommendation toward greener modes.
        """
        # Transit time constraints (days per 1000 km)
        transit_days = {
            "truck_diesel": distance_km / 800,
            "truck_electric": distance_km / 700,
            "rail_diesel": distance_km / 500,
            "rail_electric": distance_km / 500,
            "air_cargo": distance_km / 8000 + 0.5,
            "sea_container": distance_km / 400,
            "van_diesel": distance_km / 400,
            "van_electric": distance_km / 350,
        }

        df = self.calculator.mode_comparison(distance_km, weight_tonnes)
        df["transit_days"] = df["transport_mode"].map(transit_days)
        df["feasible"] = df["transit_days"] <= max_transit_days

        # Add carbon cost
        df["carbon_cost_usd"] = df["emissions_kg_co2e"] / 1000 * carbon_price_per_tonne
        df["total_cost_with_carbon"] = df["indicative_cost_usd"] + df["carbon_cost_usd"]

        feasible = df[df["feasible"]].copy()
        if feasible.empty:
            feasible = df.copy()  # Relax constraint if no feasible option

        # Recommend: minimize total cost including carbon price
        recommended = feasible.loc[feasible["total_cost_with_carbon"].idxmin()]

        return {
            "recommended_mode": recommended["transport_mode"],
            "emissions_kg_co2e": recommended["emissions_kg_co2e"],
            "indicative_cost_usd": recommended["indicative_cost_usd"],
            "carbon_cost_usd": recommended["carbon_cost_usd"],
            "transit_days": recommended["transit_days"],
            "all_modes": df.to_dict("records"),
        }

    def network_emissions_report(
        self,
        df_shipments: pd.DataFrame,
        df_warehouses: pd.DataFrame,
        df_sales: pd.DataFrame,
        period_label: str = "annual",
    ) -> EmissionsSummary:
        """
        Full network-level emissions report.

        Args:
            df_shipments: Columns: distance_km, weight_tonnes, transport_mode
            df_warehouses: Columns: size_sqft, is_cold_storage, renewable_energy_pct
            df_sales: Columns: sales (units sold — for packaging calc)
        """
        # Transport
        df_ship_calc = self.calculator.calculate_batch(df_shipments)
        transport_kg = float(df_ship_calc["emissions_kg"].sum())

        # Warehouse
        warehouse_kg = 0.0
        for _, wh in df_warehouses.iterrows():
            whe = WarehouseEmissions(
                warehouse_id=str(wh.get("warehouse_id", "")),
                size_sqft=float(wh.get("size_sqft", 0)),
                is_cold_storage=bool(wh.get("is_cold_storage", False)),
                renewable_energy_pct=float(wh.get("renewable_energy_pct", 0.0)),
            )
            warehouse_kg += whe.annual_emissions_kg

        # Packaging
        total_units = float(df_sales["sales"].sum()) if "sales" in df_sales.columns else 0
        packaging_kg = total_units * PACKAGING_FACTOR

        summary = EmissionsSummary(
            period_label=period_label,
            transport_kg=transport_kg,
            warehouse_kg=warehouse_kg,
            packaging_kg=packaging_kg,
        )

        log.info(
            "emissions_report_generated",
            period=period_label,
            total_tonnes=round(summary.total_tonnes, 2),
            transport_pct=round(summary.breakdown_pct["transport"], 1),
        )
        return summary

    def reduction_scenarios(
        self,
        baseline: EmissionsSummary,
    ) -> pd.DataFrame:
        """
        Model emissions reduction scenarios and their estimated impact.
        """
        rows = [
            {
                "scenario": "Baseline",
                "description": "Current operations",
                "emissions_kg": baseline.total_kg,
                "reduction_pct": 0.0,
                "cost_impact_pct": 0.0,
            },
            {
                "scenario": "Modal shift (30% truck → rail)",
                "description": "Move 30% of truck freight to rail",
                "emissions_kg": baseline.total_kg * 0.87,
                "reduction_pct": 13.0,
                "cost_impact_pct": -3.0,   # Rail is cheaper
            },
            {
                "scenario": "Electric last-mile fleet",
                "description": "Replace diesel vans with EVs for last-mile",
                "emissions_kg": baseline.total_kg * 0.93,
                "reduction_pct": 7.0,
                "cost_impact_pct": 2.5,
            },
            {
                "scenario": "100% renewable warehouse energy",
                "description": "Switch all warehouse electricity to renewables",
                "emissions_kg": (baseline.total_kg
                                  - baseline.warehouse_kg * 0.85),
                "reduction_pct": baseline.warehouse_kg * 0.85 / baseline.total_kg * 100,
                "cost_impact_pct": 4.0,
            },
            {
                "scenario": "Consolidated deliveries (+load factor to 95%)",
                "description": "Optimize routing to increase vehicle utilisation",
                "emissions_kg": baseline.total_kg * 0.91,
                "reduction_pct": 9.0,
                "cost_impact_pct": -5.0,   # Fewer trips = lower cost
            },
            {
                "scenario": "Combined: Rail + EV + Renewables",
                "description": "All three initiatives combined",
                "emissions_kg": baseline.total_kg * 0.72,
                "reduction_pct": 28.0,
                "cost_impact_pct": 3.0,
            },
        ]

        df = pd.DataFrame(rows)
        df["emissions_tonnes"] = (df["emissions_kg"] / 1000).round(1)
        df["reduction_kg"] = (baseline.total_kg - df["emissions_kg"]).round(0)
        return df
