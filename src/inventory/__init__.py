"""Inventory optimisation: ABC/XYZ classification, safety stock, service-level
scenario analysis and working-capital impact.

This module sits *downstream* of the forecasting platform. It consumes point
forecasts, backtest residuals and probabilistic quantile outputs, and converts
them into actionable replenishment recommendations with explicit cost/risk
trade-offs.

Data constraints
----------------
ABC classification and working-capital calculations require sell prices, which
are available only for CA_1 and CA_2 in the supplied M5 workbook (15.3%
coverage). XYZ classification and safety-stock computations work on unit sales
alone, so they cover all ten stores.
"""

from src.inventory.abc_xyz import ABCXYZClassifier
from src.inventory.safety_stock import SafetyStockCalculator
from src.inventory.service_level import ServiceLevelAnalyzer
from src.inventory.working_capital import WorkingCapitalAnalyzer

__all__ = [
    "ABCXYZClassifier",
    "SafetyStockCalculator",
    "ServiceLevelAnalyzer",
    "WorkingCapitalAnalyzer",
]
